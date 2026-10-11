"""Predeclared finite-difference and scalar mechanisms for the distance loss.

Run with ``python -m iclr_algorithm1_experiments.projection_distance_mechanisms``.
The formal hydro diagnostic reuses the existing 32-path mechanism setup; it is
not a training-performance experiment. No existing experiment files are edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from algorithm1_gradient_study.experiments import _base_config, _dataset_fingerprint
from algorithm1_gradient_study.models import build_policy, initialize_channel_policy
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.simulator import simulate

from .config import ICLRProtocol
from .projection_distance import (
    estimate_lambda_ref,
    projection_distance,
    regularized_objective,
)


DEFAULT_OUTPUT = Path(__file__).parent / "outputs/projection_distance_v1/mechanisms"
GRADIENT_MODES = ("full", "no_moving_recourse")
SCALAR_ARMS = ("zero", "fixed_one", "decay")
SCALAR_LABELS = {
    "zero": r"$\lambda=0$",
    "fixed_one": r"$\lambda=1$",
    "decay": r"$\lambda_k=(k+1)^{-1/2}$",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mechanism_protocol(protocol: ICLRProtocol, n_directions: int = 3) -> dict:
    """Specify settings before data generation or objective evaluation."""
    if n_directions < 3:
        raise ValueError("The distance diagnostic requires at least three directions.")
    study = protocol.study_protocol()
    return {
        "version": 1,
        "study_protocol": asdict(study),
        "hydro_config": asdict(_base_config(study)),
        "dtype": "torch.float64",
        "torch_threads": 1,
        "lambda_arms": ["zero", "lambda_ref"],
        "lambda_ref": {
            "data": "same 32 fitting diagnostic paths (4 in smoke)",
            "formula": "median(per-path ||grad_theta C||) / (median(per-path ||grad_theta d_pd||) + 1e-12)",
            "frozen_during_parameter_perturbations": True,
            "normalization": "half squared distance summed over all stages and reservoirs",
        },
        "n_directions": n_directions,
        "direction_seed_formula": "20000 + train_seed + 1000 * direction_index",
        "policy_initialization": "torch.manual_seed(train_seed); build_policy(mlp); double; initialize_channel_policy(10000+train_seed)",
        "gradient_modes": list(GRADIENT_MODES),
        "forward_equality_tolerance": 0.0,
        "finite_difference_error": "abs(autograd-fd)/max(1,abs(autograd),abs(fd))",
        "primary_fd_summary_step": study.finite_difference_steps[-1],
        "scalar": {
            "feasible_set": [0.0, 1.0],
            "operating_cost": "(clip(u,0,1)-1)^2",
            "projection_distance": "0.5*(u-clip(u,0,1))^2",
            "u_initial": -1.0,
            "arms": list(SCALAR_ARMS),
            "updates": 200,
            "learning_rate": 0.1,
            "primary_optimizer": "SGD (no momentum or weight decay)",
            "optimizer_sensitivity": "Adam (betas=(0.9,0.999), eps=1e-8, no weight decay)",
            "projection": "torch.clamp(u, min=0, max=1)",
            "boundary_derivative": "PyTorch clamp derivative is 1 at u=0 and u=1; 0 outside; 1 inside",
            "interpretation": "For u<0 plain SGD gives u_next=(1-eta*lambda_k)*u; eta*lambda_k<1 implies no finite-step boundary crossing in exact arithmetic.",
        },
        "data_access": "fitting diagnostic data only; no validation/test/calibration/evaluation paths",
    }


def write_preregistration(output: Path, settings: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    encoded = json.dumps(settings, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("Refusing to overwrite a different predeclared mechanism protocol.")
    if not path.exists():
        path.write_text(encoded)
    source_root = Path(__file__).resolve().parents[1]
    source_names = (
        "iclr_algorithm1_experiments/projection_distance_mechanisms.py",
        "iclr_algorithm1_experiments/projection_distance.py",
        "iclr_algorithm1_experiments/config.py",
        "algorithm1_gradient_study/models.py",
        "algorithm1_gradient_study/experiments.py",
        "efficiency_switching_hydro/simulator.py",
        "efficiency_switching_hydro/projection.py",
        "efficiency_switching_hydro/data.py",
    )
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "protocol_sha256": _sha256(path),
        "source_sha256": {name: _sha256(source_root / name) for name in source_names},
    }
    provenance_path = output / "PROVENANCE.json"
    if provenance_path.exists():
        prior = json.loads(provenance_path.read_text())
        if prior["source_sha256"] != provenance["source_sha256"]:
            raise RuntimeError("Source changed since this run was preregistered; use a fresh directory.")
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def _gradient_vector(policy, loss: torch.Tensor) -> torch.Tensor:
    parameters = tuple(p for p in policy.parameters() if p.requires_grad)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]).detach()


def regularized_finite_difference_rows(
    policy, dataset, hydro, steps, lambda_value: float, direction_seeds: list[int],
    device: str = "cpu",
) -> list[dict]:
    """Evaluate identical forwards and deliberately different backward graphs."""
    parameters = tuple(p for p in policy.parameters() if p.requires_grad)
    base = parameters_to_vector(parameters).detach().clone()
    gradients, losses = {}, {}
    reference = None
    max_forward_difference = 0.0
    for mode in GRADIENT_MODES:
        trace = simulate(policy, dataset, hydro, device=device, dtype=torch.float64, gradient_mode=mode)
        loss = regularized_objective(trace, lambda_value)
        current = (trace.total_cost.detach(), trace.actions.detach(), trace.proposals.detach())
        if reference is None:
            reference = tuple(value.clone() for value in current)
        else:
            for value, expected in zip(current, reference):
                max_forward_difference = max(max_forward_difference, float((value - expected).abs().max()))
                torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)
        losses[mode] = float(loss.detach())
        gradients[mode] = _gradient_vector(policy, loss)
    if losses["full"] != losses["no_moving_recourse"]:
        raise RuntimeError("Backward ablation unexpectedly changed the forward objective.")
    rows = []
    try:
        for direction_index, direction_seed in enumerate(direction_seeds):
            generator = torch.Generator(device=base.device).manual_seed(direction_seed)
            direction = torch.randn(base.shape, generator=generator, dtype=base.dtype, device=base.device)
            direction /= torch.linalg.vector_norm(direction).clamp(min=1e-12)
            for step in steps:
                with torch.no_grad():
                    vector_to_parameters(base + step * direction, parameters)
                    plus = float(regularized_objective(
                        simulate(policy, dataset, hydro, device=device, dtype=torch.float64), lambda_value))
                    vector_to_parameters(base - step * direction, parameters)
                    minus = float(regularized_objective(
                        simulate(policy, dataset, hydro, device=device, dtype=torch.float64), lambda_value))
                finite_difference = (plus - minus) / (2 * step)
                for mode in GRADIENT_MODES:
                    derivative = float(gradients[mode] @ direction)
                    error = abs(derivative - finite_difference)
                    rows.append({
                        "direction_index": direction_index,
                        "direction_seed": direction_seed,
                        "gradient_mode": mode,
                        "lambda_value": lambda_value,
                        "step": step,
                        "base_objective": losses[mode],
                        "autograd_directional_derivative": derivative,
                        "finite_difference_directional_derivative": finite_difference,
                        "absolute_error": error,
                        "relative_error": error / max(1.0, abs(derivative), abs(finite_difference)),
                        "gradient_norm": float(torch.linalg.vector_norm(gradients[mode])),
                        "max_forward_difference": max_forward_difference,
                    })
    finally:
        with torch.no_grad():
            vector_to_parameters(base, parameters)
    torch.testing.assert_close(parameters_to_vector(parameters), base, rtol=0.0, atol=0.0)
    return rows


def run_finite_difference(protocol: ICLRProtocol, output: Path, n_directions: int) -> pd.DataFrame:
    study = protocol.study_protocol()
    hydro = _base_config(study)
    dataset = generate_dataset("fit_gradient_diagnostic", min(study.gradient_paths, study.n_fit), study.fit_seed, hydro)
    (output / "dataset.json").write_text(json.dumps({
        "name": dataset.name, "seed": dataset.seed, "n_paths": dataset.n,
        "sha256": _dataset_fingerprint(dataset),
    }, indent=2) + "\n")
    rows, estimates, per_path = [], [], []
    for seed in study.train_seeds:
        print(f"Finite difference: seed {seed}, fitting lambda_ref", flush=True)
        torch.manual_seed(seed)
        policy = build_policy("mlp", hydro, study).to(device=study.device, dtype=torch.float64)
        initialize_channel_policy(policy, 10_000 + seed)
        estimate = estimate_lambda_ref(policy, dataset, hydro, device=study.device)
        estimates.append({"train_seed": seed, **{
            key: value for key, value in estimate.items()
            if key not in {"cost_gradient_norms", "distance_gradient_norms"}
        }})
        for path_index, (cost_norm, distance_norm) in enumerate(zip(
            estimate["cost_gradient_norms"], estimate["distance_gradient_norms"]
        )):
            per_path.append({"train_seed": seed, "fit_path": path_index,
                             "cost_gradient_norm": cost_norm, "distance_gradient_norm": distance_norm})
        with torch.no_grad():
            trace = simulate(policy, dataset, hydro, device=study.device, dtype=torch.float64)
            diagnostics = {
                "base_operating_cost": float(trace.mean_cost),
                "base_projection_distance": float(projection_distance(trace).mean()),
                "max_action_violation": float(trace.max_action_violation),
                "max_storage_violation": float(trace.max_storage_violation),
            }
        for arm, lambda_value in (("zero", 0.0), ("lambda_ref", estimate["lambda_ref"])):
            print(f"Finite difference: seed {seed}, arm {arm}, lambda={lambda_value:.8g}", flush=True)
            seed_rows = regularized_finite_difference_rows(
                policy, dataset, hydro, study.finite_difference_steps, lambda_value,
                [20_000 + seed + 1000 * index for index in range(n_directions)], study.device,
            )
            for row in seed_rows:
                row.update({"train_seed": seed, "lambda_arm": arm,
                            "n_fit_diagnostic_paths": dataset.n, **diagnostics})
            rows.extend(seed_rows)
        pd.DataFrame(rows).to_csv(output / "finite_difference_check.csv", index=False)
        pd.DataFrame(estimates).to_csv(output / "lambda_reference.csv", index=False)
        pd.DataFrame(per_path).to_csv(output / "lambda_reference_per_path.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["lambda_arm", "gradient_mode", "step"], sort=False).agg(
        n_seeds=("train_seed", "nunique"), n_directions_total=("direction_seed", "count"),
        relative_error_mean=("relative_error", "mean"),
        relative_error_median=("relative_error", "median"),
        relative_error_max=("relative_error", "max"),
        max_forward_difference=("max_forward_difference", "max"),
        max_action_violation=("max_action_violation", "max"),
        max_storage_violation=("max_storage_violation", "max"),
    ).reset_index()
    summary.to_csv(output / "finite_difference_summary.csv", index=False)
    _plot_finite_difference(summary, output)
    _compare_original_zero(frame, output, protocol.smoke)
    return frame


def _compare_original_zero(frame: pd.DataFrame, output: Path, smoke: bool) -> None:
    original_path = Path(__file__).parent / "outputs/formal_protocol_v1/01_mechanisms/01_finite_difference/finite_difference_check.csv"
    if smoke or not original_path.exists():
        return
    original = pd.read_csv(original_path)
    current = frame[(frame.lambda_arm == "zero") & (frame.direction_index == 0)]
    joined = current.merge(original, on=["train_seed", "direction_seed", "gradient_mode", "step"], suffixes=("_new", "_original"), validate="one_to_one")
    metrics = ("autograd_directional_derivative", "finite_difference_directional_derivative", "gradient_norm")
    comparisons = [{
        "metric": metric,
        "n_rows": len(joined),
        "maximum_absolute_difference": float((joined[f"{metric}_new"] - joined[f"{metric}_original"]).abs().max()),
        "matches_rtol_1e-7_atol_1e-5": bool(np.allclose(joined[f"{metric}_new"], joined[f"{metric}_original"], rtol=1e-7, atol=1e-5)),
    } for metric in metrics]
    pd.DataFrame(comparisons).to_csv(output / "original_zero_reproduction.csv", index=False)


def scalar_lambda(arm: str, update: int) -> float:
    if arm == "zero":
        return 0.0
    if arm == "fixed_one":
        return 1.0
    if arm == "decay":
        return (update + 1) ** -0.5
    raise ValueError(arm)


def scalar_path(arm: str, optimizer_name: str, updates: int = 200, learning_rate: float = 0.1) -> list[dict]:
    """Include k=0 before training and k=updates after the last actual update."""
    u = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD([u], lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam([u], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError(optimizer_name)
    rows = []
    for update in range(updates + 1):
        optimizer.zero_grad(set_to_none=True)
        x = u.clamp(min=0.0, max=1.0)
        cost = (x - 1).square()
        distance = 0.5 * (u - x).square()
        lambda_value = scalar_lambda(arm, update)
        objective = cost + lambda_value * distance
        objective.backward()
        rows.append({
            "optimizer": optimizer_name, "lambda_arm": arm,
            "update": update, "lambda_value": lambda_value,
            "u": float(u.detach()), "x": float(x.detach()),
            "operating_cost": float(cost.detach()),
            "projection_distance": float(distance.detach()),
            "objective": float(objective.detach()), "gradient": float(u.grad),
            "active_projection": int(float(u.detach()) < 0 or float(u.detach()) > 1),
            "learning_rate": learning_rate,
        })
        if update < updates:
            optimizer.step()
    return rows


def run_scalar(output: Path) -> pd.DataFrame:
    rows = [row for optimizer in ("SGD", "Adam") for arm in SCALAR_ARMS for row in scalar_path(arm, optimizer)]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "scalar_trajectories.csv", index=False)
    summaries = []
    for (optimizer, arm), group in frame.groupby(["optimizer", "lambda_arm"], sort=False):
        final = group.iloc[-1]
        positive = group[group.x > 0]
        summaries.append({
            "optimizer": optimizer, "lambda_arm": arm,
            "first_positive_action_update": int(positive.iloc[0]["update"]) if len(positive) else None,
            "final_u": final.u, "final_x": final.x,
            "final_operating_cost": final.operating_cost,
            "final_projection_distance": final.projection_distance,
            "final_gradient": final.gradient,
        })
    pd.DataFrame(summaries).to_csv(output / "scalar_summary.csv", index=False)
    locations = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(locations.clamp(0, 1).sum(), locations)[0]
    pd.DataFrame({"u": locations.detach().numpy(), "clamp_derivative": derivative.numpy()}).to_csv(output / "scalar_boundary_derivative.csv", index=False)
    _plot_scalar(frame, output)
    return frame


def _plot_finite_difference(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    for ax, arm in zip(axes, ("zero", "lambda_ref")):
        for mode, label, marker in (("full", "Complete derivative", "o"), ("no_moving_recourse", "Without moving recourse", "s")):
            selected = summary[(summary.lambda_arm == arm) & (summary.gradient_mode == mode)].sort_values("step")
            ax.loglog(selected.step, selected.relative_error_mean, marker=marker, label=label)
        ax.set_title(r"$\lambda=0$" if arm == "zero" else r"$\lambda=\lambda_{\mathrm{ref}}$")
        ax.set_xlabel("Central finite-difference step")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Mean normalized directional error")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(output / f"finite_difference.{extension}", dpi=180)
    plt.close(fig)


def _plot_scalar(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), sharex=True)
    metrics = (("u", "Proposal $u_k$"), ("x", "Action $x_k$"), ("operating_cost", "Operating cost"), ("projection_distance", "Projection distance"))
    for row, optimizer in enumerate(("SGD", "Adam")):
        for column, (metric, title) in enumerate(metrics):
            ax = axes[row, column]
            for arm in SCALAR_ARMS:
                selected = frame[(frame.optimizer == optimizer) & (frame.lambda_arm == arm)]
                ax.plot(selected["update"], selected[metric], label=SCALAR_LABELS[arm])
            ax.set_title(f"{optimizer}: {title}")
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("Completed updates")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(output / f"scalar_plateau.{extension}", dpi=180)
    plt.close(fig)


def write_report(output: Path, elapsed: float) -> None:
    summary = pd.read_csv(output / "finite_difference_summary.csv")
    scalar = pd.read_csv(output / "scalar_summary.csv")
    finest = summary[summary.step == summary.step.min()]
    lines = [
        "# Projection-distance mechanism experiments", "",
        "These diagnostics use fitting paths only; they do not establish held-out training gains.", "",
        "The hydro check preserves the original mechanism study's 32 paths, five initializations, 12 reservoirs, 48 stages, stress-ramp fraction 0.10 and seven central-difference steps. Three random directions are used per initialization. The first direction reproduces the prior diagnostic. Lambda is estimated at initialization on the diagnostic fitting paths and held fixed under parameter perturbations. Both proposed and projected actions remain in the gradient graph.", "",
        "## Directional derivative at the predeclared smallest step", "",
        "| Lambda | Backward graph | Mean normalized error | Maximum error |", "|---|---|---:|---:|",
    ]
    for row in finest.itertuples():
        lines.append(f"| {row.lambda_arm} | {row.gradient_mode} | {row.relative_error_mean:.6g} | {row.relative_error_max:.6g} |")
    lines += ["", "Every full/ablated forward action, proposal, and path cost was compared exactly. The auxiliary loss was included in both autodiff and finite differences. All seven step sizes and all directions are retained in finite_difference_check.csv; the summary does not select each direction's best step.", "",
              "## Scalar plateau", "", "| Optimizer | Lambda | First x > 0 update | Final u | Final cost | Final distance |", "|---|---|---:|---:|---:|---:|"]
    for row in scalar.itertuples():
        crossing = "none" if pd.isna(row.first_positive_action_update) else str(int(row.first_positive_action_update))
        lines.append(f"| {row.optimizer} | {row.lambda_arm} | {crossing} | {row.final_u:.8g} | {row.final_operating_cost:.8g} | {row.final_projection_distance:.8g} |")
    lines += ["", "Both optimizers use 200 updates and learning rate 0.1, declared before execution. SGD is the primary illustration; Adam is a separately labeled optimizer sensitivity. No clipping of the parameter or artificial boundary crossing is introduced.", "",
              "For u < 0, x = 0 and the cost gradient is zero. The distance gradient is u, so plain SGD gives u_(k+1) = (1 - 0.1 lambda_k) u_k. Both positive-lambda schedules therefore approach the feasible boundary while remaining negative at every finite exact-arithmetic iteration. This illustrates a nonzero corrective gradient, not a universal escape from the plateau. Adam's moment and normalization dynamics can cross the boundary; its outcome must not be attributed to the regularizer independently of the optimizer.", "",
              "Projection uses torch.clamp: the selected derivative is 1 at exactly 0 or 1. The measured boundary derivatives are saved separately. SGD never reaches those exact boundaries in this run.", "",
              "## Artifacts", "", "- PROTOCOL.json and PROVENANCE.json: predeclared settings, source hashes, software versions.",
              "- finite_difference_check.csv / finite_difference_summary.csv: all finite differences and fixed-step summaries.",
              "- original_zero_reproduction.csv: agreement of the first direction with the original formal lambda-zero diagnostic.",
              "- lambda_reference.csv / lambda_reference_per_path.csv: fitting-only lambda estimates and their gradient-norm inputs.",
              "- scalar_trajectories.csv / scalar_summary.csv: complete scalar trajectories and endpoints.",
              "- finite_difference.pdf and scalar_plateau.pdf: standalone figures (PNG copies included).", "",
              f"Total elapsed wall time: {elapsed:.2f} seconds (including lambda estimation, diagnostics and figures).", ""]
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = ICLRProtocol.smoke_protocol() if args.smoke else ICLRProtocol()
    settings = mechanism_protocol(protocol)
    write_preregistration(args.output, settings)
    if (args.output / "COMPLETED.json").exists():
        print(f"Completed mechanism artifacts already exist: {args.output}", flush=True)
        return
    started = time.perf_counter()
    run_scalar(args.output)
    run_finite_difference(protocol, args.output, settings["n_directions"])
    elapsed = time.perf_counter() - started
    write_report(args.output, elapsed)
    artifacts = {str(path.relative_to(args.output)): _sha256(path)
                 for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "COMPLETED.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed, "artifacts_sha256": artifacts,
    }, indent=2) + "\n")
    print(f"Completed mechanism experiments: {args.output}", flush=True)


if __name__ == "__main__":
    main()
