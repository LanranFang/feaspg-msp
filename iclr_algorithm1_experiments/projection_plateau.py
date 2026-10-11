"""Separate, explicitly constructed drought/initialization mechanism experiment.

Reuses the original actor, moving-recourse simulator, loss, and Adam optimizer.
Does not modify the paper or the frozen nominal benchmark. Test evaluation is
a separate command, after checkpoints and the protocol have been frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.simulator import simulate
from efficiency_switching_hydro.train import unused_batches
from .config import ICLRProtocol
from .io import dataset_fingerprint
from .projection_distance import regularized_objective, trace_metrics
from .projection_distance_run import (
    configure, gradient_diagnostic, initial_policy, save_json, sha256, state_hash,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "iclr_algorithm1_experiments/outputs/projection_plateau_v1"
SETTINGS = {
    "drought_01": (0.01, 0.01),
    "drought_02": (0.02, 0.01),
    "drought_05": (0.05, 0.01),
    "drought_10": (0.10, 0.01),
    "nominal": (1.0, 0.64),
}
ARMS = ("zero", "fixed_one", "decay_one")


def hydro_config(setting):
    q, initial_fraction = SETTINGS[setting]
    h = ICLRProtocol().hydro_config()
    return replace(
        h,
        initial_storage=tuple(initial_fraction * np.asarray(h.storage_capacity)),
        base_inflow=tuple(q * np.asarray(h.base_inflow)),
        inflow_amplitude=tuple(q * np.asarray(h.inflow_amplitude)),
        factor_loadings=tuple(map(tuple, q * np.asarray(h.factor_loadings))),
        idiosyncratic_low=q * h.idiosyncratic_low,
        idiosyncratic_high=q * h.idiosyncratic_high,
    )


def weight(arm, update):
    if arm == "zero":
        return 0.0
    if arm == "fixed_one":
        return 1.0
    if arm == "decay_one":
        return 1.0 / math.sqrt(update)
    raise ValueError(arm)


def fitting_data(setting):
    p = ICLRProtocol()
    h = hydro_config(setting)
    pool = generate_dataset("fit_pool", 2500, p.budget_seeds.fit, h)
    return h, pool.take(np.arange(1000), "fit_n1000")


def support_certificate(h):
    """Sufficient all-shock-support certificate for x_t=s_t at initialization.

    Inductively assume all water is released. Storage at the next stage is
    inflow plus the delayed routed release. Nonnegative routing preserves the
    coordinatewise upper bound. Three positive margins verify that assumption:
    proposal > storage, ramp > storage, shared capacity > weighted storage.
    If any fails, this calculation is inconclusive, not a certificate.
    """
    upper_inflow = (h.inflow_mean()
                    + 4 * h.factor_std * np.abs(h.factor_loadings).sum(axis=1)
                    + h.idiosyncratic_high)
    s = np.asarray(h.initial_storage).copy()
    queue = np.zeros((h.routing_delay, h.n_reservoirs))
    rows = []
    for t in range(h.T):
        rows.append({
            "stage": t,
            "proposal_storage_margin": float((h.reference_release()[t] - s).min()),
            "ramp_storage_margin": float((h.ramp_fraction * np.asarray(h.release_capacity) - s).min()),
            "shared_capacity_margin": float(h.shared_release_capacity - s @ np.asarray(h.generation_coefficient)),
        })
        arrival = queue[0].copy()
        queue = np.concatenate((queue[1:], (s @ np.asarray(h.routing))[None, :]))
        s = upper_inflow[t] + arrival
    margins = {key: min(row[key] for row in rows) for key in rows[0] if key != "stage"}
    return {"certified": all(v > 0 for v in margins.values()),
            "minimum_margins": margins, "per_stage": rows,
            "claim_if_certified": "All proposals strictly exceed all-water actions, for every shock in the generator support; full cost gradient is zero at original initialization."}


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "protocol.json"
    if destination.exists():
        return
    p = ICLRProtocol()
    sources = [Path(__file__), ROOT / "efficiency_switching_hydro/simulator.py",
               ROOT / "efficiency_switching_hydro/projection.py",
               ROOT / "efficiency_switching_hydro/policies.py",
               ROOT / "iclr_algorithm1_experiments/projection_distance.py"]
    save_json(destination, {
        "role": "Constructed projection-plateau stress test, separate from nominal benchmark",
        "primary_setting": "drought_02: analytically certified before training outcomes",
        "settings": {k: {"inflow_multiplier": q, "initial_storage_fraction": f}
                     for k, (q, f) in SETTINGS.items()},
        "screening_disclosure": "Before training, inspected first 32 fitting-seed paths at inflow multipliers .01,.02,.05,.10 with initial storage .01 capacity. Cost-gradient norms: 0,0,0,6501.60. All four settings retained. Support proof succeeds for .01,.02; .02 chosen as the less severe certified case. No held-out outcomes inspected for this choice.",
        "protocol": asdict(p), "policy": "projected_mlp_matched",
        "fitting": "Original 2500 pool prefix 1000; 500 updates of two previously unused paths",
        "learning_rate": 6e-4, "learning_rate_selection": "Fixed in advance, highest original MLP candidate; identical for all arms",
        "arms": {a: "0" if a == "zero" else "1" if a == "fixed_one" else "1/sqrt(k+1), k starts at zero" for a in ARMS},
        "lambda_scale": "Absolute positive weight 1, as in existing scalar mechanism. No gradient-ratio calibration: its numerator is exactly zero on a certified plateau, so it would produce zero weight.",
        "checkpoint_selection": "Lowest validation operating cost, strict <, initialization eligible, checked every 10 updates",
        "validation": {"n": 500, "seed": 35202},
        "test": {"n": 4096, "seed": 35303, "role": "Generated only by evaluate after training checkpoints are frozen; no test-based selection"},
        "seeds": list(range(5)),
        "success_criteria": "Zero arm unchanged actions/cost with zero cost gradient; positive arm changes actions and lowers independently held-out operating cost. Report magnitude and all five seeds, even if weak or unsuccessful.",
        "derivative": "Full gradients through both proposal and action, states and moving recourse; original Adam weight decay 1e-6 and gradient clipping 10",
        "source_sha256": {str(f.relative_to(ROOT)): sha256(f) for f in sources},
        "artifact_note": "Paper-file hashes omitted from anonymous code artifact; experimental protocol unchanged.",
        "created_unix_time": time.time(),
    })
    for setting in SETTINGS:
        save_json(output / "support" / f"{setting}.json", support_certificate(hydro_config(setting)))


@torch.no_grad()
def evaluate_dataset(actor, dataset, h):
    actor.eval()
    tr = simulate(actor, dataset, h)
    metrics = trace_metrics(tr)
    costs = tr.total_cost.cpu().numpy().astype(np.float64)
    metrics["mean_path_cost_float64"] = float(costs.mean())
    return metrics, costs


def fit_one(task):
    setting, seed, arm, output = task
    configure()
    output = Path(output)
    out = output / "runs" / setting / f"seed{seed}" / arm
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    p = ICLRProtocol()
    h, fit = fitting_data(setting)
    validation = generate_dataset("validation", 500, 35202, h)
    monitor = fit.take(np.arange(32), "fixed_fit_monitor")
    actor = initial_policy(p, h, seed, "projected_mlp_matched")
    initial_hash = state_hash(actor.state_dict())
    initial_diagnostic = gradient_diagnostic(actor, monitor, h, "cpu", weight(arm, 1))
    with torch.no_grad():
        initial_trace = simulate(actor, monitor, h)
        initial_actions = initial_trace.actions.clone()
        initial_costs = initial_trace.total_cost.clone()
    best_state = copy.deepcopy(actor.state_dict())
    initial_validation, _ = evaluate_dataset(actor, validation, h)
    best_validation = initial_validation["mean_cost"]
    selected_update = 0
    history = [{"update": 0, **initial_validation, **initial_diagnostic,
                "max_monitor_action_change": 0.0, "max_monitor_cost_change": 0.0}]
    batches = unused_batches(1000, 500, 2, np.random.default_rng(seed))
    optimizer = torch.optim.Adam(actor.parameters(), lr=6e-4, weight_decay=p.weight_decay)
    train_rows = []
    t0 = time.perf_counter()
    for update in range(1, 501):
        actor.train()
        batch = fit.take(batches[update - 1])
        optimizer.zero_grad(set_to_none=True)
        tr = simulate(actor, batch, h, gradient_mode="full")
        lam = weight(arm, update)
        objective = regularized_objective(tr, lam)
        objective.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), p.gradient_clip))
        optimizer.step()
        train_rows.append({"update": update, "lambda": lam,
                           "cost": float(tr.mean_cost.detach()),
                           "objective": float(objective.detach()),
                           "gradient_norm_preclip": norm})
        if update % 10 == 0:
            metrics, _ = evaluate_dataset(actor, validation, h)
            if metrics["mean_cost"] < best_validation:
                best_validation = metrics["mean_cost"]
                best_state = copy.deepcopy(actor.state_dict())
                selected_update = update
            with torch.no_grad():
                mtr = simulate(actor, monitor, h)
                action_change = float((mtr.actions - initial_actions).abs().max())
                cost_change = float((mtr.total_cost - initial_costs).abs().max())
            row = {"update": update, **metrics,
                   "max_monitor_action_change": action_change,
                   "max_monitor_cost_change": cost_change}
            if update % 50 == 0:
                row.update(gradient_diagnostic(actor, monitor, h, "cpu", lam))
            history.append(row)
    seconds = time.perf_counter() - t0
    final_hash = state_hash(actor.state_dict())
    torch.save(actor.state_dict(), out / "final.pt")
    actor.load_state_dict(best_state)
    selected_diagnostic = gradient_diagnostic(actor, monitor, h, "cpu", weight(arm, max(1, selected_update)))
    selected_metrics, _ = evaluate_dataset(actor, validation, h)
    result = {
        "setting": setting, "seed": seed, "arm": arm,
        "selected_update": selected_update, "iterations": 500,
        "initial_validation": initial_validation,
        "selected_validation": selected_metrics,
        "initial_diagnostic": initial_diagnostic,
        "selected_diagnostic": selected_diagnostic,
        "runtime_seconds_including_monitoring": seconds,
        "initial_state_sha256": initial_hash,
        "selected_state_sha256": state_hash(best_state), "final_state_sha256": final_hash,
        "fit_sha256": dataset_fingerprint(fit),
        "validation_sha256": dataset_fingerprint(validation),
        "batches_sha256": hashlib.sha256(batches.tobytes()).hexdigest(),
        "max_training_gradient_norm": max(r["gradient_norm_preclip"] for r in train_rows),
        "first_observed_action_change_update": next((r["update"] for r in history if r["max_monitor_action_change"] > 1e-6), None),
        "max_observed_action_change": max(r["max_monitor_action_change"] for r in history),
        "max_observed_cost_change": max(r["max_monitor_cost_change"] for r in history),
    }
    torch.save(best_state, out / "selected.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(train_rows).to_csv(out / "training.csv", index=False)
    save_json(out / "result.json", result)
    return result


def evaluate_one(task):
    setting, seed, arm, output = task
    configure()
    out = Path(output) / "runs" / setting / f"seed{seed}" / arm
    if (out / "test.json").exists():
        return json.loads((out / "test.json").read_text())
    result = json.loads((out / "result.json").read_text())
    h = hydro_config(setting)
    p = ICLRProtocol()
    actor = initial_policy(p, h, seed, "projected_mlp_matched")
    test = generate_dataset("independent_test", 4096, 35303, h)
    initial, initial_costs = evaluate_dataset(actor, test, h)
    actor.load_state_dict(torch.load(out / "selected.pt", weights_only=True))
    assert state_hash(actor.state_dict()) == result["selected_state_sha256"]
    selected, costs = evaluate_dataset(actor, test, h)
    np.savez_compressed(out / "test_paths.npz", initial=initial_costs, selected=costs)
    payload = {"setting": setting, "seed": seed, "arm": arm,
               "initial_test": initial, "selected_test": selected,
               "test_sha256": dataset_fingerprint(test),
               "selected_checkpoint_sha256": sha256(out / "selected.pt"),
               "mean_cost_change_from_initial": float((costs - initial_costs).mean()),
               "relative_improvement_percent": float(100 * (initial_costs.mean() - costs.mean()) / initial_costs.mean())}
    save_json(out / "test.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.output)
    if args.command == "prepare":
        return
    tasks = [(s, seed, arm, str(args.output)) for s in args.settings for seed in args.seeds for arm in args.arms]
    if args.command == "evaluate":
        # Freeze every requested selected checkpoint before generating test data.
        checkpoint_hashes = {}
        for s, seed, arm, output in tasks:
            folder = Path(output) / "runs" / s / f"seed{seed}" / arm
            assert (folder / "result.json").exists(), folder
            checkpoint_hashes[str(folder / "selected.pt")] = sha256(folder / "selected.pt")
        save_json(args.output / "test_checkpoint_freeze.json", {"checkpoints": checkpoint_hashes, "unix_time": time.time()})
    worker = fit_one if args.command == "train" else evaluate_one
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if args.command == "train":
                print(json.dumps({k: result[k] for k in ("setting", "seed", "arm", "selected_update", "max_training_gradient_norm", "first_observed_action_change_update")}), flush=True)
                print("validation", result["initial_validation"]["mean_cost"], result["selected_validation"]["mean_cost"], flush=True)
            else:
                print(json.dumps({k: result[k] for k in ("setting", "seed", "arm", "relative_improvement_percent")}), flush=True)


if __name__ == "__main__":
    main()
