"""Run every Algorithm 1, OR, and model-free RL hydro comparator together."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import gurobipy as gp
import matplotlib
import numpy as np
import pandas as pd
import stable_baselines3
import torch

from algorithm1_gradient_study.training import tune_and_select
from efficiency_switching_hydro.data import Dataset, generate_splits
from efficiency_switching_hydro.policies import (
    AffinePolicy,
    GlobalGraphNeuralPolicy,
    MyopicPolicy,
    PathwiseMLPPolicy,
    PolynomialPolicy,
    count_parameters,
)
from efficiency_switching_hydro.simulator import simulate

from .cfa import make_lookahead_policy, tune_and_select_cfa
from .config import FullComparisonProtocol
from .evaluation import (
    all_pairwise_ci,
    evaluate_test,
    summarize_pairwise,
    summarize_train_seeds,
)
from .plotting import cost_time_frontier, policy_comparison, training_curves


POLICY_ORDER = (
    "myopic",
    "projected_affine",
    "projected_polynomial",
    "projected_mlp_pathwise",
    "projected_graph_neural_global",
    "deterministic_lookahead",
    "constant_cfa",
    "rl_ppo",
    "rl_sac",
    "rl_td3",
    "linearized_empirical_sddp",
)

DIRECT_POLICIES = (
    "projected_affine",
    "projected_polynomial",
    "projected_mlp_pathwise",
    "projected_graph_neural_global",
)

FAMILIES = {
    "myopic": "fixed_rule",
    "projected_affine": "algorithm1_pathwise",
    "projected_polynomial": "algorithm1_pathwise",
    "projected_mlp_pathwise": "algorithm1_pathwise",
    "projected_graph_neural_global": "algorithm1_pathwise",
    "deterministic_lookahead": "deterministic_optimization",
    "constant_cfa": "parametric_cost_function_approximation",
    "rl_ppo": "model_free_policy_gradient",
    "rl_sac": "model_free_actor_critic",
    "rl_td3": "model_free_actor_critic",
    "linearized_empirical_sddp": "linearized_decomposition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=POLICY_ORDER,
        default=list(POLICY_ORDER),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rl-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore partial CSVs and recompute every requested run.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive.")
    torch.set_num_threads(args.torch_threads)
    protocol = (
        FullComparisonProtocol.smoke_protocol(args.device, args.rl_device)
        if args.smoke
        else FullComparisonProtocol(device=args.device, rl_device=args.rl_device)
    )
    package_root = Path(__file__).resolve().parent
    default_name = "all_models_smoke" if args.smoke else "formal_5seed_500updates"
    output = Path(args.output_dir) if args.output_dir else package_root / "outputs" / default_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(output, protocol, args)
    _write_registered_candidates(output, protocol, args.models)

    hydro = protocol.experiment_config()
    splits = generate_splits(protocol.data_config(), hydro)
    _write_data_manifest(output, splits)
    state = _load_partial(output) if not args.no_resume else _empty_state()
    rows = state["rows"]
    histories = state["histories"]
    selections = state["selections"]
    test_costs = state["test_costs"]
    completed = {
        (str(row["policy"]), str(row["train_seed"])) for row in rows
    }

    def is_complete(policy: str, seed: int | str) -> bool:
        return (policy, str(seed)) in completed

    def record(
        row: dict[str, object],
        costs: np.ndarray,
    ) -> None:
        rows.append(row)
        key = (str(row["policy"]), str(row["train_seed"]))
        test_costs[key] = costs
        completed.add(key)
        _persist_partial(output, rows, histories, selections, test_costs)
        print(
            f"completed {row['policy']} seed={row['train_seed']} "
            f"test_cost={float(row['test_mean_cost']):.6f}",
            flush=True,
        )

    if "myopic" in args.models and not is_complete("myopic", "fixed"):
        policy = MyopicPolicy(hydro).to(protocol.device)
        validation = _validation_cost(policy, splits.validation, hydro, protocol.device)
        metrics, costs = evaluate_test(
            policy,
            splits.test,
            hydro,
            protocol,
            count_parameters(policy),
            count_parameters(policy),
        )
        row = _base_row(
            "myopic",
            "fixed",
            validation,
            selected_iteration=0,
            selected_hyperparameter="fixed",
            n_candidates=0,
            fit_unique_paths=0,
            fit_path_evaluations=0,
            fit_environment_steps=0,
            gradient_updates=0,
            selected_fit_time=0.0,
            total_tuning_time=0.0,
            checkpoint_path="",
        )
        row.update(metrics)
        record(row, costs)

    for policy_name in DIRECT_POLICIES:
        if policy_name not in args.models:
            continue
        for train_seed in protocol.train_seeds:
            if is_complete(policy_name, train_seed):
                continue
            selected = tune_and_select(
                lambda name=policy_name: _build_direct_policy(name, hydro, protocol),
                splits.fit,
                splits.validation,
                hydro,
                protocol,
                train_seed,
                protocol.direct_learning_rates(policy_name),
                "full",
                checkpoint_dir,
                policy_name,
            )
            for candidate_index, candidate in enumerate(selected.candidates):
                candidate_selected = int(candidate is min(
                    selected.candidates,
                    key=lambda item: item.best_validation_cost,
                ))
                selections.append(
                    {
                        "policy": policy_name,
                        "train_seed": train_seed,
                        "candidate_index": candidate_index,
                        "hyperparameter": "learning_rate",
                        "candidate_value": candidate.learning_rate,
                        "best_validation_cost": candidate.best_validation_cost,
                        "selected_iteration": candidate.selected_iteration,
                        "fit_time_seconds": candidate.fit_time_seconds,
                        "selected": candidate_selected,
                        "checkpoint_path": candidate.checkpoint_path,
                    }
                )
                for history in candidate.history:
                    entry = dict(history)
                    entry["policy"] = policy_name
                    entry["candidate_index"] = candidate_index
                    entry["candidate_selected"] = candidate_selected
                    histories.append(entry)
            parameter_count = count_parameters(selected.policy)
            metrics, costs = evaluate_test(
                selected.policy,
                splits.test,
                hydro,
                protocol,
                parameter_count,
                parameter_count,
            )
            row = _base_row(
                policy_name,
                train_seed,
                selected.best_validation_cost,
                selected.selected_iteration,
                selected.learning_rate,
                len(selected.candidates),
                protocol.n_fit,
                protocol.n_fit,
                protocol.n_fit * hydro.T,
                protocol.direct_iterations,
                selected.selected_fit_time_seconds,
                selected.total_tuning_time_seconds,
                selected.checkpoint_path,
            )
            row.update(metrics)
            record(row, costs)

    if (
        "deterministic_lookahead" in args.models
        and not is_complete("deterministic_lookahead", "fixed")
    ):
        policy = make_lookahead_policy(
            "deterministic_lookahead",
            hydro,
            protocol.cfa_config(),
        )
        try:
            validation = _validation_cost(
                policy, splits.validation, hydro, protocol.device
            )
            metrics, costs = evaluate_test(
                policy,
                splits.test,
                hydro,
                protocol,
                0,
                0,
            )
        finally:
            policy.close()
        row = _base_row(
            "deterministic_lookahead",
            "fixed",
            validation,
            0,
            "forecast_multiplier=1",
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            "",
        )
        row.update(metrics)
        record(row, costs)

    if "constant_cfa" in args.models:
        cfa = protocol.cfa_config()
        for train_seed in protocol.train_seeds:
            if is_complete("constant_cfa", train_seed):
                continue
            selected = tune_and_select_cfa(
                splits.fit,
                splits.validation,
                hydro,
                cfa,
                train_seed,
                checkpoint_dir,
                protocol.device,
            )
            best_candidate = min(
                selected.candidates,
                key=lambda item: item.best_validation_cost,
            )
            for candidate_index, candidate in enumerate(selected.candidates):
                candidate_selected = int(candidate is best_candidate)
                selections.append(
                    {
                        "policy": "constant_cfa",
                        "train_seed": train_seed,
                        "candidate_index": candidate_index,
                        "hyperparameter": "learning_rate",
                        "candidate_value": candidate.learning_rate,
                        "best_validation_cost": candidate.best_validation_cost,
                        "selected_iteration": candidate.selected_iteration,
                        "fit_time_seconds": candidate.fit_time_seconds,
                        "selected": candidate_selected,
                        "checkpoint_path": candidate.checkpoint_path,
                        "selected_theta": float(candidate.best_theta[0]),
                    }
                )
                for history in candidate.history:
                    entry = dict(history)
                    entry["candidate_index"] = candidate_index
                    entry["candidate_selected"] = candidate_selected
                    histories.append(entry)
            try:
                metrics, costs = evaluate_test(
                    selected.policy,
                    splits.test,
                    hydro,
                    protocol,
                    selected.policy.n_policy_parameters,
                    selected.policy.n_policy_parameters,
                )
            finally:
                selected.policy.close()
            row = _base_row(
                "constant_cfa",
                train_seed,
                selected.best_validation_cost,
                selected.selected_iteration,
                f"lr={selected.learning_rate};theta={float(selected.best_theta[0])}",
                len(selected.candidates),
                protocol.n_fit,
                2 * protocol.n_fit,
                2 * protocol.n_fit * hydro.T,
                cfa.iterations,
                selected.selected_fit_time_seconds,
                selected.total_tuning_time_seconds,
                selected.checkpoint_path,
            )
            row.update(metrics)
            record(row, costs)

    for algorithm in ("ppo", "sac", "td3"):
        policy_name = f"rl_{algorithm}"
        if policy_name not in args.models:
            continue
        from efficiency_switching_hydro.rl import fit_checkpoint as fit_rl

        for train_seed in protocol.train_seeds:
            if is_complete(policy_name, train_seed):
                continue
            candidate_results = []
            candidate_times = []
            rates = protocol.rl_learning_rates(algorithm)
            for candidate_index, learning_rate in enumerate(rates):
                tag = f"{learning_rate:.8g}".replace(".", "p")
                start = time.perf_counter()
                result = fit_rl(
                    algorithm,
                    splits.fit,
                    splits.validation,
                    hydro,
                    protocol.rl_config(algorithm, learning_rate),
                    train_seed,
                    checkpoint_dir
                    / f"{policy_name}_candidate{candidate_index}_lr{tag}_seed{train_seed}.zip",
                    evaluation_device=protocol.device,
                )
                elapsed = time.perf_counter() - start
                candidate_results.append(result)
                candidate_times.append(elapsed)
            selected_index = min(
                range(len(candidate_results)),
                key=lambda index: candidate_results[index].best_validation_cost,
            )
            selected = candidate_results[selected_index]
            for candidate_index, (candidate, elapsed, learning_rate) in enumerate(
                zip(candidate_results, candidate_times, rates)
            ):
                candidate_selected = int(candidate_index == selected_index)
                selections.append(
                    {
                        "policy": policy_name,
                        "train_seed": train_seed,
                        "candidate_index": candidate_index,
                        "hyperparameter": "learning_rate",
                        "candidate_value": learning_rate,
                        "best_validation_cost": candidate.best_validation_cost,
                        "selected_iteration": candidate.selected_iteration,
                        "fit_time_seconds": elapsed,
                        "selected": candidate_selected,
                        "checkpoint_path": candidate.checkpoint_path,
                    }
                )
                for history in candidate.history:
                    entry = dict(history)
                    entry["candidate_index"] = candidate_index
                    entry["candidate_selected"] = candidate_selected
                    histories.append(entry)
            for candidate_index, candidate in enumerate(candidate_results):
                if candidate_index != selected_index:
                    candidate.policy.close()
            try:
                metrics, costs = evaluate_test(
                    selected.policy,
                    splits.test,
                    hydro,
                    protocol,
                    selected.deployment_parameters,
                    selected.training_parameters,
                )
            finally:
                selected.policy.close()
            row = _base_row(
                policy_name,
                train_seed,
                selected.best_validation_cost,
                selected.selected_iteration,
                rates[selected_index],
                len(candidate_results),
                protocol.n_fit,
                protocol.n_fit,
                selected.environment_steps,
                selected.gradient_updates,
                candidate_times[selected_index],
                sum(candidate_times),
                selected.checkpoint_path,
            )
            row.update(metrics)
            record(row, costs)

    if "linearized_empirical_sddp" in args.models:
        from efficiency_switching_hydro.sddp import fit_checkpoint as fit_sddp

        sddp = protocol.sddp_config()
        for train_seed in protocol.train_seeds:
            if is_complete("linearized_empirical_sddp", train_seed):
                continue
            start = time.perf_counter()
            result = fit_sddp(
                splits.fit,
                splits.validation,
                hydro,
                sddp,
                train_seed,
                checkpoint_dir / f"linearized_empirical_sddp_seed{train_seed}.pt",
            )
            elapsed = time.perf_counter() - start
            for history in result.history:
                entry = dict(history)
                entry["candidate_index"] = 0
                entry["candidate_selected"] = 1
                histories.append(entry)
            selections.append(
                {
                    "policy": "linearized_empirical_sddp",
                    "train_seed": train_seed,
                    "candidate_index": 0,
                    "hyperparameter": "noise_scenarios",
                    "candidate_value": sddp.noise_scenarios,
                    "best_validation_cost": result.best_validation_cost,
                    "selected_iteration": result.selected_iteration,
                    "fit_time_seconds": elapsed,
                    "selected": 1,
                    "checkpoint_path": result.checkpoint_path,
                }
            )
            try:
                metrics, costs = evaluate_test(
                    result.policy,
                    splits.test,
                    hydro,
                    protocol,
                    0,
                    0,
                )
            finally:
                result.policy.close()
            row = _base_row(
                "linearized_empirical_sddp",
                train_seed,
                result.best_validation_cost,
                result.selected_iteration,
                f"noise_scenarios={sddp.noise_scenarios}",
                1,
                protocol.n_fit,
                protocol.n_fit + sddp.iterations,
                (protocol.n_fit + sddp.iterations) * hydro.T,
                sddp.iterations,
                elapsed,
                elapsed,
                result.checkpoint_path,
            )
            row["number_of_cuts"] = result.number_of_cuts
            row.update(metrics)
            record(row, costs)

    requested_order = [policy for policy in POLICY_ORDER if policy in args.models]
    missing = [
        (policy, seed)
        for policy in requested_order
        for seed in (
            ("fixed",)
            if policy in {"myopic", "deterministic_lookahead"}
            else tuple(str(value) for value in protocol.train_seeds)
        )
        if (policy, str(seed)) not in completed
    ]
    if missing:
        raise RuntimeError(f"Comparison ended with missing runs: {missing}")
    _finalize(output, rows, histories, selections, test_costs, requested_order)
    print(f"completed unified comparison at {output}", flush=True)


def _build_direct_policy(name, hydro, protocol):
    if name == "projected_affine":
        return AffinePolicy(hydro)
    if name == "projected_polynomial":
        return PolynomialPolicy(hydro)
    if name == "projected_mlp_pathwise":
        return PathwiseMLPPolicy(
            hydro,
            width=protocol.mlp_width,
            depth=protocol.network_depth,
        )
    if name == "projected_graph_neural_global":
        return GlobalGraphNeuralPolicy(hydro, width=protocol.graph_width)
    raise ValueError(f"Unknown direct policy: {name}")


@torch.no_grad()
def _validation_cost(policy, validation, hydro, device):
    if hasattr(policy, "eval"):
        policy.eval()
    return float(simulate(policy, validation, hydro, device=device).mean_cost.cpu())


def _base_row(
    policy,
    train_seed,
    best_validation_cost,
    selected_iteration,
    selected_hyperparameter,
    n_candidates,
    fit_unique_paths,
    fit_path_evaluations,
    fit_environment_steps,
    gradient_updates,
    selected_fit_time,
    total_tuning_time,
    checkpoint_path,
):
    return {
        "policy": policy,
        "family": FAMILIES[policy],
        "role": "deployable_policy",
        "train_seed": train_seed,
        "best_validation_cost": best_validation_cost,
        "selected_iteration": selected_iteration,
        "selected_hyperparameter": selected_hyperparameter,
        "n_hyperparameter_candidates": n_candidates,
        "fit_unique_paths_per_candidate": fit_unique_paths,
        "fit_path_evaluations": fit_path_evaluations,
        "fit_environment_steps": fit_environment_steps,
        "gradient_updates": gradient_updates,
        "selected_fit_time_seconds": selected_fit_time,
        "total_tuning_time_seconds": total_tuning_time,
        "number_of_cuts": 0,
        "checkpoint_path": checkpoint_path,
    }


def _write_metadata(output, protocol, args):
    with (output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": asdict(protocol),
                "requested_models": args.models,
                "primary_metric": "test cost",
                "selection_rule": "minimum validation cost",
                "trajectory_information_rule": (
                    "Each candidate reads the same n_fit trajectories. Direct policies "
                    "use each once; RL completes each once; CFA evaluates two perturbed "
                    "policies on each path and reports those extra function evaluations."
                ),
                "reserved_interfaces": {
                    "calibration": "generated with an independent seed and never read",
                    "evaluation": "generated with an independent seed and never read",
                },
                "uncertainty": {
                    "test_path": "paired test-path CI within training seed",
                    "training": "SD of selected-policy test means across training seeds",
                },
                "sddp_limit": (
                    "The nonlinear model uses a planned-state first-order SDDP "
                    "linearization with two empirical stagewise scenarios to fit "
                    "the size-limited Gurobi license; it is not an exact optimum "
                    "benchmark."
                ),
                "cfa_limit": (
                    "Lookahead uses a planned-state LP approximation with positive "
                    "linear marginal penalties; all reported "
                    "costs are recomputed in the shared nonlinear simulator."
                ),
            },
            handle,
            indent=2,
        )
    with (output / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "gurobi": gp.gurobi.version(),
                "torch_threads": torch.get_num_threads(),
                "device": protocol.device,
                "rl_device": protocol.rl_device,
            },
            handle,
            indent=2,
        )


def _write_registered_candidates(output, protocol, models):
    rows = []
    for policy in models:
        if policy in DIRECT_POLICIES:
            values = protocol.direct_learning_rates(policy)
            parameter = "learning_rate"
        elif policy.startswith("rl_"):
            values = protocol.rl_learning_rates(policy.removeprefix("rl_"))
            parameter = "learning_rate"
        elif policy == "constant_cfa":
            values = protocol.cfa_learning_rates
            parameter = "learning_rate"
        elif policy == "linearized_empirical_sddp":
            values = (protocol.sddp_noise_scenarios,)
            parameter = "noise_scenarios"
        else:
            values = ("fixed",)
            parameter = "none"
        for index, value in enumerate(values):
            rows.append(
                {
                    "policy": policy,
                    "candidate_index": index,
                    "hyperparameter": parameter,
                    "candidate_value": value,
                }
            )
    pd.DataFrame(rows).to_csv(output / "pre_registered_candidates.csv", index=False)


def _write_data_manifest(output, splits):
    manifest = {
        name: {
            "seed": getattr(splits, name).seed,
            "n_paths": getattr(splits, name).n,
            "sha256": _dataset_fingerprint(getattr(splits, name)),
            "role": "fit" if name == "fit" else (
                "selection" if name == "validation" else "reporting_only"
            ),
        }
        for name in ("fit", "validation", "test")
    }
    manifest["calibration"] = {
        "seed": splits.calibration.seed,
        "n_paths": splits.calibration.n,
        "role": "reserved_unread",
    }
    manifest["evaluation"] = {
        "seed": splits.evaluation.seed,
        "n_paths": splits.evaluation.n,
        "role": "reserved_unread",
    }
    with (output / "data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _dataset_fingerprint(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for values in (dataset.inflow, dataset.demand, dataset.price):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _empty_state():
    return {"rows": [], "histories": [], "selections": [], "test_costs": {}}


def _load_partial(output):
    result = _empty_state()
    paths = {
        "rows": output / "partial_per_run_results.csv",
        "histories": output / "partial_training_curves.csv",
        "selections": output / "partial_hyperparameter_selection.csv",
    }
    for key, path in paths.items():
        if path.exists() and path.stat().st_size:
            result[key] = pd.read_csv(path).to_dict("records")
    cost_path = output / "partial_test_path_costs.csv"
    if cost_path.exists() and cost_path.stat().st_size:
        frame = pd.read_csv(cost_path)
        for column in frame.columns:
            if column == "test_path":
                continue
            policy, seed = column.split("@@", maxsplit=1)
            result["test_costs"][(policy, seed)] = frame[column].to_numpy(
                dtype=np.float64
            )
    return result


def _persist_partial(output, rows, histories, selections, test_costs):
    pd.DataFrame(rows).to_csv(output / "partial_per_run_results.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "partial_training_curves.csv", index=False)
    pd.DataFrame(selections).to_csv(
        output / "partial_hyperparameter_selection.csv", index=False
    )
    columns = {"test_path": np.arange(len(next(iter(test_costs.values()))))}
    for (policy, seed), costs in test_costs.items():
        columns[f"{policy}@@{seed}"] = costs
    pd.DataFrame(columns).to_csv(output / "partial_test_path_costs.csv", index=False)


def _finalize(output, rows, histories, selections, test_costs, policy_order):
    per_run = pd.DataFrame(rows)
    history = pd.DataFrame(histories)
    selection = pd.DataFrame(selections)
    summary = summarize_train_seeds(per_run)
    pairwise = all_pairwise_ci(test_costs, policy_order)
    pairwise_summary = summarize_pairwise(pairwise)
    per_run.to_csv(output / "per_run_results.csv", index=False)
    history.to_csv(output / "training_curves.csv", index=False)
    selection.to_csv(output / "hyperparameter_selection.csv", index=False)
    summary.to_csv(output / "main_results.csv", index=False)
    pairwise.to_csv(output / "paired_test_path_ci.csv", index=False)
    pairwise_summary.to_csv(output / "paired_comparison_summary.csv", index=False)
    columns = {"test_path": np.arange(len(next(iter(test_costs.values()))))}
    for (policy, seed), costs in test_costs.items():
        columns[f"{policy}@@{seed}"] = costs
    pd.DataFrame(columns).to_csv(output / "test_path_costs.csv", index=False)
    policy_comparison(summary, output / "figures" / "policy_comparison.png")
    cost_time_frontier(summary, output / "figures" / "cost_time_frontier.png")
    training_curves(history, output / "figures" / "training_curves.png")
    with (output / "COMPLETED.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "completed": True,
                "models": policy_order,
                "n_per_run_rows": len(per_run),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    run()
