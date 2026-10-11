"""Shared test evaluation, feasibility checks, timing, and uncertainty tables."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.projection import action_bounds, project_action
from efficiency_switching_hydro.simulator import simulate

from .config import FullComparisonProtocol


@torch.no_grad()
def evaluate_test(
    policy,
    test: Dataset,
    hydro: ExperimentConfig,
    protocol: FullComparisonProtocol,
    deployment_parameters: int,
    training_parameters: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Evaluate a frozen policy once on the untouched common test block."""

    if hasattr(policy, "eval"):
        policy.eval()
    start = time.perf_counter()
    trace = simulate(policy, test, hydro, device=protocol.device)
    test_wall_time = time.perf_counter() - start
    costs = trace.total_cost.detach().cpu().numpy().astype(np.float64)
    lower, upper = action_bounds(
        trace.storage_before_action,
        trace.previous_release_before_action,
        hydro,
    )
    tolerance = 1.0e-5
    action_bound_violations = (
        (trace.actions < lower - tolerance) | (trace.actions > upper + tolerance)
    )
    weights = torch.as_tensor(
        hydro.generation_coefficient,
        dtype=trace.actions.dtype,
        device=trace.actions.device,
    )
    shared_violations = (
        (trace.actions * weights).sum(dim=-1)
        > hydro.shared_release_capacity + tolerance
    )
    capacity = torch.as_tensor(
        hydro.storage_capacity,
        dtype=trace.storage.dtype,
        device=trace.storage.device,
    )
    storage_violations = (
        (trace.storage < -tolerance) | (trace.storage > capacity + tolerance)
    )
    timing = benchmark_action(policy, trace, hydro, protocol)
    row: dict[str, float | int] = {
        "n_test_paths": test.n,
        "test_path_seed": test.seed,
        "test_mean_cost": float(costs.mean()),
        "test_path_cost_sd": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "test_wall_time_seconds": test_wall_time,
        "test_seconds_per_path_stage": test_wall_time / max(test.n * hydro.T, 1),
        "num_parameters": int(deployment_parameters),
        "training_num_parameters": int(training_parameters),
        "max_action_violation": float(trace.max_action_violation.cpu()),
        "max_storage_violation": float(trace.max_storage_violation.cpu()),
        "action_bound_violation_count": int(action_bound_violations.sum().cpu()),
        "shared_capacity_violation_count": int(shared_violations.sum().cpu()),
        "storage_violation_count": int(storage_violations.sum().cpu()),
        "total_constraint_violation_count": int(
            action_bound_violations.sum().cpu()
            + shared_violations.sum().cpu()
            + storage_violations.sum().cpu()
        ),
        "test_mean_terminal_cost": float(trace.terminal_cost.mean().cpu()),
        "test_mean_thermal_generation": float(
            trace.thermal_generation.mean().cpu()
        ),
        "test_mean_peaker_generation": float(
            trace.peaker_generation.mean().cpu()
        ),
        "test_mean_ramp_cost": float(trace.ramp_cost.mean().cpu()),
        "test_mean_storage_penalty": float(trace.storage_penalty.mean().cpu()),
    }
    row.update(timing)
    return row, costs


@torch.no_grad()
def benchmark_action(
    policy,
    trace,
    hydro: ExperimentConfig,
    protocol: FullComparisonProtocol,
) -> dict[str, float | int]:
    """Measure batch-one latency separately from fixed-batch throughput."""

    stage = hydro.T // 2
    batch_size = min(protocol.latency_batch_size, trace.actions.shape[0])
    storage = trace.storage_before_action[:batch_size, stage]
    previous = trace.previous_release_before_action[:batch_size, stage]
    queue = trace.queue_before_action[:batch_size, stage]

    def action(current_storage, current_previous, current_queue):
        proposal = policy.propose(
            stage,
            current_storage,
            current_previous,
            current_queue,
        )
        return project_action(
            proposal,
            current_storage,
            current_previous,
            hydro,
        )

    action(storage[:1], previous[:1], queue[:1])
    action(storage, previous, queue)
    _synchronize(protocol.device)
    single_samples = []
    for _ in range(protocol.latency_repetitions):
        start = time.perf_counter()
        action(storage[:1], previous[:1], queue[:1])
        _synchronize(protocol.device)
        single_samples.append(time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(protocol.latency_repetitions):
        action(storage, previous, queue)
    _synchronize(protocol.device)
    batch_elapsed = time.perf_counter() - start
    action_vectors = batch_size * protocol.latency_repetitions
    return {
        "latency_stage": stage,
        "latency_repetitions": protocol.latency_repetitions,
        "batch1_action_latency_ms_median": 1000.0
        * float(np.median(single_samples)),
        "batch1_action_latency_ms_p90": 1000.0
        * float(np.quantile(single_samples, 0.90)),
        "throughput_batch_size": batch_size,
        "batched_action_vectors_per_second": action_vectors
        / max(batch_elapsed, 1.0e-12),
        "batched_reservoir_actions_per_second": action_vectors
        * hydro.n_reservoirs
        / max(batch_elapsed, 1.0e-12),
    }


def summarize_train_seeds(per_run: pd.DataFrame) -> pd.DataFrame:
    """Keep path variability and train-seed variability in separate columns."""

    rows: list[dict[str, float | int | str]] = []
    for policy, group in per_run.groupby("policy", sort=False):
        seed_costs = group["test_mean_cost"].to_numpy(dtype=np.float64)
        trained = group["train_seed"].astype(str).ne("fixed")
        rows.append(
            {
                "policy": policy,
                "family": group["family"].iloc[0],
                "role": group["role"].iloc[0],
                "n_train_seeds": int(trained.sum()),
                "n_reported_runs": len(group),
                "test_cost_mean_over_train_seeds": float(seed_costs.mean()),
                "test_cost_train_seed_sd": (
                    float(seed_costs.std(ddof=1)) if len(seed_costs) > 1 else 0.0
                ),
                "test_path_cost_sd_mean": float(group["test_path_cost_sd"].mean()),
                "selected_validation_cost_mean": float(
                    group["best_validation_cost"].mean()
                ),
                "num_parameters": int(group["num_parameters"].iloc[0]),
                "training_num_parameters": int(
                    group["training_num_parameters"].iloc[0]
                ),
                "number_of_cuts_mean": float(group["number_of_cuts"].mean()),
                "selected_iteration_mean": float(
                    group["selected_iteration"].mean()
                ),
                "gradient_updates_mean": float(group["gradient_updates"].mean()),
                "fit_unique_paths_per_candidate": int(
                    group["fit_unique_paths_per_candidate"].iloc[0]
                ),
                "fit_path_evaluations_mean": float(
                    group["fit_path_evaluations"].mean()
                ),
                "selected_fit_time_seconds_mean": float(
                    group["selected_fit_time_seconds"].mean()
                ),
                "total_tuning_time_seconds_mean": float(
                    group["total_tuning_time_seconds"].mean()
                ),
                "test_wall_time_seconds_mean": float(
                    group["test_wall_time_seconds"].mean()
                ),
                "batch1_action_latency_ms_median_across_seeds": float(
                    group["batch1_action_latency_ms_median"].median()
                ),
                "batched_action_vectors_per_second_mean": float(
                    group["batched_action_vectors_per_second"].mean()
                ),
                "max_action_violation": float(group["max_action_violation"].max()),
                "max_storage_violation": float(
                    group["max_storage_violation"].max()
                ),
                "total_constraint_violation_count": int(
                    group["total_constraint_violation_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("test_cost_mean_over_train_seeds")


def all_pairwise_ci(
    costs: Mapping[tuple[str, str], np.ndarray],
    policy_order: Sequence[str],
) -> pd.DataFrame:
    """Compare every policy pair on identical test paths and matched train seeds."""

    rows: list[dict[str, float | int | str]] = []
    available = set(costs)
    for candidate_index, candidate in enumerate(policy_order):
        for baseline in policy_order[:candidate_index]:
            candidate_seeds = {seed for policy, seed in available if policy == candidate}
            baseline_seeds = {seed for policy, seed in available if policy == baseline}
            common = sorted(
                (candidate_seeds - {"fixed"})
                & (baseline_seeds - {"fixed"}),
                key=int,
            )
            if not common:
                nonfixed = sorted(
                    (candidate_seeds | baseline_seeds) - {"fixed"},
                    key=int,
                )
                common = nonfixed or ["fixed"]
            for seed in common:
                candidate_key = (
                    candidate,
                    seed if (candidate, seed) in available else "fixed",
                )
                baseline_key = (
                    baseline,
                    seed if (baseline, seed) in available else "fixed",
                )
                if candidate_key not in available or baseline_key not in available:
                    continue
                difference = costs[candidate_key] - costs[baseline_key]
                mean = float(difference.mean())
                path_sd = (
                    float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
                )
                path_se = path_sd / max(len(difference), 1) ** 0.5
                rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "train_seed": seed,
                        "n_paired_test_paths": len(difference),
                        "mean_candidate_minus_baseline_cost": mean,
                        "paired_test_path_sd": path_sd,
                        "paired_test_path_se": path_se,
                        "paired_test_ci95_lower": mean - 1.96 * path_se,
                        "paired_test_ci95_upper": mean + 1.96 * path_se,
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (candidate, baseline), group in pairwise.groupby(
        ["candidate", "baseline"], sort=False
    ):
        differences = group["mean_candidate_minus_baseline_cost"].to_numpy(
            dtype=np.float64
        )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "n_train_seed_comparisons": len(group),
                "mean_candidate_minus_baseline_cost_over_train_seeds": float(
                    differences.mean()
                ),
                "cost_difference_train_seed_sd": (
                    float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                ),
                "all_within_seed_paired_ci_exclude_zero": int(
                    (
                        (group["paired_test_ci95_lower"] > 0.0)
                        | (group["paired_test_ci95_upper"] < 0.0)
                    ).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
