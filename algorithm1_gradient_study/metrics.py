"""Evaluation, uncertainty, feasibility, and deployment timing metrics."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.policies import count_parameters
from efficiency_switching_hydro.projection import action_bounds, project_action
from efficiency_switching_hydro.simulator import simulate

from .config import StudyProtocol


@torch.no_grad()
def evaluate_test(
    policy,
    test: Dataset,
    config: ExperimentConfig,
    protocol: StudyProtocol,
    measure_latency: bool = True,
) -> tuple[dict[str, float | int], np.ndarray]:
    policy.eval()
    trace = simulate(policy, test, config, device=protocol.device)
    costs = trace.total_cost.detach().cpu().numpy().astype(np.float64)
    lower, upper = action_bounds(
        trace.storage_before_action,
        trace.previous_release_before_action,
        config,
    )
    tolerance = 1.0e-5
    action_bound_violations = (
        (trace.actions < lower - tolerance) | (trace.actions > upper + tolerance)
    )
    weights = torch.as_tensor(
        config.generation_coefficient,
        dtype=trace.actions.dtype,
        device=trace.actions.device,
    )
    shared_violations = (
        (trace.actions * weights).sum(dim=-1)
        > config.shared_release_capacity + tolerance
    )
    capacity = torch.as_tensor(
        config.storage_capacity,
        dtype=trace.storage.dtype,
        device=trace.storage.device,
    )
    storage_violations = (
        (trace.storage < -tolerance) | (trace.storage > capacity + tolerance)
    )
    latency = (
        benchmark_action(policy, trace, config, protocol)
        if measure_latency
        else {
            "latency_stage": config.T // 2,
            "batch1_action_latency_ms_median": float("nan"),
            "batch1_action_latency_ms_p90": float("nan"),
            "throughput_batch_size": min(protocol.latency_batch_size, test.n),
            "batched_action_vectors_per_second": float("nan"),
            "batched_reservoir_actions_per_second": float("nan"),
        }
    )
    row: dict[str, float | int] = {
        "n_test_paths": test.n,
        "test_path_seed": test.seed,
        "test_mean_cost": float(costs.mean()),
        "test_path_cost_sd": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "num_parameters": count_parameters(policy),
        "max_action_violation": float(trace.max_action_violation.detach().cpu()),
        "max_storage_violation": float(trace.max_storage_violation.detach().cpu()),
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
    }
    row.update(latency)
    return row, costs


@torch.no_grad()
def benchmark_action(
    policy,
    trace,
    config: ExperimentConfig,
    protocol: StudyProtocol,
) -> dict[str, float | int]:
    stage = config.T // 2
    batch_size = min(protocol.latency_batch_size, trace.actions.shape[0])
    storage = trace.storage_before_action[:batch_size, stage]
    previous = trace.previous_release_before_action[:batch_size, stage]
    queue = trace.queues[:batch_size, stage]

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
            config,
        )

    for _ in range(3):
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
        "batch1_action_latency_ms_median": 1000.0
        * float(np.median(single_samples)),
        "batch1_action_latency_ms_p90": 1000.0
        * float(np.quantile(single_samples, 0.90)),
        "throughput_batch_size": batch_size,
        "batched_action_vectors_per_second": action_vectors
        / max(batch_elapsed, 1.0e-12),
        "batched_reservoir_actions_per_second": action_vectors
        * config.n_reservoirs
        / max(batch_elapsed, 1.0e-12),
    }


def paired_ci(
    test_costs: Mapping[tuple[str, int], np.ndarray],
    baseline: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    variants = sorted({variant for variant, _ in test_costs})
    seeds = sorted({seed for _, seed in test_costs})
    for variant in variants:
        if variant == baseline:
            continue
        for seed in seeds:
            candidate_key = (variant, seed)
            baseline_key = (baseline, seed)
            if candidate_key not in test_costs or baseline_key not in test_costs:
                continue
            difference = test_costs[candidate_key] - test_costs[baseline_key]
            mean = float(difference.mean())
            path_sd = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
            se = path_sd / max(len(difference), 1) ** 0.5
            rows.append(
                {
                    "variant": variant,
                    "baseline": baseline,
                    "train_seed": seed,
                    "n_paired_test_paths": len(difference),
                    "mean_cost_difference": mean,
                    "paired_test_path_sd": path_sd,
                    "paired_test_path_se": se,
                    "paired_test_ci95_lower": mean - 1.96 * se,
                    "paired_test_ci95_upper": mean + 1.96 * se,
                }
            )
    return pd.DataFrame(rows)


def summarize_train_seeds(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variant, group in per_run.groupby("variant", sort=False):
        seed_costs = group["test_mean_cost"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "variant": variant,
                "n_train_seeds": len(group),
                "test_cost_mean_over_train_seeds": float(seed_costs.mean()),
                "test_cost_train_seed_sd": (
                    float(seed_costs.std(ddof=1)) if len(seed_costs) > 1 else 0.0
                ),
                "selected_validation_cost_mean": float(
                    group["best_validation_cost"].mean()
                ),
                "num_parameters": int(group["num_parameters"].iloc[0]),
                "gradient_updates_per_selected_fit": int(
                    group["gradient_updates"].iloc[0]
                ),
                "tuning_gradient_updates_mean": float(
                    group["tuning_gradient_updates"].mean()
                ),
                "environment_transitions_per_selected_fit": int(
                    group["environment_transitions"].iloc[0]
                ),
                "tuning_environment_transitions_mean": float(
                    group["tuning_environment_transitions"].mean()
                ),
                "selected_fit_time_seconds_mean": float(
                    group["selected_fit_time_seconds"].mean()
                ),
                "total_tuning_time_seconds_mean": float(
                    group["total_tuning_time_seconds"].mean()
                ),
                "batch1_action_latency_ms_median_across_seeds": float(
                    group["batch1_action_latency_ms_median"].median()
                ),
                "batched_action_vectors_per_second_mean": float(
                    group["batched_action_vectors_per_second"].mean()
                ),
                "max_action_violation": float(group["max_action_violation"].max()),
                "max_storage_violation": float(group["max_storage_violation"].max()),
                "total_constraint_violation_count": int(
                    group["total_constraint_violation_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("test_cost_mean_over_train_seeds")


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
