"""Common evaluation, timing, feasibility, and uncertainty reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from torch import nn

from algorithm1_full_comparison.evaluation import (
    all_pairwise_ci,
    evaluate_test,
    summarize_pairwise,
)
from efficiency_switching_hydro.policies import count_parameters


def evaluate_policy(
    policy,
    test,
    config,
    protocol,
    deployment_parameters: int | None = None,
    training_parameters: int | None = None,
) -> tuple[dict[str, float | int], np.ndarray]:
    if deployment_parameters is None:
        deployment_parameters = (
            count_parameters(policy) if isinstance(policy, nn.Module) else 0
        )
    if training_parameters is None:
        training_parameters = deployment_parameters
    metrics, costs = evaluate_test(
        policy,
        test,
        config,
        protocol,
        deployment_parameters,
        training_parameters,
    )
    if (
        metrics["total_constraint_violation_count"] != 0
        or metrics["max_action_violation"] != 0.0
        or metrics["max_storage_violation"] != 0.0
    ):
        raise RuntimeError("A projected policy violated a hard hydro constraint.")
    return metrics, costs


def summarize_runs(
    per_run: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    columns = [*group_columns, "policy"]
    for keys, group in per_run.groupby(columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, float | int | str] = dict(zip(columns, key_values))
        costs = group["test_mean_cost"].to_numpy(dtype=np.float64)
        row.update(
            {
                "n_train_seeds": int(group["train_seed"].astype(str).ne("fixed").sum()),
                "n_reported_runs": len(group),
                "test_cost_mean_over_train_seeds": float(costs.mean()),
                "test_cost_train_seed_sd": (
                    float(costs.std(ddof=1)) if len(costs) > 1 else 0.0
                ),
                "test_path_cost_sd_mean": float(group["test_path_cost_sd"].mean()),
                "selected_validation_cost_mean": float(
                    group["best_validation_cost"].mean()
                ),
                "deployment_parameters": int(group["num_parameters"].iloc[0]),
                "training_parameters": int(group["training_num_parameters"].iloc[0]),
                "environment_transitions_per_candidate": int(
                    group["environment_transitions_per_candidate"].iloc[0]
                ),
                "actor_gradient_updates_per_candidate": float(
                    group["actor_gradient_updates"].mean()
                ),
                "critic_gradient_updates_per_candidate": float(
                    group["critic_gradient_updates"].mean()
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
                "max_storage_violation": float(group["max_storage_violation"].max()),
                "total_constraint_violation_count": int(
                    group["total_constraint_violation_count"].sum()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [*group_columns, "test_cost_mean_over_train_seeds"]
    )


def pairwise_by_group(
    costs: Mapping[tuple[object, str, str], np.ndarray],
    policy_order: Sequence[str],
    group_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for group_value in dict.fromkeys(key[0] for key in costs):
        local = {
            (policy, seed): values
            for (group, policy, seed), values in costs.items()
            if group == group_value
        }
        frame = all_pairwise_ci(local, policy_order)
        frame.insert(0, group_name, group_value)
        frames.append(frame)
    pairwise = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summaries = []
    for group_value, group in pairwise.groupby(group_name, sort=False):
        summary = summarize_pairwise(group)
        summary.insert(0, group_name, group_value)
        summaries.append(summary)
    return pairwise, pd.concat(summaries, ignore_index=True)
