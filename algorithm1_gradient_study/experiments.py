"""Implement the five pre-registered Algorithm 1 mechanism experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import generate_dataset, generate_splits
from efficiency_switching_hydro.policies import count_parameters
from efficiency_switching_hydro.simulator import simulate

from .config import StudyProtocol
from .gradients import (
    finite_difference_rows,
    geometric_slope,
    gradient_comparison,
    nested_horizon_datasets,
)
from .metrics import evaluate_test, paired_ci, summarize_train_seeds
from .models import (
    build_policy,
    copy_trainable_parameters,
    initialize_channel_policy,
    permute_config,
    permute_dataset,
    perturb_routing,
)
from .plotting import (
    ablation_figure,
    equivariance_figure,
    finite_difference_figure,
    graph_stability_figure,
    horizon_figure,
)
from .training import SelectedFit, tune_and_select


ABLATION_MODES = ("full", "no_proposal_state", "no_moving_recourse")


def run_finite_difference(protocol: StudyProtocol, root: Path) -> Path:
    output = root / "01_finite_difference"
    output.mkdir(parents=True, exist_ok=True)
    config = _base_config(protocol)
    dataset = generate_dataset(
        "fit_gradient_diagnostic",
        min(protocol.gradient_paths, protocol.n_fit),
        protocol.fit_seed,
        config,
    )
    rows: list[dict[str, float | int | str]] = []
    for train_seed in protocol.train_seeds:
        torch.manual_seed(train_seed)
        policy = build_policy("mlp", config, protocol).to(
            device=protocol.device,
            dtype=torch.float64,
        )
        initialize_channel_policy(policy, 10_000 + train_seed)
        seed_rows = finite_difference_rows(
            policy,
            dataset,
            config,
            protocol.finite_difference_steps,
            direction_seed=20_000 + train_seed,
            device=protocol.device,
        )
        with torch.no_grad():
            trace = simulate(
                policy,
                dataset,
                config,
                device=protocol.device,
                dtype=torch.float64,
            )
        for row in seed_rows:
            row.update(
                {
                    "train_seed": train_seed,
                    "n_fit_diagnostic_paths": dataset.n,
                    "num_parameters": count_parameters(policy),
                    "max_action_violation": float(trace.max_action_violation.cpu()),
                    "max_storage_violation": float(trace.max_storage_violation.cpu()),
                }
            )
        rows.extend(seed_rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "finite_difference_check.csv", index=False)
    best = frame.loc[
        frame.groupby(["train_seed", "gradient_mode"])["relative_error"].idxmin()
    ]
    summary = (
        best.groupby("gradient_mode", sort=False)
        .agg(
            n_seeds=("train_seed", "nunique"),
            best_relative_error_mean=("relative_error", "mean"),
            best_relative_error_train_seed_sd=("relative_error", "std"),
            selected_step_median=("step", "median"),
            max_action_violation=("max_action_violation", "max"),
            max_storage_violation=("max_storage_violation", "max"),
        )
        .reset_index()
    )
    summary["best_relative_error_train_seed_sd"] = summary[
        "best_relative_error_train_seed_sd"
    ].fillna(0.0)
    summary.to_csv(output / "finite_difference_summary.csv", index=False)
    finite_difference_figure(frame, output / "figures" / "finite_difference.png")
    _write_experiment_note(
        output,
        protocol,
        {
            "experiment": "finite_difference",
            "selection_data": "none",
            "data_used": ["fit_gradient_diagnostic"],
            "claim": (
                "Only the full gradient is expected to match the finite difference of "
                "the true forward objective. Ablated gradients intentionally target a "
                "different backward graph."
            ),
        },
    )
    return output


def run_channel_ablations(protocol: StudyProtocol, root: Path) -> dict[str, Path]:
    shared = root / "_shared_gradient_channel_runs"
    shared.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = shared / "checkpoints"
    config = _base_config(protocol)
    splits = generate_splits(protocol.data_config(), config)
    per_run_rows: list[dict[str, float | int | str]] = []
    hyperparameter_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []
    test_costs: dict[tuple[str, int], np.ndarray] = {}

    for mode in ABLATION_MODES:
        for train_seed in protocol.train_seeds:
            def policy_builder(seed: int = train_seed):
                policy = build_policy("mlp", config, protocol)
                initialize_channel_policy(policy, 40_000 + seed)
                return policy

            selected = tune_and_select(
                policy_builder,
                splits.fit,
                splits.validation,
                config,
                protocol,
                train_seed,
                protocol.ablation_learning_rates,
                mode,
                checkpoint_dir,
                f"projected_mlp_pathwise_{mode}",
            )
            test_metrics, costs = evaluate_test(
                selected.policy,
                splits.test,
                config,
                protocol,
            )
            test_costs[(mode, train_seed)] = costs
            row: dict[str, float | int | str] = {
                "variant": mode,
                "policy": "projected_mlp_pathwise",
                "gradient_mode": mode,
                "train_seed": train_seed,
                "selected_learning_rate": selected.learning_rate,
                "n_hyperparameter_candidates": len(selected.candidates),
                "best_validation_cost": selected.best_validation_cost,
                "selected_iteration": selected.selected_iteration,
                "checkpoint_path": selected.checkpoint_path,
                "fit_trajectory_information_budget": (
                    protocol.iterations * protocol.batch_size
                ),
                "tuning_trajectory_information_budget": (
                    protocol.iterations
                    * protocol.batch_size
                    * len(selected.candidates)
                ),
                "gradient_updates": protocol.iterations,
                "tuning_gradient_updates": (
                    protocol.iterations * len(selected.candidates)
                ),
                "environment_transitions": (
                    protocol.iterations * protocol.batch_size * config.T
                ),
                "tuning_environment_transitions": (
                    protocol.iterations
                    * protocol.batch_size
                    * config.T
                    * len(selected.candidates)
                ),
                "selected_fit_time_seconds": selected.selected_fit_time_seconds,
                "total_tuning_time_seconds": selected.total_tuning_time_seconds,
            }
            row.update(test_metrics)
            per_run_rows.append(row)
            _collect_fit_records(
                selected,
                mode,
                train_seed,
                hyperparameter_rows,
                history_rows,
            )

    per_run = pd.DataFrame(per_run_rows)
    hyperparameters = pd.DataFrame(hyperparameter_rows)
    histories = pd.DataFrame(history_rows)
    per_run.to_csv(shared / "per_run_results.csv", index=False)
    hyperparameters.to_csv(shared / "hyperparameter_candidates.csv", index=False)
    histories.to_csv(shared / "training_curves.csv", index=False)
    _wide_cost_frame(test_costs, splits.test.n).to_csv(
        shared / "test_path_costs.csv", index=False
    )

    destinations: dict[str, Path] = {}
    definitions = (
        ("no_proposal_state", "02_no_proposal_state"),
        ("no_moving_recourse", "03_no_moving_recourse"),
    )
    for ablation, directory_name in definitions:
        output = root / directory_name
        output.mkdir(parents=True, exist_ok=True)
        subset = per_run[per_run["variant"].isin(("full", ablation))].copy()
        subset_hyperparameters = hyperparameters[
            hyperparameters["variant"].isin(("full", ablation))
        ].copy()
        subset_histories = histories[
            histories["variant"].isin(("full", ablation))
        ].copy()
        subset_costs = {
            key: value
            for key, value in test_costs.items()
            if key[0] in ("full", ablation)
        }
        summary = summarize_train_seeds(subset)
        paired = paired_ci(subset_costs, baseline="full")
        subset.to_csv(output / "per_run_results.csv", index=False)
        summary.to_csv(output / "main_results.csv", index=False)
        paired.to_csv(output / "paired_test_path_ci.csv", index=False)
        subset_hyperparameters.to_csv(
            output / "hyperparameter_candidates.csv", index=False
        )
        subset_histories.to_csv(output / "training_curves.csv", index=False)
        _wide_cost_frame(subset_costs, splits.test.n).to_csv(
            output / "test_path_costs.csv", index=False
        )
        ablation_figure(summary, output / "figures" / "test_cost.png")
        _write_experiment_note(
            output,
            protocol,
            {
                "experiment": ablation,
                "selection_data": "validation only",
                "reported_data": "test after checkpoint and learning-rate selection",
                "reserved_blocks_not_read": ["calibration", "evaluation"],
                "paired_ci_unit": "common test trajectories within train seed",
                "seed_sd_unit": "selected policy across independent train seeds",
                "data_fingerprints": _split_fingerprints(splits),
            },
        )
        destinations[ablation] = output
    return destinations


def run_horizon_amplification(protocol: StudyProtocol, root: Path) -> Path:
    output = root / "04_horizon_amplification"
    output.mkdir(parents=True, exist_ok=True)
    configs = [
        replace(
            ExperimentConfig.make(
                protocol.profile,
                protocol.n_reservoirs,
                horizon,
                protocol.shock_factors,
            ),
            ramp_fraction=protocol.study_ramp_fraction,
        )
        for horizon in protocol.horizon_values
    ]
    datasets = nested_horizon_datasets(
        configs,
        protocol.gradient_paths,
        protocol.fit_seed,
    )
    rows: list[dict[str, float | int | str]] = []
    for train_seed in protocol.train_seeds:
        for config in configs:
            torch.manual_seed(train_seed)
            policy = build_policy("mlp", config, protocol).to(
                device=protocol.device,
                dtype=torch.float64,
            )
            initialize_channel_policy(policy, 30_000 + train_seed)
            comparison = gradient_comparison(
                policy,
                datasets[config.T],
                config,
                device=protocol.device,
            )
            with torch.no_grad():
                trace = simulate(
                    policy,
                    datasets[config.T],
                    config,
                    device=protocol.device,
                    dtype=torch.float64,
                )
            for row in comparison:
                row.update(
                    {
                        "train_seed": train_seed,
                        "horizon": config.T,
                        "n_fit_diagnostic_paths": datasets[config.T].n,
                        "num_parameters": count_parameters(policy),
                        "max_action_violation": float(
                            trace.max_action_violation.cpu()
                        ),
                        "max_storage_violation": float(
                            trace.max_storage_violation.cpu()
                        ),
                    }
                )
            rows.extend(comparison)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "horizon_gradient_diagnostics.csv", index=False)
    summary = (
        frame.groupby(["horizon", "gradient_mode"], sort=False)
        .agg(
            n_seeds=("train_seed", "nunique"),
            relative_gradient_error_mean=(
                "relative_difference_from_full",
                "mean",
            ),
            relative_gradient_error_train_seed_sd=(
                "relative_difference_from_full",
                "std",
            ),
            cosine_with_full_mean=("cosine_with_full", "mean"),
            full_gradient_norm_mean=("full_gradient_norm", "mean"),
            max_action_violation=("max_action_violation", "max"),
            max_storage_violation=("max_storage_violation", "max"),
        )
        .reset_index()
    )
    summary["relative_gradient_error_train_seed_sd"] = summary[
        "relative_gradient_error_train_seed_sd"
    ].fillna(0.0)
    summary.to_csv(output / "horizon_summary.csv", index=False)
    slope_rows = []
    for mode, group in summary.groupby("gradient_mode", sort=False):
        slope_rows.append(
            {
                "gradient_mode": mode,
                "log_log_amplification_slope": geometric_slope(
                    group["horizon"].to_numpy(dtype=np.float64),
                    group["relative_gradient_error_mean"].to_numpy(
                        dtype=np.float64
                    ),
                ),
            }
        )
    pd.DataFrame(slope_rows).to_csv(
        output / "amplification_slopes.csv", index=False
    )
    horizon_figure(frame, output / "figures" / "horizon_amplification.png")
    _write_experiment_note(
        output,
        protocol,
        {
            "experiment": "horizon_amplification",
            "data_used": "common standardized fit-shock prefixes",
            "selection_data": "none",
            "metric": "relative gradient-vector error against full pathwise gradient",
            "not_an_optimality_claim": True,
        },
    )
    return output


def run_graph_stability(protocol: StudyProtocol, root: Path) -> Path:
    output = root / "05_graph_stability"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    config = _graph_config(protocol)
    splits = generate_splits(protocol.data_config(), config)
    policies = (
        "projected_mlp_pathwise",
        "projected_graph_neural_global",
    )
    selected_fits: dict[tuple[str, int], SelectedFit] = {}
    base_costs: dict[tuple[str, int], np.ndarray] = {}
    base_rows: list[dict[str, float | int | str]] = []
    hyperparameter_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int | str]] = []

    for policy_name in policies:
        learning_rates = (
            protocol.mlp_learning_rates
            if policy_name == "projected_mlp_pathwise"
            else protocol.graph_learning_rates
        )
        for train_seed in protocol.train_seeds:
            selected = tune_and_select(
                lambda name=policy_name: build_policy(name, config, protocol),
                splits.fit,
                splits.validation,
                config,
                protocol,
                train_seed,
                learning_rates,
                "full",
                checkpoint_dir,
                policy_name,
            )
            selected_fits[(policy_name, train_seed)] = selected
            test_metrics, costs = evaluate_test(
                selected.policy,
                splits.test,
                config,
                protocol,
            )
            base_costs[(policy_name, train_seed)] = costs
            row: dict[str, float | int | str] = {
                "variant": policy_name,
                "policy": policy_name,
                "train_seed": train_seed,
                "selected_learning_rate": selected.learning_rate,
                "n_hyperparameter_candidates": len(selected.candidates),
                "best_validation_cost": selected.best_validation_cost,
                "selected_iteration": selected.selected_iteration,
                "checkpoint_path": selected.checkpoint_path,
                "fit_trajectory_information_budget": (
                    protocol.iterations * protocol.batch_size
                ),
                "tuning_trajectory_information_budget": (
                    protocol.iterations
                    * protocol.batch_size
                    * len(selected.candidates)
                ),
                "gradient_updates": protocol.iterations,
                "tuning_gradient_updates": (
                    protocol.iterations * len(selected.candidates)
                ),
                "environment_transitions": (
                    protocol.iterations * protocol.batch_size * config.T
                ),
                "tuning_environment_transitions": (
                    protocol.iterations
                    * protocol.batch_size
                    * config.T
                    * len(selected.candidates)
                ),
                "selected_fit_time_seconds": selected.selected_fit_time_seconds,
                "total_tuning_time_seconds": selected.total_tuning_time_seconds,
            }
            row.update(test_metrics)
            base_rows.append(row)
            _collect_fit_records(
                selected,
                policy_name,
                train_seed,
                hyperparameter_rows,
                history_rows,
            )

    base_frame = pd.DataFrame(base_rows)
    base_summary = summarize_train_seeds(base_frame)
    base_summary.to_csv(output / "base_topology_main_results.csv", index=False)
    base_frame.to_csv(output / "base_topology_per_run.csv", index=False)
    paired_ci(base_costs, baseline="projected_mlp_pathwise").to_csv(
        output / "base_policy_paired_test_path_ci.csv", index=False
    )
    pd.DataFrame(hyperparameter_rows).to_csv(
        output / "hyperparameter_candidates.csv", index=False
    )
    pd.DataFrame(history_rows).to_csv(
        output / "training_curves.csv", index=False
    )
    _wide_cost_frame(base_costs, splits.test.n).to_csv(
        output / "base_topology_test_path_costs.csv", index=False
    )

    topology_rows: list[dict[str, float | int | str]] = []
    paired_rows: list[dict[str, float | int | str]] = []
    for severity in protocol.graph_severities:
        topology_seeds = (
            protocol.graph_topology_seeds[:1]
            if severity == 0.0
            else protocol.graph_topology_seeds
        )
        for topology_seed in topology_seeds:
            target_config = perturb_routing(config, severity, topology_seed)
            for policy_name in policies:
                for train_seed in protocol.train_seeds:
                    source = selected_fits[(policy_name, train_seed)].policy
                    target = build_policy(policy_name, target_config, protocol).to(
                        protocol.device
                    )
                    copy_trainable_parameters(source, target)
                    metrics, costs = evaluate_test(
                        target,
                        splits.test,
                        target_config,
                        protocol,
                        measure_latency=False,
                    )
                    base = base_costs[(policy_name, train_seed)]
                    difference = costs - base
                    path_sd = (
                        float(difference.std(ddof=1))
                        if len(difference) > 1
                        else 0.0
                    )
                    path_se = path_sd / max(len(difference), 1) ** 0.5
                    mean_difference = float(difference.mean())
                    row: dict[str, float | int | str] = {
                        "policy": policy_name,
                        "train_seed": train_seed,
                        "topology_seed": topology_seed,
                        "severity": severity,
                        "base_test_mean_cost": float(base.mean()),
                        "cost_change": mean_difference,
                        "cost_change_percent": 100.0
                        * mean_difference
                        / max(abs(float(base.mean())), 1.0e-12),
                    }
                    row.update(metrics)
                    topology_rows.append(row)
                    paired_rows.append(
                        {
                            "policy": policy_name,
                            "train_seed": train_seed,
                            "topology_seed": topology_seed,
                            "severity": severity,
                            "n_paired_test_paths": len(difference),
                            "mean_perturbed_minus_base_cost": mean_difference,
                            "paired_test_path_sd": path_sd,
                            "paired_test_path_se": path_se,
                            "paired_test_ci95_lower": mean_difference
                            - 1.96 * path_se,
                            "paired_test_ci95_upper": mean_difference
                            + 1.96 * path_se,
                        }
                    )

    topology = pd.DataFrame(topology_rows)
    topology.to_csv(output / "routing_perturbation_results.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(
        output / "routing_perturbation_paired_test_path_ci.csv", index=False
    )
    topology_summary = _topology_summary(topology)
    base_metadata = base_summary.rename(columns={"variant": "policy"})[
        [
            "policy",
            "num_parameters",
            "gradient_updates_per_selected_fit",
            "tuning_gradient_updates_mean",
            "selected_fit_time_seconds_mean",
            "total_tuning_time_seconds_mean",
            "batch1_action_latency_ms_median_across_seeds",
            "batched_action_vectors_per_second_mean",
        ]
    ]
    topology_summary = topology_summary.merge(
        base_metadata, on="policy", how="left", validate="many_to_one"
    )
    topology_summary.to_csv(output / "main_results.csv", index=False)

    equivariance = _permutation_equivariance(
        selected_fits,
        policies,
        splits.test,
        config,
        protocol,
    )
    equivariance.to_csv(output / "permutation_equivariance.csv", index=False)
    graph_stability_figure(
        topology,
        output / "figures" / "routing_stability.png",
    )
    equivariance_figure(
        equivariance,
        output / "figures" / "permutation_equivariance.png",
    )
    _write_experiment_note(
        output,
        protocol,
        {
            "experiment": "graph_stability",
            "definition": [
                "exact reservoir-label permutation equivariance diagnostic",
                "zero-shot routing reweighting/rerouting stress test",
            ],
            "ramp_fraction": config.ramp_fraction,
            "selection_data": "base-topology validation only",
            "reported_data": "common test paths after all selections are frozen",
            "reserved_blocks_not_read": ["calibration", "evaluation"],
            "data_fingerprints": _split_fingerprints(splits),
            "optimality_language": (
                "This nonlinear hydro experiment compares implemented policies only; "
                "it has no exact optimum."
            ),
        },
    )
    return output


def _permutation_equivariance(
    selected_fits: dict[tuple[str, int], SelectedFit],
    policies: tuple[str, ...],
    test,
    config: ExperimentConfig,
    protocol: StudyProtocol,
) -> pd.DataFrame:
    subset = test.take(np.arange(min(test.n, 64)), name="test_permutation_subset")
    rows: list[dict[str, float | int | str]] = []
    for permutation_seed in protocol.graph_topology_seeds:
        permutation = np.random.default_rng(permutation_seed).permutation(
            config.n_reservoirs
        )
        if np.array_equal(permutation, np.arange(config.n_reservoirs)):
            permutation = np.roll(permutation, 1)
        inverse = np.argsort(permutation)
        target_config = permute_config(config, permutation)
        target_data = permute_dataset(subset, permutation)
        for policy_name in policies:
            for train_seed in protocol.train_seeds:
                source = selected_fits[(policy_name, train_seed)].policy
                target = build_policy(policy_name, target_config, protocol).to(
                    protocol.device
                )
                copy_trainable_parameters(source, target)
                with torch.no_grad():
                    source_trace = simulate(
                        source,
                        subset,
                        config,
                        device=protocol.device,
                    )
                    target_trace = simulate(
                        target,
                        target_data,
                        target_config,
                        device=protocol.device,
                    )
                reordered = target_trace.actions[..., inverse]
                reordered_proposals = target_trace.proposals[..., inverse]
                action_error = (reordered - source_trace.actions).abs()
                proposal_error = (
                    reordered_proposals - source_trace.proposals
                ).abs()
                cost_error = (
                    target_trace.total_cost - source_trace.total_cost
                ).abs()
                rows.append(
                    {
                        "policy": policy_name,
                        "train_seed": train_seed,
                        "permutation_seed": permutation_seed,
                        "n_test_paths": subset.n,
                        "max_action_equivariance_error": float(action_error.max().cpu()),
                        "mean_action_equivariance_error": float(
                            action_error.mean().cpu()
                        ),
                        "max_proposal_equivariance_error": float(
                            proposal_error.max().cpu()
                        ),
                        "mean_proposal_equivariance_error": float(
                            proposal_error.mean().cpu()
                        ),
                        "max_path_cost_invariance_error": float(cost_error.max().cpu()),
                        "max_target_action_violation": float(
                            target_trace.max_action_violation.cpu()
                        ),
                        "max_target_storage_violation": float(
                            target_trace.max_storage_violation.cpu()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _topology_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (policy, severity), group in frame.groupby(
        ["policy", "severity"], sort=False
    ):
        seed_means = group.groupby("train_seed").agg(
            test_mean_cost=("test_mean_cost", "mean"),
            cost_change_percent=("cost_change_percent", "mean"),
        )
        rows.append(
            {
                "policy": policy,
                "severity": severity,
                "n_train_seeds": seed_means.shape[0],
                "n_topology_instances": group["topology_seed"].nunique(),
                "test_cost_mean_over_train_seeds": float(
                    seed_means["test_mean_cost"].mean()
                ),
                "test_cost_train_seed_sd": (
                    float(seed_means["test_mean_cost"].std(ddof=1))
                    if len(seed_means) > 1
                    else 0.0
                ),
                "cost_change_percent_mean": float(
                    seed_means["cost_change_percent"].mean()
                ),
                "cost_change_percent_train_seed_sd": (
                    float(seed_means["cost_change_percent"].std(ddof=1))
                    if len(seed_means) > 1
                    else 0.0
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
    return pd.DataFrame(rows)


def _collect_fit_records(
    selected: SelectedFit,
    variant: str,
    train_seed: int,
    hyperparameter_rows: list[dict[str, float | int | str]],
    history_rows: list[dict[str, float | int | str]],
) -> None:
    for candidate_index, candidate in enumerate(selected.candidates):
        is_selected = int(
            candidate.learning_rate == selected.learning_rate
            and candidate.best_validation_cost == selected.best_validation_cost
        )
        hyperparameter_rows.append(
            {
                "variant": variant,
                "train_seed": train_seed,
                "candidate_index": candidate_index,
                "learning_rate": candidate.learning_rate,
                "best_validation_cost": candidate.best_validation_cost,
                "selected_iteration": candidate.selected_iteration,
                "fit_time_seconds": candidate.fit_time_seconds,
                "selected": is_selected,
                "checkpoint_path": candidate.checkpoint_path,
            }
        )
        for history in candidate.history:
            row = dict(history)
            row["variant"] = variant
            row["candidate_index"] = candidate_index
            row["candidate_selected"] = is_selected
            history_rows.append(row)


def _wide_cost_frame(
    costs: dict[tuple[str, int], np.ndarray], n_test: int
) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {"test_path": np.arange(n_test)}
    for (variant, seed), values in sorted(costs.items()):
        columns[f"{variant}_seed{seed}"] = values
    return pd.DataFrame(columns)


def _base_config(protocol: StudyProtocol) -> ExperimentConfig:
    return replace(
        ExperimentConfig.make(
            protocol.profile,
            protocol.n_reservoirs,
            protocol.horizon,
            protocol.shock_factors,
        ),
        ramp_fraction=protocol.study_ramp_fraction,
    )


def _graph_config(protocol: StudyProtocol) -> ExperimentConfig:
    return ExperimentConfig.make(
        protocol.profile,
        protocol.n_reservoirs,
        protocol.horizon,
        protocol.shock_factors,
    )


def _write_experiment_note(
    output: Path,
    protocol: StudyProtocol,
    details: dict[str, object],
) -> None:
    with (output / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"protocol": asdict(protocol), **details},
            handle,
            indent=2,
        )


def _split_fingerprints(splits) -> dict[str, str]:
    return {
        name: _dataset_fingerprint(getattr(splits, name))
        for name in ("fit", "validation", "test")
    }


def _dataset_fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    for values in (dataset.inflow, dataset.demand, dataset.price):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()
