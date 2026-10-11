"""Nested trajectory-budget comparison with common paths and fresh test data."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algorithm1_gradient_study.training import tune_and_select
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.policies import MyopicPolicy, count_parameters
from efficiency_switching_hydro.rl import fit_checkpoint as fit_rl_checkpoint

from .config import ICLRProtocol
from .evaluation import evaluate_policy, pairwise_by_group, summarize_runs
from .io import (
    canonical_seed,
    write_completion,
    write_data_manifest,
    write_frame,
    write_protocol,
)
from .policies import build_policy, parameter_matched_mlp_width
from .shac import SHACConfig, tune_shac


POLICY_ORDER = (
    "myopic",
    "projected_mlp_matched",
    "projected_graph_neural_global",
    "projected_shac",
    "rl_ppo",
    "rl_sac",
)


def _learning_rates(protocol: ICLRProtocol, policy: str) -> tuple[float, ...]:
    return {
        "projected_mlp_matched": protocol.mlp_learning_rates,
        "projected_graph_neural_global": protocol.graph_learning_rates,
        "projected_shac": protocol.shac_learning_rates,
        "rl_ppo": protocol.ppo_learning_rates,
        "rl_sac": protocol.sac_learning_rates,
    }[policy]


def pre_registered_candidates(protocol: ICLRProtocol) -> pd.DataFrame:
    rows = []
    for n_fit in protocol.fit_budgets:
        rows.append(
            {
                "n_fit": n_fit,
                "policy": "myopic",
                "train_seed": "fixed",
                "candidate_index": 0,
                "learning_rate": np.nan,
                "shac_window": np.nan,
                "status": "fixed",
            }
        )
        for policy in POLICY_ORDER[1:]:
            for train_seed in protocol.train_seeds:
                for index, learning_rate in enumerate(_learning_rates(protocol, policy)):
                    rows.append(
                        {
                            "n_fit": n_fit,
                            "policy": policy,
                            "train_seed": train_seed,
                            "candidate_index": index,
                            "learning_rate": learning_rate,
                            "shac_window": (
                                protocol.shac_window
                                if policy == "projected_shac"
                                else np.nan
                            ),
                            "status": "fit_candidate",
                        }
                    )
    return pd.DataFrame(rows)


def _direct_run(
    policy_name: str,
    fit,
    validation,
    test,
    hydro,
    protocol: ICLRProtocol,
    train_seed: int,
    checkpoint_dir: Path,
):
    training = protocol.direct_training(fit.n)
    selected = tune_and_select(
        lambda: build_policy(policy_name, hydro, protocol),
        fit,
        validation,
        hydro,
        training,
        train_seed,
        _learning_rates(protocol, policy_name),
        "full",
        checkpoint_dir,
        policy_name,
    )
    parameters = count_parameters(selected.policy)
    metrics, costs = evaluate_policy(
        selected.policy,
        test,
        hydro,
        protocol,
        deployment_parameters=parameters,
        training_parameters=parameters,
    )
    row = {
        "n_fit": fit.n,
        "policy": policy_name,
        "family": "algorithm1_pathwise",
        "train_seed": train_seed,
        "selected_learning_rate": selected.learning_rate,
        "selected_iteration": selected.selected_iteration,
        "best_validation_cost": selected.best_validation_cost,
        "n_hyperparameter_candidates": len(selected.candidates),
        "environment_transitions_per_candidate": fit.n * hydro.T,
        "actor_gradient_updates": training.iterations,
        "critic_gradient_updates": 0,
        "selected_fit_time_seconds": selected.selected_fit_time_seconds,
        "total_tuning_time_seconds": selected.total_tuning_time_seconds,
        "checkpoint_path": selected.checkpoint_path,
    }
    row.update(metrics)
    candidates = []
    histories = []
    for index, candidate in enumerate(selected.candidates):
        is_selected = int(candidate.checkpoint_path == selected.candidates[
            min(
                range(len(selected.candidates)),
                key=lambda item: selected.candidates[item].best_validation_cost,
            )
        ].checkpoint_path)
        candidates.append(
            {
                "n_fit": fit.n,
                "policy": policy_name,
                "train_seed": train_seed,
                "candidate_index": index,
                "learning_rate": candidate.learning_rate,
                "shac_window": np.nan,
                "best_validation_cost": candidate.best_validation_cost,
                "selected_iteration": candidate.selected_iteration,
                "fit_time_seconds": candidate.fit_time_seconds,
                "environment_transitions": fit.n * hydro.T,
                "actor_gradient_updates": training.iterations,
                "critic_gradient_updates": 0,
                "selected": is_selected,
                "checkpoint_path": candidate.checkpoint_path,
            }
        )
        for record in candidate.history:
            history = dict(record)
            history.update(
                {
                    "n_fit": fit.n,
                    "policy": policy_name,
                    "candidate_index": index,
                    "candidate_selected": is_selected,
                    "environment_transitions": int(
                        history.get("fit_trajectories_seen", 0)
                    )
                    * hydro.T,
                    "actor_gradient_updates": int(history.get("gradient_updates", 0)),
                    "critic_gradient_updates": 0,
                }
            )
            histories.append(history)
    return row, costs, candidates, histories


def _shac_run(
    fit,
    validation,
    test,
    hydro,
    protocol: ICLRProtocol,
    train_seed: int,
    checkpoint_dir: Path,
):
    width, _, _ = parameter_matched_mlp_width(
        hydro,
        graph_width=protocol.graph_width,
        depth=protocol.network_depth,
    )
    config = SHACConfig(
        iterations=protocol.iterations(fit.n),
        batch_size=protocol.batch_size,
        validation_every=protocol.validation_every(fit.n),
        window=min(protocol.shac_window, hydro.T),
        actor_learning_rate=protocol.shac_learning_rates[0],
        critic_learning_rate=protocol.shac_critic_learning_rate,
        critic_epochs=protocol.shac_critic_epochs,
        critic_tau=protocol.shac_critic_tau,
        cost_scale=protocol.shac_cost_scale,
        weight_decay=protocol.weight_decay,
        gradient_clip=protocol.gradient_clip,
        critic_width=width,
        critic_depth=protocol.network_depth,
        device=protocol.device,
    )
    selected = tune_shac(
        lambda: build_policy("projected_shac", hydro, protocol),
        fit,
        validation,
        hydro,
        config,
        train_seed,
        protocol.shac_learning_rates,
        checkpoint_dir,
    )
    chosen = min(selected.candidates, key=lambda item: item.best_validation_cost)
    metrics, costs = evaluate_policy(
        selected.policy,
        test,
        hydro,
        protocol,
        deployment_parameters=chosen.deployment_parameters,
        training_parameters=chosen.training_parameters,
    )
    row = {
        "n_fit": fit.n,
        "policy": "projected_shac",
        "family": "first_order_actor_critic",
        "train_seed": train_seed,
        "selected_learning_rate": selected.actor_learning_rate,
        "selected_iteration": selected.selected_iteration,
        "best_validation_cost": selected.best_validation_cost,
        "n_hyperparameter_candidates": len(selected.candidates),
        "shac_window": config.window,
        "environment_transitions_per_candidate": chosen.environment_transitions,
        "actor_gradient_updates": chosen.actor_gradient_updates,
        "critic_gradient_updates": chosen.critic_gradient_updates,
        "selected_fit_time_seconds": selected.selected_fit_time_seconds,
        "total_tuning_time_seconds": selected.total_tuning_time_seconds,
        "checkpoint_path": selected.checkpoint_path,
    }
    row.update(metrics)
    candidates = []
    histories = []
    for index, candidate in enumerate(selected.candidates):
        is_selected = int(candidate.checkpoint_path == chosen.checkpoint_path)
        candidates.append(
            {
                "n_fit": fit.n,
                "policy": "projected_shac",
                "train_seed": train_seed,
                "candidate_index": index,
                "learning_rate": candidate.actor_learning_rate,
                "shac_window": config.window,
                "best_validation_cost": candidate.best_validation_cost,
                "selected_iteration": candidate.selected_iteration,
                "fit_time_seconds": candidate.fit_time_seconds,
                "environment_transitions": candidate.environment_transitions,
                "actor_gradient_updates": candidate.actor_gradient_updates,
                "critic_gradient_updates": candidate.critic_gradient_updates,
                "selected": is_selected,
                "checkpoint_path": candidate.checkpoint_path,
            }
        )
        for record in candidate.history:
            history = dict(record)
            history.update(
                {
                    "n_fit": fit.n,
                    "policy": "projected_shac",
                    "candidate_index": index,
                    "candidate_selected": is_selected,
                }
            )
            histories.append(history)
    return row, costs, candidates, histories


def _rl_run(
    algorithm: str,
    fit,
    validation,
    test,
    hydro,
    protocol: ICLRProtocol,
    train_seed: int,
    checkpoint_dir: Path,
):
    policy_name = f"rl_{algorithm}"
    rates = _learning_rates(protocol, policy_name)
    candidate_records = []
    candidate_results = []
    history_records = []
    fit_times = []
    for index, learning_rate in enumerate(rates):
        rate = f"{learning_rate:.8g}".replace(".", "p")
        start = time.perf_counter()
        result = fit_rl_checkpoint(
            algorithm,
            fit,
            validation,
            hydro,
            protocol.rl_config(algorithm, learning_rate, fit.n),
            train_seed,
            checkpoint_dir
            / f"{policy_name}_candidate{index}_lr{rate}_seed{train_seed}.zip",
            evaluation_device=protocol.device,
        )
        elapsed = time.perf_counter() - start
        fit_times.append(elapsed)
        candidate_results.append(result)
    selected_index = min(
        range(len(candidate_results)),
        key=lambda index: candidate_results[index].best_validation_cost,
    )
    selected = candidate_results[selected_index]
    metrics, costs = evaluate_policy(
        selected.policy,
        test,
        hydro,
        protocol,
        deployment_parameters=selected.deployment_parameters,
        training_parameters=selected.training_parameters,
    )
    row = {
        "n_fit": fit.n,
        "policy": policy_name,
        "family": "model_free_rl",
        "train_seed": train_seed,
        "selected_learning_rate": rates[selected_index],
        "selected_iteration": selected.selected_iteration,
        "best_validation_cost": selected.best_validation_cost,
        "n_hyperparameter_candidates": len(candidate_results),
        "environment_transitions_per_candidate": selected.environment_steps,
        "actor_gradient_updates": selected.actor_gradient_updates,
        "critic_gradient_updates": selected.critic_gradient_updates,
        "selected_fit_time_seconds": fit_times[selected_index],
        "total_tuning_time_seconds": sum(fit_times),
        "checkpoint_path": selected.checkpoint_path,
    }
    row.update(metrics)
    for index, result in enumerate(candidate_results):
        is_selected = int(index == selected_index)
        candidate_records.append(
            {
                "n_fit": fit.n,
                "policy": policy_name,
                "train_seed": train_seed,
                "candidate_index": index,
                "learning_rate": rates[index],
                "shac_window": np.nan,
                "best_validation_cost": result.best_validation_cost,
                "selected_iteration": result.selected_iteration,
                "fit_time_seconds": fit_times[index],
                "environment_transitions": result.environment_steps,
                "actor_gradient_updates": result.actor_gradient_updates,
                "critic_gradient_updates": result.critic_gradient_updates,
                "selected": is_selected,
                "checkpoint_path": result.checkpoint_path,
            }
        )
        for record in result.history:
            history = dict(record)
            history.update(
                {
                    "n_fit": fit.n,
                    "policy": policy_name,
                    "candidate_index": index,
                    "candidate_selected": is_selected,
                    "environment_transitions": int(history.get("environment_steps", 0)),
                    "actor_gradient_updates": np.nan,
                    "critic_gradient_updates": np.nan,
                }
            )
            history_records.append(history)
    for result in candidate_results:
        result.policy.close()
    return row, costs, candidate_records, history_records


def _fixed_run(n_fit, test, hydro, protocol):
    policy = MyopicPolicy(hydro)
    metrics, costs = evaluate_policy(
        policy,
        test,
        hydro,
        protocol,
        deployment_parameters=0,
        training_parameters=0,
    )
    row = {
        "n_fit": n_fit,
        "policy": "myopic",
        "family": "fixed_baseline",
        "train_seed": "fixed",
        "selected_learning_rate": np.nan,
        "selected_iteration": 0,
        "best_validation_cost": np.nan,
        "n_hyperparameter_candidates": 0,
        "environment_transitions_per_candidate": 0,
        "actor_gradient_updates": 0,
        "critic_gradient_updates": 0,
        "selected_fit_time_seconds": 0.0,
        "total_tuning_time_seconds": 0.0,
        "checkpoint_path": "",
    }
    row.update(metrics)
    return row, costs


def _cost_frame(costs: dict[tuple[int, str, str], np.ndarray], n_test: int) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {"test_path": np.arange(n_test)}
    for (n_fit, policy, seed), values in sorted(costs.items()):
        columns[f"n{n_fit}_{policy}_seed{seed}"] = values
    return pd.DataFrame(columns)


def _load_partial(output: Path):
    def records(name: str):
        path = output / name
        return pd.read_csv(path).to_dict("records") if path.exists() else []

    per_run = records("partial_per_run_results.csv")
    candidates = records("partial_hyperparameter_candidates.csv")
    histories = records("partial_training_curves.csv")
    costs: dict[tuple[int, str, str], np.ndarray] = {}
    path = output / "partial_test_path_costs.csv"
    if path.exists():
        frame = pd.read_csv(path)
        for column in frame.columns[1:]:
            prefix, remainder = column.split("_", 1)
            n_fit = int(prefix[1:])
            policy, seed = remainder.rsplit("_seed", 1)
            costs[(n_fit, policy, seed)] = frame[column].to_numpy(dtype=np.float64)
    return per_run, candidates, histories, costs


def _flush(output, per_run, candidates, histories, costs, n_test):
    write_frame(pd.DataFrame(per_run), output / "partial_per_run_results.csv")
    write_frame(
        pd.DataFrame(candidates), output / "partial_hyperparameter_candidates.csv"
    )
    write_frame(pd.DataFrame(histories), output / "partial_training_curves.csv")
    write_frame(
        _cost_frame(costs, n_test), output / "partial_test_path_costs.csv"
    )


def _plot(summary: pd.DataFrame, output: Path) -> None:
    labels = {
        "myopic": "Myopic",
        "projected_mlp_matched": "Matched MLP",
        "projected_graph_neural_global": "Graph",
        "projected_shac": "Projected SHAC",
        "rl_ppo": "PPO",
        "rl_sac": "SAC",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for policy, group in summary.groupby("policy", sort=False):
        ordered = group.sort_values("n_fit")
        ax.errorbar(
            ordered["n_fit"],
            ordered["test_cost_mean_over_train_seeds"],
            yerr=ordered["test_cost_train_seed_sd"],
            marker="o",
            capsize=3,
            linewidth=2,
            label=labels.get(policy, policy),
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["n_fit"].unique()))
    ax.set_xticklabels([str(value) for value in sorted(summary["n_fit"].unique())])
    ax.set_xlabel("Unique fit trajectories")
    ax.set_ylabel("Mean test cost (lower is better)")
    ax.set_title("Trajectory-budget efficiency on nonlinear hydro")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_data_budget(
    protocol: ICLRProtocol,
    output: Path,
    resume: bool = True,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    candidates_frozen = pre_registered_candidates(protocol)
    preregistration_path = output / "pre_registered_candidates.csv"
    if preregistration_path.exists() and resume:
        existing = pd.read_csv(preregistration_path)
        if existing.fillna("NA").astype(str).to_dict("records") != candidates_frozen.fillna(
            "NA"
        ).astype(str).to_dict("records"):
            raise RuntimeError("Existing candidate preregistration does not match this protocol.")
    else:
        write_frame(candidates_frozen, preregistration_path)

    hydro = protocol.hydro_config()
    fit_pool = generate_dataset(
        "fit_pool",
        max(protocol.fit_budgets),
        protocol.budget_seeds.fit,
        hydro,
    )
    validation = generate_dataset(
        "validation",
        protocol.n_validation,
        protocol.budget_seeds.validation,
        hydro,
    )
    test = generate_dataset(
        "test",
        protocol.n_test,
        protocol.budget_seeds.test,
        hydro,
    )
    write_protocol(
        output,
        protocol,
        "trajectory_budget",
        {
            "policies": list(POLICY_ORDER),
            "nested_fit_pool": True,
            "parameter_matching": "flat MLP width minimizes parameter-count difference to graph",
            "test_seed_is_new": True,
        },
    )
    write_data_manifest(
        output,
        {"fit_pool": fit_pool, "validation": validation, "test": test},
        {
            "calibration": (protocol.budget_seeds.calibration, protocol.n_calibration),
            "evaluation": (protocol.budget_seeds.evaluation, protocol.n_evaluation),
        },
    )

    if resume:
        per_run, candidate_rows, histories, costs = _load_partial(output)
    else:
        per_run, candidate_rows, histories, costs = [], [], [], {}
    completed = {
        (int(row["n_fit"]), str(row["policy"]), canonical_seed(row["train_seed"]))
        for row in per_run
    }

    for n_fit in protocol.fit_budgets:
        fit = fit_pool.take(np.arange(n_fit), name=f"fit_n{n_fit}")
        fixed_key = (n_fit, "myopic", "fixed")
        if fixed_key not in completed:
            row, path_costs = _fixed_run(n_fit, test, hydro, protocol)
            per_run.append(row)
            costs[fixed_key] = path_costs
            completed.add(fixed_key)
            _flush(output, per_run, candidate_rows, histories, costs, test.n)

        for policy_name in POLICY_ORDER[1:]:
            for train_seed in protocol.train_seeds:
                key = (n_fit, policy_name, str(train_seed))
                if key in completed:
                    continue
                checkpoint_dir = output / "checkpoints" / f"n{n_fit}"
                print(
                    f"data budget N={n_fit}: {policy_name}, seed={train_seed}",
                    flush=True,
                )
                if policy_name in {
                    "projected_mlp_matched",
                    "projected_graph_neural_global",
                }:
                    result = _direct_run(
                        policy_name,
                        fit,
                        validation,
                        test,
                        hydro,
                        protocol,
                        train_seed,
                        checkpoint_dir,
                    )
                elif policy_name == "projected_shac":
                    result = _shac_run(
                        fit,
                        validation,
                        test,
                        hydro,
                        protocol,
                        train_seed,
                        checkpoint_dir,
                    )
                else:
                    result = _rl_run(
                        policy_name.removeprefix("rl_"),
                        fit,
                        validation,
                        test,
                        hydro,
                        protocol,
                        train_seed,
                        checkpoint_dir,
                    )
                row, path_costs, new_candidates, new_histories = result
                per_run.append(row)
                candidate_rows.extend(new_candidates)
                histories.extend(new_histories)
                costs[key] = path_costs
                completed.add(key)
                _flush(output, per_run, candidate_rows, histories, costs, test.n)

    per_run_frame = pd.DataFrame(per_run)
    candidate_frame = pd.DataFrame(candidate_rows)
    history_frame = pd.DataFrame(histories)
    summary = summarize_runs(per_run_frame, ["n_fit"])
    pairwise, pairwise_summary = pairwise_by_group(costs, POLICY_ORDER, "n_fit")
    write_frame(per_run_frame, output / "per_run_results.csv")
    write_frame(candidate_frame, output / "hyperparameter_candidates.csv")
    write_frame(history_frame, output / "training_curves.csv")
    write_frame(summary, output / "main_results.csv")
    write_frame(_cost_frame(costs, test.n), output / "test_path_costs.csv")
    write_frame(pairwise, output / "paired_test_path_ci.csv")
    write_frame(pairwise_summary, output / "paired_comparison_summary.csv")
    _plot(summary, output / "figures" / "trajectory_budget.png")
    write_completion(
        output,
        "trajectory_budget",
        {
            "n_fit_budgets": len(protocol.fit_budgets),
            "n_final_runs": len(per_run_frame),
            "n_candidates": len(candidate_frame),
            "all_constraint_violations_zero": bool(
                summary["total_constraint_violation_count"].sum() == 0
            ),
        },
    )
    return output
