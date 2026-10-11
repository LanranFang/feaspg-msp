"""Fresh-test topology and cross-size transfer for frozen graph policies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algorithm1_gradient_study.models import perturb_routing
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.policies import MyopicPolicy, count_parameters

from .config import ICLRProtocol
from .evaluation import evaluate_policy, pairwise_by_group, summarize_runs
from .io import (
    canonical_seed,
    dataset_fingerprint,
    write_completion,
    write_frame,
    write_protocol,
)
from .policies import build_policy, copy_trainable_parameters


def _load_policy(policy_name, config, protocol, checkpoint_path):
    policy = build_policy(policy_name, config, protocol)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["state_dict"])
    policy.to(protocol.device)
    policy.eval()
    return policy


def _selected_checkpoint(
    scaling_runs: pd.DataFrame,
    config_id: str,
    policy: str,
    train_seed: int,
) -> str:
    selected = scaling_runs[
        (scaling_runs["config_id"] == config_id)
        & (scaling_runs["policy"] == policy)
        & (scaling_runs["train_seed"].map(canonical_seed) == str(train_seed))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one scaling checkpoint for {config_id}, {policy}, seed {train_seed}."
        )
    return str(selected.iloc[0]["checkpoint_path"])


def _run_row(
    policy,
    policy_name,
    family,
    train_seed,
    test,
    config,
    protocol,
    checkpoint_path,
):
    parameters = count_parameters(policy)
    metrics, costs = evaluate_policy(
        policy,
        test,
        config,
        protocol,
        deployment_parameters=parameters,
        training_parameters=parameters,
    )
    row = {
        "policy": policy_name,
        "family": family,
        "train_seed": train_seed,
        "selected_learning_rate": np.nan,
        "selected_iteration": 0,
        "best_validation_cost": np.nan,
        "environment_transitions_per_candidate": 0,
        "actor_gradient_updates": 0,
        "critic_gradient_updates": 0,
        "selected_fit_time_seconds": 0.0,
        "total_tuning_time_seconds": 0.0,
        "checkpoint_path": checkpoint_path,
    }
    row.update(metrics)
    return row, costs


def _same_size_topology(
    protocol: ICLRProtocol,
    scaling_runs: pd.DataFrame,
    output: Path,
):
    base = protocol.hydro_config()
    test = generate_dataset(
        "topology_test",
        protocol.n_test,
        protocol.transfer_seeds.test,
        base,
    )
    policies = ("projected_mlp_matched", "projected_graph_neural_global")
    sources = {}
    base_costs = {}
    for policy_name in policies:
        for train_seed in protocol.train_seeds:
            checkpoint = _selected_checkpoint(
                scaling_runs,
                f"R{protocol.n_reservoirs}_T{protocol.horizon}",
                policy_name,
                train_seed,
            )
            source = _load_policy(policy_name, base, protocol, checkpoint)
            sources[(policy_name, train_seed)] = (source, checkpoint)
            _, costs = _run_row(
                source,
                policy_name,
                "frozen_base",
                train_seed,
                test,
                base,
                protocol,
                checkpoint,
            )
            base_costs[(policy_name, train_seed)] = costs

    rows = []
    paired_rows = []
    costs_by_severity = {}
    for severity in protocol.routing_severities:
        for topology_seed in protocol.routing_topology_seeds:
            target_config = perturb_routing(base, severity, topology_seed)
            group = f"severity{severity:g}_topology{topology_seed}"
            for policy_name in policies:
                for train_seed in protocol.train_seeds:
                    source, checkpoint = sources[(policy_name, train_seed)]
                    target = build_policy(policy_name, target_config, protocol).to(
                        protocol.device
                    )
                    copy_trainable_parameters(source, target)
                    row, costs = _run_row(
                        target,
                        policy_name,
                        "zero_shot_topology",
                        train_seed,
                        test,
                        target_config,
                        protocol,
                        checkpoint,
                    )
                    difference = costs - base_costs[(policy_name, train_seed)]
                    path_sd = float(difference.std(ddof=1))
                    path_se = path_sd / len(difference) ** 0.5
                    row.update(
                        {
                            "transfer_group": group,
                            "severity": severity,
                            "topology_seed": topology_seed,
                            "base_test_cost": float(
                                base_costs[(policy_name, train_seed)].mean()
                            ),
                            "cost_change_percent": 100.0
                            * float(difference.mean())
                            / max(
                                abs(float(base_costs[(policy_name, train_seed)].mean())),
                                1.0e-12,
                            ),
                        }
                    )
                    rows.append(row)
                    costs_by_severity[(group, policy_name, str(train_seed))] = costs
                    paired_rows.append(
                        {
                            "transfer_group": group,
                            "policy": policy_name,
                            "train_seed": train_seed,
                            "severity": severity,
                            "topology_seed": topology_seed,
                            "n_paired_test_paths": len(difference),
                            "mean_perturbed_minus_base_cost": float(difference.mean()),
                            "paired_test_path_sd": path_sd,
                            "paired_test_ci95_lower": float(difference.mean()) - 1.96 * path_se,
                            "paired_test_ci95_upper": float(difference.mean()) + 1.96 * path_se,
                        }
                    )
    frame = pd.DataFrame(rows)
    summary_rows = []
    for (policy, severity), group in frame.groupby(["policy", "severity"], sort=False):
        seed_means = group.groupby("train_seed").agg(
            test_cost=("test_mean_cost", "mean"),
            change=("cost_change_percent", "mean"),
        )
        summary_rows.append(
            {
                "policy": policy,
                "severity": severity,
                "n_train_seeds": len(seed_means),
                "n_topologies": group["topology_seed"].nunique(),
                "test_cost_mean_over_train_seeds": float(seed_means["test_cost"].mean()),
                "test_cost_train_seed_sd": float(seed_means["test_cost"].std(ddof=1))
                if len(seed_means) > 1
                else 0.0,
                "cost_change_percent_mean": float(seed_means["change"].mean()),
                "cost_change_percent_train_seed_sd": float(
                    seed_means["change"].std(ddof=1)
                )
                if len(seed_means) > 1
                else 0.0,
                "max_action_violation": float(group["max_action_violation"].max()),
                "max_storage_violation": float(group["max_storage_violation"].max()),
                "total_constraint_violation_count": int(
                    group["total_constraint_violation_count"].sum()
                ),
            }
        )
    pairwise, pairwise_summary = pairwise_by_group(
        costs_by_severity,
        policies,
        "transfer_group",
    )
    write_frame(frame, output / "topology_per_run_results.csv")
    write_frame(pd.DataFrame(summary_rows), output / "topology_main_results.csv")
    write_frame(pd.DataFrame(paired_rows), output / "topology_vs_base_paired_ci.csv")
    write_frame(pairwise, output / "topology_policy_paired_ci.csv")
    write_frame(pairwise_summary, output / "topology_policy_paired_summary.csv")
    return pd.DataFrame(summary_rows), test


def _target_configurations(protocol: ICLRProtocol):
    values = []
    for reservoirs in protocol.reservoir_values:
        if reservoirs != protocol.n_reservoirs:
            values.append((reservoirs, protocol.horizon, "reservoir"))
    for horizon in protocol.horizon_values:
        if horizon != protocol.horizon:
            values.append((protocol.n_reservoirs, horizon, "horizon"))
    return tuple(values)


def _cross_size_transfer(
    protocol: ICLRProtocol,
    scaling_runs: pd.DataFrame,
    output: Path,
):
    base_id = f"R{protocol.n_reservoirs}_T{protocol.horizon}"
    base_config = protocol.hydro_config()
    source_policies = {}
    for train_seed in protocol.train_seeds:
        checkpoint = _selected_checkpoint(
            scaling_runs,
            base_id,
            "projected_graph_neural_global",
            train_seed,
        )
        source_policies[train_seed] = (
            _load_policy(
                "projected_graph_neural_global",
                base_config,
                protocol,
                checkpoint,
            ),
            checkpoint,
        )

    rows = []
    costs = {}
    manifest = {}
    policy_order = ("myopic", "zero_shot_graph", "target_trained_graph")
    for reservoirs, horizon, axis in _target_configurations(protocol):
        config_id = f"R{reservoirs}_T{horizon}"
        target_config = protocol.hydro_config(reservoirs, horizon)
        test = generate_dataset(
            f"transfer_{config_id}",
            protocol.n_test,
            protocol.transfer_seeds.test,
            target_config,
        )
        manifest[config_id] = {
            "seed": test.seed,
            "n_paths": test.n,
            "sha256": dataset_fingerprint(test),
            "role": "fresh transfer test",
        }
        myopic_row, myopic_costs = _run_row(
            MyopicPolicy(target_config),
            "myopic",
            "fixed_baseline",
            "fixed",
            test,
            target_config,
            protocol,
            "",
        )
        myopic_row.update(
            {
                "config_id": config_id,
                "n_reservoirs": reservoirs,
                "horizon": horizon,
                "transfer_axis": axis,
            }
        )
        rows.append(myopic_row)
        costs[(config_id, "myopic", "fixed")] = myopic_costs

        for train_seed in protocol.train_seeds:
            source, source_checkpoint = source_policies[train_seed]
            zero_shot = build_policy(
                "projected_graph_neural_global", target_config, protocol
            ).to(protocol.device)
            copy_trainable_parameters(source, zero_shot)
            zero_row, zero_costs = _run_row(
                zero_shot,
                "zero_shot_graph",
                "zero_shot_transfer",
                train_seed,
                test,
                target_config,
                protocol,
                source_checkpoint,
            )
            target_checkpoint = _selected_checkpoint(
                scaling_runs,
                config_id,
                "projected_graph_neural_global",
                train_seed,
            )
            target_trained = _load_policy(
                "projected_graph_neural_global",
                target_config,
                protocol,
                target_checkpoint,
            )
            target_row, target_costs = _run_row(
                target_trained,
                "target_trained_graph",
                "target_trained_reference",
                train_seed,
                test,
                target_config,
                protocol,
                target_checkpoint,
            )
            for row in (zero_row, target_row):
                row.update(
                    {
                        "config_id": config_id,
                        "n_reservoirs": reservoirs,
                        "horizon": horizon,
                        "transfer_axis": axis,
                    }
                )
                rows.append(row)
            costs[(config_id, "zero_shot_graph", str(train_seed))] = zero_costs
            costs[(config_id, "target_trained_graph", str(train_seed))] = target_costs

    frame = pd.DataFrame(rows)
    summary = summarize_runs(
        frame,
        ["config_id", "n_reservoirs", "horizon", "transfer_axis"],
    )
    pairwise, pairwise_summary = pairwise_by_group(costs, policy_order, "config_id")
    write_frame(frame, output / "cross_size_per_run_results.csv")
    write_frame(summary, output / "cross_size_main_results.csv")
    write_frame(pairwise, output / "cross_size_paired_test_path_ci.csv")
    write_frame(pairwise_summary, output / "cross_size_paired_summary.csv")
    return summary, manifest


def _plot(topology: pd.DataFrame, cross_size: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for policy, group in topology.groupby("policy", sort=False):
        axes[0].errorbar(
            group["severity"],
            group["cost_change_percent_mean"],
            yerr=group["cost_change_percent_train_seed_sd"],
            marker="o",
            capsize=3,
            linewidth=2,
            label=policy.replace("projected_", ""),
        )
    axes[0].set_xlabel("Routing perturbation severity")
    axes[0].set_ylabel("Cost change from base (%)")
    axes[0].set_title("Same-size topology transfer")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    reference = cross_size[cross_size["policy"] == "target_trained_graph"][
        ["config_id", "test_cost_mean_over_train_seeds"]
    ].rename(columns={"test_cost_mean_over_train_seeds": "target_cost"})
    plotted = cross_size.merge(reference, on="config_id", how="left")
    plotted["cost_ratio"] = plotted["test_cost_mean_over_train_seeds"] / plotted[
        "target_cost"
    ]
    positions = np.arange(plotted["config_id"].nunique())
    configs = list(dict.fromkeys(plotted["config_id"]))
    width = 0.25
    for index, policy in enumerate(("myopic", "zero_shot_graph", "target_trained_graph")):
        group = plotted[plotted["policy"] == policy].set_index("config_id").loc[configs]
        axes[1].bar(
            positions + (index - 1) * width,
            group["cost_ratio"],
            width=width,
            label=policy.replace("_", " "),
        )
    axes[1].set_xticks(positions, configs, rotation=35, ha="right")
    axes[1].set_ylabel("Cost / target-trained graph cost")
    axes[1].set_title("Cross-size zero-shot transfer")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_graph_transfer(
    protocol: ICLRProtocol,
    output: Path,
    scaling_output: Path,
) -> Path:
    completed = scaling_output / "COMPLETED.json"
    if not completed.exists():
        raise FileNotFoundError("Scaling must complete before graph transfer is evaluated.")
    scaling_runs = pd.read_csv(scaling_output / "per_run_results.csv")
    output.mkdir(parents=True, exist_ok=True)
    write_protocol(
        output,
        protocol,
        "graph_transfer",
        {
            "source_configuration": f"R{protocol.n_reservoirs}_T{protocol.horizon}",
            "selection": "all checkpoints frozen by scaling validation data",
            "test_seed_is_new": True,
            "no_transfer_fine_tuning": True,
        },
    )
    topology, topology_test = _same_size_topology(protocol, scaling_runs, output)
    cross_size, cross_manifest = _cross_size_transfer(protocol, scaling_runs, output)
    manifest = {
        "topology_test": {
            "seed": topology_test.seed,
            "n_paths": topology_test.n,
            "sha256": dataset_fingerprint(topology_test),
            "role": "fresh reporting-only test",
        },
        **cross_manifest,
        "calibration": {
            "seed": protocol.transfer_seeds.calibration,
            "n_paths": protocol.n_calibration,
            "role": "reserved_unread",
        },
        "evaluation": {
            "seed": protocol.transfer_seeds.evaluation,
            "n_paths": protocol.n_evaluation,
            "role": "reserved_unread",
        },
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _plot(topology, cross_size, output / "figures" / "graph_transfer.png")
    write_completion(
        output,
        "graph_transfer",
        {
            "n_topology_summary_rows": len(topology),
            "n_cross_size_summary_rows": len(cross_size),
            "all_constraint_violations_zero": bool(
                topology["total_constraint_violation_count"].sum() == 0
                and cross_size["total_constraint_violation_count"].sum() == 0
            ),
        },
    )
    return output
