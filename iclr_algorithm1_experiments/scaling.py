"""Reservoir-count and horizon scaling for matched MLP and graph policies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from efficiency_switching_hydro.data import generate_dataset

from .config import ICLRProtocol
from .data_budget import _direct_run, _fixed_run
from .evaluation import evaluate_policy, pairwise_by_group, summarize_runs
from .io import (
    canonical_seed,
    dataset_fingerprint,
    write_completion,
    write_frame,
    write_protocol,
)
from .policies import parameter_matched_mlp_width


POLICY_ORDER = (
    "myopic",
    "projected_mlp_matched",
    "projected_graph_neural_global",
)


def scaling_configurations(protocol: ICLRProtocol) -> tuple[tuple[int, int], ...]:
    values = []
    for horizon in protocol.horizon_values:
        values.append((protocol.n_reservoirs, horizon))
    for reservoirs in protocol.reservoir_values:
        values.append((reservoirs, protocol.horizon))
    return tuple(dict.fromkeys(values))


def _config_id(reservoirs: int, horizon: int) -> str:
    return f"R{reservoirs}_T{horizon}"


def pre_registered_candidates(protocol: ICLRProtocol) -> pd.DataFrame:
    rows = []
    for reservoirs, horizon in scaling_configurations(protocol):
        config = protocol.hydro_config(reservoirs, horizon)
        mlp_width, mlp_parameters, graph_parameters = parameter_matched_mlp_width(
            config,
            graph_width=protocol.graph_width,
            depth=protocol.network_depth,
        )
        config_id = _config_id(reservoirs, horizon)
        rows.append(
            {
                "config_id": config_id,
                "n_reservoirs": reservoirs,
                "horizon": horizon,
                "policy": "myopic",
                "train_seed": "fixed",
                "candidate_index": 0,
                "learning_rate": np.nan,
                "mlp_width": mlp_width,
                "deployment_parameters": 0,
            }
        )
        for policy, rates, parameters in (
            ("projected_mlp_matched", protocol.mlp_learning_rates, mlp_parameters),
            (
                "projected_graph_neural_global",
                protocol.graph_learning_rates,
                graph_parameters,
            ),
        ):
            for train_seed in protocol.train_seeds:
                for index, learning_rate in enumerate(rates):
                    rows.append(
                        {
                            "config_id": config_id,
                            "n_reservoirs": reservoirs,
                            "horizon": horizon,
                            "policy": policy,
                            "train_seed": train_seed,
                            "candidate_index": index,
                            "learning_rate": learning_rate,
                            "mlp_width": mlp_width,
                            "deployment_parameters": parameters,
                        }
                    )
    return pd.DataFrame(rows)


def _cost_frame(costs, n_test):
    columns: dict[str, np.ndarray] = {"test_path": np.arange(n_test)}
    for (config_id, policy, seed), values in sorted(costs.items()):
        columns[f"{config_id}_{policy}_seed{seed}"] = values
    return pd.DataFrame(columns)


def _load_partial(output: Path):
    result_path = output / "partial_per_run_results.csv"
    candidate_path = output / "partial_hyperparameter_candidates.csv"
    history_path = output / "partial_training_curves.csv"
    cost_path = output / "partial_test_path_costs.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    candidates = (
        pd.read_csv(candidate_path).to_dict("records") if candidate_path.exists() else []
    )
    histories = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    costs = {}
    if cost_path.exists():
        frame = pd.read_csv(cost_path)
        for column in frame.columns[1:]:
            prefix, seed = column.rsplit("_seed", 1)
            reservoirs, horizon, policy = prefix.split("_", 2)
            config_id = f"{reservoirs}_{horizon}"
            costs[(config_id, policy, seed)] = frame[column].to_numpy(dtype=np.float64)
    return rows, candidates, histories, costs


def _flush(output, rows, candidates, histories, costs, n_test):
    write_frame(pd.DataFrame(rows), output / "partial_per_run_results.csv")
    write_frame(
        pd.DataFrame(candidates), output / "partial_hyperparameter_candidates.csv"
    )
    write_frame(pd.DataFrame(histories), output / "partial_training_curves.csv")
    write_frame(_cost_frame(costs, n_test), output / "partial_test_path_costs.csv")


def _plot(summary: pd.DataFrame, protocol: ICLRProtocol, path: Path) -> None:
    myopic = summary[summary["policy"] == "myopic"][
        ["config_id", "test_cost_mean_over_train_seeds"]
    ].rename(columns={"test_cost_mean_over_train_seeds": "myopic_cost"})
    plotted = summary.merge(myopic, on="config_id", how="left")
    plotted["relative_cost"] = (
        plotted["test_cost_mean_over_train_seeds"] / plotted["myopic_cost"]
    )
    labels = {
        "myopic": "Myopic",
        "projected_mlp_matched": "Matched MLP",
        "projected_graph_neural_global": "Graph",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    reservoir = plotted[plotted["horizon"] == protocol.horizon]
    horizon = plotted[plotted["n_reservoirs"] == protocol.n_reservoirs]
    for policy in POLICY_ORDER:
        group = reservoir[reservoir["policy"] == policy].sort_values("n_reservoirs")
        axes[0].plot(
            group["n_reservoirs"],
            group["relative_cost"],
            marker="o",
            linewidth=2,
            label=labels[policy],
        )
        group = horizon[horizon["policy"] == policy].sort_values("horizon")
        axes[1].plot(
            group["horizon"],
            group["relative_cost"],
            marker="o",
            linewidth=2,
            label=labels[policy],
        )
    axes[0].set_xlabel("Reservoirs")
    axes[0].set_ylabel("Cost / myopic cost")
    axes[0].set_title(f"System scaling at T={protocol.horizon}")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Cost / myopic cost")
    axes[1].set_title(f"Horizon scaling at R={protocol.n_reservoirs}")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_scaling(protocol: ICLRProtocol, output: Path, resume: bool = True) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    preregistration = pre_registered_candidates(protocol)
    preregistration_path = output / "pre_registered_candidates.csv"
    if not preregistration_path.exists() or not resume:
        write_frame(preregistration, preregistration_path)
    write_protocol(
        output,
        protocol,
        "r_t_scaling",
        {
            "policies": list(POLICY_ORDER),
            "configurations": [
                {"n_reservoirs": reservoirs, "horizon": horizon}
                for reservoirs, horizon in scaling_configurations(protocol)
            ],
            "parameter_matching": "MLP width selected independently at each size",
            "test_seed_is_new": True,
        },
    )

    if resume:
        rows, candidate_rows, histories, costs = _load_partial(output)
    else:
        rows, candidate_rows, histories, costs = [], [], [], {}
    completed = {
        (
            str(row["config_id"]),
            str(row["policy"]),
            canonical_seed(row["train_seed"]),
        )
        for row in rows
    }
    data_manifest: dict[str, object] = {}
    n_fit = protocol.base_n_fit

    for reservoirs, horizon in scaling_configurations(protocol):
        config_id = _config_id(reservoirs, horizon)
        hydro = protocol.hydro_config(reservoirs, horizon)
        fit = generate_dataset("fit", n_fit, protocol.scaling_seeds.fit, hydro)
        validation = generate_dataset(
            "validation",
            protocol.n_validation,
            protocol.scaling_seeds.validation,
            hydro,
        )
        test = generate_dataset(
            "test",
            protocol.n_test,
            protocol.scaling_seeds.test,
            hydro,
        )
        for name, dataset in (("fit", fit), ("validation", validation), ("test", test)):
            data_manifest[f"{config_id}_{name}"] = {
                "seed": dataset.seed,
                "n_paths": dataset.n,
                "sha256": dataset_fingerprint(dataset),
                "role": name,
            }

        fixed_key = (config_id, "myopic", "fixed")
        if fixed_key not in completed:
            row, path_costs = _fixed_run(n_fit, test, hydro, protocol)
            row.update(
                {
                    "config_id": config_id,
                    "n_reservoirs": reservoirs,
                    "horizon": horizon,
                }
            )
            rows.append(row)
            costs[fixed_key] = path_costs
            completed.add(fixed_key)
            _flush(output, rows, candidate_rows, histories, costs, test.n)

        for policy in POLICY_ORDER[1:]:
            for train_seed in protocol.train_seeds:
                key = (config_id, policy, str(train_seed))
                if key in completed:
                    continue
                print(f"scaling {config_id}: {policy}, seed={train_seed}", flush=True)
                result = _direct_run(
                    policy,
                    fit,
                    validation,
                    test,
                    hydro,
                    protocol,
                    train_seed,
                    output / "checkpoints" / config_id,
                )
                row, path_costs, new_candidates, new_histories = result
                row.update(
                    {
                        "config_id": config_id,
                        "n_reservoirs": reservoirs,
                        "horizon": horizon,
                    }
                )
                for candidate in new_candidates:
                    candidate.update(
                        {
                            "config_id": config_id,
                            "n_reservoirs": reservoirs,
                            "horizon": horizon,
                        }
                    )
                for history in new_histories:
                    history.update(
                        {
                            "config_id": config_id,
                            "n_reservoirs": reservoirs,
                            "horizon": horizon,
                        }
                    )
                rows.append(row)
                candidate_rows.extend(new_candidates)
                histories.extend(new_histories)
                costs[key] = path_costs
                completed.add(key)
                _flush(output, rows, candidate_rows, histories, costs, test.n)

    data_manifest["calibration"] = {
        "seed": protocol.scaling_seeds.calibration,
        "n_paths": protocol.n_calibration,
        "role": "reserved_unread",
    }
    data_manifest["evaluation"] = {
        "seed": protocol.scaling_seeds.evaluation,
        "n_paths": protocol.n_evaluation,
        "role": "reserved_unread",
    }
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2), encoding="utf-8"
    )

    frame = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    curves = pd.DataFrame(histories)
    summary = summarize_runs(
        frame,
        ["config_id", "n_reservoirs", "horizon"],
    )
    pairwise, pairwise_summary = pairwise_by_group(costs, POLICY_ORDER, "config_id")
    write_frame(frame, output / "per_run_results.csv")
    write_frame(candidates, output / "hyperparameter_candidates.csv")
    write_frame(curves, output / "training_curves.csv")
    write_frame(summary, output / "main_results.csv")
    write_frame(_cost_frame(costs, protocol.n_test), output / "test_path_costs.csv")
    write_frame(pairwise, output / "paired_test_path_ci.csv")
    write_frame(pairwise_summary, output / "paired_comparison_summary.csv")
    _plot(summary, protocol, output / "figures" / "r_t_scaling.png")
    write_completion(
        output,
        "r_t_scaling",
        {
            "n_configurations": len(scaling_configurations(protocol)),
            "n_final_runs": len(frame),
            "n_candidates": len(candidates),
            "all_constraint_violations_zero": bool(
                summary["total_constraint_violation_count"].sum() == 0
            ),
        },
    )
    return output
