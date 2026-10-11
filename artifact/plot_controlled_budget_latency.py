#!/usr/bin/env python3
"""Draw the controlled-benchmark trajectory-budget and latency figure.

Both panels use the frozen ``formal_protocol_v1/02_data_budget`` outputs.
The left panel compares test cost at matched distinct-fit-trajectory budgets.
The right panel reports the N_fit=2500 cost against batch-one policy proposal
plus feasibility-projection latency; it is a scatter plot, not a frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MSP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    MSP_ROOT / "reference_results"
    / "iclr_algorithm1_experiments"
    / "outputs"
    / "formal_protocol_v1"
    / "02_data_budget"
)
DEFAULT_OUTPUT_DIR = MSP_ROOT / "outputs/paper"

BUDGETS = (100, 250, 500, 1000, 2500)
PRIMARY_BUDGET = 2500
METHODS = (
    "projected_graph_neural_global",
    "projected_mlp_matched",
    "projected_shac",
    "rl_sac",
    "rl_ppo",
)

# Okabe--Ito colors, with distinct markers and line styles for grayscale use.
STYLES = {
    "projected_graph_neural_global": {
        "label": "FeasPG+Graph",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "projected_mlp_matched": {
        "label": "FeasPG+MLP (matched)",
        "color": "#56B4E9",
        "marker": "s",
        "linestyle": "--",
    },
    "projected_shac": {
        "label": "Projected SHAC",
        "color": "#E69F00",
        "marker": "^",
        "linestyle": "-.",
    },
    "rl_sac": {
        "label": "SAC",
        "color": "#D55E00",
        "marker": "X",
        "linestyle": "-",
    },
    "rl_ppo": {
        "label": "PPO",
        "color": "#009E73",
        "marker": "D",
        "linestyle": ":",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_validate(source_dir: Path) -> tuple[pd.DataFrame, dict]:
    paths = {
        "summary": source_dir / "main_results.csv",
        "per_run": source_dir / "per_run_results.csv",
        "protocol": source_dir / "protocol.json",
        "data_manifest": source_dir / "data_manifest.json",
        "environment": source_dir / "environment.json",
        "completion": source_dir / "COMPLETED.json",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"Missing frozen protocol files: {missing_files}")

    summary = pd.read_csv(paths["summary"])
    per_run = pd.read_csv(paths["per_run"])
    protocol_doc = load_json(paths["protocol"])
    protocol = protocol_doc["protocol"]
    manifest = load_json(paths["data_manifest"])
    environment = load_json(paths["environment"])
    completion = load_json(paths["completion"])

    required_summary = {
        "n_fit",
        "policy",
        "n_train_seeds",
        "n_reported_runs",
        "test_cost_mean_over_train_seeds",
        "test_cost_train_seed_sd",
        "environment_transitions_per_candidate",
        "actor_gradient_updates_per_candidate",
        "critic_gradient_updates_per_candidate",
        "deployment_parameters",
        "batch1_action_latency_ms_median_across_seeds",
        "max_action_violation",
        "max_storage_violation",
        "total_constraint_violation_count",
    }
    missing_columns = required_summary.difference(summary.columns)
    if missing_columns:
        raise RuntimeError(f"Summary is missing columns: {sorted(missing_columns)}")

    selected = summary[summary.policy.isin(METHODS)].copy()
    expected_keys = {(budget, method) for budget in BUDGETS for method in METHODS}
    actual_keys = set(selected[["n_fit", "policy"]].itertuples(index=False, name=None))
    if actual_keys != expected_keys or len(selected) != len(expected_keys):
        raise RuntimeError(
            "Learned-method budget grid is incomplete or duplicated; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}, rows={len(selected)}"
        )
    if set(selected.n_train_seeds) != {5} or set(selected.n_reported_runs) != {5}:
        raise RuntimeError("Every plotted point must summarize exactly five train seeds.")
    if not np.isfinite(
        selected[
            [
                "test_cost_mean_over_train_seeds",
                "test_cost_train_seed_sd",
                "batch1_action_latency_ms_median_across_seeds",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Plotted metrics contain non-finite values.")
    if (selected.test_cost_mean_over_train_seeds <= 0).any():
        raise RuntimeError("Test costs must be positive.")
    if (selected.test_cost_train_seed_sd < 0).any():
        raise RuntimeError("Train-seed sample SDs must be nonnegative.")
    if (selected.batch1_action_latency_ms_median_across_seeds <= 0).any():
        raise RuntimeError("Latency summaries must be positive.")
    if (
        selected.max_action_violation.ne(0).any()
        or selected.max_storage_violation.ne(0).any()
        or selected.total_constraint_violation_count.ne(0).any()
    ):
        raise RuntimeError("A plotted policy has a reported feasibility violation.")

    if tuple(protocol["fit_budgets"]) != BUDGETS:
        raise RuntimeError(f"Unexpected fit budgets: {protocol['fit_budgets']}")
    if list(protocol["train_seeds"]) != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Unexpected train seeds: {protocol['train_seeds']}")
    if protocol["horizon"] != 48 or protocol["n_reservoirs"] != 12:
        raise RuntimeError("Figure expects the registered R=12, T=48 benchmark.")
    expected_transitions = selected.n_fit.to_numpy(dtype=int) * protocol["horizon"]
    if not np.array_equal(
        selected.environment_transitions_per_candidate.to_numpy(dtype=int),
        expected_transitions,
    ):
        raise RuntimeError("Transition counts do not equal N_fit times the horizon.")
    if protocol["latency_repetitions"] != 100 or protocol["device"] != "cpu":
        raise RuntimeError("Figure expects 100-repetition CPU latency measurements.")
    if environment.get("device") != "cpu" or environment.get("torch_threads") != 1:
        raise RuntimeError("Figure expects the recorded one-thread CPU environment.")
    if manifest["test"]["n_paths"] != 4096 or manifest["test"]["role"] != "reporting only":
        raise RuntimeError("Unexpected held-out test manifest.")
    if not completion.get("all_constraint_violations_zero", False):
        raise RuntimeError("Completion record does not certify zero constraint violations.")

    per_run_primary = per_run[
        (per_run.n_fit == PRIMARY_BUDGET) & per_run.policy.isin(METHODS)
    ].copy()
    if len(per_run_primary) != len(METHODS) * 5:
        raise RuntimeError("Expected five N_fit=2500 runs for every plotted method.")
    if set(per_run_primary.train_seed.astype(str)) != {"0", "1", "2", "3", "4"}:
        raise RuntimeError("Unexpected train-seed identities in N_fit=2500 runs.")
    if set(per_run_primary.latency_repetitions) != {100}:
        raise RuntimeError("Per-run latency repetitions do not match the protocol.")
    if set(per_run_primary.latency_stage) != {protocol["horizon"] // 2}:
        raise RuntimeError("Per-run latency was not measured at the registered stage.")
    if set(per_run_primary.n_test_paths) != {manifest["test"]["n_paths"]}:
        raise RuntimeError("Per-run test-path count does not match the manifest.")
    if set(per_run_primary.test_path_seed) != {manifest["test"]["seed"]}:
        raise RuntimeError("Per-run test seed does not match the manifest.")

    recomputed = (
        per_run_primary.groupby("policy", sort=False)
        .agg(
            recomputed_cost_mean=("test_mean_cost", "mean"),
            recomputed_cost_sd=("test_mean_cost", lambda values: values.std(ddof=1)),
            recomputed_latency_median=("batch1_action_latency_ms_median", "median"),
        )
        .reset_index()
    )
    primary = selected[selected.n_fit == PRIMARY_BUDGET].merge(
        recomputed, on="policy", validate="one_to_one"
    )
    comparisons = (
        (
            "test_cost_mean_over_train_seeds",
            "recomputed_cost_mean",
        ),
        ("test_cost_train_seed_sd", "recomputed_cost_sd"),
        (
            "batch1_action_latency_ms_median_across_seeds",
            "recomputed_latency_median",
        ),
    )
    for reported, calculated in comparisons:
        if not np.allclose(primary[reported], primary[calculated], rtol=1e-12, atol=1e-9):
            raise RuntimeError(f"Reported {reported} does not reproduce from per-run data.")

    order = {method: index for index, method in enumerate(METHODS)}
    selected["method_label"] = selected.policy.map(
        {method: STYLES[method]["label"] for method in METHODS}
    )
    selected["_method_order"] = selected.policy.map(order)
    selected = selected.sort_values(["_method_order", "n_fit"]).drop(
        columns="_method_order"
    )

    provenance_context = {
        "paths": paths,
        "protocol": protocol,
        "manifest": manifest,
        "environment": environment,
        "completion": completion,
    }
    return selected, provenance_context


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    axis.tick_params(axis="both", length=2.5, width=0.65, pad=1.8)
    for spine in axis.spines.values():
        spine.set_color("#333333")


def draw_budget_axis(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    title: str | None,
) -> None:
    for method in METHODS:
        values = summary[summary.policy == method].sort_values("n_fit")
        style = STYLES[method]
        axis.errorbar(
            values.n_fit,
            values.test_cost_mean_over_train_seeds / 1000.0,
            yerr=values.test_cost_train_seed_sd / 1000.0,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            markersize=3.8,
            linestyle=style["linestyle"],
            linewidth=1.25,
            elinewidth=0.75,
            capsize=1.6,
            capthick=0.75,
            zorder=3,
        )

    if title is not None:
        axis.set_title(title, pad=4.0)
    axis.set_xscale("log")
    axis.set_xlim(83, 3000)
    axis.set_xticks(BUDGETS, labels=("100", "250", "500", "1k", "2.5k"))
    axis.minorticks_off()
    axis.set_ylim(47, 148)
    axis.set_yticks((50, 75, 100, 125, 150))
    axis.set_xlabel(r"Distinct fit trajectories, $N_{\mathrm{fit}}$")
    axis.set_ylabel(r"Test cost ($\times 10^3$; lower is better)")
    style_axis(axis)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=STYLES[method]["color"],
            linestyle=STYLES[method]["linestyle"],
            marker=STYLES[method]["marker"],
            markerfacecolor=STYLES[method]["color"],
            markeredgecolor=STYLES[method]["color"],
            markeredgewidth=0.8,
            markersize=3.8,
            linewidth=1.25,
            label=STYLES[method]["label"],
        )
        for method in METHODS
    ]


def draw_figure(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.9, 2.48),
        gridspec_kw={"width_ratios": (1.18, 1.0)},
    )
    budget_axis, latency_axis = axes

    draw_budget_axis(
        budget_axis,
        summary,
        title="(a) Trajectory-budget efficiency",
    )

    primary = summary[summary.n_fit == PRIMARY_BUDGET]
    for method in METHODS:
        value = primary[primary.policy == method]
        style = STYLES[method]
        latency_axis.errorbar(
            value.batch1_action_latency_ms_median_across_seeds,
            value.test_cost_mean_over_train_seeds / 1000.0,
            yerr=value.test_cost_train_seed_sd / 1000.0,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            markersize=5.0,
            linestyle="none",
            elinewidth=0.8,
            capsize=1.8,
            capthick=0.8,
            zorder=4,
        )

    latency_axis.set_title("(b) Cost vs. deployed-action latency", pad=4.0)
    latency_axis.set_xlim(0.835, 1.005)
    latency_axis.set_xticks((0.85, 0.90, 0.95, 1.00))
    latency_axis.set_ylim(47, 90)
    latency_axis.set_yticks((50, 60, 70, 80, 90))
    latency_axis.set_xlabel("Batch-1 proposal + projection latency (ms)")
    latency_axis.set_ylabel(r"Test cost at $N_{\mathrm{fit}}=2{,}500$ ($\times 10^3$)")
    latency_axis.text(
        0.035,
        0.955,
        "CPU; 100 repetitions per seed",
        transform=latency_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#555555",
    )
    style_axis(latency_axis)

    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        columnspacing=1.15,
        handlelength=2.0,
        handletextpad=0.45,
        frameon=False,
    )
    figure.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.31, wspace=0.29)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "controlled_budget_latency.pdf"
    png_path = output_dir / "controlled_budget_latency.png"
    metadata = {
        "Title": "Trajectory-budget efficiency and deployed-action latency",
        "Subject": "Frozen formal nonlinear-hydro protocol",
        "Creator": "plot_controlled_budget_latency.py",
    }
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return pdf_path, png_path


def draw_budget_only(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    configure_matplotlib()
    # Match the ICLR text width natively so the paper does not shrink labels.
    figure, axis = plt.subplots(figsize=(5.50, 2.55))
    draw_budget_axis(axis, summary, title=None)
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        columnspacing=1.3,
        handlelength=2.0,
        handletextpad=0.45,
        frameon=False,
    )
    figure.subplots_adjust(left=0.10, right=0.995, top=0.98, bottom=0.32)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "controlled_budget_only.pdf"
    png_path = output_dir / "controlled_budget_only.png"
    metadata = {
        "Title": "Trajectory-budget efficiency",
        "Subject": "Frozen formal nonlinear-hydro protocol",
        "Creator": "plot_controlled_budget_latency.py",
    }
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return pdf_path, png_path


def write_outputs(
    summary: pd.DataFrame,
    context: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    snapshot_columns = [
        "n_fit",
        "policy",
        "method_label",
        "n_train_seeds",
        "n_reported_runs",
        "test_cost_mean_over_train_seeds",
        "test_cost_train_seed_sd",
        "batch1_action_latency_ms_median_across_seeds",
        "environment_transitions_per_candidate",
        "actor_gradient_updates_per_candidate",
        "critic_gradient_updates_per_candidate",
        "deployment_parameters",
        "max_action_violation",
        "max_storage_violation",
        "total_constraint_violation_count",
    ]
    snapshot_path = output_dir / "controlled_budget_latency_data.csv"
    summary[snapshot_columns].to_csv(snapshot_path, index=False)

    paths = context["paths"]
    protocol = context["protocol"]
    manifest = context["manifest"]

    def portable(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(MSP_ROOT))
        except ValueError:
            return path.name

    provenance = {
        "figure": "controlled_budget_latency",
        "outputs": {
            "main_text": ["controlled_budget_only.pdf", "controlled_budget_only.png"],
            "appendix": ["controlled_budget_latency.pdf", "controlled_budget_latency.png"],
        },
        "source_files": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "filter": {
            "methods": list(METHODS),
            "fit_budgets": list(BUDGETS),
            "right_panel_fit_budget": PRIMARY_BUDGET,
            "excluded_from_both_panels": ["myopic"],
        },
        "benchmark": {
            "profile": protocol["profile"],
            "n_reservoirs": protocol["n_reservoirs"],
            "horizon": protocol["horizon"],
            "train_seeds": protocol["train_seeds"],
            "test_paths": manifest["test"]["n_paths"],
            "test_seed": manifest["test"]["seed"],
            "test_role": manifest["test"]["role"],
        },
        "uncertainty": {
            "vertical_error_bars": "sample SD of selected-policy test means across five training seeds",
            "pathwise_uncertainty_not_plotted": True,
        },
        "latency": {
            "operation": "batch-one policy proposal plus feasibility projection",
            "device": protocol["device"],
            "torch_threads": context["environment"]["torch_threads"],
            "stage": protocol["horizon"] // 2,
            "timing_state": "first test path at the midpoint stage of each policy's own rollout",
            "repetitions_per_train_seed": protocol["latency_repetitions"],
            "within_seed_statistic": "median",
            "plotted_statistic": "median of the five within-seed medians",
            "training_simulation_and_online_update_time_excluded": True,
        },
        "interpretation_limits": {
            "matched_quantity": "distinct fit trajectories",
            "not_matched": ["optimizer updates", "computation"],
            "right_panel_is_not_a_pareto_frontier": True,
            "latency_is_implementation_level": True,
            "rl_adapter_includes_numpy_sb3_torch_interface_overhead": True,
        },
        "selected_rows": len(summary),
    }
    provenance_path = output_dir / "controlled_budget_latency_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return snapshot_path, provenance_path


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, context = load_and_validate(source_dir)
    snapshot_path, provenance_path = write_outputs(summary, context, output_dir)
    pdf_path, png_path = draw_figure(summary, output_dir)
    budget_pdf_path, budget_png_path = draw_budget_only(summary, output_dir)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {budget_pdf_path}")
    print(f"wrote {budget_png_path}")
    print(f"wrote {snapshot_path}")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
