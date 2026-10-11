"""Figures for the unified policy comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def policy_comparison(summary: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("test_cost_mean_over_train_seeds", ascending=True)
    figure, axis = plt.subplots(figsize=(10.0, 6.4))
    positions = np.arange(len(ordered))
    axis.barh(
        positions,
        ordered["test_cost_mean_over_train_seeds"],
        xerr=ordered["test_cost_train_seed_sd"],
        color="#2F6B5F",
        alpha=0.88,
        capsize=3,
    )
    axis.set_yticks(positions, ordered["policy"])
    axis.set_xlabel("Test cost (lower is better; error bar = train-seed SD)")
    axis.set_title("Unified nonlinear-hydro policy comparison")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def cost_time_frontier(summary: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.4, 6.4))
    x = summary["batch1_action_latency_ms_median_across_seeds"].to_numpy()
    y = summary["test_cost_mean_over_train_seeds"].to_numpy()
    sizes = 28.0 + 10.0 * np.log10(
        np.maximum(summary["num_parameters"].to_numpy(dtype=float), 1.0)
    )
    axis.scatter(x, y, s=sizes, color="#C05A3D", alpha=0.84)
    offsets = {
        "projected_affine": (5, 10),
        "projected_polynomial": (5, 10),
        "projected_mlp_pathwise": (5, -10),
        "projected_graph_neural_global": (5, -16),
        "deterministic_lookahead": (5, 9),
        "constant_cfa": (5, -10),
    }
    for x_value, y_value, label in zip(x, y, summary["policy"]):
        offset = offsets.get(label, (4, 4))
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    if np.all(x > 0.0):
        axis.set_xscale("log")
    axis.set_xlabel("Batch-1 action latency (ms, log scale)")
    axis.set_ylabel("Test cost")
    axis.set_title("Cost and online decision-time tradeoff")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def training_curves(history: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    usable = history[
        history.get("candidate_selected", pd.Series(1, index=history.index)).eq(1)
        & history["validation_cost"].notna()
    ].copy()
    figure, axis = plt.subplots(figsize=(9.4, 6.2))
    if usable.empty:
        axis.text(0.5, 0.5, "No trainable-policy history", ha="center", va="center")
    else:
        for policy, group in usable.groupby("policy", sort=False):
            curve = group.groupby("iteration", as_index=False)["validation_cost"].mean()
            axis.plot(curve["iteration"], curve["validation_cost"], label=policy)
        axis.legend(fontsize=8, ncol=2)
    axis.set_xlabel("Algorithm iteration / completed fit episodes")
    axis.set_ylabel("Validation cost")
    axis.set_title("Validation-selected training curves")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
