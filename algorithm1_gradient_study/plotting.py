"""Paper-oriented figures for the five mechanism experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "full": "#1B6F5A",
    "no_proposal_state": "#D97706",
    "no_moving_recourse": "#2463A7",
    "projected_mlp_pathwise": "#D97706",
    "projected_graph_neural_global": "#1B6F5A",
}


def finite_difference_figure(rows: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for mode, group in rows.groupby("gradient_mode", sort=False):
        summary = group.groupby("step")["relative_error"].mean().sort_index()
        axis.loglog(
            summary.index,
            np.maximum(summary.values, 1.0e-14),
            marker="o",
            linewidth=2.0,
            label=_label(mode),
            color=COLORS.get(mode),
        )
    axis.set_xlabel("Central finite-difference step")
    axis.set_ylabel("Relative directional-derivative error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    _save(figure, path)


def ablation_figure(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("test_cost_mean_over_train_seeds")
    labels = [_label(value) for value in ordered["variant"]]
    means = ordered["test_cost_mean_over_train_seeds"].to_numpy()
    errors = ordered["test_cost_train_seed_sd"].to_numpy()
    colors = [COLORS.get(value, "#666666") for value in ordered["variant"]]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    positions = np.arange(len(ordered))
    axis.bar(positions, means, yerr=errors, capsize=4, color=colors)
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylabel("Test cost")
    axis.set_title("Error bars are train-seed SD, not test-path CI")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def horizon_figure(rows: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    ablations = rows[rows["gradient_mode"] != "full"]
    for mode, group in ablations.groupby("gradient_mode", sort=False):
        summary = group.groupby("horizon")[
            "relative_difference_from_full"
        ].agg(["mean", "std"])
        standard_deviation = summary["std"].fillna(0.0).to_numpy()
        horizons = summary.index.to_numpy()
        mean = summary["mean"].to_numpy()
        axis.plot(
            horizons,
            mean,
            marker="o",
            linewidth=2.0,
            label=_label(mode),
            color=COLORS.get(mode),
        )
        axis.fill_between(
            horizons,
            np.maximum(0.0, mean - standard_deviation),
            mean + standard_deviation,
            alpha=0.15,
            color=COLORS.get(mode),
        )
    axis.set_xlabel("Horizon T")
    axis.set_ylabel("Relative gradient error vs. full pathwise gradient")
    axis.set_xscale("log", base=2)
    horizon_ticks = sorted(ablations["horizon"].unique())
    axis.set_xticks(horizon_ticks)
    axis.set_xticklabels([str(value) for value in horizon_ticks])
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    _save(figure, path)


def graph_stability_figure(rows: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    for policy, group in rows.groupby("policy", sort=False):
        summary = group.groupby("severity")["cost_change_percent"].agg(
            ["mean", "std"]
        )
        error = summary["std"].fillna(0.0).to_numpy()
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=error,
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=_label(policy),
            color=COLORS.get(policy),
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xlabel("Routing perturbation severity")
    axis.set_ylabel("Paired test-cost change from base topology (%)")
    axis.set_title("Error bars are topology/train-seed SD")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    _save(figure, path)


def equivariance_figure(rows: pd.DataFrame, path: Path) -> None:
    summary = rows.groupby("policy")["max_proposal_equivariance_error"].mean()
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    labels = [_label(value) for value in summary.index]
    values = np.maximum(summary.to_numpy(), 1.0e-14)
    colors = [COLORS.get(value, "#666666") for value in summary.index]
    axis.bar(np.arange(len(summary)), values, color=colors)
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(summary)), labels, rotation=10, ha="right")
    axis.set_ylabel("Max proposal equivariance error")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, path)


def _label(value: str) -> str:
    labels = {
        "full": "Full pathwise gradient",
        "no_proposal_state": "Without proposal-state channel",
        "no_moving_recourse": "Without moving-recourse channel",
        "projected_mlp_pathwise": "Pathwise MLP",
        "projected_graph_neural_global": "Global graph policy",
    }
    return labels.get(value, value.replace("_", " ").title())


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
