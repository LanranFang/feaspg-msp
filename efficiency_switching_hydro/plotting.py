"""Reporting plots for one profile and the full mechanism sequence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def training_curves(history: pd.DataFrame, path: str | Path) -> None:
    data = history.dropna(subset=["validation_cost"])
    if data.empty:
        return
    if "fit_trajectories_seen" in data:
        data = data.copy()
        data["training_progress"] = data["fit_trajectories_seen"].fillna(
            data["iteration"]
        )
        x_column = "training_progress"
        x_label = "Fit trajectories consumed (SDDP: iterations)"
    else:
        x_column = "iteration"
        x_label = "Iteration"
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = (
        data.groupby(["policy", x_column], as_index=False)["validation_cost"]
        .mean()
    )
    for policy, group in grouped.groupby("policy", sort=False):
        ax.plot(
            group[x_column],
            group["validation_cost"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=policy,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation cost")
    ax.set_title("Validation-selected training curves")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def policy_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    data = summary.sort_values("test_mean_cost")
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(data))
    costs = data["test_mean_cost"].to_numpy()
    ax.scatter(costs, positions, color="#2f6b9a", s=70)
    for cost, position in zip(costs, positions):
        ax.annotate(
            f"{cost:.1f}",
            (cost, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    padding = max(1.0, 0.08 * float(costs.max() - costs.min()))
    ax.set_xlim(costs.min() - padding, costs.max() + 4.0 * padding)
    ax.set_yticks(positions)
    ax.set_yticklabels(data["policy"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean test cost (lower is better)")
    ax.set_title(f"Policy comparison: {data['profile'].iloc[0]}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sequence_comparison(data: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for policy, group in data.groupby("policy", sort=False):
        ordered = group.sort_values("profile_order")
        ax.plot(
            ordered["profile"],
            ordered["cost_relative_to_myopic"],
            marker="o",
            linewidth=2,
            label=policy,
        )
    ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_xlabel("Mechanism profile")
    ax.set_ylabel("Test cost / myopic test cost")
    ax.set_title("Mechanism-by-mechanism policy comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
