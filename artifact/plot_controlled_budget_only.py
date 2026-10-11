#!/usr/bin/env python3
"""Draw the controlled-benchmark figure from frozen results, without refitting.

The portable CSV is copied from regularized_a1_v2/report/controlled_all_methods.csv
(SHA256 54c2361ce5fd4d18c266d95fa6fc566c0693cf46bbfb41556e0148f0370eec2c).
Only labels and visual styling are changed.
Run from any directory; outputs default to extend_version/figures.
Use --compact for a shorter main-text figure with the same width and font sizes.
Use --beside-table for a half-width plot with a legend and the same printed text size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES = Path(__file__).resolve().parents[1] / "extend_version" / "figures"
BUDGETS = (100, 250, 500, 1000, 2500)
STYLES = {
    "projected_graph_neural_global": ("Graph A1", "#0072B2", "o", "-"),
    "projected_mlp_matched": ("Matched MLP A1", "#56B4E9", "s", (0, (4, 2))),
    "projected_shac": ("Projected SHAC", "#C79224", "^", (0, (4, 2, 1, 2))),
    "rl_sac": ("SAC", "#C06835", "x", "-"),
    "rl_ppo": ("PPO", "#369875", "D", (0, (1, 2))),
    "linearized_empirical_sddp": ("Linearized SDDP", "#646B73", "v", (0, (5, 3))),
}
# Spread the closely spaced endpoint labels; leaders connect to the actual means.
# These positions affect text only, never the plotted means or SD bands.
LABEL_HEIGHTS = {
    "projected_graph_neural_global": 47.5,
    "projected_mlp_matched": 54.5,
    "projected_shac": 61.5,
    "rl_sac": 68.5,
}


def draw(data: Path, output_dir: Path, *, compact: bool = False,
         beside_table: bool = False) -> None:
    frame = pd.read_csv(data)
    expected = {(method, budget) for method in STYLES for budget in BUDGETS}
    actual = set(frame[["policy", "n_fit"]].itertuples(index=False, name=None))
    if actual != expected or len(frame) != len(expected):
        raise ValueError("Expected the complete six-method, five-budget grid.")
    metrics = frame[["test_cost_mean_over_train_seeds", "test_cost_train_seed_sd"]]
    if not np.isfinite(metrics.to_numpy()).all() or (metrics < 0).any().any():
        raise ValueError("Cost means and standard deviations must be finite and nonnegative.")
    if not frame.n_train_seeds.eq(5).all():
        raise ValueError("Every point must summarize five training seeds.")

    # The existing compact PDF is 431.546 bp wide and is printed at 0.88 of
    # the ICLR template's 5.5-inch text width. Preserve its printed font sizes
    # when exporting the new plot directly at 0.48 of the text width.
    scale = 0.88 * 5.5 * 72 / 431.546 if beside_table else 1.0
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9 * scale,
        "axes.labelsize": 9 * scale,
        "xtick.labelsize": 8.5 * scale,
        "ytick.labelsize": 8.5 * scale,
        "axes.edgecolor": "#7A8189",
        "axes.linewidth": 0.55 * scale,
        "text.color": "#30343A",
        "axes.labelcolor": "#30343A",
        "xtick.color": "#50565E",
        "ytick.color": "#50565E",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if beside_table:
        # Preserve the plot's physical height and font sizes while reserving
        # less space below it for a compact two-row legend.
        fig, ax = plt.subplots(figsize=(0.48 * 5.5, 1.80))
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.355, top=0.97)
    else:
        height = 2.2 if compact else 2.65
        fig, ax = plt.subplots(figsize=(6.5, height))
        # Keep the horizontal geometry and physical top/bottom margins unchanged.
        # At the same LaTeX include width, all text retains its original point size.
        fig.subplots_adjust(left=0.09, right=0.76,
                            bottom=0.19 * 2.65 / height,
                            top=1 - 0.03 * 2.65 / height)
    label_heights = LABEL_HEIGHTS
    if compact:
        # Spread labels in the shorter plot; data and SD bands are unchanged.
        label_heights = {
            "projected_graph_neural_global": 44.5,
            "projected_mlp_matched": 54.5,
            "projected_shac": 64.5,
            "rl_sac": 74.5,
        }
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7E9ED", linewidth=0.5 * scale)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5 * scale, width=0.55 * scale, pad=3 * scale)
    ax.set_xscale("log")
    ax.set_xlim(85, 2800)
    ax.set_xticks(BUDGETS, labels=("100", "250", "500", "1k", "2.5k"))
    ax.minorticks_off()
    ax.set_ylim(44, 149)
    ax.set_yticks((50, 75, 100, 125))
    ax.set_xlabel(r"Distinct fit trajectories, $N_{\mathrm{fit}}$", labelpad=6 * scale)
    ax.set_ylabel(r"Test cost ($\times 10^3$)", labelpad=7 * scale)

    for method, (label, color, marker, linestyle) in STYLES.items():
        values = frame[frame.policy.eq(method)].sort_values("n_fit")
        mean = values.test_cost_mean_over_train_seeds.to_numpy() / 1000
        sd = values.test_cost_train_seed_sd.to_numpy() / 1000
        primary = method.startswith("projected_graph") or method == "projected_mlp_matched"
        ax.fill_between(values.n_fit, mean - sd, mean + sd,
                        color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(
            values.n_fit, mean,
            color=color, linestyle=linestyle, label=label,
            linewidth=(1.45 if primary else 1.1) * scale,
            marker=marker, markersize=3.5 * scale,
            markerfacecolor=color if primary else "white",
            markeredgecolor=color, markeredgewidth=0.85 * scale,
            zorder=4 if primary else 3,
        )
        if beside_table:
            continue
        ax.annotate(
            label,
            xy=(BUDGETS[-1], mean[-1]), xycoords="data",
            xytext=(1.055, label_heights.get(method, mean[-1])),
            textcoords=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.6,
            color="#287BA8" if method == "projected_mlp_matched" else color,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6,
                        "shrinkA": 3, "shrinkB": 4},
        )

    if beside_table:
        handles, labels = ax.get_legend_handles_labels()
        # Matplotlib fills columns first: display the two A1 variants and
        # SHAC on the first row, followed by SAC, PPO, and SDDP on the second.
        order = (0, 3, 1, 4, 2, 5)
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower center", bbox_to_anchor=(0.50, 0.025),
                   ncol=3, frameon=False, labelcolor="black",
                   fontsize=8.6 * scale, handlelength=1.6,
                   handletextpad=0.4, columnspacing=0.8,
                   labelspacing=0.25, borderpad=0, borderaxespad=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ("controlled_budget_beside_table" if beside_table else
            "controlled_budget_compact" if compact else "controlled_budget_only")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300,
                    bbox_inches=None if beside_table else "tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=FIGURES / "controlled_budget_only_data.csv")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--compact", action="store_true",
                        help="Reduce plot height while keeping width and font sizes unchanged.")
    layout.add_argument("--beside-table", action="store_true",
                        help="Use half-width geometry and a legend, preserving printed font sizes.")
    args = parser.parse_args()
    draw(args.data, args.output_dir, compact=args.compact, beside_table=args.beside_table)
