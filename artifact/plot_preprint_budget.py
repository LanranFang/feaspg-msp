#!/usr/bin/env python3
"""Draw the preprint's compact Figure 4 from its unchanged published CSV."""

from pathlib import Path
from . import ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = ROOT / "reference_results/paper"
STYLES = {
    "projected_graph_neural_global": ("FeasPG+Graph", "#1F5FA9", "o"),
    "projected_mlp_matched": ("FeasPG+MLP", "#5BA3D9", "s"),
    "projected_shac": ("Projected SHAC", "#C79224", "^"),
    "rl_sac": ("SAC", "#B5462F", "X"),
    "rl_ppo": ("PPO", "#2E8B6F", "D"),
    "linearized_empirical_sddp": ("Linearized SDDP", "#5C636B", "v"),
}


def draw(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_DIR / "controlled_budget_only_data.csv")
    # Reduce the previous seven-inch figure's width by 15%, keeping its height.
    # Include at 0.7854\linewidth instead of 0.924\linewidth so the printed
    # height and text sizes stay unchanged.
    width = 7.0 * 0.85
    height = 7.0 * (144 / 190.08) * 0.75 / 1.10
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14.5,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(width, height))
    figure.subplots_adjust(left=0.105 / 0.85, right=1 - 0.015 / 0.85,
                           top=0.965, bottom=0.35)
    for method, (label, color, marker) in STYLES.items():
        group = frame[frame.policy.eq(method)].sort_values("n_fit")
        mean = group.test_cost_mean_over_train_seeds.to_numpy() / 1000
        sd = group.test_cost_train_seed_sd.to_numpy() / 1000
        axis.fill_between(group.n_fit, mean - sd, mean + sd,
                          color=color, alpha=0.17, linewidth=0)
        axis.plot(group.n_fit, mean, color=color, marker=marker,
                  linewidth=2, markersize=5.5, markeredgewidth=0.7, label=label)
    axis.set_xscale("log")
    axis.set_xlim(85, 2800)
    axis.set_xticks((100, 250, 500, 1000, 2500),
                   labels=("100", "250", "500", "1k", "2.5k"))
    axis.minorticks_off()
    axis.set_ylim(44, 149)
    axis.set_yticks((50, 75, 100, 125))
    axis.set_xlabel(r"Number of sample paths, $N_{\rm fit}$", labelpad=5)
    axis.set_ylabel(r"Test cost (in $10^3$)", labelpad=5)
    axis.set_axisbelow(True)
    axis.grid(color="#BFC4CA", linewidth=0.5, linestyle=(0, (2, 3)))
    handles, labels = axis.get_legend_handles_labels()
    order = (0, 3, 1, 4, 2, 5)
    figure.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="lower center", bbox_to_anchor=(0.53, 0.012), ncol=3,
                  fontsize=13.5, frameon=True, fancybox=False, framealpha=1,
                  edgecolor="#30343A", handlelength=1.25, handletextpad=0.3,
                  columnspacing=0.5, labelspacing=0.3, borderpad=0.35)
    figure.savefig(output / "controlled_budget_preprint.pdf", metadata={
        "Title": "Synthetic training-budget comparison for the preprint",
        "Creator": "scripts/plot_preprint_budget.py",
        "Subject": "Published means and sample SDs; compact preprint layout",
    })
    plt.close(figure)


if __name__ == "__main__":
    draw(ROOT / "outputs/paper")
