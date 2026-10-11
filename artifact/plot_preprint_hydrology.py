#!/usr/bin/env python3
"""Render the preprint's hydrology comparison from the published frozen data.

Run from any directory. The CSV contains the same five-budget means and
sample SDs as the September 26 Overleaf figure; no experiments are rerun.
Only the preprint figure is written, leaving the shared ICLR figure intact.
"""

from pathlib import Path
from . import ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = ROOT / "reference_results/paper"
DATA = DATA_DIR / "real_hydrology_preprint_data.csv"
BUDGETS = (250, 500, 1000, 2500, 5000)
STYLES = {
    "structured": dict(color="#1F5FA9", marker="o", linestyle="-"),
    "matched_mlp": dict(color="#009E73", marker="s", linestyle="--"),
    "sddp": dict(color="#D55E00", marker="v", linestyle="-."),
}


def draw(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
    })
    # Same seven-inch width as the Overleaf figure, with 15% less height.
    figure, axes = plt.subplots(1, 2, figsize=(7, 2.55))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.215,
                           top=0.92, wspace=0.27)
    cases = (
        ("ons_subsystems", "(a) ONS", 1e9, (49.8, 66), "lower left"),
        ("camels_br_12", "(b) CAMELS-BR", 1, (163.5, 193), "upper right"),
    )
    for axis, (case, title, scale, limits, legend_location) in zip(axes, cases):
        subset = data[data.case_id.eq(case)]
        handles = {}
        for role in ("sddp", "matched_mlp", "structured"):
            group = subset[subset.role.eq(role)].sort_values("n_fit")
            if group.empty:
                continue
            if tuple(group.n_fit) != BUDGETS:
                raise ValueError(f"Incomplete published budget series: {case}/{role}")
            mean = group.mean_test_cost.to_numpy() / scale
            sd = group.seed_sd.to_numpy() / scale
            style = STYLES[role]
            label = {
                "structured": "FeasPG+MLP" if case == "ons_subsystems"
                              else "FeasPG+DeepSets",
                "matched_mlp": "FeasPG+MLP (matched)",
                "sddp": "Finite-state SDDP",
            }[role]
            axis.fill_between(BUDGETS, mean - sd, mean + sd,
                              color=style["color"], alpha=0.13, linewidth=0)
            handles[role], = axis.plot(
                BUDGETS, mean, **style, label=label, linewidth=1.45,
                markersize=3.8, markeredgewidth=0.6,
                markerfacecolor="white" if role == "matched_mlp"
                                else style["color"],
            )
        axis.set_xscale("log")
        axis.set_xlim(210, 6000)
        axis.set_xticks(BUDGETS, labels=("250", "500", "1k", "2.5k", "5k"))
        axis.minorticks_off()
        axis.set_ylim(*limits)
        axis.set_yticks(range(50, 66, 2) if scale == 1e9 else range(165, 195, 5))
        axis.set_title(title, pad=5)
        axis.set_xlabel(r"Number of sample paths, $N_{\rm fit}$", labelpad=3)
        axis.set_ylabel(r"Test cost (in $10^9$)" if scale == 1e9 else "Test cost")
        axis.set_axisbelow(True)
        axis.grid(color="#BFC4CA", linewidth=0.4, linestyle=(0, (2, 2)))
        axis.legend(handles=[handles[r] for r in STYLES if r in handles],
                    loc=legend_location, fontsize=8.2, frameon=True,
                    fancybox=False, framealpha=1, edgecolor="#AAB0B8",
                    borderpad=0.4, labelspacing=0.32, handlelength=2.7,
                    handletextpad=0.55)
    figure.savefig(output / "real_hydrology_preprint.pdf", metadata={
        "Title": "Hydrology training-budget comparison for the preprint",
        "Creator": "scripts/plot_preprint_hydrology.py",
        "Subject": "Published means and sample SDs; presentation changes only",
    })
    plt.close(figure)


if __name__ == "__main__":
    draw(ROOT / "outputs/paper")
