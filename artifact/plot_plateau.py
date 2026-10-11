"""Plot the first 300 updates of the immutable 500-update plateau histories."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import ROOT
from iclr_algorithm1_experiments.projection_distance_run import save_json, sha256


def draw(output):
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reference_results/iclr_algorithm1_experiments/outputs/projection_plateau_v1/report/all_validation_histories.csv"
    frame = pd.read_csv(source)
    frame = frame[frame.setting == "drought_02"]
    display_updates = 300
    training_updates = int(frame["update"].max())
    frame = frame[frame["update"] <= display_updates]
    plt.rcParams.update({"font.family":"serif", "font.serif":["Liberation Serif","DejaVu Serif"],
                         "mathtext.fontset":"stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.85))
    styles = {"zero": ("#555555", "-", r"$\lambda=0$"),
              "fixed_one": ("#0072B2", "-", r"$\lambda=1$"),
              "decay_one": ("#D55E00", "--", r"$\lambda_k=(k+1)^{-1/2}$")}
    columns = [("mean_cost", "Validation cost\n(thousands)", 1000),
               ("proposal_distance", "Proposal distance", 1),
               ("cost_gradient_norm", r"Cost gradient $\|\nabla C\|$", 1)]
    for ax, (column, label, scale) in zip(axes, columns):
        for arm, (color, linestyle, legend) in styles.items():
            g = frame[frame.arm == arm].dropna(subset=[column]).groupby("update")[column].agg(["mean", "std"])
            assert (frame[frame.arm == arm].dropna(subset=[column]).groupby("update").size() == 5).all()
            ax.plot(g.index, g["mean"]/scale, color=color, linestyle=linestyle, label=legend, linewidth=1.5)
            ax.fill_between(g.index, np.maximum(0, g["mean"]-g["std"])/scale,
                            (g["mean"]+g["std"])/scale, color=color, alpha=.13)
        ax.set_xlabel("Adam updates")
        ax.set_ylabel(label, labelpad=2)
        ax.set_xlim(-3, display_updates + 3)
        ax.set_xticks([0, 100, 200, 300])
        ax.grid(alpha=.2)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[2].set_yscale("symlog", linthresh=1)
    # A small linear margin separates an exactly zero curve from the axis spine.
    axes[1].set_ylim(bottom=-.3, top=5e5)
    axes[2].set_ylim(bottom=-.3, top=3e4)
    for ax in axes[1:]:
        ax.set_yticks([0, 1e2, 1e4])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.5,-.005), columnspacing=2)
    fig.subplots_adjust(left=.075,right=.99,bottom=.36,top=.97,wspace=.47)
    for ext in ("pdf", "png"):
        fig.savefig(output/f"proposal_distance_mechanism.{ext}", dpi=350, bbox_inches="tight",pad_inches=.02)
    plt.close(fig)
    save_json(output/"proposal_distance_mechanism_provenance.json",{
        "source":str(source.relative_to(ROOT)),"source_sha256":sha256(source),
        "setting":"drought_02: 12 reservoirs, 48 stages, 1% initial storage, 2% inflow",
        "curves":"Five-seed means and sample SD; unchanged measurements, compact layout",
        "displayed_updates":[0,display_updates],
        "training_updates":training_updates,
        "figure_size_inches":[6.8,1.85],
        "cost_and_distance":"500 validation paths, sampled every10 updates including initialization",
        "gradient":"32 fixed fitting paths, sampled every50 updates including initialization",
        "figure_sha256":sha256(output/"proposal_distance_mechanism.pdf")})


if __name__ == "__main__":
    draw(ROOT / "outputs/paper")
