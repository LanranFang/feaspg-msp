"""Combine the preserved C diagnostics with the isolated auxiliary check."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import ROOT
OUT = ROOT / "reference_results/iclr_algorithm1_experiments/outputs/auxiliary_gradient_v1"
from iclr_algorithm1_experiments.projection_distance_run import save_json, sha256


def draw(output):
    output.mkdir(parents=True, exist_ok=True)
    old=ROOT/"reference_results/iclr_algorithm1_experiments/outputs/formal_protocol_v1/01_mechanisms"
    sources=[old/"01_finite_difference/finite_difference_check.csv",
             OUT/"diagnostics/finite_difference.csv",old/"04_horizon_amplification/horizon_summary.csv"]
    cost,aux,horizon=[pd.read_csv(f) for f in sources]
    plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","DejaVu Serif"],
                         "mathtext.fontset":"stix","font.size":7.5,"axes.labelsize":7.5,"legend.fontsize":7.3,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(7.,2.10))
    styles={"full":("#00796B","o","-","Full gradient"),
            "no_proposal_state":("#E69F00","s","--","No proposal-state"),
            "no_moving_recourse":("#0072B2","^","-.","No moving-recourse"),
            "detach_auxiliary_action":("#AA3377","D","--","Detached auxiliary action")}
    for ax,frame,modes,title in [(axes[0],cost,("full","no_proposal_state","no_moving_recourse"),r"(a) Operating cost $C$"),
                                  (axes[1],aux,("full","detach_auxiliary_action"),r"(b) Augmented cost $C_\lambda$")]:
        for mode in modes:
            g=frame[frame.gradient_mode==mode].groupby("step").relative_error.mean()
            color,marker,ls,label=styles[mode]
            ax.plot(g.index,g.values,color=color,marker=marker,linestyle=ls,label=label,linewidth=1.1,markersize=2.7)
        ax.set_xscale("log");ax.set_yscale("log");ax.set_ylim(1e-8,2.)
        ax.set_xticks([1e-5,1e-4,1e-3,1e-2]);ax.set_yticks([1e-7,1e-4,1e-1])
        ax.set_xlabel(r"FD step $h$");ax.set_ylabel("FD relative error",labelpad=1)
        ax.set_title(title,fontsize=8);ax.grid(alpha=.2)
    for mode in ("no_proposal_state","no_moving_recourse"):
        g=horizon[horizon.gradient_mode==mode].sort_values("horizon")
        color,marker,ls,label=styles[mode]
        axes[2].plot(g.horizon,g.relative_gradient_error_mean,color=color,marker=marker,linestyle=ls,linewidth=1.1,markersize=2.7)
        axes[2].fill_between(g.horizon,g.relative_gradient_error_mean-g.relative_gradient_error_train_seed_sd,
                            g.relative_gradient_error_mean+g.relative_gradient_error_train_seed_sd,color=color,alpha=.12)
    axes[2].set_xticks([12,24,48,96]);axes[2].set_ylim(bottom=-.04)
    axes[2].set_xlabel(r"Horizon $T$");axes[2].set_ylabel("Relative gradient error",labelpad=1)
    axes[2].set_title("(c) Horizon amplification",fontsize=8);axes[2].grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    extra_h,extra_l=axes[1].get_legend_handles_labels()
    fig.legend(handles+extra_h[1:],labels+extra_l[1:],loc="lower center",ncol=2,frameon=False,
               bbox_to_anchor=(.5,-.015),columnspacing=2.,handlelength=2.2,labelspacing=.35)
    fig.subplots_adjust(left=.065,right=.99,bottom=.36,top=.88,wspace=.52)
    folder=output
    for ext in ("pdf","png"):fig.savefig(folder/f"gradient_mechanism.{ext}",dpi=350,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
    save_json(folder/"gradient_mechanism_provenance.json",{
        "sources":{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        "panel_a":"Unchanged operating-cost FD means over five seeds; all original seven steps.",
        "panel_b":"Augmented FD means over three directions at the saved seed0/update30 scarcity checkpoint; detach only x in d_pd.",
        "panel_c":"Unchanged horizon means and sample SD over five seeds.",
        "figure_sha256":sha256(folder/"gradient_mechanism.pdf")})


if __name__=="__main__":draw(ROOT / "outputs/paper")
