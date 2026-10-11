"""Summarize every frozen stress-test arm; generate standalone figures only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .projection_distance_run import save_json, sha256
from .projection_plateau import ARMS, DEFAULT_OUTPUT, ROOT, SETTINGS

COLORS = {"zero": "#555555", "fixed_one": "#1479AF", "decay_one": "#DD7722"}
LABELS = {"zero": r"$\lambda=0$", "fixed_one": r"$\lambda=1$",
          "decay_one": r"$\lambda_k=(k+1)^{-1/2}$"}


def main():
    output = DEFAULT_OUTPUT
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((output / "protocol.json").read_text())
    rows, histories = [], []
    for setting in SETTINGS:
        for seed in range(5):
            matched = []
            for arm in ARMS:
                folder = output / "runs" / setting / f"seed{seed}" / arm
                run = json.loads((folder / "result.json").read_text())
                test = json.loads((folder / "test.json").read_text())
                assert test["selected_checkpoint_sha256"] == sha256(folder / "selected.pt")
                matched.append((run["initial_state_sha256"], run["fit_sha256"], run["validation_sha256"], run["batches_sha256"], test["test_sha256"]))
                zero = np.load(output / "runs" / setting / f"seed{seed}" / "zero/test_paths.npz")["selected"]
                paths = np.load(folder / "test_paths.npz")
                differences = paths["selected"] - zero
                row = {"setting": setting, "seed": seed, "arm": arm,
                       "test_cost": test["selected_test"]["mean_path_cost_float64"],
                       "validation_cost": run["selected_validation"]["mean_path_cost_float64"],
                       "initial_test_cost": test["initial_test"]["mean_path_cost_float64"],
                       "test_proposal_distance": test["selected_test"]["proposal_distance"],
                       "initial_proposal_distance": test["initial_test"]["proposal_distance"],
                       "test_active_projection_fraction": test["selected_test"]["active_projection_fraction"],
                       "action_feasibility_residual": test["selected_test"]["max_action_violation"],
                       "storage_feasibility_residual": test["selected_test"]["max_storage_violation"],
                       "selected_update": run["selected_update"],
                       "initial_cost_gradient_norm": run["initial_diagnostic"]["cost_gradient_norm"],
                       "initial_distance_gradient_norm": run["initial_diagnostic"]["distance_gradient_norm"],
                       "selected_cost_gradient_norm": run["selected_diagnostic"]["cost_gradient_norm"],
                       "runtime_seconds_including_monitoring": run["runtime_seconds_including_monitoring"],
                       "max_training_gradient_norm": run["max_training_gradient_norm"],
                       "max_observed_action_change": run["max_observed_action_change"],
                       "first_observed_action_change_update": run["first_observed_action_change_update"],
                       "improvement_from_initial_percent": test["relative_improvement_percent"],
                       "improvement_vs_zero_percent": float(100 * (zero.mean() - paths["selected"].mean()) / zero.mean()),
                       "paired_path_difference": float(differences.mean()),
                       "paired_path_difference_ci_halfwidth": float(student_t.ppf(.975, len(differences)-1) * differences.std(ddof=1) / np.sqrt(len(differences)))}
                rows.append(row)
                hist = pd.read_csv(folder / "history.csv")
                hist["setting"], hist["seed"], hist["arm"] = setting, seed, arm
                histories.append(hist)
                if setting in ("drought_01", "drought_02") and arm == "zero":
                    assert run["max_training_gradient_norm"] == 0
                    assert run["max_observed_action_change"] == 0
                    assert run["max_observed_cost_change"] == 0
                    assert np.array_equal(paths["initial"], paths["selected"])
            assert len(set(matched)) == 1, (setting, seed)
    frame = pd.DataFrame(rows)
    frame.to_csv(report / "per_seed.csv", index=False)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(report / "all_validation_histories.csv", index=False)
    summary = frame.groupby(["setting", "arm"], sort=False).agg(
        n_seeds=("seed", "count"), test_mean=("test_cost", "mean"), test_sd=("test_cost", "std"),
        validation_mean=("validation_cost", "mean"), proposal_distance=("test_proposal_distance", "mean"),
        active_projection_fraction=("test_active_projection_fraction", "mean"),
        improvement_vs_zero_percent=("improvement_vs_zero_percent", "mean"),
        improvement_vs_zero_percent_sd=("improvement_vs_zero_percent", "std"),
        seeds_improved=("improvement_vs_zero_percent", lambda s: int((s > 0).sum())),
        mean_runtime_seconds=("runtime_seconds_including_monitoring", "mean"),
        max_action_residual=("action_feasibility_residual", "max"),
        max_storage_residual=("storage_feasibility_residual", "max"),
    ).reset_index()
    summary.to_csv(report / "summary.csv", index=False)
    paired = []
    for (setting, arm), g in frame.groupby(["setting", "arm"], sort=False):
        delta = g.paired_path_difference.to_numpy()
        halfwidth = student_t.ppf(.975, 4) * delta.std(ddof=1) / np.sqrt(5)
        paired.append({"setting": setting, "arm": arm, "mean_cost_difference_vs_zero": delta.mean(),
                       "paired_seed_ci_lower": delta.mean()-halfwidth,
                       "paired_seed_ci_upper": delta.mean()+halfwidth})
    pd.DataFrame(paired).to_csv(report / "paired_seed_intervals.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    main_history = history[history.setting == "drought_02"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    columns = [("mean_cost", "Validation operating cost", 1),
               ("proposal_distance", "Proposal distance", 1),
               ("cost_gradient_norm", r"Operating-cost gradient norm $\|\nabla C\|$", 1)]
    for ax, (column, ylabel, scale) in zip(axes, columns):
        for arm in ARMS:
            series = main_history[main_history.arm == arm].dropna(subset=[column]).groupby("update")[column].agg(["mean", "std"])
            linestyle = "--" if arm == "decay_one" else "-"
            ax.plot(series.index, series["mean"]/scale, color=COLORS[arm], label=LABELS[arm], linestyle=linestyle, linewidth=2)
            ax.fill_between(series.index, np.maximum(0, series["mean"]-series["std"])/scale,
                            (series["mean"]+series["std"])/scale, color=COLORS[arm], alpha=.12)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes[1].set_yscale("symlog", linthresh=1)
    axes[2].set_yscale("symlog", linthresh=1)
    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Projection plateau: 12 reservoirs, 48 stages; 1% initial storage, 2% inflow", fontsize=11)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(report / f"plateau_mechanism.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 4.2))
    for i, arm in enumerate(("fixed_one", "decay_one")):
        g = summary[summary.arm == arm].set_index("setting").loc[list(SETTINGS)]
        x = np.arange(len(SETTINGS)) + (i-.5)*.33
        ax.bar(x, g.improvement_vs_zero_percent, .31, color=COLORS[arm], label=LABELS[arm],
               yerr=g.improvement_vs_zero_percent_sd, capsize=3)
    ax.set_xticks(np.arange(len(SETTINGS)), ["1% inflow", "2% inflow\n(primary)", "5% inflow", "10% inflow", "Nominal"])
    ax.set_ylabel("Test cost reduction relative to trained λ=0 (%)")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    ax.set_title("Initial storage: 1% for drought cases, 64% for nominal\nFive training seeds; error bars show seed SD", fontsize=11)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(report / f"setting_comparison.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    checks = json.loads((output / "audit/checks.json").read_text())
    source_checks = {name: sha256(ROOT / name) == value for name, value in protocol["source_sha256"].items()}
    paper_checks = {name: sha256(Path(name)) == value for name, value in protocol["paper_before_sha256"].items()}
    assert all(source_checks.values())
    save_json(report / "verification.json", {
        "n_runs": len(frame), "paired_initialization_data_batches_match": True,
        "certified_zero_arms_have_exactly_unchanged_actions_and_costs": True,
        "source_hashes_unchanged": source_checks,
        "paper_hashes_unchanged": paper_checks,
        "independent_replay_and_gradient_checks": checks,
        "max_action_residual": float(frame.action_feasibility_residual.max()),
        "max_storage_residual": float(frame.storage_feasibility_residual.max()),
    })
    lines = [
        "# Projection-distance penalty: drought startup mechanism experiment", "",
        "This is a deliberately constructed stress test of projection-induced flat regions. It is separate from the original nominal benchmark. The current manuscript and PDF have not been edited.", "",
        "The primary case retains the original nonlinear, coupled 12-reservoir/48-stage hydro model, matched MLP actor and default zero-output initialization. Only initial storage (1% of capacity) and the full inflow law (2% of nominal) change. Demand, prices, physical constraints, cost coefficients and architecture remain those of the existing benchmark. Every arm uses identical initialization, fitting paths, batch order, Adam (lr 6e-4, weight decay 1e-6), gradient clipping at 10, and 500 updates. The 1,000 fitting trajectories are the prefix of the original-seed 2,500-path pool. Validation has 500 fresh paths (seed 35202); test has 4,096 fresh paths (seed 35303).", "",
        "The primary case was chosen before observing training outcomes because an all-support bound proves the initial projection plateau. The fitting-only diagnostic inspected inflow scales 1%, 2%, 5%, and 10%; all are reported, together with a nominal control. These are synthetic stress settings, not a claim of realistic drought frequency.", "",
        "| Setting | Weight | Test cost, mean ± seed SD | Reduction vs trained λ=0 | Seeds improved |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.setting} | {row.arm} | {row.test_mean:,.2f} ± {row.test_sd:.2f} | {row.improvement_vs_zero_percent:.3f}% | {row.seeds_improved}/5 |")
    main = summary[summary.setting == "drought_02"].set_index("arm")
    lines += ["", "## Mechanism and mathematical scope", "",
        "At initialization, the unchanged demand-based reference proposal exceeds available water at every stage. A support bound verifies a minimum proposal/storage margin of 3.871, ramp/storage margin of 13.982, and shared-capacity slack of 374.786. Thus x_t=s_t, and the next state depends only on exogenous inflow and delayed routing of those state-determined releases. Induction gives zero parameter sensitivity of every deployed action and of operating cost, including moving recourse. The same proof also succeeds at 1% inflow. It is inconclusive at 5% and 10%; observed training outcomes for those cases are empirical only.", "",
        "The original MLP output layer starts with zero weights and biases. For λ=0, zero loss gradients keep that layer exactly zero under the original Adam optimizer; weight decay on hidden weights does not change the zero network output. In both certified settings, all five seeds have zero loss-gradient norm at every update, and monitored actions/costs and selected test path costs remain exactly unchanged.", "",
        f"The positive penalty supplies a proposal gradient, while retaining gradients through both u and x and moving recourse. In the independent seed-0 replay, the first changed action and first nonzero operating-cost gradient occur after updates {checks['first_action_change_update_seed0_fixed_one']} and {checks['first_nonzero_cost_gradient_update_seed0_fixed_one']}, respectively. λ=1 lowers primary-case test cost by {main.loc['fixed_one', 'improvement_vs_zero_percent']:.3f}%; the decaying schedule lowers it by {main.loc['decay_one', 'improvement_vs_zero_percent']:.3f}%.", "",
        "This is an empirical escape result for the specified Adam training setup, not a guarantee that every optimizer or every positive weight escapes. The existing scalar check is a useful caution: fixed-step plain SGD approaches the boundary without crossing, while Adam crosses. A different feasible initialization can also avoid this particular plateau.", "",
        "The previous gradient-ratio calibration cannot initialize a positive weight here: median ||∇C|| is zero, so λ_ref=0. Accordingly the weights are explicitly prescribed as λ=1 and λ_k=(k+1)^(-1/2), as in the existing scalar mechanism. They are not presented as the earlier calibrated schedule. Algorithm 1 already permits prescribed nonnegative weights; no new derivative formula is needed. A positive floor for the ratio would be a separate method change, not part of this experiment.", "",
        "## Checks and interpretation", "",
        "An independent 30-update replay writes the auxiliary formula directly and reproduces the recorded costs, objectives and gradient norms. At the mixed-projection update-30 checkpoint, central finite differences use 32 fitting paths, three fixed random directions and the original seven step sizes. At h=1e-5 the full derivative has mean normalized error 2.12e-7 and maximum 5.67e-7; omitting moving recourse gives mean error 1.03. All directions and step sizes, including larger-step discrepancies, are retained. Full and ablated forward values are identical.", "",
        "The test set was generated only after the requested selected checkpoints were frozen. Checkpoints are selected on validation operating cost alone, not the penalized loss or test results. Per-seed paired path intervals and paired training-seed intervals are reported separately; they are pointwise, unadjusted descriptions and the settings share underlying random numbers. Reported SD is across five training seeds, not a population confidence bound. Runtimes include diagnostics under concurrent execution and are not an isolated speed comparison.", "",
        "The result supports adding a labeled mechanism/stress experiment and allowing a positive λ in that experiment. It does not justify rewriting the old nominal λ=0 runs as regularized, removing unsuccessful controls, or claiming a universal advantage. Original nominal results should remain visible. No main-paper edits were made.", "",
        "A post-training, validation-only cost decomposition explains the primary-case seed-0 gain: thermal cost rises from 178,472 to 183,343, but storage penalties fall from 92,586 to 65,612 and terminal cost falls from 24,646 to 21,412. The policy retains water instead of repeatedly emptying reservoirs. A simple no-release reference costs 270,507, compared with 270,366 for trained λ=1: most of the primary-case gain is basic water conservation, rather than evidence of sophisticated dispatch near the optimum. At 5% inflow, the corresponding reference costs 239,655 versus 236,383 for the trained policy. This reference was inspected after training for interpretation only, not used for selecting settings or checkpoints, and is not an optimality certificate. Details are in audit/cost_decomposition_and_no_release_reference.csv.", "",
        "Artifacts: summary.csv, per_seed.csv, paired_seed_intervals.csv, all_validation_histories.csv, verification.json; plateau_mechanism.pdf/png and setting_comparison.pdf/png. Raw histories, checkpoints, per-path test costs, protocol, support calculations and numerical audits are in the parent folder.", "",
        "Reproduce from .:", "", "```bash",
        "python -m iclr_algorithm1_experiments.projection_plateau train --workers 15",
        "python -m iclr_algorithm1_experiments.projection_plateau_audit",
        "python -m iclr_algorithm1_experiments.projection_plateau evaluate --workers 10",
        "python -m iclr_algorithm1_experiments.projection_plateau_report", "```", "",
    ]
    (report / "README.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
