"""Report the frozen projection-distance ablation without further training.

Training seeds, rather than pooled seed-by-test-path observations, are the
replicates for the main paired comparison. Test-path uncertainty is recorded
separately conditional on each pair of trained policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ARM_ORDER = ("zero", "tenth", "fixed", "decay")
LABELS = {
    "zero": r"$\lambda=0$",
    "tenth": r"$0.1\lambda_{\rm ref}$",
    "fixed": r"$\lambda_{\rm ref}$",
    "decay": r"$\lambda_{\rm ref}/\sqrt{k+1}$",
}
COLORS = {"zero": "#222222", "tenth": "#4477AA", "fixed": "#EE6677", "decay": "#228833"}
METRICS = (
    "test_cost", "validation_cost", "validation_cost_float64",
    "test_proposal_distance", "validation_proposal_distance",
    "test_active_projection_fraction", "validation_active_projection_fraction",
    "cost_gradient_norm", "distance_gradient_norm", "total_gradient_norm",
    "test_max_action_violation", "test_max_storage_violation",
    "validation_max_action_violation", "validation_max_storage_violation",
    "selected_fit_seconds", "all_candidates_fit_seconds",
    "selected_monitoring_seconds", "all_candidates_monitoring_seconds",
    "lambda_initialization_seconds_shared", "standalone_lambda_initialization_seconds",
    "all_candidates_plus_initialization_seconds", "evaluation_seconds",
    "lambda_ref", "selected_iteration", "training_gradient_norm_preclip_mean",
    "training_gradient_clipped_fraction",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_interval(values: np.ndarray) -> dict[str, float | int | None]:
    """Unadjusted two-sided Student-t interval, undefined for one replicate."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired interval requires a nonempty finite vector.")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    margin = float(student_t.ppf(0.975, n - 1)) * se if se is not None else None
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
    }


def summarize(per_seed: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        group = per_seed.loc[per_seed.arm == arm]
        row = {"arm": arm, "n_seeds": len(group), "n_fit": int(group.n_fit.iloc[0]), "policy": group.policy.iloc[0]}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
            if "violation" in metric:
                row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def collect(output: Path):
    protocol = read_json(output / "PROTOCOL.json")
    selection = read_json(output / "SELECTION.json")
    completed = read_json(output / "COMPLETED.json")
    evaluations = read_json(output / "evaluations.json")
    reproduction = read_json(output / "BASELINE_REPRODUCTION.json")
    smoke = bool(protocol["protocol"]["smoke"])
    if not smoke and not reproduction.get("all_bitwise_equal", False):
        raise ValueError("Formal baseline reproduction must pass before a report is generated.")
    arms = [arm for arm in ARM_ORDER if arm in protocol["arms"]]
    seeds = protocol["protocol"]["train_seeds"]
    if "zero" not in arms:
        raise ValueError("A zero-lambda arm is required for paired reporting.")
    expected = {(seed, arm) for seed in seeds for arm in arms}
    indexed = {(row["seed"], row["arm"]): row for row in evaluations}
    selected = {(row["seed"], row["arm"]): row for row in selection["selected"]}
    if set(indexed) != expected or set(selected) != expected or len(evaluations) != len(expected):
        raise ValueError("Selected evaluations do not form the complete prespecified seed-by-arm grid.")
    if completed["n_selected"] != len(expected):
        raise ValueError("Completion record disagrees with the selected-policy count.")
    sources = {output / name for name in ("PROTOCOL.json", "SELECTION.json", "COMPLETED.json", "evaluations.json", "BASELINE_REPRODUCTION.json")}
    rows, validation_rows = [], []
    costs = {}
    for seed in seeds:
        for arm in arms:
            evaluation = indexed[seed, arm]
            chosen = selected[seed, arm]
            for key in ("path", "lr_index", "learning_rate", "selected_iteration", "selected_state_sha256", "initial_state_sha256", "fit_sha256", "batches_sha256", "best_validation_cost"):
                if evaluation[key] != chosen[key]:
                    raise ValueError(f"Evaluation differs from frozen selection: {(seed, arm, key)}")
            directory = output / evaluation["path"]
            path_costs = np.load(directory / "test_costs.npy", allow_pickle=False)
            if path_costs.ndim != 1 or len(path_costs) != evaluation["test"]["n_paths"] or not np.isfinite(path_costs).all():
                raise ValueError(f"Invalid test-path cost array: {directory}")
            if path_costs.dtype != np.float64:
                raise ValueError("Stored path costs must retain the runner's float64 aggregation dtype.")
            if float(path_costs.mean()) != evaluation["test"]["mean_path_cost_float64"]:
                raise ValueError("Stored test mean does not exactly reproduce the saved path array.")
            costs[seed, arm] = path_costs
            candidate_files = sorted(directory.parent.glob("lr*/result.json"))
            candidates = [read_json(path) for path in candidate_files]
            expected_rates = protocol["protocol"]["graph_learning_rates"] if protocol["policy"] == "projected_graph_neural_global" else protocol["protocol"]["mlp_learning_rates"]
            if len(candidates) != len(expected_rates):
                raise ValueError(f"Missing LR candidates for {(seed, arm)}")
            minimum = min(candidates, key=lambda value: (value["best_validation_cost"], value["lr_index"]))
            if minimum["path"] != evaluation["path"]:
                raise ValueError("Saved checkpoint does not match validation-only LR selection.")
            training = pd.read_csv(directory / "training.csv")
            validation = pd.read_csv(directory / "validation.csv")
            validation["seed"] = seed
            validation["arm"] = arm
            validation["selected_learning_rate"] = evaluation["learning_rate"]
            validation_rows.append(validation)
            init_path = output / "initialization" / f"seed{seed}.json"
            initialization = read_json(init_path)
            sources.update(candidate_files)
            sources.update([directory / "test_costs.npy", directory / "training.csv", directory / "validation.csv", directory / "evaluation.json", init_path])
            test, val, diagnostic = evaluation["test"], evaluation["validation"], evaluation["fit_gradient_diagnostic"]
            rows.append({
                "seed": seed, "arm": arm, "n_fit": evaluation["n_fit"], "policy": evaluation["policy"],
                "learning_rate": evaluation["learning_rate"], "lr_index": evaluation["lr_index"],
                "selected_iteration": evaluation["selected_iteration"], "candidate_count": len(candidates),
                "test_n_paths": len(path_costs), "test_cost": float(path_costs.mean()),
                "validation_cost": evaluation["best_validation_cost"],
                "validation_cost_float64": val["mean_path_cost_float64"],
                "test_proposal_distance": test["proposal_distance"],
                "validation_proposal_distance": val["proposal_distance"],
                "test_active_projection_fraction": test["active_projection_fraction"],
                "validation_active_projection_fraction": val["active_projection_fraction"],
                "active_projection_tolerance": test["active_projection_tolerance"],
                "cost_gradient_norm": diagnostic["cost_gradient_norm"],
                "distance_gradient_norm": diagnostic["distance_gradient_norm"],
                "total_gradient_norm": diagnostic["total_gradient_norm"],
                "diagnostic_lambda": diagnostic["lambda_value"],
                "gradient_diagnostic_n_paths": diagnostic["n_paths"],
                "test_max_action_violation": test["max_action_violation"],
                "test_max_storage_violation": test["max_storage_violation"],
                "validation_max_action_violation": val["max_action_violation"],
                "validation_max_storage_violation": val["max_storage_violation"],
                "selected_fit_seconds": evaluation["fit_time_seconds"],
                "all_candidates_fit_seconds": sum(candidate["fit_time_seconds"] for candidate in candidates),
                "selected_monitoring_seconds": evaluation["monitoring_seconds"],
                "all_candidates_monitoring_seconds": sum(candidate["monitoring_seconds"] for candidate in candidates),
                "lambda_initialization_seconds_shared": initialization["elapsed_seconds"],
                "standalone_lambda_initialization_seconds": initialization["elapsed_seconds"] if arm != "zero" else 0.0,
                "all_candidates_plus_initialization_seconds": sum(candidate["fit_time_seconds"] for candidate in candidates) + (initialization["elapsed_seconds"] if arm != "zero" else 0.0),
                "evaluation_seconds": evaluation["evaluation_seconds"],
                "lambda_ref": initialization["lambda_ref"],
                "training_gradient_norm_preclip_mean": float(training.gradient_norm_preclip.mean()),
                "training_gradient_clipped_fraction": float(training.gradient_clipped.mean()),
                "selected_state_sha256": evaluation["selected_state_sha256"],
                "initial_state_sha256": evaluation["initial_state_sha256"],
                "fit_sha256": evaluation["fit_sha256"], "batches_sha256": evaluation["batches_sha256"],
                "test_sha256": evaluation["test_sha256"], "path": evaluation["path"],
            })
    per_seed = pd.DataFrame(rows)
    for seed in seeds:
        group = per_seed.loc[per_seed.seed == seed]
        for key in ("initial_state_sha256", "fit_sha256", "batches_sha256", "test_sha256", "test_n_paths", "lambda_ref"):
            if group[key].nunique() != 1:
                raise ValueError(f"Unmatched arms for seed {seed}: {key}")
    for arm in arms:
        mean_validation = float(np.mean([selected[seed, arm]["best_validation_cost"] for seed in seeds]))
        if mean_validation != selection["mean_validation"][arm]:
            raise ValueError("Selection's validation mean does not match the selected candidates.")
    return protocol, selection, reproduction, per_seed, pd.concat(validation_rows, ignore_index=True), costs, sources, arms


def pair_comparisons(per_seed, costs, arms):
    paired_rows, summary_rows = [], []
    seeds = sorted(per_seed.seed.unique())
    indexed = per_seed.set_index(["seed", "arm"])
    for arm in arms:
        if arm == "zero":
            continue
        deltas, validation_deltas = [], []
        for seed in seeds:
            baseline, treatment = indexed.loc[(seed, "zero")], indexed.loc[(seed, arm)]
            path_deltas = costs[seed, arm] - costs[seed, "zero"]
            conditional = paired_interval(path_deltas)
            # Averaging pairwise path differences and differencing the two means
            # are numerically equivalent, but may round by a few ulps.
            if not np.isclose(conditional["mean"], treatment.test_cost - baseline.test_cost, rtol=1e-12, atol=1e-9):
                raise ValueError("Paired test delta does not match the policy means.")
            delta = float(treatment.test_cost - baseline.test_cost)
            validation_delta = float(treatment.validation_cost - baseline.validation_cost)
            deltas.append(delta)
            validation_deltas.append(validation_delta)
            paired_rows.append({
                "seed": int(seed), "arm": arm,
                "baseline_test_cost": float(baseline.test_cost), "regularized_test_cost": float(treatment.test_cost),
                "test_delta": delta, "test_delta_percent": 100.0 * delta / float(baseline.test_cost),
                "validation_delta": validation_delta,
                "within_seed_n_test_paths": conditional["n"],
                "within_seed_path_delta_sd": conditional["sd"],
                "within_seed_path_delta_se": conditional["se"],
                "within_seed_path_ci95_low": conditional["ci95_low"],
                "within_seed_path_ci95_high": conditional["ci95_high"],
            })
        aggregate = paired_interval(np.asarray(deltas))
        validation_aggregate = paired_interval(np.asarray(validation_deltas))
        summary_rows.append({
            "arm": arm, "n_seeds": aggregate["n"],
            "test_delta_mean": aggregate["mean"], "test_delta_sd": aggregate["sd"],
            "test_delta_se": aggregate["se"], "test_seed_ci95_low": aggregate["ci95_low"],
            "test_seed_ci95_high": aggregate["ci95_high"],
            "validation_delta_mean": validation_aggregate["mean"],
            "validation_delta_sd": validation_aggregate["sd"],
            "test_improved_seed_count": int(np.sum(np.asarray(deltas) < 0.0)),
            "validation_improved_seed_count": int(np.sum(np.asarray(validation_deltas) < 0.0)),
        })
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def decide(protocol, selection, paired_summary):
    arm = selection["selected_positive_arm"]
    positive = [value for value in protocol["arms"] if value != "zero"]
    expected = min(positive, key=lambda value: selection["mean_validation"][value]) if positive else None
    if arm != expected:
        raise ValueError("Frozen positive arm is not the validation-selected arm.")
    if arm is None:
        return {"selected_positive_arm": None, "extension_criterion_met": False, "extend_displayed_mlp_budgets": False, "reason": "No positive-lambda arm was evaluated."}
    row = paired_summary.loc[paired_summary.arm == arm].iloc[0]
    validation_delta = float(selection["mean_validation"][arm] - selection["mean_validation"]["zero"])
    if bool(selection["validation_improves"]) != (validation_delta < 0):
        raise ValueError("Frozen validation improvement flag is inconsistent.")
    test_delta = float(row.test_delta_mean)
    low = float(row.test_seed_ci95_low) if pd.notna(row.test_seed_ci95_low) else None
    high = float(row.test_seed_ci95_high) if pd.notna(row.test_seed_ci95_high) else None
    criterion = validation_delta < 0 and test_delta < 0
    smoke = bool(protocol["protocol"]["smoke"])
    return {
        "selected_positive_arm": arm,
        "selection_source": "SELECTION.json, frozen before test generation/access",
        "validation_mean_delta": validation_delta, "test_mean_delta": test_delta,
        "test_seed_ci95": [low, high], "test_improved_seed_count": int(row.test_improved_seed_count),
        "n_training_seeds": int(row.n_seeds),
        "extension_criterion_met": bool(criterion),
        "extend_displayed_mlp_budgets": bool(criterion and not smoke and protocol["policy"] == "projected_mlp_matched"),
        "remaining_displayed_budgets": [int(n) for n in protocol["protocol"]["fit_budgets"] if n != protocol["n_fit"]],
        "rule": "Both validation mean and test mean must improve over zero for the already validation-selected positive arm; no alternative arm may be chosen using test results.",
        "seed_ci_includes_zero": bool(low <= 0 <= high) if low is not None else None,
        "uncertainty": "Approximate unadjusted Student-t 95% interval across paired training-seed means conditional on the fixed test set; does not remove validation-arm selection uncertainty or combine finite test-path uncertainty.",
        "smoke": smoke,
        "action": "Extend only the chosen MLP A1 configuration to the remaining displayed controlled budgets." if criterion and not smoke and protocol["policy"] == "projected_mlp_matched" else "Retain the current main results; report the auxiliary-loss mechanism and ablation separately.",
    }


def plots(report, summary, curves, arms, n_seeds):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for axis, metric, label in zip(axes, ("mean_cost", "proposal_distance"), ("Validation operating cost", r"Validation $d_{\rm pd}$")):
        for arm in arms:
            grouped = curves.loc[curves.arm == arm].groupby("iteration")[metric]
            mean = grouped.mean()
            sd = grouped.std(ddof=1).fillna(0.0)
            x = mean.index.to_numpy()
            axis.plot(x, mean.to_numpy(), label=LABELS[arm], color=COLORS[arm])
            if n_seeds > 1:
                axis.fill_between(x, (mean - sd).to_numpy(), (mean + sd).to_numpy(), color=COLORS[arm], alpha=0.12)
        axis.set(xlabel="Optimizer updates", ylabel=label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Validation-only selected learning rates; mean over {n_seeds} training seeds" + (" (shading: ±1 SD)" if n_seeds > 1 else ""), fontsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(report / f"training_curves.{suffix}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    frame = summary.set_index("arm").loc[arms]
    for axis, metric, label in zip(axes, ("test_cost", "test_proposal_distance", "test_active_projection_fraction"), ("Test operating cost", r"Test $d_{\rm pd}$", "Active projection fraction")):
        means = frame[f"{metric}_mean"].to_numpy()
        deviations = frame[f"{metric}_sd"].fillna(0).to_numpy()
        x = np.arange(len(arms))
        axis.bar(x, means, yerr=deviations if n_seeds > 1 else None, capsize=3, color=[COLORS[arm] for arm in arms], alpha=0.82)
        axis.set_xticks(x, [LABELS[arm] for arm in arms], fontsize=9)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        if metric == "test_active_projection_fraction":
            axis.set_ylim(0, max(1.0, float((means + deviations).max()) * 1.05))
    fig.suptitle(f"Selected policies, {n_seeds} training seeds" + ("; error bars: ±1 SD across seeds" if n_seeds > 1 else ""), fontsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(report / f"ablation.{suffix}")
    plt.close(fig)


def value_sd(mean, sd, precision=3):
    return f"{mean:.{precision}f} ± {sd:.{precision}f}" if pd.notna(sd) else f"{mean:.{precision}f} (one seed)"


def text_reports(report, protocol, selection, reproduction, summary, paired_summary, decision, per_seed):
    n_seeds = len(protocol["protocol"]["train_seeds"])
    smoke = protocol["protocol"]["smoke"]
    lines = [
        "# 投影距离辅助损失消融", "",
        ("**这是 smoke 检查，不能作为正式经验结论。**" if smoke else "这是预先固定协议下的正式消融；主比较的重复单位为训练种子。"), "",
        f"模型：`{protocol['policy']}`；训练轨迹 {protocol['n_fit']} 条；{n_seeds} 个训练种子；每组候选学习率数 {int(per_seed.candidate_count.iloc[0])}。四组共享初始化、拟合路径、批次顺序、优化器与训练预算。",
        "训练目标为原始运行成本加辅助损失，验证选模型和测试评估仍使用原始运行成本。测试成本采用保存的逐路径 float32 成本，以 float64 求均值；不是重新用 float64 模拟。", "",
        "| λ 版本 | 验证成本（均值 ± SD） | 测试成本（均值 ± SD） | 测试投影距离（均值 ± SD） | 投影修正比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.arm} | {value_sd(row.validation_cost_mean, row.validation_cost_sd)} | {value_sd(row.test_cost_mean, row.test_cost_sd)} | {value_sd(row.test_proposal_distance_mean, row.test_proposal_distance_sd)} | {100 * row.test_active_projection_fraction_mean:.2f}% |")
    lines += ["", "以下差值均为正 λ 组减去同种子的 λ=0 组；负值表示运行成本降低。", "", "| λ 版本 | 测试成本配对差值 | 跨训练种子 95% t 区间 | 测试改善种子数 |", "|---|---:|---:|---:|"]
    for row in paired_summary.itertuples(index=False):
        interval = f"[{row.test_seed_ci95_low:.3f}, {row.test_seed_ci95_high:.3f}]" if pd.notna(row.test_seed_ci95_low) else "不可计算（一个种子）"
        lines.append(f"| {row.arm} | {row.test_delta_mean:+.3f} | {interval} | {row.test_improved_seed_count}/{row.n_seeds} |")
    arm = decision["selected_positive_arm"]
    lines += ["", f"测试前，按平均验证成本选择的正 λ 版本是 **{arm}**。"]
    if arm is not None:
        lines.append(f"该版本平均验证成本差值为 {decision['validation_mean_delta']:+.3f}，平均测试成本差值为 {decision['test_mean_delta']:+.3f}。")
        if decision["extend_displayed_mlp_budgets"]:
            lines.append("按预设均值改善规则，满足扩展到其余已展示 MLP A1 训练预算的条件；这不等同于已证明稳定或显著改善，其他基线无需重跑。")
        else:
            lines.append("当前不触发正式主图预算扩展。保留原主结果，将机制示例与本消融单独报告；不能因为其他版本的测试结果更好而改选版本。")
        if decision["seed_ci_includes_zero"]:
            lines.append("所选版本的跨种子区间包含 0，现有五种子证据不足以排除没有改善；均值方向与稳定收益需区分。" if n_seeds == 5 else "所选版本的跨种子区间包含 0，均值方向与稳定收益需区分。")
    lines += [
        "", "配对区间基于训练种子的成本均值，固定测试集上的有限路径误差另列在 `paired_seed.csv`。相同测试路径被多个训练种子复用，不能将种子 × 测试路径的数量当作独立重复数。上述 t 区间是未作多重比较或验证选版本校正的近似描述，不覆盖所有选择不确定性。", "",
        "| λ 版本 | ∥∇C∥ | ∥∇d_pd∥ | ∥∇Cλ∥ | 最大动作违反 | 最大库容违反 | 所选候选耗时（秒） | 全部学习率候选耗时（秒） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.cost_gradient_norm_mean:.3g} | {row.distance_gradient_norm_mean:.3g} | {row.total_gradient_norm_mean:.3g} | {row.test_max_action_violation_max:.3g} | {row.test_max_storage_violation_max:.3g} | {value_sd(row.selected_fit_seconds_mean, row.selected_fit_seconds_sd, 1)} | {value_sd(row.all_candidates_fit_seconds_mean, row.all_candidates_fit_seconds_sd, 1)} |")
    shared = per_seed.drop_duplicates("seed").lambda_initialization_seconds_shared
    lines += [
        "", f"初始化 λ_ref 估计每个种子只计算一次，三个正 λ 组共享；平均额外耗时为 {shared.mean():.2f} 秒，总计 {shared.sum():.2f} 秒。若将某个正 λ 版本单独部署，在该版本全部学习率候选耗时之外只加一次初始化开销（不按学习率重复加）；λ=0 的这一开销为 0。实际实验总开销按每个种子只计一次，不能再把三个版本的独立部署估算相加。",
        "候选训练耗时包含监控和并行竞争，不是隔离环境下的速度基准；监控耗时另存于 CSV。梯度范数在所选模型上用固定前 32 条拟合路径（smoke 时为全部较小拟合池）计算，未裁剪，未用于更新；训练曲线另记录每步裁剪前范数与裁剪比例。",
        "投影距离严格为每条路径的 ½Σₜ∥uₜ−xₜ∥²，再对路径取平均；未除以阶段数或水库数。投影修正比例按路径–阶段动作向量 ∥uₜ−xₜ∥₂ > 10⁻⁶ 计算，不等于绑定约束的比例。",
        "每条训练曲线选用该种子最终由验证集选中的学习率候选，显示各更新时点的模型诊断；不是把不同候选的最好时点拼成一条曲线。阴影为种子间 ±1 SD，不是置信区间。",
        "", f"原 λ=0 参数逐位复现：{reproduction.get('all_bitwise_equal', 'smoke 不适用')}。数据和选择身份校验见 `PROVENANCE.json`；扩展判定见 `DECISION.json`。", "",
    ]
    (report / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    readme = [
        "# Projection-distance auxiliary-loss ablation", "",
        "This report reads completed, frozen artifacts only. It performs no fitting, retuning, test-driven configuration selection, or manuscript editing.", "",
        "- `summary.csv`: arm means and sample SD across training seeds. Test C is the float64 mean of saved per-path costs from the original float32 simulator; validation C is the original checkpoint-selection metric. Additional float64-aggregated validation means are explicitly named.",
        "- `per_seed.csv`: selected candidate identities, original costs, raw half-squared-sum distance, active projection fractions, gradients, feasibility residuals, selected-fit and all-LR-candidate runtimes, and separately charged initialization overhead.",
        "- `paired_summary.csv`: regularized-minus-zero differences and unadjusted 95% Student-t intervals over paired training-seed means. These intervals condition on the fixed test set and do not remove validation-arm selection uncertainty.",
        "- `paired_seed.csv`: paired test-path uncertainty conditional on each trained policy pair. Paths shared across seeds are never pooled as independent seed-by-path replicates.",
        "- `selected_validation_curves.csv` and `training_curves.pdf/png`: validation cost/distance by updates for each seed's validation-selected LR candidate. Shading is one SD across seeds, without smoothing or best-so-far stitching.",
        "- `ablation.pdf/png`: held-out operating cost, projection distance, and fraction of path-stage action vectors corrected by more than absolute L2 tolerance 1e-6. Error bars are one SD across training seeds.",
        "- `DECISION.json`: only the positive arm frozen in SELECTION.json can trigger extension. Both its validation and test mean must beat zero under the declared rule; uncertainty is reported even if the mean criterion passes. Smoke runs cannot trigger extension.",
        "- `PROVENANCE.json`: hashes of report inputs and generated artifacts; `REPORT_ZH.md`: detailed Chinese interpretation.", "",
        "The final-model gradient diagnostic uses the first 32 fitting paths (the full smaller fitting pool in smoke runs), no clipping and no updates. It reports gradients of C, d_pd and their weighted sum, with the coefficient at the selected checkpoint's last update. The regularizer coefficient estimate is a frozen ratio of median per-path parameter-gradient norms at initialization, using all fitting paths only.", "",
        "Initialization calibration is physically performed once per seed and shared across the three positive arms. `lambda_initialization_seconds_shared` records this measured value, while `standalone_lambda_initialization_seconds` charges it once per positive arm considered in isolation (not once per LR), and zero to the zero arm. `all_candidates_plus_initialization_seconds` adds that standalone charge to all candidate fit times. Do not sum these standalone charges across arms to estimate the actual experiment total; the actual total is recorded separately in PROVENANCE.json. Wall times include monitoring and concurrent workers, so they are not isolated latency comparisons.", "",
        f"Validation-selected positive arm: `{arm}`. Extension criterion met: `{decision['extension_criterion_met']}`. Formal MLP budget extension triggered: `{decision['extend_displayed_mlp_budgets']}`.", "",
    ]
    (report / "README.md").write_text("\n".join(readme), encoding="utf-8")


def run(output: Path) -> Path:
    output = output.resolve()
    protocol, selection, reproduction, per_seed, curves, costs, sources, arms = collect(output)
    summary = summarize(per_seed, arms)
    paired_seed, paired_summary = pair_comparisons(per_seed, costs, arms)
    decision = decide(protocol, selection, paired_summary)
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    for name, frame in (("per_seed", per_seed), ("summary", summary), ("paired_seed", paired_seed), ("paired_summary", paired_summary), ("selected_validation_curves", curves)):
        frame.to_csv(report / f"{name}.csv", index=False)
    write_json(report / "DECISION.json", decision)
    plots(report, summary, curves, arms, len(protocol["protocol"]["train_seeds"]))
    text_reports(report, protocol, selection, reproduction, summary, paired_summary, decision, per_seed)
    write_json(report / "PROVENANCE.json", {
        "report_source_sha256": file_hash(Path(__file__)),
        "input_sha256": {str(path.relative_to(output)): file_hash(path) for path in sorted(sources)},
        "artifact_sha256": {path.name: file_hash(path) for path in sorted(report.iterdir()) if path.is_file() and path.name != "PROVENANCE.json"},
        "checks": {
            "complete_prespecified_seed_arm_grid": True,
            "evaluations_match_frozen_selection": True,
            "selected_lr_minimizes_original_validation_cost": True,
            "stored_test_means_exactly_match_float64_path_array_means": True,
            "arms_share_initialization_fit_batches_and_test": True,
            "paired_path_deltas_match_differences_of_means": True,
            "positive_arm_chosen_only_by_frozen_validation": True,
            "formal_zero_parameters_bitwise_reproduce_original": reproduction.get("all_bitwise_equal"),
        },
        "main_uncertainty_unit": "paired training-seed mean, conditional on the common fixed test set",
        "test_path_uncertainty": "within each seed, reported separately; not pooled across seeds",
        "actual_shared_calibration_seconds_total": float(per_seed.drop_duplicates("seed").lambda_initialization_seconds_shared.sum()),
        "actual_shared_calibration_seed_count": int(per_seed.seed.nunique()),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed controlled_n1000 (or smoke) output directory")
    args = parser.parse_args()
    print(run(args.output))


if __name__ == "__main__":
    main()
