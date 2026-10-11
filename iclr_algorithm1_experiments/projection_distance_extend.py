"""Conditionally extend the validation-selected positive A1 loss across budgets.

This runner never trains the zero arm and never selects another lambda version.
It reuses the original lambda=0 checkpoints/results and the existing N=1000
positive-arm results. Each remaining budget freezes all LR/checkpoint choices
before any new policy evaluation on test data.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import ICLRProtocol
from .io import dataset_fingerprint
from .projection_distance_report import paired_interval
from .projection_distance_run import (
    ORIGINAL, ROOT, configure, data, evaluate_one, fit_one, parallel,
    prepare_seed, save_json, sha256,
)


POLICY = "projected_mlp_matched"
DEFAULT_PARENT = ROOT / "iclr_algorithm1_experiments/outputs/projection_distance_v1/controlled_n1000"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_json(path: Path, value) -> None:
    """Only one root process writes each independently sharded budget."""
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if load(path) != normalized:
            raise RuntimeError(f"Frozen extension artifact changed: {path}")
    else:
        save_json(path, value)


def authorized_parent(parent: Path):
    decision_path = parent / "report/DECISION.json"
    decision = load(decision_path)
    if not decision.get("extend_displayed_mlp_budgets", False) or decision.get("smoke", True):
        raise RuntimeError("The completed formal ablation did not trigger the prespecified extension rule.")
    protocol = load(parent / "PROTOCOL.json")
    selection = load(parent / "SELECTION.json")
    if protocol["policy"] != POLICY or protocol["n_fit"] != 1000 or protocol["protocol"]["smoke"]:
        raise RuntimeError("Extension requires the formal controlled N=1000 MLP ablation.")
    current = json.loads(json.dumps(asdict(ICLRProtocol())))
    if protocol["protocol"] != current:
        raise RuntimeError("Current experiment configuration differs from the frozen parent protocol.")
    if not load(parent / "BASELINE_REPRODUCTION.json").get("all_bitwise_equal", False):
        raise RuntimeError("Original zero-lambda reproduction must have passed.")
    audit_path = parent / "INDEPENDENT_AUDIT.json"
    if not audit_path.exists():
        raise RuntimeError("A completed independent audit with all 20 selected-policy replays is required.")
    audit = load(audit_path)
    expected_audit = {
        "status": "passed", "n_candidates": 60, "n_selected": 20,
        "original_baselines_bitwise_verified": 5,
        "independently_replayed_selected_evaluations": 20,
        "protocol_sha256": sha256(parent / "PROTOCOL.json"),
    }
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise RuntimeError("Independent parent audit must pass with 60 candidates, 20 selected-policy replays, and 5 original baseline identities.")
    if load(parent / "COMPLETED.json")["n_candidates"] != 60:
        raise RuntimeError("Parent controlled ablation is incomplete.")
    arm = selection["selected_positive_arm"]
    positive = [value for value in protocol["arms"] if value != "zero"]
    validation_choice = min(positive, key=lambda value: selection["mean_validation"][value])
    if arm != validation_choice or arm != decision["selected_positive_arm"]:
        raise RuntimeError("Extension arm differs from the frozen validation-only choice.")
    validation_delta = selection["mean_validation"][arm] - selection["mean_validation"]["zero"]
    evaluations = load(parent / "evaluations.json")
    selected_mean = np.mean([row["test"]["mean_path_cost_float64"] for row in evaluations if row["arm"] == arm])
    zero_mean = np.mean([row["test"]["mean_path_cost_float64"] for row in evaluations if row["arm"] == "zero"])
    test_delta = float(selected_mean - zero_mean)
    if validation_delta >= 0 or test_delta >= 0:
        raise RuntimeError("Actual frozen validation/test means do not meet the extension criterion.")
    if not np.isclose(test_delta, decision["test_mean_delta"], rtol=1e-12, atol=1e-9):
        raise RuntimeError("Parent test decision is inconsistent with saved evaluation means.")
    provenance = load(parent / "report/PROVENANCE.json")
    if provenance["artifact_sha256"]["DECISION.json"] != sha256(decision_path):
        raise RuntimeError("Parent decision differs from its report provenance.")
    for relative, digest in protocol["source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Frozen training source changed before extension: {relative}")
    remaining = tuple(n for n in ICLRProtocol().fit_budgets if n != protocol["n_fit"])
    if list(remaining) != decision["remaining_displayed_budgets"]:
        raise RuntimeError("Extension budgets differ from the prespecified displayed budgets.")
    return protocol, selection, decision, arm, remaining


def original_checkpoint(n_fit: int, seed: int) -> Path:
    return ORIGINAL / "checkpoints" / f"n{n_fit}" / f"{POLICY}_selected_seed{seed}.pt"


def freeze_budget(output: Path, parent: Path, n_fit: int, workers: int, arm: str, parent_protocol) -> None:
    protocol = ICLRProtocol()
    _, pool, fit, validation = data(protocol, n_fit)
    manifest = load(ORIGINAL / "data_manifest.json")
    if dataset_fingerprint(pool) != manifest["fit_pool"]["sha256"] or dataset_fingerprint(validation) != manifest["validation"]["sha256"]:
        raise RuntimeError("Extension does not reproduce the original pool and validation data.")
    original_hashes = {str(original_checkpoint(n_fit, seed).relative_to(ROOT)): sha256(original_checkpoint(n_fit, seed)) for seed in protocol.train_seeds}
    source_hashes = {**parent_protocol["source_sha256"],
                     "iclr_algorithm1_experiments/projection_distance_extend.py": sha256(Path(__file__)),
                     "iclr_algorithm1_experiments/projection_distance_report.py": sha256(ROOT / "iclr_algorithm1_experiments/projection_distance_report.py")}
    freeze_json(output / "PROTOCOL.json", {
        "experiment": "conditional_projection_distance_budget_extension",
        "protocol": asdict(protocol), "n_fit": n_fit, "policy": POLICY,
        "arms": [arm], "selected_positive_arm": arm,
        "parent": str(parent), "parent_decision_sha256": sha256(parent / "report/DECISION.json"),
        "parent_selection_sha256": sha256(parent / "SELECTION.json"),
        "parent_protocol_sha256": sha256(parent / "PROTOCOL.json"),
        "parent_independent_audit_sha256": sha256(parent / "INDEPENDENT_AUDIT.json"),
        "source_sha256": source_hashes,
        "original_selected_checkpoint_sha256": original_hashes,
        "original_results_sha256": sha256(ORIGINAL / "per_run_results.csv"),
        "original_test_path_costs_sha256": sha256(ORIGINAL / "test_path_costs.csv") if (ORIGINAL / "test_path_costs.csv").exists() else None,
        "original_data_manifest_sha256": sha256(ORIGINAL / "data_manifest.json"),
        "fit_pool_sha256": dataset_fingerprint(pool), "fit_sha256": dataset_fingerprint(fit),
        "validation_sha256": dataset_fingerprint(validation), "expected_test_sha256": manifest["test"]["sha256"],
        "selection": "Lambda arm is inherited unchanged from the N=1000 validation selection; checkpoint and LR selected by original validation cost separately per seed; all choices frozen before test evaluation.",
        "baseline": "Reuse original zero-lambda selected checkpoint and archived costs; no baseline training or invented gradient/trajectory diagnostics.",
        "lambda_reference": "Re-estimated once per seed at initialization from all fitting paths of this budget; same nested 2500-path original pool; raw per-path gradient medians.",
        "training_candidates": len(protocol.train_seeds) * len(protocol.mlp_learning_rates),
        "workers": workers, "torch_threads_per_worker": 1,
        "timing": "Parallel wall times include monitoring; calibration once per seed; no cross-era runtime speedup claim.",
        "inference": "Budgets share paths, test data, and seeds. Pointwise seed CIs are descriptive, not independent confirmations or simultaneous guarantees.",
    })


def run_budget(parent: Path, extension: Path, n_fit: int, workers: int = 8) -> Path:
    parent = parent.resolve()
    extension = extension.resolve()
    parent_protocol, _, _, arm, remaining = authorized_parent(parent)
    if n_fit not in remaining:
        raise ValueError(f"Run only a remaining budget from {remaining}; N=1000 is reused.")
    configure()
    protocol = ICLRProtocol()
    output = extension / f"n{n_fit}"
    output.mkdir(parents=True, exist_ok=True)
    freeze_budget(output, parent, n_fit, workers, arm, parent_protocol)
    started = time.perf_counter()
    parallel(prepare_seed, [(protocol, n_fit, seed, POLICY, str(output)) for seed in protocol.train_seeds],
             min(workers, len(protocol.train_seeds)), f"n{n_fit} initialization")
    tasks = [(protocol, n_fit, seed, POLICY, arm, index, str(output))
             for index in range(len(protocol.mlp_learning_rates)) for seed in protocol.train_seeds]
    np.random.default_rng(92751 + n_fit).shuffle(tasks)
    results = parallel(fit_one, tasks, workers, f"n{n_fit} fit")
    results.sort(key=lambda value: (value["seed"], value["lr_index"]))
    selected = [min((row for row in results if row["seed"] == seed),
                    key=lambda row: (row["best_validation_cost"], row["lr_index"])) for seed in protocol.train_seeds]
    for row in selected:
        initial = load(output / "initialization" / f"seed{row['seed']}.json")
        for field in ("initial_state_sha256", "fit_sha256", "batches_sha256"):
            if row[field] != initial[field]:
                raise RuntimeError(f"Initialization identity mismatch: {field}")
    freeze_json(output / "SELECTION.json", {
        "selected_positive_arm": arm,
        "parent_decision_sha256": sha256(parent / "report/DECISION.json"),
        "mean_validation": {arm: float(np.mean([row["best_validation_cost"] for row in selected]))},
        "selected": selected,
        "role": "Checkpoint and LR selection complete for every seed before new-policy test evaluation; lambda version never reconsidered.",
    })
    evaluated = parallel(evaluate_one, [(protocol, str(output), row) for row in selected], workers, f"n{n_fit} evaluate")
    evaluated.sort(key=lambda row: row["seed"])
    save_json(output / "evaluations.json", evaluated)
    save_json(output / "COMPLETED.json", {
        "n_candidates": len(results), "n_selected": len(evaluated), "n_baseline_training_runs": 0,
        "arm": arm, "n_fit": n_fit, "wall_seconds_this_invocation": time.perf_counter() - started,
    })
    return output


def report_extension(parent: Path, extension: Path) -> Path:
    parent = parent.resolve()
    extension = extension.resolve()
    _, _, decision, arm, remaining = authorized_parent(parent)
    protocol = ICLRProtocol()
    if not all((extension / f"n{n}" / "COMPLETED.json").exists() for n in remaining):
        raise RuntimeError("All four remaining budget shards must complete before the extension report.")
    original = pd.read_csv(ORIGINAL / "per_run_results.csv", float_precision="round_trip")
    original = original.loc[original.policy == POLICY].copy()
    original["train_seed"] = original.train_seed.astype(int)
    originals = original.set_index(["n_fit", "train_seed"])
    path_csv = ORIGINAL / "test_path_costs.csv"
    archived_paths = pd.read_csv(path_csv, float_precision="round_trip") if path_csv.exists() else None
    if archived_paths is not None and not np.array_equal(archived_paths.test_path.to_numpy(), np.arange(protocol.n_test)):
        raise RuntimeError("Archived test path identifiers do not match the common generated order.")
    rows, per_path_uncertainty, input_hashes = [], [], {}
    input_paths = [parent / "report/DECISION.json", parent / "SELECTION.json", parent / "PROTOCOL.json", parent / "INDEPENDENT_AUDIT.json", ORIGINAL / "per_run_results.csv", ORIGINAL / "data_manifest.json"]
    if path_csv.exists():
        input_paths.append(path_csv)
    for n_fit in protocol.fit_budgets:
        directory = parent if n_fit == 1000 else extension / f"n{n_fit}"
        evaluations = [row for row in load(directory / "evaluations.json") if row["arm"] == arm]
        selection = load(directory / "SELECTION.json")
        selected = {row["seed"]: row for row in selection["selected"] if row["arm"] == arm}
        if len(evaluations) != 5 or set(selected) != set(protocol.train_seeds):
            raise RuntimeError(f"Incomplete selected positive policies at budget {n_fit}.")
        if n_fit != 1000:
            frozen = load(directory / "PROTOCOL.json")
            if frozen["parent_decision_sha256"] != sha256(parent / "report/DECISION.json") or frozen["selected_positive_arm"] != arm:
                raise RuntimeError("An extension shard was run under another decision.")
            if frozen["parent_independent_audit_sha256"] != sha256(parent / "INDEPENDENT_AUDIT.json"):
                raise RuntimeError("Parent independent audit changed after extension freeze.")
            for key, archive in (("original_results_sha256", ORIGINAL / "per_run_results.csv"),
                                 ("original_data_manifest_sha256", ORIGINAL / "data_manifest.json")):
                if sha256(archive) != frozen[key]:
                    raise RuntimeError("Original result/data archive changed after extension freeze.")
            if frozen["original_test_path_costs_sha256"] is not None and (not path_csv.exists() or sha256(path_csv) != frozen["original_test_path_costs_sha256"]):
                raise RuntimeError("Original per-path costs changed after extension freeze.")
            for relative, digest in frozen["source_sha256"].items():
                if sha256(ROOT / relative) != digest:
                    raise RuntimeError(f"Extension source changed: {relative}")
            for relative, digest in frozen["original_selected_checkpoint_sha256"].items():
                if sha256(ROOT / relative) != digest:
                    raise RuntimeError("Original checkpoint changed after protocol freeze.")
        input_paths += [directory / "evaluations.json", directory / "SELECTION.json", directory / "PROTOCOL.json", directory / "COMPLETED.json"]
        for value in evaluations:
            seed = value["seed"]
            if value["path"] != selected[seed]["path"] or value["selected_state_sha256"] != selected[seed]["selected_state_sha256"]:
                raise RuntimeError("Evaluated model differs from the frozen selection.")
            old = originals.loc[(n_fit, seed)]
            checkpoint = original_checkpoint(n_fit, seed)
            old_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if old_payload["selected_iteration"] != old.selected_iteration or old_payload["learning_rate"] != old.selected_learning_rate or old_payload["best_validation_cost"] != old.best_validation_cost:
                raise RuntimeError("Archived baseline checkpoint metadata disagree with the original result table.")
            positive_dir = directory / value["path"]
            individual_evaluation_path = positive_dir / "evaluation.json"
            if load(individual_evaluation_path) != value:
                raise RuntimeError("Aggregate evaluation differs from the selected policy's exact evaluation JSON.")
            positive_costs = np.load(positive_dir / "test_costs.npy", allow_pickle=False)
            if positive_costs.shape != (protocol.n_test,) or not np.isfinite(positive_costs).all():
                raise RuntimeError("Invalid positive-arm test-path cost array.")
            if float(positive_costs.mean()) != value["test"]["mean_path_cost_float64"]:
                raise RuntimeError("Positive-arm saved test mean differs from its per-path array.")
            old_column = f"n{n_fit}_{POLICY}_seed{seed}"
            old_paths = archived_paths[old_column].to_numpy(dtype=np.float64) if archived_paths is not None and old_column in archived_paths else None
            if old_paths is not None and not np.isclose(old_paths.mean(), old.test_mean_cost, rtol=1e-13, atol=1e-9):
                raise RuntimeError("Archived lambda=0 path costs disagree with its original reported mean.")
            if value["test_sha256"] != load(ORIGINAL / "data_manifest.json")["test"]["sha256"]:
                raise RuntimeError("Extension test paths differ from the original baseline test set.")
            baseline_cost = float(old.test_mean_cost)
            positive_cost = float(positive_costs.mean())
            candidate_result_paths = sorted(positive_dir.parent.glob("lr*/result.json"))
            fit_candidates = [load(path) for path in candidate_result_paths]
            if len(fit_candidates) != 3:
                raise RuntimeError("Missing positive LR candidate results.")
            init_path = directory / "initialization" / f"seed{seed}.json"
            init = load(init_path)
            rows.append({
                "n_fit": n_fit, "seed": seed, "arm": arm,
                "baseline_test_cost": baseline_cost, "regularized_test_cost": positive_cost,
                "test_delta": positive_cost - baseline_cost,
                "test_delta_percent": 100.0 * (positive_cost - baseline_cost) / baseline_cost,
                "baseline_validation_cost": float(old.best_validation_cost),
                "regularized_validation_cost": value["best_validation_cost"],
                "validation_delta": value["best_validation_cost"] - float(old.best_validation_cost),
                "baseline_learning_rate": float(old.selected_learning_rate), "regularized_learning_rate": value["learning_rate"],
                "baseline_selected_iteration": int(old.selected_iteration), "regularized_selected_iteration": value["selected_iteration"],
                "baseline_max_action_violation": float(old.max_action_violation), "baseline_max_storage_violation": float(old.max_storage_violation),
                "regularized_max_action_violation": value["test"]["max_action_violation"], "regularized_max_storage_violation": value["test"]["max_storage_violation"],
                "regularized_proposal_distance": value["test"]["proposal_distance"],
                "regularized_active_projection_fraction": value["test"]["active_projection_fraction"],
                "regularized_cost_gradient_norm": value["fit_gradient_diagnostic"]["cost_gradient_norm"],
                "regularized_distance_gradient_norm": value["fit_gradient_diagnostic"]["distance_gradient_norm"],
                "regularized_total_gradient_norm": value["fit_gradient_diagnostic"]["total_gradient_norm"],
                "regularized_selected_fit_seconds": value["fit_time_seconds"],
                "regularized_all_candidates_fit_seconds": sum(row["fit_time_seconds"] for row in fit_candidates),
                "lambda_initialization_seconds_once": init["elapsed_seconds"],
                "lambda_ref": init["lambda_ref"], "n_test_paths": len(positive_costs),
                "baseline_source": "unchanged original archived selected checkpoint and results",
                "regularized_source": "reused N=1000 parent" if n_fit == 1000 else "conditional extension",
                "baseline_checkpoint_sha256": sha256(checkpoint),
                "regularized_checkpoint_sha256": sha256(positive_dir / "selected.pt"),
            })
            conditional = paired_interval(positive_costs - old_paths) if old_paths is not None else None
            per_path_uncertainty.append({
                "n_fit": n_fit, "seed": seed, "arm": arm,
                "archived_baseline_path_costs_available": old_paths is not None,
                "n_test_paths": len(positive_costs),
                "within_seed_paired_path_delta_mean": conditional["mean"] if conditional else None,
                "within_seed_paired_path_delta_sd": conditional["sd"] if conditional else None,
                "within_seed_paired_path_ci95_low": conditional["ci95_low"] if conditional else None,
                "within_seed_paired_path_ci95_high": conditional["ci95_high"] if conditional else None,
            })
            input_paths += [checkpoint, positive_dir / "selected.pt", positive_dir / "test_costs.npy", init_path, individual_evaluation_path, *candidate_result_paths]
    per_seed = pd.DataFrame(rows).sort_values(["n_fit", "seed"])
    summaries, curves = [], []
    for n_fit, group in per_seed.groupby("n_fit", sort=True):
        delta = paired_interval(group.test_delta.to_numpy())
        summaries.append({
            "n_fit": int(n_fit), "arm": arm, "n_seeds": len(group),
            "baseline_test_cost_mean": float(group.baseline_test_cost.mean()), "baseline_test_cost_sd": float(group.baseline_test_cost.std(ddof=1)),
            "regularized_test_cost_mean": float(group.regularized_test_cost.mean()), "regularized_test_cost_sd": float(group.regularized_test_cost.std(ddof=1)),
            "test_delta_mean": delta["mean"], "test_delta_sd": delta["sd"], "test_seed_ci95_low": delta["ci95_low"], "test_seed_ci95_high": delta["ci95_high"],
            "validation_delta_mean": float(group.validation_delta.mean()),
            "test_improved_seed_count": int((group.test_delta < 0).sum()),
            "regularized_max_action_violation": float(group.regularized_max_action_violation.max()),
            "regularized_max_storage_violation": float(group.regularized_max_storage_violation.max()),
        })
        for version, column in (("zero", "baseline_test_cost"), (arm, "regularized_test_cost")):
            curves.append({"n_fit": int(n_fit), "arm": version, "n_seeds": len(group),
                           "test_cost_mean": float(group[column].mean()), "test_cost_sd": float(group[column].std(ddof=1)),
                           "source": "original unchanged archive" if version == "zero" else ("parent ablation" if n_fit == 1000 else "conditional extension")})
    summary, curve_frame = pd.DataFrame(summaries), pd.DataFrame(curves)
    report = extension / "report"
    report.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(report / "extension_per_seed.csv", index=False)
    summary.to_csv(report / "extension_summary.csv", index=False)
    curve_frame.to_csv(report / "cost_curves.csv", index=False)
    pd.DataFrame(per_path_uncertainty).to_csv(report / "within_seed_test_path_uncertainty.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    for version, color, label in (("zero", "#222222", "Original A1, lambda=0"), (arm, "#228833", f"A1, frozen {arm} schedule")):
        group = curve_frame.loc[curve_frame.arm == version]
        axes[0].errorbar(group.n_fit, group.test_cost_mean, yerr=group.test_cost_sd, color=color, marker="o", capsize=3, label=label)
    axes[0].set(xlabel="Fitting trajectories", ylabel="Test operating cost", xscale="log", title="Five-seed mean ±1 SD")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(summary.n_fit, summary.test_delta_mean,
                     yerr=np.vstack((summary.test_delta_mean - summary.test_seed_ci95_low, summary.test_seed_ci95_high - summary.test_delta_mean)),
                     color="#228833", marker="o", capsize=3)
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Fitting trajectories", ylabel="Regularized minus original cost", xscale="log", title="Paired seed means, pointwise 95% t CI")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xticks(protocol.fit_budgets, [str(n) for n in protocol.fit_budgets])
    for suffix in ("pdf", "png"):
        fig.savefig(report / f"cost_curves.{suffix}", dpi=180)
    plt.close(fig)
    improves = int((summary.test_delta_mean < 0).sum())
    note = {
        "frozen_positive_arm": arm, "parent_decision_sha256": sha256(parent / "report/DECISION.json"),
        "selection_rule": "Positive lambda schedule selected from N=1000 validation results before parent test access; held fixed across all remaining budgets.",
        "original_baseline_training_runs_added": 0, "new_positive_candidate_training_runs": 60,
        "new_positive_selected_policies": 20, "reused_parent_positive_policies": 5,
        "budgets_with_lower_mean_test_cost": improves, "n_budgets": len(summary),
        "mean_comparison_by_budget": summary[["n_fit", "test_delta_mean", "test_seed_ci95_low", "test_seed_ci95_high"]].to_dict(orient="records"),
        "limitations": [
            "The schedule was chosen and extension triggered using the parent ablation; this is a conditional follow-up, not a new independent confirmatory experiment.",
            "Nested training pools, common test paths and matched seeds make budgets dependent. Do not count all seed-by-budget-by-path rows as independent replicates.",
            "Pointwise seed t intervals are descriptive and unadjusted for multiple budgets or the earlier selection step. Within-seed path uncertainty is separate.",
            "No claim is made for other policies, datasets, reservoir counts, horizons or regularizer schedules.",
            "Old proposal-distance and gradient histories were not recorded; no baseline values for those diagnostics are invented.",
            "Current concurrent runtime is not directly compared to historical baseline timing as a speedup claim.",
        ],
        "actual_extension_calibration_seconds": float(per_seed.loc[per_seed.n_fit != 1000, "lambda_initialization_seconds_once"].sum()),
        "parent_calibration_seconds_reused": float(per_seed.loc[per_seed.n_fit == 1000, "lambda_initialization_seconds_once"].sum()),
    }
    save_json(report / "DECISION_NOTE.json", note)
    lines = ["# Conditional projection-distance extension", "",
             f"Only the `{arm}` schedule selected by the N=1000 validation results was extended. The original zero-lambda baselines were reused without training. New work: 4 budgets × 5 seeds × 3 LR candidates = 60 candidates, yielding 20 selected positive policies; the 5 selected N=1000 positive policies are reused.", "",
             "| Fitting paths | Original cost mean | Regularized cost mean | Paired difference | Seed 95% t interval | Improved seeds |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.n_fit} | {row.baseline_test_cost_mean:.3f} | {row.regularized_test_cost_mean:.3f} | {row.test_delta_mean:+.3f} | [{row.test_seed_ci95_low:.3f}, {row.test_seed_ci95_high:.3f}] | {row.test_improved_seed_count}/5 |")
    lines += ["", f"The frozen positive schedule has a lower mean test cost at {improves}/{len(summary)} budgets. This is a budget-specific result on one benchmark, not a universal training improvement.", "",
              "The primary differences use original archived zero-lambda means and float64 means of the new float32-simulator path costs. Archived per-path costs permit paired test-path intervals; these are recorded separately in `within_seed_test_path_uncertainty.csv`. The main uncertainty unit is a training seed within a budget. Budgets share nested fitting data and test paths and are not independent confirmations.", "",
              "Original baseline checkpoints, result tables, and test arrays are hashed in PROVENANCE.json. No old gradient or proposal-distance history is reconstructed or imputed. Original baselines and manuscript files remain unchanged. Current and historical run times should not be interpreted as an isolated speed comparison.", "",
              "The cost curve shows mean ±1 SD; the paired panel shows pointwise, unadjusted 95% t intervals over five seed means. Selection and conditional extension are disclosed in DECISION_NOTE.json.", ""]
    (report / "README.md").write_text("\n".join(lines), encoding="utf-8")
    exact_json_inputs = {}
    for path in set(input_paths):
        input_hashes[str(path)] = sha256(path)
        if path.suffix == ".json":
            exact_json_inputs[str(path)] = {"sha256": input_hashes[str(path)], "text": path.read_text(encoding="utf-8")}
    save_json(report / "REPORT_INPUTS.json", {
        "role": "Exact JSON text used by this report, including every LR candidate result and selected individual evaluation. Large arrays/checkpoints and original CSVs remain referenced by immutable-content SHA256 in PROVENANCE.json.",
        "json_inputs": exact_json_inputs,
    })
    save_json(report / "PROVENANCE.json", {
        "source_sha256": sha256(Path(__file__)), "input_sha256": input_hashes,
        "artifact_sha256": {path.name: sha256(path) for path in report.iterdir() if path.is_file() and path.name != "PROVENANCE.json"},
        "checks": {"same_frozen_lambda_arm_all_budgets": True, "zero_lambda_not_retrained": True,
                   "archived_checkpoint_metadata_match_original_results": True,
                   "common_test_dataset_hash_matches_original": True,
                   "new_test_means_match_saved_path_arrays": True,
                   "available_archived_path_means_match_original_result_table": True},
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PARENT.parent / "budget_extension")
    parser.add_argument("--n-fit", type=int, help="Run just one remaining displayed budget; useful for independent sharding.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-only", action="store_true", help="Aggregate completed shards without any training.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.report_only:
        print(report_extension(args.parent, args.output))
        return
    _, _, _, _, remaining = authorized_parent(args.parent.resolve())
    if args.n_fit is not None:
        print(run_budget(args.parent, args.output, args.n_fit, args.workers))
    else:
        for n_fit in remaining:
            print(run_budget(args.parent, args.output, n_fit, args.workers))
        print(report_extension(args.parent, args.output))


if __name__ == "__main__":
    main()
