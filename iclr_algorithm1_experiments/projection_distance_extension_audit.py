"""Independently audit every conditional budget shard and its paired report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .projection_distance_audit import ORIGINAL, audit, close, file_hash, require


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    width = float(student_t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return mean, sd, mean - width, mean + width


def audit_extension(parent, output, replay=False, reuse_shard_audits=False):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    parent_protocol = read(parent / "PROTOCOL.json")
    protocol = parent_protocol["protocol"]
    budgets = protocol["fit_budgets"]
    seeds = protocol["train_seeds"]
    inherited = read(parent / "SELECTION.json")["selected_positive_arm"]
    require(budgets == [100, 250, 500, 1000, 2500] and seeds == [0, 1, 2, 3, 4], "Original displayed grid changed")
    shards = []
    for n_fit in budgets:
        if n_fit == 1000:
            continue
        directory = output / f"n{n_fit}"
        if reuse_shard_audits:
            result = read(directory / "INDEPENDENT_AUDIT.json")
            require(result["status"] == "passed" and result["protocol_sha256"] == file_hash(directory / "PROTOCOL.json"),
                    "Saved shard audit is missing, failed, or refers to another protocol")
            if replay:
                require(result["independently_replayed_selected_evaluations"] == 5,
                        "Saved shard audit omitted requested evaluation replay")
        else:
            result = audit(directory, replay=replay)
        require(result["n_candidates"] == 15 and result["n_selected"] == 5, "Extension shard is incomplete")
        if not reuse_shard_audits:
            write(directory / "INDEPENDENT_AUDIT.json", result)
        shards.append(result)
    report = output / "report"
    provenance = read(report / "PROVENANCE.json")
    for name, digest in provenance["input_sha256"].items():
        require(file_hash(Path(name)) == digest, f"Extension report input changed: {name}")
    for name, digest in provenance["artifact_sha256"].items():
        require(file_hash(report / name) == digest, f"Extension report artifact changed: {name}")
    per_seed = pd.read_csv(report / "extension_per_seed.csv", float_precision="round_trip")
    summary = pd.read_csv(report / "extension_summary.csv", float_precision="round_trip").set_index("n_fit")
    paths = pd.read_csv(report / "within_seed_test_path_uncertainty.csv", float_precision="round_trip")
    curves = pd.read_csv(report / "cost_curves.csv", float_precision="round_trip")
    require(set(zip(per_seed.n_fit, per_seed.seed)) == {(n, seed) for n in budgets for seed in seeds}
            and len(per_seed) == 25, "Budget-by-seed report omitted or duplicated a result")
    require(set(per_seed.arm) == {inherited}, "Report changed the inherited lambda arm")
    require(set(summary.index) == set(budgets) and len(summary) == 5, "Summary omitted or duplicated a budget")
    require(set(zip(paths.n_fit, paths.seed)) == {(n, seed) for n in budgets for seed in seeds}
            and len(paths) == 25, "Path uncertainty report omitted or duplicated a result")
    require(set(zip(curves.n_fit, curves.arm)) == {(n, arm) for n in budgets for arm in ("zero", inherited)}
            and len(curves) == 10, "Cost curves omitted a budget/arm")
    original = pd.read_csv(ORIGINAL / "per_run_results.csv", float_precision="round_trip")
    original = original.loc[original.policy == parent_protocol["policy"]].copy()
    original["train_seed"] = original.train_seed.astype(int)
    original = original.set_index(["n_fit", "train_seed"])
    archive = pd.read_csv(ORIGINAL / "test_path_costs.csv", float_precision="round_trip")
    require(np.array_equal(archive.test_path, np.arange(protocol["n_test"])), "Archived test path ordering changed")
    for n_fit in budgets:
        directory = parent if n_fit == 1000 else output / f"n{n_fit}"
        evaluations = {item["seed"]: item for item in read(directory / "evaluations.json") if item["arm"] == inherited}
        differences, old_means, new_means, validation_differences = [], [], [], []
        for seed in seeds:
            value = evaluations[seed]
            new_paths = np.load(directory / value["path"] / "test_costs.npy")
            old_paths = archive[f"n{n_fit}_{parent_protocol['policy']}_seed{seed}"].to_numpy()
            archived_row = original.loc[(n_fit, seed), :]
            close(old_paths.mean(), archived_row.test_mean_cost, "Original saved path mean differs from baseline table")
            old_mean, new_mean = float(archived_row.test_mean_cost), float(new_paths.mean())
            difference = new_mean - old_mean
            val_difference = value["best_validation_cost"] - float(archived_row.best_validation_cost)
            row = per_seed.loc[(per_seed.n_fit == n_fit) & (per_seed.seed == seed)].iloc[0]
            for field, expected in (("baseline_test_cost", old_mean), ("regularized_test_cost", new_mean),
                                    ("test_delta", difference), ("validation_delta", val_difference),
                                    ("test_delta_percent", 100 * difference / old_mean)):
                close(row[field], expected, f"Per-seed report differs: {n_fit}/{seed}/{field}")
            conditional = paths.loc[(paths.n_fit == n_fit) & (paths.seed == seed)].iloc[0]
            require(bool(conditional.archived_baseline_path_costs_available), "Available archived path costs omitted")
            for field, expected in zip(("within_seed_paired_path_delta_mean", "within_seed_paired_path_delta_sd",
                                        "within_seed_paired_path_ci95_low", "within_seed_paired_path_ci95_high"),
                                       interval(new_paths - old_paths)):
                close(conditional[field], expected, f"Within-seed path interval differs: {n_fit}/{seed}/{field}")
            differences.append(difference)
            old_means.append(old_mean)
            new_means.append(new_mean)
            validation_differences.append(val_difference)
        row = summary.loc[n_fit]
        require(row.n_seeds == 5, "Seed CI used wrong independent replicate count")
        for field, expected in zip(("test_delta_mean", "test_delta_sd", "test_seed_ci95_low", "test_seed_ci95_high"), interval(differences)):
            close(row[field], expected, f"Across-seed interval differs: {n_fit}/{field}")
        close(row.validation_delta_mean, np.mean(validation_differences), "Validation seed aggregation differs")
        require(row.test_improved_seed_count == sum(value < 0 for value in differences), "Improved seed count differs")
        for arm, means in (("zero", old_means), (inherited, new_means)):
            curve = curves.loc[(curves.n_fit == n_fit) & (curves.arm == arm)].iloc[0]
            close(curve.test_cost_mean, np.mean(means), "Cost curve mean differs")
            close(curve.test_cost_sd, np.std(means, ddof=1), "Cost curve SD differs")
    decision = read(report / "DECISION_NOTE.json")
    require(decision["new_positive_candidate_training_runs"] == sum(row["n_candidates"] for row in shards) == 60,
            "New training count differs")
    require(decision["original_baseline_training_runs_added"] == 0 and decision["reused_parent_positive_policies"] == 5,
            "Baseline/parent reuse accounting differs")
    require(decision["budgets_with_lower_mean_test_cost"] == int((summary.test_delta_mean < 0).sum()),
            "Aggregate improvement count differs")
    return {"status": "passed", "n_new_candidates": 60, "n_new_selected": 20,
            "n_reused_parent_selected": 5, "n_budgets": 5,
            "new_selected_evaluations_independently_replayed": sum(row["independently_replayed_selected_evaluations"] for row in shards),
            "paired_seed_intervals_independently_recomputed": 5,
            "conditional_test_path_intervals_independently_recomputed": 25,
            "auditor_sha256": file_hash(__file__), "shard_audits": shards}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--reuse-shard-audits", action="store_true", help="Reuse completed shard integrity/replay checks; only audit the final aggregate report.")
    args = parser.parse_args()
    try:
        result = audit_extension(args.parent, args.output, args.replay, args.reuse_shard_audits)
    except Exception as error:
        write(args.output / "INDEPENDENT_AUDIT.json", {"status": "failed", "error": str(error)})
        raise
    write(args.output / "INDEPENDENT_AUDIT.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
