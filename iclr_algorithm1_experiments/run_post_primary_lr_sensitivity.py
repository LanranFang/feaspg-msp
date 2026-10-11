"""Post-primary upper-learning-rate sensitivity for the controlled benchmark.

This script leaves the frozen ``formal_protocol_v1`` outputs untouched.  It
adds one candidate above each registered upper learning-rate boundary at
``N_fit=2500`` while reusing the original fit, validation, and test seeds.
Candidate selection continues to use validation cost only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.data import generate_dataset

from .config import ICLRProtocol
from .data_budget import _rl_run, _shac_run
from .io import write_frame, write_protocol


ADDED_RATES = {
    "projected_shac": 1.0e-3,
    "rl_ppo": 3.0e-3,
    "rl_sac": 3.0e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "post_primary_lr_sensitivity_v1",
    )
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "formal_protocol_v1"
        / "02_data_budget",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(ADDED_RATES),
        default=list(ADDED_RATES),
        help="Policy shards to run; use disjoint output directories in parallel.",
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        help="Training-seed subset; defaults to all registered seeds.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional completed-run CSV whose keys should be skipped.",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Skip expanded selection tables for a partial shard.",
    )
    return parser.parse_args()


def _policy_protocol(base: ICLRProtocol, policy: str) -> ICLRProtocol:
    updates = {
        "fit_budgets": (2500,),
        "base_n_fit": 2500,
    }
    rate = ADDED_RATES[policy]
    if policy == "projected_shac":
        updates["shac_learning_rates"] = (rate,)
    elif policy == "rl_ppo":
        updates["ppo_learning_rates"] = (rate,)
    elif policy == "rl_sac":
        updates["sac_learning_rates"] = (rate,)
    else:  # pragma: no cover - fixed table above
        raise ValueError(policy)
    return replace(base, **updates)


def _records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def _record_key(row: dict) -> tuple[str, int]:
    return str(row["policy"]), int(float(row["train_seed"]))


def _valid_complete_keys(directory: Path, n_test: int) -> set[tuple[str, int]]:
    runs = _records(directory / "added_per_run_results.csv")
    candidates = _records(directory / "added_hyperparameter_candidates.csv")
    histories = _records(directory / "added_training_curves.csv")
    candidate_groups: dict[tuple[str, int], list[dict]] = {}
    for row in candidates:
        candidate_groups.setdefault(_record_key(row), []).append(row)
    history_keys = {_record_key(row) for row in histories}
    run_groups: dict[tuple[str, int], list[dict]] = {}
    for row in runs:
        run_groups.setdefault(_record_key(row), []).append(row)

    valid: set[tuple[str, int]] = set()
    for key, run_rows in run_groups.items():
        policy, seed = key
        candidate_rows = candidate_groups.get(key, [])
        if (
            policy not in ADDED_RATES
            or len(run_rows) != 1
            or len(candidate_rows) != 1
            or key not in history_keys
        ):
            continue
        run_row = run_rows[0]
        candidate_row = candidate_rows[0]
        expected_rate = ADDED_RATES[policy]
        if (
            int(run_row.get("n_fit", -1)) != 2500
            or int(candidate_row.get("n_fit", -1)) != 2500
            or not np.isclose(
                float(run_row.get("selected_learning_rate", np.nan)),
                expected_rate,
            )
            or not np.isclose(
                float(candidate_row.get("learning_rate", np.nan)),
                expected_rate,
            )
        ):
            continue
        costs_path = directory / f"test_costs_{policy}_seed{seed}.npy"
        try:
            costs = np.load(costs_path)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if costs.shape != (n_test,) or not np.isfinite(costs).all():
            continue
        checkpoints = [
            Path(str(run_row.get("checkpoint_path", ""))),
            Path(str(candidate_row.get("checkpoint_path", ""))),
        ]
        if any(not path.is_file() for path in checkpoints):
            continue
        if policy.startswith("rl_") and any(
            not path.with_suffix(".json").is_file() for path in checkpoints
        ):
            continue
        valid.add(key)
    return valid


def _write_or_check_protocol(
    output: Path,
    base: ICLRProtocol,
    selected_policies: tuple[str, ...],
    selected_seeds: tuple[int, ...],
) -> None:
    details = {
        "n_fit": 2500,
        "candidate_selection": "minimum validation cost only",
        "added_rates": {
            policy: ADDED_RATES[policy] for policy in selected_policies
        },
        "selected_policies": list(selected_policies),
        "selected_train_seeds": list(selected_seeds),
        "primary_outputs_untouched": True,
        "same_fit_validation_test_seeds_as_primary": True,
    }
    protocol_path = output / "protocol.json"
    if not protocol_path.exists():
        write_protocol(
            output,
            base,
            "post_primary_upper_learning_rate_sensitivity",
            details,
        )
        return
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    existing_policies = existing.get(
        "selected_policies", list(ADDED_RATES)
    )
    existing_seeds = existing.get(
        "selected_train_seeds", list(base.train_seeds)
    )
    checks = {
        "experiment": existing.get("experiment")
        == "post_primary_upper_learning_rate_sensitivity",
        "n_fit": int(existing.get("n_fit", -1)) == 2500,
        "candidate_selection": existing.get("candidate_selection")
        == details["candidate_selection"],
        "added_rates": existing.get("added_rates") == details["added_rates"],
        "selected_policies": list(existing_policies)
        == details["selected_policies"],
        "selected_train_seeds": [int(seed) for seed in existing_seeds]
        == details["selected_train_seeds"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            f"Existing shard protocol is incompatible in: {failed}"
        )


def _write_expanded_selection(
    output: Path,
    primary_dir: Path,
    added_candidates: pd.DataFrame,
    added_runs: pd.DataFrame,
) -> None:
    primary_candidates = pd.read_csv(primary_dir / "hyperparameter_candidates.csv")
    primary_runs = pd.read_csv(primary_dir / "per_run_results.csv")
    primary_runs["train_seed_numeric"] = pd.to_numeric(
        primary_runs["train_seed"], errors="coerce"
    )
    primary_candidates = primary_candidates[
        (primary_candidates["n_fit"] == 2500)
        & primary_candidates["policy"].isin(ADDED_RATES)
    ].copy()
    added_candidates = added_candidates.copy()
    primary_candidates["candidate_source"] = "registered"
    added_candidates["candidate_source"] = "post_primary_upper_sensitivity"
    expanded = pd.concat([primary_candidates, added_candidates], ignore_index=True)
    expanded["selected_expanded"] = 0

    selections: list[dict] = []
    for (policy, train_seed), indices in expanded.groupby(
        ["policy", "train_seed"], sort=True
    ).groups.items():
        selected_index = expanded.loc[list(indices), "best_validation_cost"].idxmin()
        expanded.loc[selected_index, "selected_expanded"] = 1
        selected = expanded.loc[selected_index]
        if selected["candidate_source"] == "registered":
            run = primary_runs[
                (primary_runs["n_fit"] == 2500)
                & (primary_runs["policy"] == policy)
                & (primary_runs["train_seed_numeric"] == int(train_seed))
            ].iloc[0]
        else:
            run = added_runs[
                (added_runs["policy"] == policy)
                & (added_runs["train_seed"] == train_seed)
            ].iloc[0]
        selections.append(
            {
                "policy": policy,
                "train_seed": int(train_seed),
                "selected_learning_rate_expanded": selected["learning_rate"],
                "candidate_source": selected["candidate_source"],
                "best_validation_cost": selected["best_validation_cost"],
                "test_mean_cost": run["test_mean_cost"],
            }
        )

    selection_frame = pd.DataFrame(selections)
    summary = (
        selection_frame.groupby("policy", sort=True)
        .agg(
            test_cost_mean_over_train_seeds=("test_mean_cost", "mean"),
            test_cost_train_seed_sd=("test_mean_cost", "std"),
            post_primary_candidate_selected_count=(
                "candidate_source",
                lambda values: int(
                    sum(value == "post_primary_upper_sensitivity" for value in values)
                ),
            ),
        )
        .reset_index()
    )
    write_frame(expanded, output / "expanded_hyperparameter_candidates.csv")
    write_frame(selection_frame, output / "expanded_selection_by_seed.csv")
    write_frame(summary, output / "expanded_selection_summary.csv")


def run() -> None:
    args = parse_args()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    base = ICLRProtocol(device=args.device, rl_device=args.device)
    selected_policies = tuple(dict.fromkeys(args.policies))
    selected_seeds = (
        tuple(base.train_seeds)
        if args.train_seeds is None
        else tuple(dict.fromkeys(args.train_seeds))
    )
    if not set(selected_seeds).issubset(base.train_seeds):
        raise ValueError(
            f"Training seeds must be drawn from {tuple(base.train_seeds)}"
        )
    hydro = base.hydro_config()
    fit = generate_dataset("fit_n2500", 2500, base.budget_seeds.fit, hydro)
    validation = generate_dataset(
        "validation", base.n_validation, base.budget_seeds.validation, hydro
    )
    test = generate_dataset("test", base.n_test, base.budget_seeds.test, hydro)
    _write_or_check_protocol(output, base, selected_policies, selected_seeds)

    all_runs = _records(output / "added_per_run_results.csv")
    all_candidates = _records(output / "added_hyperparameter_candidates.csv")
    all_histories = _records(output / "added_training_curves.csv")
    local_complete = _valid_complete_keys(output, base.n_test)
    runs = [row for row in all_runs if _record_key(row) in local_complete]
    candidates = [
        row for row in all_candidates if _record_key(row) in local_complete
    ]
    histories = [
        row for row in all_histories if _record_key(row) in local_complete
    ]
    complete = set(local_complete)
    if args.resume_from is not None:
        resume_directory = args.resume_from.resolve().parent
        complete.update(_valid_complete_keys(resume_directory, base.n_test))

    for policy in selected_policies:
        protocol = _policy_protocol(base, policy)
        for train_seed in selected_seeds:
            key = (policy, train_seed)
            if key in complete:
                continue
            print(
                f"post-primary LR sensitivity: {policy}, seed={train_seed}, "
                f"lr={ADDED_RATES[policy]:g}",
                flush=True,
            )
            checkpoint_dir = output / "checkpoints" / "n2500"
            if policy == "projected_shac":
                result = _shac_run(
                    fit,
                    validation,
                    test,
                    hydro,
                    protocol,
                    train_seed,
                    checkpoint_dir,
                )
            else:
                result = _rl_run(
                    policy.removeprefix("rl_"),
                    fit,
                    validation,
                    test,
                    hydro,
                    protocol,
                    train_seed,
                    checkpoint_dir,
                )
            row, path_costs, new_candidates, new_histories = result
            row["candidate_role"] = "post_primary_upper_sensitivity"
            for candidate in new_candidates:
                candidate["candidate_role"] = "post_primary_upper_sensitivity"
                candidate["candidate_index"] = 3
            runs.append(row)
            candidates.extend(new_candidates)
            histories.extend(new_histories)
            np.save(
                output / f"test_costs_{policy}_seed{train_seed}.npy",
                np.asarray(path_costs, dtype=np.float64),
            )
            write_frame(pd.DataFrame(runs), output / "added_per_run_results.csv")
            write_frame(
                pd.DataFrame(candidates), output / "added_hyperparameter_candidates.csv"
            )
            write_frame(pd.DataFrame(histories), output / "added_training_curves.csv")
            complete.add(key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.no_finalize:
        expected = {
            (policy, int(seed))
            for policy in ADDED_RATES
            for seed in base.train_seeds
        }
        local_complete = {
            (str(row["policy"]), int(row["train_seed"])) for row in runs
        }
        missing = sorted(expected - local_complete)
        if missing:
            raise RuntimeError(
                "Cannot finalize an incomplete sensitivity output; "
                f"missing keys: {missing}"
            )
        _write_expanded_selection(
            output,
            args.primary_dir.resolve(),
            pd.DataFrame(candidates),
            pd.DataFrame(runs),
        )
        print(f"Completed post-primary LR sensitivity at {output}", flush=True)
    else:
        print(f"Completed post-primary LR sensitivity shard at {output}", flush=True)


if __name__ == "__main__":
    run()
