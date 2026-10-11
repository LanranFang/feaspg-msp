"""Combine the nested-pool seed-0 and seed-1/2 hydro benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import policy_comparison, training_curves
from .run_experiment import summarize


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SOURCES = (
    OUTPUTS / "data_scaling_n1000_seeds_0",
    OUTPUTS / "data_scaling_n1000_seeds_1-2",
)
DESTINATION = OUTPUTS / "data_scaling_n1000_multiseed"
CANDIDATES = (
    "projected_mlp_pathwise",
    "projected_graph_neural_global",
)
BASELINES = (
    "projected_mlp_pathwise",
    "projected_graph_neural_global",
    "rl_ppo",
    "rl_sac",
    "rl_td3",
    "myopic",
)


def _require_sources() -> None:
    for source in SOURCES:
        for filename in (
            "per_run_results.csv",
            "training_curves.csv",
            "test_path_costs.csv",
        ):
            path = source / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing multiseed source: {path}")


def _combine_rows() -> pd.DataFrame:
    pieces = []
    for source_index, source in enumerate(SOURCES):
        data = pd.read_csv(source / "per_run_results.csv")
        if source_index:
            data = data[data["policy"] != "myopic"]
        data = data.copy()
        data["source_directory"] = source.name
        pieces.append(data)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _combine_histories() -> pd.DataFrame:
    pieces = []
    for source in SOURCES:
        data = pd.read_csv(source / "training_curves.csv")
        data = data.copy()
        data["source_directory"] = source.name
        pieces.append(data)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _combine_hyperparameters() -> pd.DataFrame:
    pieces = []
    for source in SOURCES:
        path = source / "rl_hyperparameter_selection.csv"
        if path.exists() and path.stat().st_size:
            data = pd.read_csv(path)
            data = data.copy()
            data["source_directory"] = source.name
            pieces.append(data)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True, sort=False)


def _combine_test_costs() -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for source in SOURCES:
        data = pd.read_csv(source / "test_path_costs.csv")
        if combined is None:
            combined = data
            continue
        if not np.array_equal(combined["trajectory"], data["trajectory"]):
            raise ValueError(f"Test trajectories do not align for {source}.")
        duplicate = set(combined.columns) & set(data.columns) - {"trajectory"}
        for column in duplicate:
            if not np.array_equal(combined[column], data[column]):
                raise ValueError(f"Duplicate test column differs: {column}")
        new_columns = [
            column
            for column in data.columns
            if column != "trajectory" and column not in duplicate
        ]
        combined = pd.concat(
            [combined, data[new_columns].reset_index(drop=True)], axis=1
        )
    if combined is None:
        raise RuntimeError("No test costs were loaded.")
    return combined


def _column(policy: str, seed: int | None) -> str:
    if policy == "myopic":
        return "myopic_fixed"
    if seed is None:
        raise ValueError("A train seed is required for a fitted policy.")
    return f"{policy}_seed{seed}"


def _paired_comparisons(costs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for seed in (0, 1, 2):
        for candidate in CANDIDATES:
            candidate_cost = costs[_column(candidate, seed)].to_numpy()
            for baseline in BASELINES:
                if baseline == candidate:
                    continue
                baseline_seed = None if baseline == "myopic" else seed
                difference = candidate_cost - costs[
                    _column(baseline, baseline_seed)
                ].to_numpy()
                path_se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
                mean = float(difference.mean())
                per_seed_rows.append(
                    {
                        "candidate_policy": candidate,
                        "baseline_policy": baseline,
                        "train_seed": seed,
                        "n_test_paths": len(difference),
                        "mean_candidate_minus_baseline": mean,
                        "paired_path_se": path_se,
                        "paired_path_ci95_lower": mean - 1.96 * path_se,
                        "paired_path_ci95_upper": mean + 1.96 * path_se,
                        "candidate_win_on_this_seed": int(
                            mean + 1.96 * path_se < 0.0
                        ),
                    }
                )
    per_seed = pd.DataFrame(per_seed_rows)
    grouped_rows = []
    for (candidate, baseline), group in per_seed.groupby(
        ["candidate_policy", "baseline_policy"], sort=False
    ):
        differences = group["mean_candidate_minus_baseline"].to_numpy()
        grouped_rows.append(
            {
                "candidate_policy": candidate,
                "baseline_policy": baseline,
                "n_train_seeds": len(differences),
                "mean_difference_across_seeds": float(differences.mean()),
                "std_difference_across_seeds": float(differences.std(ddof=1)),
                "se_difference_across_seeds": float(
                    differences.std(ddof=1) / np.sqrt(len(differences))
                ),
                "min_difference_across_seeds": float(differences.min()),
                "max_difference_across_seeds": float(differences.max()),
                "candidate_wins_all_seeds": int(np.all(differences < 0.0)),
            }
        )
    return per_seed, pd.DataFrame(grouped_rows)


def _multiseed_plot(rows: pd.DataFrame, path: Path) -> None:
    fitted = rows[rows["policy"] != "myopic"].copy()
    order = (
        fitted.groupby("policy")["test_mean_cost"].mean().sort_values().index
    )
    labels = {
        "projected_graph_neural_global": "Projected graph neural",
        "projected_mlp_pathwise": "Projected pathwise MLP",
        "rl_sac": "SAC",
        "rl_td3": "TD3",
        "rl_ppo": "PPO",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for position, policy in enumerate(order):
        values = fitted.loc[fitted["policy"] == policy, "test_mean_cost"].to_numpy()
        ax.scatter(
            np.full(len(values), position),
            values,
            color="#4a78a8",
            alpha=0.75,
            zorder=2,
        )
        ax.errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            color="#b13f3f",
            capsize=5,
            linewidth=2,
            zorder=3,
        )
    ax.set_xticks(range(len(order)), [labels.get(policy, policy) for policy in order])
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylabel("Mean test cost (lower is better)")
    ax.set_title("Three training seeds: points and mean +/- one SD")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    _require_sources()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    figures = DESTINATION / "figures"
    figures.mkdir(exist_ok=True)
    rows = _combine_rows()
    history = _combine_histories()
    hyperparameters = _combine_hyperparameters()
    costs = _combine_test_costs()
    summary = summarize(rows)
    per_seed_paired, across_seed_paired = _paired_comparisons(costs)

    rows.to_csv(DESTINATION / "per_run_results.csv", index=False)
    history.to_csv(DESTINATION / "training_curves.csv", index=False)
    hyperparameters.to_csv(
        DESTINATION / "rl_hyperparameter_selection.csv", index=False
    )
    costs.to_csv(DESTINATION / "test_path_costs.csv", index=False)
    summary.to_csv(DESTINATION / "main_results.csv", index=False)
    per_seed_paired.to_csv(
        DESTINATION / "paired_test_comparisons_by_seed.csv", index=False
    )
    across_seed_paired.to_csv(
        DESTINATION / "paired_test_comparisons_across_seeds.csv", index=False
    )
    (DESTINATION / "sources.json").write_text(
        json.dumps({"sources": [str(source) for source in SOURCES]}, indent=2),
        encoding="utf-8",
    )
    training_curves(history, figures / "training_curves.png")
    policy_comparison(summary, figures / "policy_comparison.png")
    _multiseed_plot(rows, figures / "multiseed_comparison.png")
    print(f"Wrote three-seed comparison to {DESTINATION}")
    print(
        summary[
            [
                "policy",
                "n_runs",
                "test_mean_cost",
                "test_std_across_runs",
                "fit_wall_time_seconds_mean",
                "max_action_violation",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
