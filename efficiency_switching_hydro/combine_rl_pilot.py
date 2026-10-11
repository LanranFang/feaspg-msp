"""Combine validation-selected seed-0 direct, RL, and SDDP pilot outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import policy_comparison, training_curves
from .run_experiment import paired_comparisons, summarize


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DESTINATION = OUTPUTS / "rl_pilot_combined_seed0"

SOURCES = {
    "direct_default": OUTPUTS / "pilot_decision_switching_r12_t48",
    "direct_tuned": OUTPUTS / "pilot_decision_switching_lr_alt",
    "mlp_ppo": OUTPUTS / "rl_pilot_step1_mlp_ppo",
    "sac": OUTPUTS / "rl_pilot_step2_sac",
    "td3": OUTPUTS / "rl_pilot_step3_td3",
}

SELECTED = {
    "direct_tuned": (
        "projected_affine",
        "projected_polynomial",
        "projected_graph_neural_global",
    ),
    "mlp_ppo": ("myopic", "projected_mlp_pathwise", "rl_ppo"),
    "sac": ("rl_sac",),
    "td3": ("rl_td3",),
    "direct_default": ("linearized_empirical_sddp",),
}

TEST_COLUMNS = {
    "direct_tuned": (
        "projected_affine_seed0",
        "projected_polynomial_seed0",
        "projected_graph_neural_global_seed0",
    ),
    "mlp_ppo": (
        "myopic_fixed",
        "projected_mlp_pathwise_seed0",
        "rl_ppo_seed0",
    ),
    "sac": ("rl_sac_seed0",),
    "td3": ("rl_td3_seed0",),
    "direct_default": ("linearized_empirical_sddp_seed0",),
}

DIRECT_LEARNING_RATES = {
    "projected_affine": (5e-4, 2e-3),
    "projected_polynomial": (1.5e-4, 3e-4),
    "projected_graph_neural_global": (3e-4, 6e-4),
}


def _require_sources() -> None:
    for name, directory in SOURCES.items():
        for filename in ("per_run_results.csv", "test_path_costs.csv"):
            path = directory / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} source: {path}")


def _selected_rows() -> pd.DataFrame:
    pieces = []
    for source, policies in SELECTED.items():
        data = pd.read_csv(SOURCES[source] / "per_run_results.csv")
        selected = data[data["policy"].isin(policies)].copy()
        missing = set(policies) - set(selected["policy"])
        if missing:
            raise ValueError(f"Missing policies in {source}: {sorted(missing)}")
        pieces.append(selected)
    rows = pd.concat(pieces, ignore_index=True, sort=False)
    rows["training_num_parameters"] = rows["training_num_parameters"].fillna(
        rows["num_parameters"]
    )
    rows["fit_trajectories"] = rows["fit_trajectories"].fillna(
        rows["fit_function_evaluations"]
    )
    rows["fit_environment_steps"] = rows["fit_environment_steps"].fillna(
        rows["fit_trajectories"] * 48
    )
    rows["gradient_updates"] = rows["gradient_updates"].fillna(
        rows["completed_iterations"]
    )

    default = pd.read_csv(SOURCES["direct_default"] / "per_run_results.csv")
    tuned = pd.read_csv(SOURCES["direct_tuned"] / "per_run_results.csv")
    for policy, rates in DIRECT_LEARNING_RATES.items():
        mask = rows["policy"] == policy
        selected_rate = rates[1]
        rows.loc[mask, "selected_learning_rate"] = selected_rate
        rows.loc[mask, "n_hyperparameter_candidates"] = 2
        rows.loc[mask, "tuning_environment_steps"] = 2 * 1000 * 48
        rows.loc[mask, "tuning_gradient_updates"] = 2 * 500
        rows.loc[mask, "tuning_fit_trajectory_uses"] = 2 * 1000
        elapsed = float(
            default.loc[default["policy"] == policy, "fit_wall_time_seconds"].iloc[0]
            + tuned.loc[tuned["policy"] == policy, "fit_wall_time_seconds"].iloc[0]
        )
        rows.loc[mask, "fit_wall_time_seconds"] = elapsed
    rows.loc[rows["policy"] == "linearized_empirical_sddp", "gradient_updates"] = np.nan
    return rows


def _test_costs() -> dict[str, np.ndarray]:
    costs: dict[str, np.ndarray] = {}
    reference_trajectory = None
    for source, columns in TEST_COLUMNS.items():
        data = pd.read_csv(SOURCES[source] / "test_path_costs.csv")
        trajectory = data["trajectory"].to_numpy()
        if reference_trajectory is None:
            reference_trajectory = trajectory
        elif not np.array_equal(reference_trajectory, trajectory):
            raise ValueError(f"Test trajectories do not align for {source}.")
        for column in columns:
            costs[column] = data[column].to_numpy(dtype=np.float64)
    return costs


def _histories() -> pd.DataFrame:
    pieces = []
    allowed = set().union(*SELECTED.values())
    for source_name, source in SOURCES.items():
        path = source / "training_curves.csv"
        if path.exists() and path.stat().st_size:
            data = pd.read_csv(path)
            if source_name in {"direct_default", "direct_tuned"}:
                data = data[data["policy"].isin(allowed)].copy()
                rate_index = int(source_name == "direct_tuned")
                for policy, rates in DIRECT_LEARNING_RATES.items():
                    mask = data["policy"] == policy
                    data.loc[mask, "policy"] = (
                        f"{policy}_lr{rates[rate_index]:.8g}"
                    )
            data["source_directory"] = source.name
            pieces.append(data)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _hyperparameters(rows: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for source in ("mlp_ppo", "sac", "td3"):
        path = SOURCES[source] / "rl_hyperparameter_selection.csv"
        pieces.append(pd.read_csv(path))
    for policy, rates in DIRECT_LEARNING_RATES.items():
        for index, rate in enumerate(rates):
            source = "direct_tuned" if index == 1 else "direct_default"
            source_rows = pd.read_csv(SOURCES[source] / "per_run_results.csv")
            result = source_rows[source_rows["policy"] == policy].iloc[0]
            pieces.append(
                pd.DataFrame(
                    [
                        {
                            "profile": "decision_switching",
                            "algorithm": "pathwise_adam",
                            "policy": policy,
                            "train_seed": 0,
                            "candidate_index": index,
                            "learning_rate": rate,
                            "best_validation_cost": result["best_validation_cost"],
                            "selected": int(index == 1),
                            "selected_iteration": result["selected_iteration"],
                            "environment_steps": 1000 * 48,
                            "gradient_updates": 500,
                            "fit_wall_time_seconds": result[
                                "fit_wall_time_seconds"
                            ],
                            "checkpoint_path": result["checkpoint_path"],
                        }
                    ]
                )
            )
    return pd.concat(pieces, ignore_index=True, sort=False)


def main() -> None:
    _require_sources()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    figures = DESTINATION / "figures"
    figures.mkdir(exist_ok=True)
    rows = _selected_rows()
    costs = _test_costs()
    history = _histories()
    hyperparameters = _hyperparameters(rows)
    summary = summarize(rows)
    paired = paired_comparisons(costs)

    rows.to_csv(DESTINATION / "per_run_results.csv", index=False)
    summary.to_csv(DESTINATION / "main_results.csv", index=False)
    history.to_csv(DESTINATION / "training_curves.csv", index=False)
    hyperparameters.to_csv(
        DESTINATION / "hyperparameter_selection.csv", index=False
    )
    pd.DataFrame(
        {"trajectory": np.arange(len(next(iter(costs.values())))), **costs}
    ).to_csv(DESTINATION / "test_path_costs.csv", index=False)
    paired.to_csv(DESTINATION / "paired_test_comparisons.csv", index=False)
    training_curves(history, figures / "training_curves.png")
    policy_comparison(summary, figures / "policy_comparison.png")
    print(f"Wrote combined RL pilot to {DESTINATION}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
