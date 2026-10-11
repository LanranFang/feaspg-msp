"""Combine validation-selected learning-rate variants into final pilot tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import policy_comparison


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
BASE = OUTPUTS / "pilot_decision_switching_r12_t48"
ALTERNATIVE = OUTPUTS / "pilot_decision_switching_lr_alt"

TUNED_POLICIES = (
    "projected_affine",
    "projected_polynomial",
    "projected_graph_neural_global",
)


def run() -> None:
    base_results = pd.read_csv(BASE / "main_results.csv")
    alternative_results = pd.read_csv(ALTERNATIVE / "main_results.csv")
    candidate_rows = []
    selected_rows = []
    learning_rates = {
        "projected_affine": (5e-4, 2e-3),
        "projected_polynomial": (1.5e-4, 3e-4),
        "projected_graph_neural_global": (3e-4, 6e-4),
    }
    for policy in TUNED_POLICIES:
        candidates = []
        for label, frame, learning_rate in (
            ("base_lr", base_results, learning_rates[policy][0]),
            ("higher_lr", alternative_results, learning_rates[policy][1]),
        ):
            row = frame.loc[frame.policy == policy].iloc[0].copy()
            candidates.append((label, row, learning_rate))
        best_label, best_row, best_lr = min(
            candidates, key=lambda item: float(item[1].validation_mean_cost)
        )
        for label, row, learning_rate in candidates:
            candidate_rows.append(
                {
                    "policy": policy,
                    "candidate": label,
                    "learning_rate": learning_rate,
                    "validation_mean_cost": row.validation_mean_cost,
                    "test_mean_cost": row.test_mean_cost,
                    "selected_iteration": row.selected_iteration_mean,
                    "selected_by_validation": int(label == best_label),
                }
            )
        selected = best_row.copy()
        selected["source_run"] = best_label
        selected["selected_learning_rate"] = best_lr
        selected_rows.append(selected)

    fixed = base_results.loc[
        base_results.policy.isin(
            ("myopic", "projected_graph_neural", "linearized_empirical_sddp")
        )
    ].copy()
    fixed["source_run"] = "base_lr"
    fixed["selected_learning_rate"] = np.nan
    tuned = pd.concat((fixed, pd.DataFrame(selected_rows)), ignore_index=True)
    tuned = tuned.sort_values("test_mean_cost")
    tuned.to_csv(OUTPUTS / "tuned_main_results.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(
        OUTPUTS / "hyperparameter_selection.csv", index=False
    )

    base_costs = pd.read_csv(BASE / "test_path_costs.csv")
    alternative_costs = pd.read_csv(ALTERNATIVE / "test_path_costs.csv")
    tuned_costs = pd.DataFrame({"trajectory": base_costs.trajectory})
    tuned_costs["myopic_fixed"] = base_costs["myopic_fixed"]
    tuned_costs["projected_graph_neural_seed0"] = base_costs[
        "projected_graph_neural_seed0"
    ]
    tuned_costs["linearized_empirical_sddp_seed0"] = base_costs[
        "linearized_empirical_sddp_seed0"
    ]
    for policy in TUNED_POLICIES:
        tuned_costs[f"{policy}_seed0"] = alternative_costs[f"{policy}_seed0"]
    tuned_costs.to_csv(OUTPUTS / "tuned_test_path_costs.csv", index=False)

    candidate = "projected_graph_neural_global_seed0"
    comparisons = []
    for baseline in tuned_costs.columns:
        if baseline in {"trajectory", candidate}:
            continue
        difference = tuned_costs[candidate] - tuned_costs[baseline]
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        mean = float(difference.mean())
        comparisons.append(
            {
                "candidate_run": candidate,
                "baseline_run": baseline,
                "n_test": len(difference),
                "mean_candidate_minus_baseline": mean,
                "paired_se": standard_error,
                "ci95_lower": mean - 1.96 * standard_error,
                "ci95_upper": mean + 1.96 * standard_error,
                "candidate_win": int(mean + 1.96 * standard_error < 0.0),
            }
        )
    pd.DataFrame(comparisons).to_csv(
        OUTPUTS / "tuned_paired_comparisons.csv", index=False
    )

    figures = OUTPUTS / "figures"
    figures.mkdir(exist_ok=True)
    policy_comparison(tuned, figures / "tuned_policy_comparison.png")
    print(tuned[["policy", "validation_mean_cost", "test_mean_cost"]].to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    run()
