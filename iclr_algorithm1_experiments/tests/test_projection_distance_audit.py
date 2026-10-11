"""Reject consequential budget, coefficient, and model-selection corruption."""
import copy

import numpy as np
import pandas as pd
import pytest

from iclr_algorithm1_experiments.projection_distance_audit import (
    AuditError, verify_history, verify_norms,
)


def _fixture():
    result = {"arm": "decay", "lambda_ref": 2.0, "gradient_clip": 10.0,
              "selected_iteration": 2, "best_validation_cost": 3.0}
    lam = 2.0 / np.sqrt(np.arange(1, 5))
    history = pd.DataFrame({
        "iteration": [1, 2, 3, 4], "fit_trajectories_seen": [2, 4, 6, 8],
        "lambda_value": lam, "train_cost": [10.0] * 4, "mean_cost": [10.0] * 4,
        "train_objective": 10.0 + 3.0 * lam, "proposal_distance": [3.0] * 4,
        "gradient_norm_preclip": [12.0] * 4, "gradient_clipped": [1] * 4,
        "active_projection_fraction": [0.5] * 4,
        "max_action_violation": [0.0] * 4, "max_storage_violation": [0.0] * 4,
        "validation_cost": [np.nan, 3.0, np.nan, 3.0],
    })
    validation = pd.DataFrame({"iteration": [0, 2, 4], "mean_cost": [5.0, 3.0, 3.0]})
    return result, history, validation


def test_accepts_strict_validation_tie_and_complete_budget():
    verify_history(*_fixture(), iterations=4, batch_size=2, validation_every=2)


def test_rejects_shifted_decay_and_wrong_checkpoint_selection():
    result, history, validation = _fixture()
    history["lambda_value"] = 2.0 / np.sqrt(np.arange(2, 6))
    with pytest.raises(AuditError, match="lambda schedule"):
        verify_history(result, history, validation, 4, 2, 2)
    result, history, validation = _fixture()
    result["selected_iteration"] = 4
    with pytest.raises(AuditError, match="first-tie"):
        verify_history(result, history, validation, 4, 2, 2)


def test_rejects_objective_normalized_over_stages():
    result, history, validation = _fixture()
    history["train_objective"] = history.train_cost + history.lambda_value * history.proposal_distance / 48
    with pytest.raises(AuditError, match="half-summed"):
        verify_history(result, history, validation, 4, 2, 2)


def test_rejects_median_of_ratios_and_missing_fitting_paths():
    info = {"cost_gradient_norms": [2.0, 3.0, 100.0],
            "distance_gradient_norms": [1.0, 10.0, 50.0],
            "median_cost_gradient_norm": 3.0, "median_distance_gradient_norm": 10.0,
            "lambda_ref": 3.0 / (10.0 + 1e-12), "n_paths": 3, "gradient_mode": "full"}
    verify_norms(info, 3)
    wrong = copy.deepcopy(info)
    wrong["lambda_ref"] = np.median(np.array(info["cost_gradient_norms"]) / np.array(info["distance_gradient_norms"]))
    with pytest.raises(AuditError, match="ratio of medians"):
        verify_norms(wrong, 3)
    with pytest.raises(AuditError, match="every fitting path"):
        verify_norms(info, 4)
