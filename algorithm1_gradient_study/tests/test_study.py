from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.simulator import simulate

from algorithm1_gradient_study.config import StudyProtocol
from algorithm1_gradient_study.gradients import (
    finite_difference_rows,
    objective_gradient,
)
from algorithm1_gradient_study.models import (
    build_policy,
    copy_trainable_parameters,
    initialize_channel_policy,
    permute_config,
    permute_dataset,
    perturb_parameters,
)


class GradientStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.protocol = StudyProtocol.smoke_protocol()
        self.config = replace(
            ExperimentConfig.make("decision_switching", 4, 6, 2),
            ramp_fraction=self.protocol.study_ramp_fraction,
        )
        self.data = generate_dataset("fit", 3, 4101, self.config)

    def test_gradient_modes_preserve_forward_actions_and_feasibility(self) -> None:
        policy = build_policy("mlp", self.config, self.protocol).to(torch.float64)
        perturb_parameters(policy, 111)
        traces = {
            mode: simulate(
                policy,
                self.data,
                self.config,
                dtype=torch.float64,
                gradient_mode=mode,
            )
            for mode in ("full", "no_proposal_state", "no_moving_recourse")
        }
        for mode, trace in traces.items():
            self.assertTrue(
                torch.equal(trace.actions, traces["full"].actions), msg=mode
            )
            self.assertTrue(
                torch.equal(trace.total_cost, traces["full"].total_cost), msg=mode
            )
            self.assertLessEqual(
                float(trace.max_action_violation.detach()), 1.0e-8
            )
            self.assertLessEqual(
                float(trace.max_storage_violation.detach()), 1.0e-8
            )

    def test_both_ablation_gradients_differ_from_full(self) -> None:
        torch.manual_seed(0)
        policy = build_policy("mlp", self.config, self.protocol).to(torch.float64)
        initialize_channel_policy(policy, 222)
        _, full = objective_gradient(policy, self.data, self.config, "full")
        for mode in ("no_proposal_state", "no_moving_recourse"):
            _, ablated = objective_gradient(policy, self.data, self.config, mode)
            self.assertTrue(torch.isfinite(ablated).all())
            relative = torch.linalg.vector_norm(ablated - full) / torch.clamp(
                torch.linalg.vector_norm(full), min=1.0e-12
            )
            self.assertGreater(float(relative), 1.0e-7, msg=mode)

    def test_full_gradient_matches_a_finite_difference_scale(self) -> None:
        torch.manual_seed(0)
        policy = build_policy("mlp", self.config, self.protocol).to(torch.float64)
        initialize_channel_policy(policy, 333)
        rows = finite_difference_rows(
            policy,
            self.data,
            self.config,
            (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4),
            direction_seed=444,
        )
        full_errors = [
            float(row["relative_error"])
            for row in rows
            if row["gradient_mode"] == "full"
        ]
        self.assertLess(min(full_errors), 5.0e-3)

    def test_shared_capacity_projection_uses_kkt_backward(self) -> None:
        config = replace(self.config, ramp_fraction=0.70)
        data = generate_dataset("fit", 3, 4101, config)
        torch.manual_seed(0)
        policy = build_policy("mlp", config, self.protocol).to(torch.float64)
        initialize_channel_policy(policy, 333, normalized_bias=0.80)
        rows = finite_difference_rows(
            policy,
            data,
            config,
            (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4),
            direction_seed=444,
        )
        full_errors = [
            float(row["relative_error"])
            for row in rows
            if row["gradient_mode"] == "full"
        ]
        self.assertLess(min(full_errors), 5.0e-3)

    def test_graph_policy_is_reservoir_permutation_equivariant(self) -> None:
        policy = build_policy("graph", self.config, self.protocol).to(torch.float64)
        perturb_parameters(policy, 555)
        permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
        inverse = np.argsort(permutation)
        target_config = permute_config(self.config, permutation)
        target_data = permute_dataset(self.data, permutation)
        target = build_policy("graph", target_config, self.protocol).to(torch.float64)
        copy_trainable_parameters(policy, target)
        with torch.no_grad():
            source_trace = simulate(
                policy, self.data, self.config, dtype=torch.float64
            )
            target_trace = simulate(
                target, target_data, target_config, dtype=torch.float64
            )
        error = (
            target_trace.actions[..., inverse] - source_trace.actions
        ).abs().max()
        cost_error = (
            target_trace.total_cost - source_trace.total_cost
        ).abs().max()
        self.assertLess(float(error), 1.0e-6)
        self.assertLess(float(cost_error), 1.0e-6)

    def test_formal_protocol_has_five_seeds_and_reserved_split_interfaces(self) -> None:
        protocol = StudyProtocol()
        self.assertEqual(len(protocol.train_seeds), 5)
        data = protocol.data_config()
        seeds = {
            data.fit_seed,
            data.validation_seed,
            data.test_seed,
            data.calibration_seed,
            data.evaluation_seed,
        }
        self.assertEqual(len(seeds), 5)

    def test_unknown_gradient_mode_is_rejected(self) -> None:
        policy = build_policy("mlp", self.config, self.protocol)
        with self.assertRaises(ValueError):
            simulate(
                policy,
                self.data,
                self.config,
                gradient_mode="not_a_gradient_mode",
            )


if __name__ == "__main__":
    unittest.main()
