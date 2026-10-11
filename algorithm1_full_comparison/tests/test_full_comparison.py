from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from algorithm1_full_comparison.cfa import make_lookahead_policy
from algorithm1_full_comparison.config import FullComparisonProtocol
from algorithm1_full_comparison.evaluation import all_pairwise_ci
from algorithm1_full_comparison.run_comparison import _build_direct_policy
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.policies import MyopicPolicy
from efficiency_switching_hydro.projection import action_bounds
from efficiency_switching_hydro.simulator import simulate


class FullComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.protocol = FullComparisonProtocol.smoke_protocol()
        cls.hydro = cls.protocol.experiment_config()
        cls.dataset = generate_dataset("unit", 3, 9917, cls.hydro)

    def assert_trace_feasible(self, trace) -> None:
        lower, upper = action_bounds(
            trace.storage_before_action,
            trace.previous_release_before_action,
            self.hydro,
        )
        tolerance = 1.0e-5
        self.assertTrue(torch.all(trace.actions >= lower - tolerance))
        self.assertTrue(torch.all(trace.actions <= upper + tolerance))
        weights = torch.as_tensor(
            self.hydro.generation_coefficient,
            dtype=trace.actions.dtype,
        )
        shared = (trace.actions * weights).sum(dim=-1)
        self.assertTrue(
            torch.all(shared <= self.hydro.shared_release_capacity + tolerance)
        )
        self.assertEqual(float(trace.max_action_violation), 0.0)
        self.assertEqual(float(trace.max_storage_violation), 0.0)

    def test_protocol_has_independent_interfaces_and_equal_path_budget(self) -> None:
        protocol = self.protocol
        seeds = {
            protocol.fit_seed,
            protocol.validation_seed,
            protocol.test_seed,
            protocol.calibration_seed,
            protocol.evaluation_seed,
        }
        self.assertEqual(len(seeds), 5)
        self.assertEqual(
            protocol.direct_iterations * protocol.direct_batch_size,
            protocol.n_fit,
        )
        self.assertEqual(
            protocol.cfa_iterations * protocol.cfa_batch_size,
            protocol.n_fit,
        )

    def test_every_direct_policy_uses_feasible_projected_actions(self) -> None:
        policies = [MyopicPolicy(self.hydro)]
        policies.extend(
            _build_direct_policy(name, self.hydro, self.protocol)
            for name in (
                "projected_affine",
                "projected_polynomial",
                "projected_mlp_pathwise",
                "projected_graph_neural_global",
            )
        )
        for policy in policies:
            with self.subTest(policy=policy.policy_name):
                with torch.no_grad():
                    for parameter in policy.parameters():
                        parameter.add_(25.0 * torch.randn_like(parameter))
                    trace = simulate(policy, self.dataset, self.hydro)
                self.assert_trace_feasible(trace)

    def test_lookahead_and_cfa_use_feasible_projected_actions(self) -> None:
        for name, theta in (
            ("deterministic_lookahead", None),
            ("constant_cfa", np.asarray([1.35])),
        ):
            policy = make_lookahead_policy(
                name,
                self.hydro,
                self.protocol.cfa_config(),
                theta,
            )
            try:
                with self.subTest(policy=name):
                    trace = simulate(policy, self.dataset, self.hydro)
                    self.assert_trace_feasible(trace)
            finally:
                policy.close()

    def test_pairwise_ci_keeps_path_and_seed_units_separate(self) -> None:
        costs = {
            ("fixed", "fixed"): np.asarray([2.0, 3.0, 4.0]),
            ("learned", "0"): np.asarray([1.0, 3.0, 3.0]),
            ("learned", "1"): np.asarray([2.0, 2.0, 3.0]),
        }
        frame = all_pairwise_ci(costs, ("fixed", "learned"))
        self.assertEqual(set(frame["train_seed"]), {"0", "1"})
        self.assertTrue((frame["n_paired_test_paths"] == 3).all())


if __name__ == "__main__":
    unittest.main()
