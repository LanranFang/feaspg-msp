from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from algorithm1_gradient_study.models import initialize_channel_policy
from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.simulator import simulate

from iclr_algorithm1_experiments.config import ICLRProtocol
from iclr_algorithm1_experiments.data_budget import pre_registered_candidates
from iclr_algorithm1_experiments.evaluation import evaluate_policy
from iclr_algorithm1_experiments.io import canonical_seed
from iclr_algorithm1_experiments.policies import (
    build_policy,
    parameter_matched_mlp_width,
)
from iclr_algorithm1_experiments.shac import SHACConfig, fit_shac_candidate
from iclr_algorithm1_experiments.scaling import _load_partial as load_scaling_partial


class ICLRProtocolTests(unittest.TestCase):
    def test_all_formal_data_seeds_are_new_and_distinct(self):
        protocol = ICLRProtocol()
        old_seeds = {4101, 4202, 4303, 4404, 4505}
        self.assertEqual(len(protocol.all_seed_values), 20)
        self.assertEqual(len(set(protocol.all_seed_values)), 20)
        self.assertTrue(old_seeds.isdisjoint(protocol.all_seed_values))

    def test_parameter_matched_mlp_is_close_to_graph(self):
        protocol = ICLRProtocol()
        width, mlp_parameters, graph_parameters = parameter_matched_mlp_width(
            protocol.hydro_config(),
            protocol.graph_width,
            protocol.network_depth,
        )
        self.assertEqual(width, 203)
        self.assertLess(abs(mlp_parameters - graph_parameters), 0.005 * graph_parameters)

    def test_smoke_preregistration_is_complete(self):
        protocol = ICLRProtocol.smoke_protocol()
        frame = pre_registered_candidates(protocol)
        expected_per_budget = 1 + 5 * len(protocol.train_seeds)
        self.assertEqual(len(frame), len(protocol.fit_budgets) * expected_per_budget)
        self.assertEqual(frame["status"].eq("fixed").sum(), len(protocol.fit_budgets))

    def test_partial_scaling_keys_and_csv_seeds_are_canonical(self):
        self.assertEqual(canonical_seed(0.0), "0")
        self.assertEqual(canonical_seed("fixed"), "fixed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pd.DataFrame(
                {
                    "test_path": [0, 1],
                    "R12_T48_projected_mlp_matched_seed0": [1.0, 2.0],
                }
            ).to_csv(output / "partial_test_path_costs.csv", index=False)
            _, _, _, costs = load_scaling_partial(output)
        self.assertIn(("R12_T48", "projected_mlp_matched", "0"), costs)


class SimulatorWindowTests(unittest.TestCase):
    def setUp(self):
        self.protocol = ICLRProtocol.smoke_protocol()
        self.config = self.protocol.hydro_config()
        self.data = generate_dataset("test", 4, 9001, self.config)

    def test_window_detachment_preserves_forward_values(self):
        torch.manual_seed(3)
        policy = build_policy("matched_mlp", self.config, self.protocol).double()
        initialize_channel_policy(policy, 1003)
        full = simulate(policy, self.data, self.config, dtype=torch.float64)
        windowed = simulate(
            policy,
            self.data,
            self.config,
            dtype=torch.float64,
            detach_state_every=3,
        )
        torch.testing.assert_close(full.actions, windowed.actions, rtol=0.0, atol=0.0)
        torch.testing.assert_close(full.total_cost, windowed.total_cost, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            full.queue_before_action,
            windowed.queue_before_action,
            rtol=0.0,
            atol=0.0,
        )

    def test_window_detachment_changes_long_horizon_gradient(self):
        torch.manual_seed(4)
        policy = build_policy("matched_mlp", self.config, self.protocol).double()
        initialize_channel_policy(policy, 1004)
        full_loss = simulate(
            policy,
            self.data,
            self.config,
            dtype=torch.float64,
        ).mean_cost
        full_gradient = torch.autograd.grad(full_loss, tuple(policy.parameters()))
        window_loss = simulate(
            policy,
            self.data,
            self.config,
            dtype=torch.float64,
            detach_state_every=3,
        ).mean_cost
        window_gradient = torch.autograd.grad(window_loss, tuple(policy.parameters()))
        full_vector = torch.cat([value.reshape(-1) for value in full_gradient])
        window_vector = torch.cat([value.reshape(-1) for value in window_gradient])
        self.assertGreater(float(torch.linalg.vector_norm(full_vector - window_vector)), 1.0e-8)

    def test_queue_before_action_has_correct_timing(self):
        policy = build_policy("matched_mlp", self.config, self.protocol)
        trace = simulate(policy, self.data, self.config)
        self.assertTrue(torch.equal(trace.queue_before_action[:, 0], torch.zeros_like(trace.queue_before_action[:, 0])))
        torch.testing.assert_close(
            trace.queue_before_action[:, 1],
            trace.queues[:, 0],
            rtol=0.0,
            atol=0.0,
        )


class ProjectedSHACTests(unittest.TestCase):
    def test_smoke_fit_uses_exact_budget_and_is_feasible(self):
        protocol = ICLRProtocol.smoke_protocol()
        config = protocol.hydro_config()
        fit = generate_dataset("fit", 8, 9101, config)
        validation = generate_dataset("validation", 4, 9202, config)
        test = generate_dataset("test", 8, 9303, config)
        width, _, _ = parameter_matched_mlp_width(
            config,
            protocol.graph_width,
            protocol.network_depth,
        )
        shac = SHACConfig(
            iterations=4,
            batch_size=2,
            validation_every=2,
            window=3,
            actor_learning_rate=3.0e-4,
            critic_learning_rate=1.0e-3,
            critic_epochs=1,
            critic_tau=0.05,
            cost_scale=100_000.0,
            weight_decay=1.0e-6,
            gradient_clip=10.0,
            critic_width=width,
            critic_depth=2,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            result = fit_shac_candidate(
                build_policy("shac", config, protocol),
                fit,
                validation,
                config,
                shac,
                train_seed=0,
                checkpoint_path=Path(directory) / "shac.pt",
            )
            self.assertTrue(Path(result.checkpoint_path).exists())
        self.assertEqual(result.environment_transitions, fit.n * config.T)
        self.assertEqual(result.actor_gradient_updates, 4)
        self.assertEqual(result.critic_gradient_updates, 4)
        metrics, costs = evaluate_policy(
            result.policy,
            test,
            config,
            protocol,
            result.deployment_parameters,
            result.training_parameters,
        )
        self.assertEqual(len(costs), test.n)
        self.assertEqual(metrics["total_constraint_violation_count"], 0)
        self.assertEqual(metrics["max_action_violation"], 0.0)
        self.assertEqual(metrics["max_storage_violation"], 0.0)


if __name__ == "__main__":
    unittest.main()
