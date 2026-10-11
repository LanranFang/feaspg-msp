"""Core and solver tests for efficiency-switching hydro."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from efficiency_switching_hydro.config import (
    DataConfig,
    ExperimentConfig,
    PROFILES,
    RLConfig,
    SDDPConfig,
)
from efficiency_switching_hydro.data import generate_dataset, generate_splits
from efficiency_switching_hydro.evaluation import evaluate_with_costs
from efficiency_switching_hydro.physics import (
    generation_np,
    total_generation_linearization,
    transition_np,
    turbine_efficiency_np,
)
from efficiency_switching_hydro.policy_features import (
    flat_markov_observation_np,
    flat_markov_observation_torch,
)
from efficiency_switching_hydro.policies import make_policy
from efficiency_switching_hydro.projection import action_bounds, project_action
from efficiency_switching_hydro.rl import fit_checkpoint as fit_rl
from efficiency_switching_hydro.rl_env import HydroTrajectoryEnv
from efficiency_switching_hydro.sddp import fit_checkpoint as fit_sddp
from efficiency_switching_hydro.simulator import simulate


class CoreTests(unittest.TestCase):
    def test_profiles_activate_mechanisms_in_order(self) -> None:
        configs = {name: ExperimentConfig.make(name, 8, 10, 3) for name in PROFILES}
        self.assertEqual(max(configs["linear"].band1_amplitude), 0.0)
        self.assertGreater(max(configs["single_band"].band1_amplitude), 0.0)
        self.assertEqual(max(configs["single_band"].band2_amplitude), 0.0)
        self.assertGreater(max(configs["double_band"].band2_amplitude), 0.0)
        self.assertEqual(configs["double_band"].routing_delay, 0)
        self.assertEqual(configs["double_band_network"].routing_delay, 2)
        routing = np.asarray(configs["double_band_network"].routing)
        self.assertGreater(int((routing > 0.0).sum(axis=0).max()), 1)
        self.assertGreater(max(configs["double_band_full"].evaporation_rate), 0.0)
        self.assertLess(configs["double_band_full"].ramp_fraction, 1.0)
        self.assertGreater(configs["double_band_full"].peaker_quadratic_cost, 0.0)
        self.assertLess(
            max(configs["decision_switching"].band1_width),
            min(configs["double_band_network"].band1_width),
        )
        self.assertGreater(
            min(configs["decision_switching"].head_downstream_loss),
            max(configs["double_band_network"].head_downstream_loss),
        )

    def test_double_band_has_two_efficiency_peaks(self) -> None:
        config = ExperimentConfig.make("double_band", 4, 6, 2)
        capacity = np.asarray(config.storage_capacity)
        release = np.asarray(config.release_capacity)
        fractions = np.linspace(0.0, 1.0, 1001)
        storage = np.broadcast_to(0.65 * capacity, (len(fractions), 4)).copy()
        action = fractions[:, None] * release[None, :]
        efficiency = turbine_efficiency_np(storage, action, config)[:, 0]
        local_maxima = np.flatnonzero(
            (efficiency[1:-1] > efficiency[:-2])
            & (efficiency[1:-1] > efficiency[2:])
        ) + 1
        self.assertGreaterEqual(len(local_maxima), 2)
        peak_fractions = fractions[local_maxima]
        self.assertTrue(np.any((peak_fractions > 0.25) & (peak_fractions < 0.45)))
        self.assertTrue(np.any((peak_fractions > 0.55) & (peak_fractions < 0.80)))

    def test_downstream_storage_reduces_upstream_head(self) -> None:
        config = ExperimentConfig.make("double_band_network", 12, 8, 3)
        source = next(index for index in range(12) if config.downstream_index[index] >= 0)
        target = config.downstream_index[source]
        capacity = np.asarray(config.storage_capacity)
        release = np.asarray(config.release_capacity)
        low = 0.60 * capacity
        high = low.copy()
        high[target] = 0.95 * capacity[target]
        action = np.zeros(12)
        action[source] = 0.65 * release[source]
        self.assertGreater(
            generation_np(low, action, config)[source],
            generation_np(high, action, config)[source],
        )

    def test_linear_profile_linearization_is_exact(self) -> None:
        config = ExperimentConfig.make("linear", 5, 7, 2)
        storage = 0.61 * np.asarray(config.storage_capacity)
        action = 0.43 * np.asarray(config.release_capacity)
        value, storage_gradient, action_gradient = total_generation_linearization(
            storage, action, config
        )
        alternative_storage = 0.31 * np.asarray(config.storage_capacity)
        alternative_action = 0.71 * np.asarray(config.release_capacity)
        affine = (
            value
            + storage_gradient @ (alternative_storage - storage)
            + action_gradient @ (alternative_action - action)
        )
        exact = generation_np(alternative_storage, alternative_action, config).sum()
        self.assertAlmostEqual(float(affine), float(exact), places=7)

    def test_independent_data_interfaces_have_no_context(self) -> None:
        config = ExperimentConfig.make("double_band_network", 4, 6, 3)
        splits = generate_splits(
            DataConfig(
                n_fit=8,
                n_validation=7,
                n_test=6,
                n_calibration=5,
                n_evaluation=4,
            ),
            config,
        )
        self.assertEqual(splits.fit.inflow.shape, (8, 6, 4))
        self.assertFalse(hasattr(splits.fit, "context"))
        seeds = [dataset.seed for dataset in splits.as_dict().values()]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_fit_pool_prefixes_are_nested(self) -> None:
        config = ExperimentConfig.make("decision_switching", 4, 6, 2)
        pool = generate_dataset("fit_pool", 20, 57, config)
        small = pool.take(np.arange(7), name="fit")
        medium = pool.take(np.arange(13), name="fit")
        np.testing.assert_array_equal(small.inflow, medium.inflow[:7])
        np.testing.assert_array_equal(small.demand, medium.demand[:7])

    def test_projection_satisfies_every_action_constraint(self) -> None:
        generator = torch.Generator().manual_seed(29)
        for profile in PROFILES:
            with self.subTest(profile=profile):
                config = ExperimentConfig.make(profile, 12, 8, 3)
                capacity = torch.as_tensor(config.storage_capacity)
                release = torch.as_tensor(config.release_capacity)
                storage = torch.rand(256, 12, generator=generator) * capacity
                previous = torch.rand(256, 12, generator=generator) * release
                proposal = torch.randn(256, 12, generator=generator) * 200.0
                action = project_action(proposal, storage, previous, config)
                lower, upper = action_bounds(storage, previous, config)
                weights = torch.as_tensor(config.generation_coefficient)
                self.assertTrue(torch.all(action >= lower - 1e-5))
                self.assertTrue(torch.all(action <= upper + 1e-5))
                self.assertTrue(torch.all(action <= storage + 1e-5))
                self.assertTrue(
                    torch.all(
                        (action * weights).sum(dim=-1)
                        <= config.shared_release_capacity + 1e-4
                    )
                )

    def test_two_stage_routing_delay(self) -> None:
        config = ExperimentConfig.make("double_band_network", 4, 5, 2)
        storage = np.asarray(config.initial_storage, dtype=np.float64)
        previous = np.zeros(4)
        queue = np.zeros((2, 4))
        action = np.zeros(4)
        source = 0
        target = config.downstream_index[source]
        action[source] = 10.0
        zero_inflow = np.zeros(4)
        storage, previous, queue, _ = transition_np(
            storage, previous, queue, action, zero_inflow, 0.0, 1.0, config
        )
        self.assertAlmostEqual(
            queue[1, target], 10.0 * config.routing[source][target]
        )
        downstream_before = storage[target]
        for _ in range(2):
            storage, previous, queue, _ = transition_np(
                storage,
                previous,
                queue,
                np.zeros(4),
                zero_inflow,
                0.0,
                1.0,
                config,
            )
        self.assertGreater(storage[target], downstream_before)

    def test_every_policy_uses_shared_simulator_and_projection(self) -> None:
        config = ExperimentConfig.make("double_band_full", 4, 5, 3)
        data = generate_dataset("unit", 12, 41, config)
        for name in (
            "myopic",
            "projected_affine",
            "projected_polynomial",
            "projected_mlp_pathwise",
            "projected_graph_neural",
            "projected_graph_neural_global",
        ):
            with self.subTest(policy=name):
                policy = make_policy(name, config)
                trace = simulate(policy, data, config)
                self.assertLessEqual(float(trace.max_action_violation.detach()), 1e-5)
                self.assertLessEqual(float(trace.max_storage_violation.detach()), 1e-5)
                if name != "myopic":
                    trace.mean_cost.backward()
                    gradients = [
                        parameter.grad
                        for parameter in policy.parameters()
                        if parameter.grad is not None
                    ]
                    self.assertTrue(gradients)
                    self.assertTrue(
                        all(torch.isfinite(gradient).all() for gradient in gradients)
                    )

    def test_pathwise_mlp_matches_rl_observation_and_reference_center(self) -> None:
        config = ExperimentConfig.make("decision_switching", 5, 8, 3)
        generator = torch.Generator().manual_seed(71)
        capacity = torch.as_tensor(config.storage_capacity, dtype=torch.float32)
        release = torch.as_tensor(config.release_capacity, dtype=torch.float32)
        storage = torch.rand(7, 5, generator=generator) * capacity
        previous = torch.rand(7, 5, generator=generator) * release
        queue_capacity = torch.as_tensor(
            config.queue_capacity(), dtype=torch.float32
        )
        queue = torch.rand(7, 2, 5, generator=generator) * queue_capacity
        stage = 3
        torch_observation = flat_markov_observation_torch(
            stage, storage, previous, queue, config
        )
        numpy_observation = flat_markov_observation_np(
            stage,
            storage.numpy(),
            previous.numpy(),
            queue.numpy(),
            config,
        )
        np.testing.assert_allclose(
            torch_observation.numpy(), numpy_observation, rtol=1e-6, atol=1e-6
        )
        policy = make_policy("projected_mlp_pathwise", config)
        proposal = policy.propose(stage, storage, previous, queue)
        expected = torch.as_tensor(
            config.reference_release()[stage], dtype=storage.dtype
        ).expand_as(storage)
        torch.testing.assert_close(proposal, expected)


class RLTests(unittest.TestCase):
    @staticmethod
    def small_rl_config() -> RLConfig:
        return RLConfig(
            validation_every_episodes=3,
            network_width=16,
            network_depth=1,
            ppo_rollout_episodes=3,
            ppo_epochs=1,
            ppo_batch_size=12,
            replay_buffer_size=100,
            learning_starts=4,
            off_policy_batch_size=8,
            gradient_steps=1,
            device="cpu",
        )

    def test_rl_environment_projects_every_action_and_uses_unique_paths(self) -> None:
        config = ExperimentConfig.make("decision_switching", 4, 5, 2)
        fit = generate_dataset("fit", 6, 81, config)
        env = HydroTrajectoryEnv(fit, config, reward_scale=1000.0, seed=17)
        rng = np.random.default_rng(19)
        for _ in range(fit.n):
            env.reset()
            terminated = False
            while not terminated:
                _, _, terminated, truncated, info = env.step(
                    rng.uniform(-2.0, 2.0, config.n_reservoirs).astype(np.float32)
                )
                self.assertFalse(truncated)
                self.assertLessEqual(info["action_violation"], 1e-8)
        self.assertEqual(env.transitions_taken, fit.n * config.T)
        self.assertEqual(len(set(env.completed_path_indices)), fit.n)

    def test_rl_environment_cost_matches_shared_batch_simulator(self) -> None:
        config = ExperimentConfig.make("decision_switching", 4, 6, 2)
        dataset = generate_dataset("unit", 3, 91, config)
        env = HydroTrajectoryEnv(dataset, config, reward_scale=1000.0, seed=23)

        class ReferencePolicy:
            def eval(self):
                return self

            def propose(self, stage, storage, previous_release, queue):
                reference = torch.as_tensor(
                    config.reference_release()[stage],
                    dtype=storage.dtype,
                    device=storage.device,
                )
                return reference.expand_as(storage)

        for path_index in range(dataset.n):
            env.reset(options={"path_index": path_index})
            scaled_reward = 0.0
            for _ in range(config.T):
                _, reward, terminated, _, info = env.step(
                    np.zeros(config.n_reservoirs, dtype=np.float32)
                )
                scaled_reward -= 1000.0 * reward
            self.assertTrue(terminated)
            trace = simulate(
                ReferencePolicy(),
                dataset.take(np.asarray([path_index])),
                config,
            )
            self.assertAlmostEqual(
                scaled_reward, float(trace.total_cost[0]), delta=2e-3
            )
            self.assertAlmostEqual(
                info["episode_cost"], float(trace.total_cost[0]), delta=2e-3
            )

    def test_all_rl_algorithms_respect_the_fixed_fit_budget(self) -> None:
        config = ExperimentConfig.make("decision_switching", 3, 4, 2)
        fit = generate_dataset("fit", 6, 101, config)
        validation = generate_dataset("validation", 4, 102, config)
        test = generate_dataset("test", 5, 103, config)
        rl_config = self.small_rl_config()
        with tempfile.TemporaryDirectory() as directory:
            for algorithm in ("ppo", "sac", "td3"):
                with self.subTest(algorithm=algorithm):
                    result = fit_rl(
                        algorithm,
                        fit,
                        validation,
                        config,
                        rl_config,
                        train_seed=0,
                        checkpoint_path=Path(directory) / f"{algorithm}.zip",
                    )
                    try:
                        self.assertEqual(result.completed_episodes, fit.n)
                        self.assertEqual(
                            result.environment_steps, fit.n * config.T
                        )
                        self.assertGreater(result.actor_gradient_updates, 0)
                        self.assertGreater(result.critic_gradient_updates, 0)
                        self.assertTrue(Path(result.checkpoint_path).exists())
                        metrics, _ = evaluate_with_costs(
                            result.policy, test, config
                        )
                        self.assertLessEqual(metrics.max_action_violation, 1e-6)
                        self.assertLessEqual(metrics.max_storage_violation, 1e-6)
                    finally:
                        result.policy.close()


class SolverTests(unittest.TestCase):
    def test_exact_and_linearized_sddp_smoke(self) -> None:
        for profile in ("linear", "double_band_full"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig.make(profile, 3, 3, 2)
                splits = generate_splits(
                    DataConfig(
                        n_fit=20,
                        n_validation=4,
                        n_test=5,
                        n_calibration=2,
                        n_evaluation=2,
                    ),
                    config,
                )
                result = fit_sddp(
                    splits.fit,
                    splits.validation,
                    config,
                    SDDPConfig(iterations=1, validation_every=1, noise_scenarios=2),
                    0,
                    Path(directory) / "sddp.pt",
                )
                try:
                    metrics, _ = evaluate_with_costs(result.policy, splits.test, config)
                    expected = (
                        "empirical_sddp"
                        if profile == "linear"
                        else "linearized_empirical_sddp"
                    )
                    self.assertEqual(result.policy.policy_name, expected)
                    self.assertLessEqual(metrics.max_action_violation, 1e-6)
                    self.assertLessEqual(metrics.max_storage_violation, 1e-6)
                finally:
                    result.policy.close()


if __name__ == "__main__":
    unittest.main()
