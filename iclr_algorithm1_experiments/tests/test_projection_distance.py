from __future__ import annotations

import copy
import json
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.policies import PathwiseMLPPolicy
from efficiency_switching_hydro.projection import project_action
from efficiency_switching_hydro.simulator import simulate
from iclr_algorithm1_experiments.projection_distance import (
    estimate_lambda_ref,
    lambda_at_step,
    projection_distance,
    regularized_objective,
    trace_metrics,
)


class ProjectionDistanceTests(unittest.TestCase):
    def test_distance_sums_stages_and_reservoirs_and_zero_is_exact_original(self) -> None:
        proposals = torch.full((2, 3, 4), 2.0, requires_grad=True)
        actions = torch.zeros_like(proposals)
        cost = proposals.sum() / 3.0
        trace = SimpleNamespace(proposals=proposals, actions=actions, mean_cost=cost)
        torch.testing.assert_close(projection_distance(trace), torch.tensor([24.0, 24.0]))
        self.assertIs(regularized_objective(trace, 0.0), cost)
        torch.testing.assert_close(regularized_objective(trace, 0.5), cost + 12.0)
        base_grad, = torch.autograd.grad(cost, proposals, retain_graph=True)
        zero_grad, = torch.autograd.grad(regularized_objective(trace, 0.0), proposals)
        self.assertTrue(torch.equal(base_grad, zero_grad))

    def test_schedule_uses_zero_based_optimizer_updates(self) -> None:
        self.assertEqual(lambda_at_step("zero", 10, 0), 0)
        self.assertEqual(lambda_at_step("tenth", 10, 999), 1)
        self.assertEqual(lambda_at_step("fixed", 10, 999), 10)
        self.assertEqual(lambda_at_step("decay", 10, 0), 10)
        self.assertEqual(lambda_at_step("decay", 10, 3), 5)
        for arm, reference, step in [("bad", 1, 0), ("fixed", -1, 0), ("zero", float("nan"), 0), ("fixed", float("inf"), 0), ("fixed", 1, -1), ("fixed", 1, 1.5)]:
            with self.assertRaises(ValueError):
                lambda_at_step(arm, reference, step)

    def test_moving_recourse_and_action_gradients_match_finite_difference(self) -> None:
        # Only the fields consumed by the real shared projection are needed
        # for this analytically solvable two-stage, one-reservoir example.
        config = SimpleNamespace(
            release_capacity=(10.0,), ramp_fraction=1.0,
            shared_release_capacity=100.0, generation_coefficient=(1.0,),
        )

        def trace_at(theta: torch.Tensor, omit_moving: bool = False):
            zero = torch.zeros((1, 1), dtype=torch.float64)
            first_proposal = theta[0].reshape(1, 1)
            first_action = project_action(first_proposal, zero + 5, zero, config)
            storage = 5.0 - first_action
            second_proposal = theta[1].reshape(1, 1)
            second_action = project_action(
                second_proposal,
                storage.detach() if omit_moving else storage,
                first_action.detach() if omit_moving else first_action,
                config,
            )
            actions = torch.stack((first_action, second_action), dim=1)
            return SimpleNamespace(
                actions=actions,
                proposals=torch.stack((first_proposal, second_proposal), dim=1),
                mean_cost=0.1 * actions.square().sum(),
            )

        theta = torch.tensor([1.0, 8.0], dtype=torch.float64, requires_grad=True)
        full = trace_at(theta)
        omitted = trace_at(theta, omit_moving=True)
        self.assertTrue(torch.equal(full.actions, omitted.actions))
        full_gradient, = torch.autograd.grad(regularized_objective(full, 2.0), theta)
        omitted_gradient, = torch.autograd.grad(regularized_objective(omitted, 2.0), theta)
        torch.testing.assert_close(full_gradient, torch.tensor([7.4, 8.0], dtype=torch.float64))
        self.assertGreater(float(torch.linalg.vector_norm(full_gradient - omitted_gradient)), 7.0)
        direction = torch.tensor([0.7, -0.3], dtype=torch.float64)
        for step in [1e-3, 1e-4, 1e-5]:
            fd = (regularized_objective(trace_at(theta + step * direction), 2.0) - regularized_objective(trace_at(theta - step * direction), 2.0)) / (2 * step)
            self.assertAlmostEqual(float(full_gradient @ direction), float(fd.detach()), places=7)
        full = trace_at(theta)
        detached_action_distance = 0.5 * (full.proposals - full.actions.detach()).square().sum()
        detached_gradient, = torch.autograd.grad(detached_action_distance, theta)
        complete_gradient, = torch.autograd.grad(projection_distance(trace_at(theta)).sum(), theta)
        self.assertEqual(float(detached_gradient[0]), 0.0)
        self.assertEqual(float(complete_gradient[0]), 4.0)

    def test_metrics_count_action_vectors_not_binding_constraints(self) -> None:
        trace = SimpleNamespace(
            proposals=torch.tensor([[[0.0, 0.0], [3.0, 4.0]]]),
            actions=torch.zeros((1, 2, 2)),
            mean_cost=torch.tensor(7.0),
            max_action_violation=torch.tensor(0.0),
            max_storage_violation=torch.tensor(0.0),
        )
        metrics = trace_metrics(trace)
        self.assertEqual(metrics["proposal_distance"], 12.5)
        self.assertEqual(metrics["active_projection_fraction"], 0.5)
        self.assertEqual(trace_metrics(trace, active_tolerance=5.0)["active_projection_fraction"], 0.0)

    def test_lambda_estimate_uses_per_path_gradients_and_preserves_state_rng_dtype(self) -> None:
        config = ExperimentConfig.make("decision_switching", 2, 3, 1)
        torch.manual_seed(4)
        policy = PathwiseMLPPolicy(config, width=5, depth=1).double()
        policy.train()
        fit = generate_dataset("fit", 3, 11, config)
        for parameter in policy.parameters():
            parameter.grad = torch.ones_like(parameter)
        state_before = copy.deepcopy(policy.state_dict())
        grads_before = [parameter.grad.clone() for parameter in policy.parameters()]
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = copy.deepcopy(np.random.get_state())
        python_before = random.getstate()
        estimate = estimate_lambda_ref(policy, fit, config)
        self.assertTrue(policy.training)
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertEqual(numpy_before[2:], np.random.get_state()[2:])
        for key, tensor in policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, state_before[key]))
        for parameter, grad in zip(policy.parameters(), grads_before):
            self.assertTrue(torch.equal(parameter.grad, grad))
        manual_cost = []
        manual_distance = []
        parameters = tuple(policy.parameters())
        for index in range(fit.n):
            trace = simulate(policy, fit.take(np.array([index])), config, dtype=torch.float64)
            for values, output, retain in [(manual_cost, trace.mean_cost, True), (manual_distance, projection_distance(trace).sum(), False)]:
                gradients = torch.autograd.grad(output, parameters, retain_graph=retain)
                values.append(float(torch.cat([g.reshape(-1) for g in gradients]).norm()))
        np.testing.assert_allclose(estimate["cost_gradient_norms"], manual_cost, rtol=1e-14)
        np.testing.assert_allclose(estimate["distance_gradient_norms"], manual_distance, rtol=1e-14)
        self.assertAlmostEqual(estimate["lambda_ref"], np.median(manual_cost) / (np.median(manual_distance) + 1e-12), places=12)
        self.assertEqual(estimate["gradient_dtype"], "torch.float64")
        self.assertEqual(estimate["n_paths"], 3)
        json.dumps(estimate, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
