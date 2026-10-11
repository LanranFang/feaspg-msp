"""Checks for the scalar mechanism's exact recurrence and optimizer labels."""

import unittest

import numpy as np
import torch

from iclr_algorithm1_experiments.projection_distance_mechanisms import scalar_path


class ScalarProjectionMechanismTests(unittest.TestCase):
    def test_zero_lambda_stays_at_original_flat_point(self):
        for optimizer in ("SGD", "Adam"):
            rows = scalar_path("zero", optimizer)
            self.assertTrue(all(row["u"] == -1 and row["operating_cost"] == 1 for row in rows))
            self.assertTrue(all(row["gradient"] == 0 for row in rows))

    def test_fixed_lambda_sgd_matches_analytic_no_crossing_recurrence(self):
        rows = scalar_path("fixed_one", "SGD")
        observed = np.asarray([row["u"] for row in rows])
        expected = -np.power(0.9, np.arange(len(rows)))
        np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=0)
        self.assertTrue(np.all(observed < 0))
        self.assertTrue(all(row["operating_cost"] == 1 for row in rows))

    def test_decay_sgd_matches_product_recurrence(self):
        rows = scalar_path("decay", "SGD")
        expected = np.concatenate(([-1.0], -np.cumprod(1 - 0.1 / np.sqrt(np.arange(1, 201)))))
        np.testing.assert_allclose([row["u"] for row in rows], expected, rtol=2e-14)
        self.assertTrue(all(row["u"] < 0 for row in rows))

    def test_documented_boundary_derivative(self):
        points = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.grad(points.clamp(0, 1).sum(), points)[0]
        torch.testing.assert_close(derivative, torch.tensor([0, 1, 1, 1, 0], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
