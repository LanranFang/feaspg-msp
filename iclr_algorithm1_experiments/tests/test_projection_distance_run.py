"""Check that instrumentation preserves the established optimization protocol."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from algorithm1_gradient_study.training import fit_candidate
from iclr_algorithm1_experiments.config import ICLRProtocol
from iclr_algorithm1_experiments.io import dataset_fingerprint
from iclr_algorithm1_experiments.projection_distance_run import (
    configure, data, initial_policy, prepare_seed, fit_one, evaluate_one,
)


class ProjectionDistanceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_zero_arm_reproduces_existing_training_bitwise(self):
        protocol = ICLRProtocol.smoke_protocol()
        n_fit, seed, name = protocol.base_n_fit, 0, "projected_mlp_matched"
        hydro, _, fit, validation = data(protocol, n_fit)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old = fit_candidate(
                initial_policy(protocol, hydro, seed, name), fit, validation, hydro,
                protocol.direct_training(n_fit), seed, protocol.mlp_learning_rates[0],
                "full", output / "original.pt",
            )
            prepare_seed((protocol, n_fit, seed, name, str(output)))
            new = fit_one((protocol, n_fit, seed, name, "zero", 0, str(output)))
            payload = torch.load(output / new["path"] / "selected.pt", weights_only=False)
            self.assertEqual(old.best_validation_cost, new["best_validation_cost"])
            self.assertEqual(old.selected_iteration, new["selected_iteration"])
            for key, value in old.policy.state_dict().items():
                self.assertTrue(torch.equal(value, payload["state_dict"][key]), key)

    def test_original_formal_pool_and_validation_fingerprints(self):
        protocol = ICLRProtocol()
        _, pool, fit, validation = data(protocol, 1000)
        self.assertEqual(dataset_fingerprint(pool), "c1074a1b914d5b70e98e1a0ba83e787e2e3f5fbb60ebde21323bc4341f31d234")
        self.assertEqual(dataset_fingerprint(validation), "ffce11e4551ddb649350d770151bf5abedead90e1aec266c2fad297f9eec4ea7")
        self.assertEqual(dataset_fingerprint(fit), "39c9987da837da525c79115dfe473f67ff1e88bc060d07769c9d7dfd3229f798")

    def test_evaluation_requires_frozen_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "frozen selection"):
                evaluate_one((ICLRProtocol.smoke_protocol(), directory, {"path": "candidate"}))


if __name__ == "__main__":
    unittest.main()
