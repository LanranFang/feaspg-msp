"""Supplementary full-comparison A1 shard (affine, polynomial, MLP, Graph)."""
import argparse
import json
import pickle
import time
import numpy as np
import torch
from . import regularized_a1_v2 as core
from .projection_distance_run import configure, save_json, sha256

BASE_BUILD = core.build
NAME = "full_comparison_n1000"
METHODS = ("projected_affine", "projected_polynomial", "projected_mlp_pathwise", "projected_graph_neural_global")


def full_build(b, method, seed):
    if b["kind"] != "full":
        return BASE_BUILD(b, method, seed)
    from algorithm1_full_comparison.run_comparison import _build_direct_policy
    torch.manual_seed(seed)
    np.random.seed(seed)
    return _build_direct_policy(method, b["system"], b["protocol"])


core.build = full_build


def init_worker(task):
    return core.calibrate(task)


def fit_worker(task):
    return core.fit(task)


def test_worker(task):
    return core.test_selected(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate"))
    args = parser.parse_args()
    configure()
    if not (core.OUT / "bundles" / f"{NAME}.pkl").exists():
        from algorithm1_full_comparison.config import FullComparisonProtocol
        from efficiency_switching_hydro.data import generate_splits
        p = FullComparisonProtocol()
        h = p.experiment_config()
        splits = generate_splits(p.data_config(), h)
        b = {"kind": "full", "system": h, "fit": splits.fit, "validation": splits.validation,
             "test": splits.test, "horizon": h.T, "protocol": p,
             "settings": {m: {"iterations": 500, "validation_every": 10,
                                "learning_rates": list(p.direct_learning_rates(m))} for m in METHODS}}
        path = core.OUT / "bundles" / f"{NAME}.pkl"
        with path.open("wb") as f:
            pickle.dump(b, f)
        save_json(core.OUT / "full_comparison_manifest.json", {"name": NAME, "methods": METHODS,
                  "bundle_sha256": sha256(path), "rule": "Same positive-strength and original LR selection as main reruns",
                  "added_unix_time": time.time()})
    if args.command == "train":
        core.parallel(init_worker, [(NAME, m, seed) for m in METHODS for seed in range(5)], 8)
        tasks = [(NAME, m, seed, a, lr, "full") for m in METHODS for seed in range(5)
                 for a in core.STRENGTHS for lr in range(2)]
        np.random.default_rng(9172028).shuffle(tasks)
        core.parallel(fit_worker, tasks, 8)
    else:
        tasks = []
        for m in METHODS:
            for seed in range(5):
                parent = core.OUT / "runs" / NAME / m / f"seed{seed}" / "full"
                candidates = [json.loads(f.read_text()) for f in parent.glob("*/result.json")]
                assert len(candidates) == 6
                chosen = min(candidates, key=lambda c: (c["best_validation_cost"], c["strength"], c["lr_index"]))
                dest = core.OUT / "selected" / NAME / m / f"seed{seed}" / "full/positive"
                save_json(dest / "selection.json", {"chosen": chosen, "candidates": candidates,
                          "frozen_before_test_unix_time": time.time()})
                tasks.append((NAME, m, seed, "full", False))
        core.parallel(test_worker, tasks, 6)


if __name__ == "__main__":
    main()
