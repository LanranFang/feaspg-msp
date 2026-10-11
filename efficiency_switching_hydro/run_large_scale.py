"""Run the R=24, T=96 scalability benchmark with fixed hyperparameters."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUTS = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fit", type=int, default=1000)
    parser.add_argument("--fit-pool-size", type=int, default=5000)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        n_fit = 6
        fit_pool_size = 8
        n_validation, n_test, n_calibration, n_evaluation = 4, 8, 2, 2
        iterations, validation_every = 3, 1
        rl_validation_every = 3
        output = OUTPUTS / "smoke_large_scale_r24_t96"
    else:
        n_fit = args.n_fit
        fit_pool_size = args.fit_pool_size
        if n_fit <= 0 or n_fit % 2:
            raise ValueError("n-fit must be a positive even number.")
        if fit_pool_size < n_fit:
            raise ValueError("fit-pool-size must be at least n-fit.")
        n_validation, n_test, n_calibration, n_evaluation = 256, 1024, 256, 512
        iterations, validation_every = n_fit // 2, 10
        rl_validation_every = 50
        seed_suffix = "-".join(str(seed) for seed in args.train_seeds)
        output = OUTPUTS / (
            f"large_scale_r24_t96_n{n_fit}_seeds_{seed_suffix}"
        )

    result = output / "main_results.csv"
    if result.exists() and not args.rerun:
        print(f"skip completed {output}; pass --rerun to train again", flush=True)
        return

    command = [
        sys.executable,
        "-m",
        "efficiency_switching_hydro.run_experiment",
        "--profile",
        "decision_switching",
        "--output-dir",
        str(output),
        "--n-reservoirs",
        "24",
        "--horizon",
        "96",
        "--shock-factors",
        "4",
        "--n-fit",
        str(n_fit),
        "--fit-pool-size",
        str(fit_pool_size),
        "--n-validation",
        str(n_validation),
        "--n-test",
        str(n_test),
        "--n-calibration",
        str(n_calibration),
        "--n-evaluation",
        str(n_evaluation),
        "--iterations",
        str(iterations),
        "--batch-size",
        "2",
        "--validation-every",
        str(validation_every),
        "--policies",
        "myopic",
        "projected_mlp_pathwise",
        "projected_graph_neural_global",
        "--lr-neural",
        "0.0006",
        "--lr-mlp",
        "0.0003",
        "--rl-algorithms",
        "ppo",
        "sac",
        "td3",
        "--rl-validation-every",
        str(rl_validation_every),
        "--rl-network-width",
        "256",
        "--rl-network-depth",
        "2",
        "--ppo-lr",
        "0.0003",
        "--ppo-rollout-episodes",
        "10",
        "--ppo-epochs",
        "10",
        "--ppo-batch-size",
        "256",
        "--sac-lr",
        "0.0003",
        "--td3-lr",
        "0.001",
        "--rl-replay-buffer-size",
        "100000",
        "--rl-learning-starts",
        "1000",
        "--rl-batch-size",
        "256",
        "--rl-gradient-steps",
        "1",
        "--rl-n-steps",
        "3",
        "--td3-action-noise",
        "0.1",
        "--rl-reward-scale",
        "1000",
        "--rl-gamma",
        "1",
        "--rl-device",
        "auto",
        "--ppo-device",
        "cpu",
        "--off-policy-device",
        "auto",
        "--train-seeds",
        *(str(seed) for seed in args.train_seeds),
        "--skip-sddp",
    ]
    print(f"start large-scale benchmark: {output}", flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


if __name__ == "__main__":
    main()
