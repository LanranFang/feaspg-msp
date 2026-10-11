"""Run the unified direct-policy, RL, and optional SDDP hydro comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUTS = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-sddp", action="store_true")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    if args.smoke:
        output = OUTPUTS / "smoke_rl_benchmarks"
        scale = [
            "--n-reservoirs", "3",
            "--horizon", "4",
            "--shock-factors", "2",
            "--n-fit", "6",
            "--n-validation", "4",
            "--n-test", "8",
            "--n-calibration", "2",
            "--n-evaluation", "2",
            "--iterations", "3",
            "--batch-size", "2",
            "--validation-every", "1",
            "--rl-validation-every", "3",
            "--rl-network-width", "16",
            "--rl-network-depth", "1",
            "--ppo-rollout-episodes", "3",
            "--ppo-epochs", "1",
            "--ppo-batch-size", "12",
            "--rl-replay-buffer-size", "100",
            "--rl-learning-starts", "4",
            "--rl-batch-size", "8",
            "--rl-device", "cpu",
            "--skip-sddp",
        ]
    else:
        seed_suffix = "-".join(str(seed) for seed in args.train_seeds)
        output = OUTPUTS / f"rl_decision_switching_r12_t48_seeds_{seed_suffix}"
        scale = [
            "--n-reservoirs", "12",
            "--horizon", "48",
            "--shock-factors", "4",
            "--n-fit", "1000",
            "--n-validation", "256",
            "--n-test", "1024",
            "--n-calibration", "256",
            "--n-evaluation", "512",
            "--iterations", "500",
            "--batch-size", "2",
            "--validation-every", "10",
            "--lr-affine", "0.002",
            "--lr-polynomial", "0.0003",
            "--lr-neural", "0.0006",
            "--lr-mlp", "0.0003",
            "--rl-validation-every", "50",
            "--ppo-lr-candidates", "0.0001", "0.0003",
            "--sac-lr-candidates", "0.0001", "0.0003",
            "--td3-lr-candidates", "0.0003", "0.001",
            "--rl-device", "auto",
            "--ppo-device", "cpu",
            "--off-policy-device", "auto",
            "--sddp-iterations", "30",
            "--sddp-validation-every", "10",
            "--sddp-noise-scenarios", "1",
        ]
        if args.skip_sddp:
            scale.append("--skip-sddp")

    result = output / "main_results.csv"
    if result.exists() and not args.rerun:
        print(f"skip completed {output.name}; pass --rerun to fit it again")
        return
    command = [
        sys.executable,
        "-m",
        "efficiency_switching_hydro.run_experiment",
        "--profile",
        "decision_switching",
        "--output-dir",
        str(output),
        "--policies",
        "myopic",
        "projected_affine",
        "projected_polynomial",
        "projected_mlp_pathwise",
        "projected_graph_neural_global",
        "--rl-algorithms",
        "ppo",
        "sac",
        "td3",
        "--train-seeds",
        *(str(seed) for seed in args.train_seeds),
        *scale,
    ]
    subprocess.run(command, cwd=WORKSPACE, check=True)


if __name__ == "__main__":
    main()
