"""Run the validation-tuned decision-switching pilot in reproducible order."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUTPUTS = ROOT / "outputs"


def _run(arguments: list[str], result: Path, rerun: bool) -> None:
    if result.exists() and not rerun:
        print(f"skip completed {result.parent.name}")
        return
    subprocess.run(
        [sys.executable, "-m", "efficiency_switching_hydro.run_experiment", *arguments],
        cwd=WORKSPACE,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    base = OUTPUTS / "pilot_decision_switching_r12_t48"
    alternative = OUTPUTS / "pilot_decision_switching_lr_alt"
    common = [
        "--profile", "decision_switching",
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
        "--train-seeds", "0",
    ]
    _run(
        [
            *common,
            "--output-dir", str(base),
            "--policies",
            "myopic",
            "projected_affine",
            "projected_polynomial",
            "projected_graph_neural",
            "projected_graph_neural_global",
            "--sddp-iterations", "30",
            "--sddp-validation-every", "10",
            "--sddp-noise-scenarios", "1",
        ],
        base / "main_results.csv",
        args.rerun,
    )
    _run(
        [
            *common,
            "--output-dir", str(alternative),
            "--policies",
            "projected_affine",
            "projected_polynomial",
            "projected_graph_neural_global",
            "--skip-sddp",
            "--lr-affine", "0.002",
            "--lr-polynomial", "0.0003",
            "--lr-neural", "0.0006",
        ],
        alternative / "main_results.csv",
        args.rerun,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "efficiency_switching_hydro.diagnostics",
            "--experiment-dir",
            str(alternative),
            "--neural-policy",
            "projected_graph_neural_global",
        ],
        cwd=WORKSPACE,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "efficiency_switching_hydro.summarize_pilot"],
        cwd=WORKSPACE,
        check=True,
    )


if __name__ == "__main__":
    main()
