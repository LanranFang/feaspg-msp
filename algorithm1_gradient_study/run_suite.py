"""Run the five Algorithm 1 pathwise-gradient mechanism experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from .config import StudyProtocol
from .experiments import (
    run_channel_ablations,
    run_finite_difference,
    run_graph_stability,
    run_horizon_amplification,
)


EXPERIMENTS = (
    "finite_difference",
    "no_proposal_state",
    "no_moving_recourse",
    "horizon_amplification",
    "graph_stability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive.")
    torch.set_num_threads(args.torch_threads)
    protocol = (
        StudyProtocol.smoke_protocol(device=args.device)
        if args.smoke
        else StudyProtocol(device=args.device)
    )
    package_root = Path(__file__).resolve().parent
    default_name = "smoke" if args.smoke else "formal_protocol_v1"
    output = Path(args.output_dir) if args.output_dir else package_root / "outputs" / default_name
    output.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(output, protocol, args)

    requested = set(args.experiments)
    completed: list[Path] = []
    if "finite_difference" in requested:
        completed.append(run_finite_difference(protocol, output))
    if requested.intersection({"no_proposal_state", "no_moving_recourse"}):
        destinations = run_channel_ablations(protocol, output)
        for name in ("no_proposal_state", "no_moving_recourse"):
            if name in requested:
                completed.append(destinations[name])
    if "horizon_amplification" in requested:
        completed.append(run_horizon_amplification(protocol, output))
    if "graph_stability" in requested:
        completed.append(run_graph_stability(protocol, output))

    print(f"Completed Algorithm 1 study at {output}")
    for path in completed:
        print(f"  {path.name}")


def _write_run_metadata(
    output: Path,
    protocol: StudyProtocol,
    args: argparse.Namespace,
) -> None:
    with (output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": asdict(protocol),
                "requested_experiments": args.experiments,
                "primary_metric": "test cost",
                "selection_rule": "minimum validation cost",
                "test_role": "reporting only after all selections are frozen",
                "reserved_blocks": {
                    "calibration": "generated interface, not read in this study",
                    "evaluation": "generated interface, not read in this study",
                },
                "uncertainty_reporting": {
                    "test_path": "paired 95% CI within train seed",
                    "training": "SD across independent train seeds",
                },
                "nonlinear_hydro_claim_limit": (
                    "No exact optimum is available; compare only with implemented baselines."
                ),
            },
            handle,
            indent=2,
        )
    with (output / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "torch_threads": torch.get_num_threads(),
                "device": protocol.device,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    run()
