"""Run the frozen ICLR-focused Algorithm 1 experiment sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config import ICLRProtocol
from .data_budget import run_data_budget
from .graph_transfer import run_graph_transfer
from .io import write_completion, write_protocol
from .mechanisms import run_mechanisms
from .scaling import run_scaling


EXPERIMENTS = ("mechanisms", "data_budget", "scaling", "graph_transfer")


def _text_table(frame: pd.DataFrame) -> str:
    return f"```text\n{frame.to_string(index=False)}\n```"


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
    parser.add_argument("--rl-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _write_summary(root: Path) -> None:
    sections = ["# ICLR Algorithm 1 Experiment Results", ""]
    mechanism = root / "01_mechanisms" / "01_finite_difference" / "finite_difference_summary.csv"
    if mechanism.exists():
        data = pd.read_csv(mechanism)
        sections.extend(
            [
                "## Gradient check",
                "",
                _text_table(data),
                "",
            ]
        )
    budget = root / "02_data_budget" / "main_results.csv"
    if budget.exists():
        data = pd.read_csv(budget)
        columns = [
            "n_fit",
            "policy",
            "test_cost_mean_over_train_seeds",
            "test_cost_train_seed_sd",
            "environment_transitions_per_candidate",
            "actor_gradient_updates_per_candidate",
            "total_tuning_time_seconds_mean",
        ]
        sections.extend(
            [
                "## Trajectory budget",
                "",
                _text_table(data[columns]),
                "",
            ]
        )
    scaling = root / "03_scaling" / "main_results.csv"
    if scaling.exists():
        data = pd.read_csv(scaling)
        columns = [
            "config_id",
            "policy",
            "test_cost_mean_over_train_seeds",
            "test_cost_train_seed_sd",
            "deployment_parameters",
            "batch1_action_latency_ms_median_across_seeds",
        ]
        sections.extend(
            [
                "## Scaling",
                "",
                _text_table(data[columns]),
                "",
            ]
        )
    transfer = root / "04_graph_transfer" / "cross_size_main_results.csv"
    if transfer.exists():
        data = pd.read_csv(transfer)
        columns = [
            "config_id",
            "policy",
            "test_cost_mean_over_train_seeds",
            "test_cost_train_seed_sd",
        ]
        sections.extend(
            [
                "## Graph transfer",
                "",
                _text_table(data[columns]),
                "",
            ]
        )
    (root / "RESULTS.md").write_text("\n".join(sections), encoding="utf-8")


def run() -> None:
    args = parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive.")
    torch.set_num_threads(args.torch_threads)
    protocol = (
        ICLRProtocol.smoke_protocol(args.device, args.rl_device)
        if args.smoke
        else ICLRProtocol(device=args.device, rl_device=args.rl_device)
    )
    package_root = Path(__file__).resolve().parent
    default_name = "all_experiments_smoke" if args.smoke else "formal_protocol_v1"
    root = Path(args.output_dir) if args.output_dir else package_root / "outputs" / default_name
    root.mkdir(parents=True, exist_ok=True)
    write_protocol(
        root,
        protocol,
        "iclr_algorithm1_suite",
        {
            "execution_order": list(EXPERIMENTS),
            "requested_experiments": args.experiments,
            "old_test_seed_reused": False,
        },
    )
    resume = not args.no_resume
    requested = set(args.experiments)
    completed = []
    if "mechanisms" in requested:
        run_mechanisms(protocol, root / "01_mechanisms", resume=resume)
        completed.append("mechanisms")
    if "data_budget" in requested:
        run_data_budget(protocol, root / "02_data_budget", resume=resume)
        completed.append("data_budget")
    if "scaling" in requested:
        run_scaling(protocol, root / "03_scaling", resume=resume)
        completed.append("scaling")
    if "graph_transfer" in requested:
        scaling_output = root / "03_scaling"
        run_graph_transfer(
            protocol,
            root / "04_graph_transfer",
            scaling_output,
        )
        completed.append("graph_transfer")
    _write_summary(root)
    write_completion(
        root,
        "iclr_algorithm1_suite",
        {
            "completed_experiments": completed,
            "smoke": protocol.smoke,
        },
    )
    print(f"Completed ICLR Algorithm 1 suite at {root}", flush=True)


if __name__ == "__main__":
    run()
