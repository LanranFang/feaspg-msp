"""Formal wrapper around the five existing gradient-mechanism experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from algorithm1_gradient_study.experiments import (
    run_channel_ablations,
    run_finite_difference,
    run_graph_stability,
    run_horizon_amplification,
)

from .config import ICLRProtocol
from .io import write_completion, write_frame, write_protocol


def pre_registered_candidates(protocol: ICLRProtocol) -> pd.DataFrame:
    study = protocol.study_protocol()
    rows = []
    for mode in ("full", "no_proposal_state", "no_moving_recourse"):
        for train_seed in study.train_seeds:
            for index, learning_rate in enumerate(study.ablation_learning_rates):
                rows.append(
                    {
                        "experiment": "gradient_channel_ablation",
                        "variant": mode,
                        "policy": "projected_mlp_pathwise",
                        "train_seed": train_seed,
                        "candidate_index": index,
                        "learning_rate": learning_rate,
                    }
                )
    for policy, rates in (
        ("projected_mlp_pathwise", study.mlp_learning_rates),
        ("projected_graph_neural_global", study.graph_learning_rates),
    ):
        for train_seed in study.train_seeds:
            for index, learning_rate in enumerate(rates):
                rows.append(
                    {
                        "experiment": "graph_stability",
                        "variant": policy,
                        "policy": policy,
                        "train_seed": train_seed,
                        "candidate_index": index,
                        "learning_rate": learning_rate,
                    }
                )
    return pd.DataFrame(rows)


def run_mechanisms(
    protocol: ICLRProtocol,
    output: Path,
    resume: bool = True,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    study = protocol.study_protocol()
    write_frame(
        pre_registered_candidates(protocol),
        output / "pre_registered_candidates.csv",
    )
    write_protocol(
        output,
        protocol,
        "gradient_mechanisms",
        {
            "study_protocol": study.__dict__,
            "experiments": [
                "finite_difference",
                "no_proposal_state",
                "no_moving_recourse",
                "horizon_amplification",
                "graph_stability",
            ],
            "test_seed_is_new": True,
        },
    )

    finite = output / "01_finite_difference" / "finite_difference_summary.csv"
    if not resume or not finite.exists():
        print("mechanism 1/5: finite-difference check", flush=True)
        run_finite_difference(study, output)

    channel_outputs = (
        output / "02_no_proposal_state" / "main_results.csv",
        output / "03_no_moving_recourse" / "main_results.csv",
    )
    if not resume or not all(path.exists() for path in channel_outputs):
        print("mechanisms 2-3/5: gradient-channel ablations", flush=True)
        run_channel_ablations(study, output)

    horizon = output / "04_horizon_amplification" / "horizon_summary.csv"
    if not resume or not horizon.exists():
        print("mechanism 4/5: horizon amplification", flush=True)
        run_horizon_amplification(study, output)

    graph = output / "05_graph_stability" / "main_results.csv"
    if not resume or not graph.exists():
        print("mechanism 5/5: graph stability", flush=True)
        run_graph_stability(study, output)

    violations = 0
    for path in (
        output / "02_no_proposal_state" / "main_results.csv",
        output / "03_no_moving_recourse" / "main_results.csv",
        output / "05_graph_stability" / "main_results.csv",
    ):
        frame = pd.read_csv(path)
        if "total_constraint_violation_count" in frame:
            violations += int(frame["total_constraint_violation_count"].sum())
        if "max_action_violation" in frame:
            if float(frame["max_action_violation"].max()) != 0.0:
                raise RuntimeError(f"Nonzero action violation in {path}.")
        if "max_storage_violation" in frame:
            if float(frame["max_storage_violation"].max()) != 0.0:
                raise RuntimeError(f"Nonzero storage violation in {path}.")
    if violations:
        raise RuntimeError("A formal mechanism experiment violated a hard constraint.")
    write_completion(
        output,
        "gradient_mechanisms",
        {
            "n_train_seeds": len(study.train_seeds),
            "n_experiments": 5,
            "all_constraint_violations_zero": True,
        },
    )
    return output
