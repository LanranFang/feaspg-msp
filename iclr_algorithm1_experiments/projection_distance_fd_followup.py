"""Explicit post-hoc small-step diagnosis of the predeclared FD outlier.

This preserves the primary seven-step results and their source hashes. The
case is selected because it has the largest complete-derivative error in the
positive-lambda arm at the predeclared smallest step, not to replace that error.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from algorithm1_gradient_study.experiments import _base_config
from algorithm1_gradient_study.models import build_policy, initialize_channel_policy
from efficiency_switching_hydro.data import generate_dataset
from efficiency_switching_hydro.projection import action_bounds
from efficiency_switching_hydro.simulator import simulate
from .config import ICLRProtocol
from .projection_distance_mechanisms import regularized_finite_difference_rows


def _signature(policy, dataset, hydro):
    relu_inputs = []
    hooks = [module.register_forward_pre_hook(
        lambda module, arguments: relu_inputs.append(arguments[0].detach().reshape(-1).clone())
    ) for module in policy.modules() if isinstance(module, torch.nn.ReLU)]
    try:
        with torch.no_grad():
            trace = simulate(policy, dataset, hydro, dtype=torch.float64)
    finally:
        for hook in hooks:
            hook.remove()
    lower, upper = action_bounds(trace.storage_before_action, trace.previous_release_before_action, hydro)
    capacity = torch.tensor(hydro.storage_capacity, dtype=torch.float64)
    return {
        "relu": torch.cat(relu_inputs) > 0,
        "at_lower": (trace.actions - lower).abs() <= 1e-9,
        "at_upper": (trace.actions - upper).abs() <= 1e-9,
        "storage_empty": trace.storage <= 0,
        "storage_full": trace.storage >= capacity,
        "spill_positive": trace.spill > 0,
        "thermal_positive": trace.thermal_generation > 0,
        "peaker_positive": trace.peaker_generation > 0,
    }


def main():
    torch.set_num_threads(1)
    base_output = Path(__file__).parent / "outputs/projection_distance_v1"
    primary = base_output / "mechanisms"
    output = base_output / "mechanisms_fd_followup"
    output.mkdir(exist_ok=True)
    if (output / "COMPLETED.json").exists():
        print("Follow-up already completed.")
        return
    original = pd.read_csv(primary / "finite_difference_check.csv")
    chosen = original[(original.lambda_arm == "lambda_ref") & (original.gradient_mode == "full") & (original.step == 1e-5)].sort_values("relative_error", ascending=False).iloc[0]
    seed, direction_seed = int(chosen.train_seed), int(chosen.direction_seed)
    steps = (1e-5, 3e-6, 1e-6, 3e-7)
    settings = {
        "diagnostic_status": "post-hoc follow-up; primary seven-step results remain unchanged",
        "selection": "largest positive-lambda complete-derivative error at primary h=1e-5",
        "train_seed": seed, "direction_seed": direction_seed,
        "lambda_ref": float(chosen.lambda_value), "steps": steps,
        "primary_relative_error": float(chosen.relative_error),
        "signature_at_bound_tolerance": 1e-9,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_results_sha256": hashlib.sha256((primary / "finite_difference_check.csv").read_bytes()).hexdigest(),
    }
    (output / "PROTOCOL.json").write_text(json.dumps(settings, indent=2) + "\n")
    study = ICLRProtocol().study_protocol()
    hydro = _base_config(study)
    dataset = generate_dataset("fit_gradient_diagnostic", 32, study.fit_seed, hydro)
    torch.manual_seed(seed)
    policy = build_policy("mlp", hydro, study).double()
    initialize_channel_policy(policy, 10_000 + seed)
    rows = []
    for arm, coefficient in (("zero", 0.0), ("lambda_ref", float(chosen.lambda_value))):
        current = regularized_finite_difference_rows(policy, dataset, hydro, steps, coefficient, [direction_seed])
        for row in current:
            row.update({"lambda_arm": arm, "train_seed": seed})
        rows.extend(current)
    pd.DataFrame(rows).to_csv(output / "finite_difference_followup.csv", index=False)
    parameters = tuple(p for p in policy.parameters() if p.requires_grad)
    base = parameters_to_vector(parameters).detach().clone()
    generator = torch.Generator().manual_seed(direction_seed)
    direction = torch.randn(base.shape, dtype=base.dtype, generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    signature_base = _signature(policy, dataset, hydro)
    signatures = []
    try:
        for step in steps:
            for sign in (-1, 1):
                with torch.no_grad():
                    vector_to_parameters(base + sign * step * direction, parameters)
                current = _signature(policy, dataset, hydro)
                signatures.append({"step": step, "perturbation_sign": sign, **{
                    name + "_changed": int((value != signature_base[name]).sum())
                    for name, value in current.items()
                }})
    finally:
        with torch.no_grad():
            vector_to_parameters(base, parameters)
    pd.DataFrame(signatures).to_csv(output / "branch_signatures.csv", index=False)
    (output / "COMPLETED.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file()},
    }, indent=2) + "\n")
    print(pd.DataFrame(rows).query("lambda_arm=='lambda_ref' and gradient_mode=='full'")[["step", "relative_error"]].to_string(index=False))
    print(pd.DataFrame(signatures).to_string(index=False))


if __name__ == "__main__":
    main()
