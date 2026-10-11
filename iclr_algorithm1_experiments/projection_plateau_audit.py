"""Independent checks and numerical derivative diagnostic for the stress test."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efficiency_switching_hydro.simulator import simulate
from efficiency_switching_hydro.train import unused_batches
from .config import ICLRProtocol
from .projection_distance import projection_distance
from .projection_distance_mechanisms import regularized_finite_difference_rows
from .projection_distance_run import configure, initial_policy, save_json, state_hash
from .projection_plateau import DEFAULT_OUTPUT, fitting_data


def main():
    configure()
    p = ICLRProtocol()
    h, fit = fitting_data("drought_02")
    monitor = fit.take(np.arange(32), "audit_fit32")
    out = DEFAULT_OUTPUT / "audit"
    out.mkdir(parents=True, exist_ok=True)
    actor = initial_policy(p, h, 0, "projected_mlp_matched")
    initial_state = copy.deepcopy(actor.state_dict())
    with torch.no_grad():
        initial = simulate(actor, monitor, h)
    perturbations = []
    # Large enough changes in proposals to exclude mere rounding of a tiny
    # numerical cost change; all remain within the certified clipping region.
    for bias_change in (-0.05, -0.01, 0.01, 0.05):
        actor.load_state_dict(initial_state)
        with torch.no_grad():
            actor.actor[-1].bias.add_(bias_change)
            trace = simulate(actor, monitor, h)
        perturbations.append({
            "output_bias_change": bias_change,
            "maximum_proposal_change": float((trace.proposals - initial.proposals).abs().max()),
            "maximum_action_change": float((trace.actions - initial.actions).abs().max()),
            "maximum_operating_cost_change": float((trace.total_cost - initial.total_cost).abs().max()),
            "mean_proposal_distance": float(projection_distance(trace).mean()),
        })
        assert torch.equal(trace.actions, initial.actions)
        assert torch.equal(trace.total_cost, initial.total_cost)
    pd.DataFrame(perturbations).to_csv(out / "plateau_perturbations.csv", index=False)

    # Independently replay the first 30 updates with the auxiliary formula
    # written directly, instead of invoking the fitting loss wrapper.
    actor.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(actor.parameters(), lr=6e-4, weight_decay=p.weight_decay)
    batches = unused_batches(1000, 500, 2, np.random.default_rng(0))
    replay_rows = []
    first_change = None
    first_cost_gradient = None
    for update in range(1, 31):
        optimizer.zero_grad(set_to_none=True)
        trace = simulate(actor, fit.take(batches[update - 1]), h)
        loss = trace.mean_cost + .5 * (trace.proposals - trace.actions).square().sum((1, 2)).mean()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 10))
        replay_rows.append({"update": update, "cost": float(trace.mean_cost.detach()),
                            "objective": float(loss.detach()), "gradient_norm_preclip": norm})
        optimizer.step()
        diagnostic = simulate(actor, monitor, h)
        gradients = torch.autograd.grad(diagnostic.mean_cost, tuple(actor.parameters()))
        cost_norm = float(sum(g.detach().double().square().sum() for g in gradients).sqrt())
        action_change = float((diagnostic.actions.detach() - initial.actions).abs().max())
        if first_change is None and action_change > 1e-6:
            first_change = update
        if first_cost_gradient is None and cost_norm > 0:
            first_cost_gradient = update
    original = pd.read_csv(DEFAULT_OUTPUT / "runs/drought_02/seed0/fixed_one/training.csv").iloc[:30]
    replay = pd.DataFrame(replay_rows)
    comparison = {}
    for key in ("cost", "objective", "gradient_norm_preclip"):
        error = float(np.max(np.abs(replay[key].to_numpy() - original[key].to_numpy())))
        comparison[key] = error
        assert np.allclose(replay[key], original[key], rtol=1e-14, atol=1e-10)
    replay.to_csv(out / "independent_replay_first30.csv", index=False)
    torch.save(actor.state_dict(), out / "seed0_fixed_one_update30.pt")
    actor.double()
    # Retain every direction and the complete original step-size grid.
    steps = p.study_protocol().finite_difference_steps
    rows = regularized_finite_difference_rows(
        actor, monitor, h, steps, 1.0, [20000, 21000, 22000], "cpu")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "finite_difference.csv", index=False)
    summary = frame.groupby(["gradient_mode", "step"]).relative_error.agg(["mean", "max"]).reset_index()
    summary.to_csv(out / "finite_difference_summary.csv", index=False)
    save_json(out / "checks.json", {
        "plateau_perturbations_preserve_cost_and_actions_exactly": True,
        "independent_first30_update_replay_max_abs_difference": comparison,
        "first_action_change_update_seed0_fixed_one": first_change,
        "first_nonzero_cost_gradient_update_seed0_fixed_one": first_cost_gradient,
        "fd_checkpoint": "Replayed seed0 fixed_one update30; mixed active projections, first32 fitting paths, float64",
        "fd_directions": [20000, 21000, 22000], "fd_steps": list(steps),
        "fd_full_omitted_moving_recourse_forward_difference_max": float(frame.max_forward_difference.max()),
        "note": "At the exact zero-output initialization, the proposal parameterization itself has a piecewise kink. Central differences for the distance term are therefore tested at an intermediate checkpoint; the initial cost plateau is checked by exact forward perturbations and support bounds.",
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
