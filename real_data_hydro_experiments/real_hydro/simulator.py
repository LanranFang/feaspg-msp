"""One differentiable simulator shared by direct gradients, SHAC, RL, and SDDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .config import ReservoirSystem
from .dataset import TrajectoryBatch
from .projection import project_release, proposal_violation


@dataclass
class SimulationTrace:
    total_cost: torch.Tensor
    mean_cost: torch.Tensor
    stage_cost: torch.Tensor
    terminal_cost: torch.Tensor
    actions: torch.Tensor
    proposals: torch.Tensor
    storage_before_action: torch.Tensor
    storage: torch.Tensor
    previous_release_before_action: torch.Tensor
    spill: torch.Tensor
    shortage: torch.Tensor
    excess_release: torch.Tensor
    low_storage: torch.Tensor
    flood_storage: torch.Tensor
    ramp: torch.Tensor
    thermal_generation: torch.Tensor
    deficit: torch.Tensor
    raw_proposal_violation: torch.Tensor
    policy_features: torch.Tensor
    system: ReservoirSystem

    @property
    def max_action_violation(self) -> torch.Tensor:
        release_capacity = torch.as_tensor(
            self.system.release_capacity,
            dtype=self.actions.dtype,
            device=self.actions.device,
        )
        violations = torch.stack(
            (
                torch.relu(-self.actions).amax(),
                torch.relu(self.actions - release_capacity).amax(),
                torch.relu(self.actions - self.storage_before_action).amax(),
            )
        )
        if self.system.shared_release_capacity is not None:
            violations = torch.cat(
                (
                    violations,
                    torch.relu(
                        self.actions.sum(dim=-1)
                        - float(self.system.shared_release_capacity)
                    ).amax().view(1),
                )
            )
        return violations.amax()

    @property
    def max_storage_violation(self) -> torch.Tensor:
        capacity = torch.as_tensor(
            self.system.storage_capacity,
            dtype=self.storage.dtype,
            device=self.storage.device,
        )
        return torch.maximum(
            torch.relu(-self.storage).amax(),
            torch.relu(self.storage - capacity).amax(),
        )


def thermal_dispatch_cost(
    net_demand: torch.Tensor,
    block_capacity: torch.Tensor,
    block_cost: torch.Tensor,
    deficit_cost: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merit-order dispatch for convex piecewise-linear thermal supply."""

    remaining = torch.relu(net_demand)
    total_cost = torch.zeros_like(remaining)
    generation = torch.zeros_like(remaining)
    for capacity, marginal_cost in zip(block_capacity, block_cost, strict=True):
        used = torch.minimum(remaining, capacity)
        generation = generation + used
        total_cost = total_cost + used * marginal_cost
        remaining = remaining - used
    deficit = remaining
    total_cost = total_cost + float(deficit_cost) * deficit
    return total_cost, generation, deficit


def _tensor_batch(
    batch: TrajectoryBatch | Mapping[str, torch.Tensor],
    device: str | torch.device,
    dtype: torch.dtype,
    n_regimes: int | None,
) -> Mapping[str, torch.Tensor]:
    if isinstance(batch, TrajectoryBatch):
        return batch.to_torch(device=device, dtype=dtype, n_regimes=n_regimes)
    return {key: value.to(device=device) for key, value in batch.items()}


def simulate(
    policy,
    batch: TrajectoryBatch | Mapping[str, torch.Tensor],
    system: ReservoirSystem,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    n_regimes: int | None = None,
    detach_state_every: int | None = None,
) -> SimulationTrace:
    """Simulate the preregistered decision-before-current-innovation timing."""

    tensors = _tensor_batch(batch, device, dtype, n_regimes)
    inflow = tensors["inflow"].to(dtype=dtype)
    features = tensors["policy_features"].to(dtype=dtype)
    if inflow.shape[1:] != (system.horizon, system.n_reservoirs):
        raise ValueError("Trajectory dimensions do not match the reservoir system.")
    n = inflow.shape[0]
    capacity = torch.as_tensor(system.storage_capacity, dtype=dtype, device=device)
    routing = torch.as_tensor(system.routing, dtype=dtype, device=device)
    initial = torch.as_tensor(system.initial_storage, dtype=dtype, device=device)
    storage = initial.expand(n, -1).clone()
    previous_release = torch.zeros_like(storage)

    objective = system.objective
    release_target = torch.as_tensor(objective.release_target, dtype=dtype, device=device)
    guide_lower = torch.as_tensor(objective.guide_lower, dtype=dtype, device=device)
    guide_upper = torch.as_tensor(objective.guide_upper, dtype=dtype, device=device)
    weights = {
        name: torch.as_tensor(getattr(objective, name), dtype=dtype, device=device)
        for name in (
            "shortage_cost",
            "excess_release_cost",
            "low_storage_cost",
            "flood_cost",
            "spill_cost",
            "ramp_cost",
        )
    }
    if objective.kind == "hydrothermal":
        generation_coefficient = torch.as_tensor(
            objective.generation_coefficient, dtype=dtype, device=device
        )
        thermal_capacity = torch.as_tensor(
            objective.thermal_block_capacity, dtype=dtype, device=device
        )
        thermal_cost = torch.as_tensor(
            objective.thermal_block_cost, dtype=dtype, device=device
        )
    else:
        generation_coefficient = torch.zeros(system.n_reservoirs, dtype=dtype, device=device)
        thermal_capacity = torch.empty(0, dtype=dtype, device=device)
        thermal_cost = torch.empty(0, dtype=dtype, device=device)

    records: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "actions",
            "proposals",
            "storage_before",
            "storage",
            "previous_release",
            "spill",
            "shortage",
            "excess",
            "low",
            "flood",
            "ramp",
            "thermal",
            "deficit",
            "raw_violation",
            "stage_cost",
        )
    }

    for stage in range(system.horizon):
        if detach_state_every and stage and stage % detach_state_every == 0:
            storage = storage.detach()
            previous_release = previous_release.detach()
        period = system.period(stage)
        records["storage_before"].append(storage)
        records["previous_release"].append(previous_release)
        proposal = policy.propose(
            stage, storage, previous_release, features[:, stage]
        )
        if proposal.shape != storage.shape:
            raise ValueError("Policy proposal has the wrong action shape.")
        records["raw_violation"].append(
            proposal_violation(proposal, storage, system)
        )
        action = project_release(proposal, storage, system)

        raw_storage = storage - action + inflow[:, stage] + action @ routing
        spill = torch.relu(raw_storage - capacity)
        storage = torch.minimum(raw_storage, capacity)
        shortage = torch.relu(release_target[period] - action)
        excess = torch.relu(action - release_target[period])
        low = torch.relu(guide_lower[period] - storage)
        flood = torch.relu(storage - guide_upper[period])
        ramp = torch.abs(action - previous_release)

        stage_cost = (
            shortage * weights["shortage_cost"]
            + excess * weights["excess_release_cost"]
            + low * weights["low_storage_cost"]
            + flood * weights["flood_cost"]
            + spill * weights["spill_cost"]
            + ramp * weights["ramp_cost"]
        ).sum(dim=-1)
        if objective.kind == "hydrothermal":
            if "demand" in tensors:
                demand = tensors["demand"][:, stage].to(dtype=dtype)
            else:
                demand = torch.full(
                    (n,), float(objective.seasonal_demand[period]), dtype=dtype, device=device
                )
            hydro_generation = (action * generation_coefficient).sum(dim=-1)
            thermal_stage_cost, thermal, deficit = thermal_dispatch_cost(
                demand - hydro_generation,
                thermal_capacity,
                thermal_cost,
                objective.deficit_cost,
            )
            stage_cost = stage_cost + thermal_stage_cost
        else:
            thermal = torch.zeros(n, dtype=dtype, device=device)
            deficit = torch.zeros(n, dtype=dtype, device=device)

        records["actions"].append(action)
        records["proposals"].append(proposal)
        records["storage"].append(storage)
        records["spill"].append(spill)
        records["shortage"].append(shortage)
        records["excess"].append(excess)
        records["low"].append(low)
        records["flood"].append(flood)
        records["ramp"].append(ramp)
        records["thermal"].append(thermal)
        records["deficit"].append(deficit)
        records["stage_cost"].append(stage_cost)
        previous_release = action

    terminal_target = torch.as_tensor(
        objective.terminal_target, dtype=dtype, device=device
    )
    terminal_shortfall = torch.as_tensor(
        objective.terminal_shortfall_cost, dtype=dtype, device=device
    )
    terminal_excess = torch.as_tensor(
        objective.terminal_excess_cost, dtype=dtype, device=device
    )
    terminal_cost = (
        terminal_shortfall * torch.relu(terminal_target - storage)
        + terminal_excess * torch.relu(storage - terminal_target)
    ).sum(dim=-1)
    stage_cost = torch.stack(records["stage_cost"], dim=1)
    total_cost = stage_cost.sum(dim=1) + terminal_cost
    return SimulationTrace(
        total_cost=total_cost,
        mean_cost=total_cost.mean(),
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        actions=torch.stack(records["actions"], dim=1),
        proposals=torch.stack(records["proposals"], dim=1),
        storage_before_action=torch.stack(records["storage_before"], dim=1),
        storage=torch.stack(records["storage"], dim=1),
        previous_release_before_action=torch.stack(
            records["previous_release"], dim=1
        ),
        spill=torch.stack(records["spill"], dim=1),
        shortage=torch.stack(records["shortage"], dim=1),
        excess_release=torch.stack(records["excess"], dim=1),
        low_storage=torch.stack(records["low"], dim=1),
        flood_storage=torch.stack(records["flood"], dim=1),
        ramp=torch.stack(records["ramp"], dim=1),
        thermal_generation=torch.stack(records["thermal"], dim=1),
        deficit=torch.stack(records["deficit"], dim=1),
        raw_proposal_violation=torch.stack(records["raw_violation"], dim=1),
        policy_features=features,
        system=system,
    )
