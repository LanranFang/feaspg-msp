"""Shared differentiable simulator for all efficiency-switching policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .data import Dataset
from .physics import generation_torch
from .projection import action_bounds, project_action


GRADIENT_MODES = (
    "full",
    "no_proposal_state",
    "no_moving_recourse",
)


@dataclass
class Trace:
    total_cost: torch.Tensor
    mean_cost: torch.Tensor
    stage_cost: torch.Tensor
    terminal_cost: torch.Tensor
    actions: torch.Tensor
    proposals: torch.Tensor
    storage_before_action: torch.Tensor
    previous_release_before_action: torch.Tensor
    queue_before_action: torch.Tensor
    storage: torch.Tensor
    queues: torch.Tensor
    spill: torch.Tensor
    evaporation: torch.Tensor
    hydro_generation: torch.Tensor
    thermal_generation: torch.Tensor
    peaker_generation: torch.Tensor
    thermal_cost: torch.Tensor
    ramp_cost: torch.Tensor
    storage_penalty: torch.Tensor
    config: ExperimentConfig

    @property
    def max_action_violation(self) -> torch.Tensor:
        lower, upper = action_bounds(
            self.storage_before_action,
            self.previous_release_before_action,
            self.config,
        )
        weights = torch.as_tensor(
            self.config.generation_coefficient,
            dtype=self.actions.dtype,
            device=self.actions.device,
        )
        return torch.stack(
            (
                torch.relu(lower - self.actions).max(),
                torch.relu(self.actions - upper).max(),
                torch.relu(
                    (self.actions * weights).sum(dim=-1)
                    - self.config.shared_release_capacity
                ).max(),
            )
        ).max()

    @property
    def max_storage_violation(self) -> torch.Tensor:
        capacity = torch.as_tensor(
            self.config.storage_capacity,
            dtype=self.storage.dtype,
            device=self.storage.device,
        )
        return torch.maximum(
            torch.relu(-self.storage).max(),
            torch.relu(self.storage - capacity).max(),
        )


def simulate(
    policy,
    data: Dataset | dict[str, torch.Tensor],
    config: ExperimentConfig,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    gradient_mode: str = "full",
    detach_state_every: int | None = None,
) -> Trace:
    if gradient_mode not in GRADIENT_MODES:
        raise ValueError(
            f"Unknown gradient mode {gradient_mode!r}; expected one of {GRADIENT_MODES}."
        )
    if detach_state_every is not None and detach_state_every < 1:
        raise ValueError("detach_state_every must be positive when provided.")
    batch = data.to_torch(device=device, dtype=dtype) if isinstance(data, Dataset) else data
    inflow = batch["inflow"].to(device=device, dtype=dtype)
    demand = batch["demand"].to(device=device, dtype=dtype)
    price = batch["price"].to(device=device, dtype=dtype)
    n = inflow.shape[0]
    storage = torch.as_tensor(
        config.initial_storage, dtype=dtype, device=device
    ).expand(n, -1).clone()
    previous_release = torch.zeros(
        n, config.n_reservoirs, dtype=dtype, device=device
    )
    queue = torch.zeros(
        n,
        config.routing_delay,
        config.n_reservoirs,
        dtype=dtype,
        device=device,
    )
    capacity = torch.as_tensor(
        config.storage_capacity, dtype=dtype, device=device
    )
    release_capacity = torch.as_tensor(
        config.release_capacity, dtype=dtype, device=device
    )
    routing = torch.as_tensor(config.routing, dtype=dtype, device=device)
    evaporation_rate = torch.as_tensor(
        config.evaporation_rate, dtype=dtype, device=device
    )
    spill_coeff = torch.as_tensor(config.spill_cost, dtype=dtype, device=device)
    flood_threshold = torch.as_tensor(
        config.flood_threshold, dtype=dtype, device=device
    )
    low_threshold = torch.as_tensor(
        config.low_storage_threshold, dtype=dtype, device=device
    )

    actions = []
    proposals = []
    storages_before = []
    previous_before = []
    queues_before = []
    storages = []
    queues = []
    spills = []
    evaporations = []
    hydro_generation = []
    thermal_generation = []
    peaker_generation = []
    thermal_costs = []
    ramp_costs = []
    storage_penalties = []
    stage_costs = []

    for stage in range(config.T):
        if (
            detach_state_every is not None
            and stage > 0
            and stage % detach_state_every == 0
        ):
            storage = storage.detach()
            previous_release = previous_release.detach()
            queue = queue.detach()
        storages_before.append(storage)
        previous_before.append(previous_release)
        queues_before.append(queue)
        if gradient_mode == "no_proposal_state":
            proposal_storage = storage.detach()
            proposal_previous_release = previous_release.detach()
            proposal_queue = queue.detach()
        else:
            proposal_storage = storage
            proposal_previous_release = previous_release
            proposal_queue = queue
        proposal = policy.propose(
            stage,
            proposal_storage,
            proposal_previous_release,
            proposal_queue,
        )
        if gradient_mode == "no_moving_recourse":
            projection_storage = storage.detach()
            projection_previous_release = previous_release.detach()
        else:
            projection_storage = storage
            projection_previous_release = previous_release
        action = project_action(
            proposal,
            projection_storage,
            projection_previous_release,
            config,
        )
        generated = generation_torch(storage, action, config).sum(dim=-1)
        thermal = torch.relu(demand[:, stage] - generated)
        peaker = torch.relu(thermal - config.peaker_threshold)
        post_release = torch.clamp(storage - action, min=0.0)
        evaporation = evaporation_rate * post_release.square() / capacity
        routed = action @ routing
        if config.routing_delay == 0:
            arrival = routed
        else:
            arrival = queue[:, 0, :]
            queue = torch.cat((queue[:, 1:, :], routed[:, None, :]), dim=1)
        raw_storage = post_release - evaporation + inflow[:, stage, :] + arrival
        spill = torch.relu(raw_storage - capacity)
        storage = torch.clamp(raw_storage, min=0.0)
        storage = torch.minimum(storage, capacity)
        ramp = ((action - previous_release) / release_capacity).square().sum(dim=-1)
        flood = torch.relu(storage - flood_threshold)
        low = torch.relu(low_threshold - storage)
        thermal_cost = (
            price[:, stage] * thermal
            + config.thermal_quadratic_cost * thermal.square()
            + config.peaker_quadratic_cost * peaker.square()
        )
        ramp_cost = config.ramp_cost * ramp
        storage_penalty = (
            config.flood_cost * flood.square().sum(dim=-1)
            + config.low_storage_cost * low.square().sum(dim=-1)
        )
        stage_cost = (
            thermal_cost
            + (spill_coeff * spill).sum(dim=-1)
            + ramp_cost
            + storage_penalty
        )
        previous_release = action
        actions.append(action)
        proposals.append(proposal)
        storages.append(storage)
        queues.append(queue)
        spills.append(spill)
        evaporations.append(evaporation)
        hydro_generation.append(generated)
        thermal_generation.append(thermal)
        peaker_generation.append(peaker)
        thermal_costs.append(thermal_cost)
        ramp_costs.append(ramp_cost)
        storage_penalties.append(storage_penalty)
        stage_costs.append(stage_cost)

    terminal_water = storage + (queue.sum(dim=1) if config.routing_delay else 0.0)
    terminal_target = torch.as_tensor(
        config.terminal_target, dtype=dtype, device=device
    )
    terminal_value = torch.as_tensor(
        config.terminal_water_value, dtype=dtype, device=device
    )
    terminal_cost = (
        terminal_value * torch.relu(terminal_target - terminal_water)
    ).sum(dim=-1)
    stage_cost = torch.stack(stage_costs, dim=1)
    total_cost = stage_cost.sum(dim=1) + terminal_cost
    queue_trace = (
        torch.stack(queues, dim=1)
        if config.routing_delay
        else torch.empty(n, config.T, 0, config.n_reservoirs, device=device)
    )
    return Trace(
        total_cost=total_cost,
        mean_cost=total_cost.mean(),
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        actions=torch.stack(actions, dim=1),
        proposals=torch.stack(proposals, dim=1),
        storage_before_action=torch.stack(storages_before, dim=1),
        previous_release_before_action=torch.stack(previous_before, dim=1),
        queue_before_action=(
            torch.stack(queues_before, dim=1)
            if config.routing_delay
            else torch.empty(
                n,
                config.T,
                0,
                config.n_reservoirs,
                dtype=dtype,
                device=device,
            )
        ),
        storage=torch.stack(storages, dim=1),
        queues=queue_trace,
        spill=torch.stack(spills, dim=1),
        evaporation=torch.stack(evaporations, dim=1),
        hydro_generation=torch.stack(hydro_generation, dim=1),
        thermal_generation=torch.stack(thermal_generation, dim=1),
        peaker_generation=torch.stack(peaker_generation, dim=1),
        thermal_cost=torch.stack(thermal_costs, dim=1),
        ramp_cost=torch.stack(ramp_costs, dim=1),
        storage_penalty=torch.stack(storage_penalties, dim=1),
        config=config,
    )
