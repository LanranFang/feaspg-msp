"""Efficiency-switching hydro physics shared by simulation and SDDP."""

from __future__ import annotations

import numpy as np
import torch

from .config import ExperimentConfig


def _downstream_storage_torch(
    storage: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    index = torch.as_tensor(config.downstream_index, device=storage.device)
    safe = torch.clamp(index, min=0)
    downstream = storage[..., safe]
    return torch.where(index >= 0, downstream, torch.zeros_like(downstream))


def _downstream_storage_np(
    storage: np.ndarray, config: ExperimentConfig
) -> np.ndarray:
    index = np.asarray(config.downstream_index, dtype=np.int64)
    downstream = np.zeros_like(storage)
    mask = index >= 0
    downstream[..., mask] = storage[..., index[mask]]
    return downstream


def turbine_efficiency_np(
    storage: np.ndarray,
    action: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    storage = np.asarray(storage, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    capacity = np.asarray(config.storage_capacity, dtype=np.float64)
    release = np.asarray(config.release_capacity, dtype=np.float64)
    storage_fraction = np.clip(storage / capacity, 0.0, 1.0)
    release_fraction = np.clip(action / release, 0.0, 1.0)
    center1 = np.asarray(config.band1_center) + np.asarray(
        config.band1_storage_shift
    ) * (1.0 - storage_fraction)
    center2 = np.asarray(config.band2_center) + np.asarray(
        config.band2_storage_shift
    ) * (1.0 - storage_fraction)
    band1 = np.asarray(config.band1_amplitude) * np.exp(
        -np.square((release_fraction - center1) / np.asarray(config.band1_width))
    )
    band2 = np.asarray(config.band2_amplitude) * np.exp(
        -np.square((release_fraction - center2) / np.asarray(config.band2_width))
    )
    return np.asarray(config.efficiency_floor) + band1 + band2


def generation_torch(
    storage: torch.Tensor,
    action: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    dtype = storage.dtype
    device = storage.device

    def tensor(values):
        return torch.as_tensor(values, dtype=dtype, device=device)

    capacity = tensor(config.storage_capacity)
    release = tensor(config.release_capacity)
    coefficient = tensor(config.generation_coefficient)
    storage_fraction = torch.clamp(storage / capacity, 0.0, 1.0)
    release_fraction = torch.clamp(action / release, 0.0, 1.0)
    downstream_index = torch.as_tensor(config.downstream_index, device=device)
    downstream_capacity = capacity[torch.clamp(downstream_index, min=0)]
    downstream_fraction = torch.where(
        downstream_index >= 0,
        _downstream_storage_torch(storage, config) / downstream_capacity,
        torch.zeros_like(storage),
    )
    head = (
        tensor(config.head_base)
        + tensor(config.head_storage_gain) * storage_fraction
        - tensor(config.head_downstream_loss) * downstream_fraction
        - tensor(config.head_release_loss) * release_fraction.square()
    )
    center1 = tensor(config.band1_center) + tensor(
        config.band1_storage_shift
    ) * (1.0 - storage_fraction)
    center2 = tensor(config.band2_center) + tensor(
        config.band2_storage_shift
    ) * (1.0 - storage_fraction)
    band1 = tensor(config.band1_amplitude) * torch.exp(
        -torch.square((release_fraction - center1) / tensor(config.band1_width))
    )
    band2 = tensor(config.band2_amplitude) * torch.exp(
        -torch.square((release_fraction - center2) / tensor(config.band2_width))
    )
    efficiency = tensor(config.efficiency_floor) + band1 + band2
    return coefficient * action * torch.clamp(head, min=0.10) * efficiency


def generation_np(
    storage: np.ndarray,
    action: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    storage = np.asarray(storage, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    capacity = np.asarray(config.storage_capacity, dtype=np.float64)
    release = np.asarray(config.release_capacity, dtype=np.float64)
    coefficient = np.asarray(config.generation_coefficient, dtype=np.float64)
    storage_fraction = np.clip(storage / capacity, 0.0, 1.0)
    release_fraction = np.clip(action / release, 0.0, 1.0)
    downstream_index = np.asarray(config.downstream_index, dtype=np.int64)
    downstream_capacity = capacity[np.maximum(downstream_index, 0)]
    downstream_fraction = np.where(
        downstream_index >= 0,
        _downstream_storage_np(storage, config) / downstream_capacity,
        0.0,
    )
    head = (
        np.asarray(config.head_base)
        + np.asarray(config.head_storage_gain) * storage_fraction
        - np.asarray(config.head_downstream_loss) * downstream_fraction
        - np.asarray(config.head_release_loss) * np.square(release_fraction)
    )
    efficiency = turbine_efficiency_np(storage, action, config)
    return coefficient * action * np.maximum(head, 0.10) * efficiency


def total_generation_linearization(
    storage: np.ndarray,
    action: np.ndarray,
    config: ExperimentConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    storage = np.asarray(storage, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    value = float(generation_np(storage, action, config).sum())
    storage_gradient = np.empty(config.n_reservoirs, dtype=np.float64)
    action_gradient = np.empty(config.n_reservoirs, dtype=np.float64)
    capacity = np.asarray(config.storage_capacity, dtype=np.float64)
    release = np.asarray(config.release_capacity, dtype=np.float64)
    for index in range(config.n_reservoirs):
        step = 1e-4 * capacity[index]
        plus = storage.copy()
        minus = storage.copy()
        plus[index] = min(capacity[index], plus[index] + step)
        minus[index] = max(0.0, minus[index] - step)
        denominator = plus[index] - minus[index]
        storage_gradient[index] = (
            generation_np(plus, action, config).sum()
            - generation_np(minus, action, config).sum()
        ) / max(denominator, 1e-12)

        step = 1e-4 * release[index]
        plus_action = action.copy()
        minus_action = action.copy()
        plus_action[index] = min(release[index], plus_action[index] + step)
        minus_action[index] = max(0.0, minus_action[index] - step)
        denominator = plus_action[index] - minus_action[index]
        action_gradient[index] = (
            generation_np(storage, plus_action, config).sum()
            - generation_np(storage, minus_action, config).sum()
        ) / max(denominator, 1e-12)
    return value, storage_gradient, action_gradient


def transition_np(
    storage: np.ndarray,
    previous_release: np.ndarray,
    queue: np.ndarray,
    action: np.ndarray,
    inflow: np.ndarray,
    demand: float,
    price: float,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | np.ndarray]]:
    capacity = np.asarray(config.storage_capacity, dtype=np.float64)
    release = np.asarray(config.release_capacity, dtype=np.float64)
    routing = np.asarray(config.routing, dtype=np.float64)
    evaporation_rate = np.asarray(config.evaporation_rate, dtype=np.float64)
    post_release = np.maximum(storage - action, 0.0)
    evaporation = evaporation_rate * np.square(post_release) / capacity
    routed = action @ routing
    if config.routing_delay == 0:
        arrival = routed
        next_queue = np.zeros((0, config.n_reservoirs), dtype=np.float64)
    else:
        arrival = queue[0]
        next_queue = np.concatenate((queue[1:], routed[None, :]), axis=0)
    raw_storage = post_release - evaporation + inflow + arrival
    spill = np.maximum(raw_storage - capacity, 0.0)
    next_storage = np.clip(raw_storage, 0.0, capacity)
    generated = float(generation_np(storage, action, config).sum())
    thermal = max(0.0, float(demand) - generated)
    peaker = max(0.0, thermal - config.peaker_threshold)
    ramp = float(
        np.square((action - previous_release) / np.maximum(release, 1e-12)).sum()
    )
    flood = np.maximum(next_storage - np.asarray(config.flood_threshold), 0.0)
    low = np.maximum(np.asarray(config.low_storage_threshold) - next_storage, 0.0)
    components = {
        "thermal": float(price) * thermal
        + config.thermal_quadratic_cost * thermal * thermal
        + config.peaker_quadratic_cost * peaker * peaker,
        "spill": float(np.asarray(config.spill_cost) @ spill),
        "ramp": config.ramp_cost * ramp,
        "storage_penalty": config.flood_cost * float(flood @ flood)
        + config.low_storage_cost * float(low @ low),
        "evaporation": evaporation,
        "generation": generated,
        "thermal_generation": thermal,
        "peaker_generation": peaker,
        "spill_volume": spill,
    }
    return next_storage, action.copy(), next_queue, components
