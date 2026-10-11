"""Explicit-block evaluation for efficiency-switching hydro policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig
from .data import Dataset
from .simulator import simulate


@dataclass(frozen=True)
class Metrics:
    block: str
    n: int
    mean_cost: float
    std_cost: float
    se_cost: float
    mean_hydro_generation: float
    mean_thermal_generation: float
    mean_peaker_generation: float
    mean_thermal_cost: float
    mean_spill: float
    mean_evaporation: float
    mean_ramp_cost: float
    mean_storage_penalty: float
    mean_terminal_cost: float
    mean_terminal_storage: float
    max_action_violation: float
    max_storage_violation: float


@torch.no_grad()
def evaluate_with_costs(
    policy,
    dataset: Dataset,
    config: ExperimentConfig,
    device: str = "cpu",
) -> tuple[Metrics, np.ndarray]:
    policy.eval()
    trace = simulate(policy, dataset, config, device=device)
    costs = trace.total_cost.detach().cpu()
    std = float(costs.std(unbiased=True)) if costs.numel() > 1 else 0.0
    metrics = Metrics(
        block=dataset.name,
        n=dataset.n,
        mean_cost=float(costs.mean()),
        std_cost=std,
        se_cost=std / max(dataset.n, 1) ** 0.5,
        mean_hydro_generation=float(trace.hydro_generation.mean()),
        mean_thermal_generation=float(trace.thermal_generation.mean()),
        mean_peaker_generation=float(trace.peaker_generation.mean()),
        mean_thermal_cost=float(trace.thermal_cost.mean()),
        mean_spill=float(trace.spill.sum(dim=-1).mean()),
        mean_evaporation=float(trace.evaporation.sum(dim=-1).mean()),
        mean_ramp_cost=float(trace.ramp_cost.mean()),
        mean_storage_penalty=float(trace.storage_penalty.mean()),
        mean_terminal_cost=float(trace.terminal_cost.mean()),
        mean_terminal_storage=float(trace.storage[:, -1, :].mean()),
        max_action_violation=float(trace.max_action_violation),
        max_storage_violation=float(trace.max_storage_violation),
    )
    return metrics, costs.numpy().astype(np.float64)


def flatten(prefix: str, metrics: Metrics) -> dict[str, float | int]:
    return {
        f"{prefix}_n": metrics.n,
        f"{prefix}_mean_cost": metrics.mean_cost,
        f"{prefix}_std_cost": metrics.std_cost,
        f"{prefix}_se_cost": metrics.se_cost,
        f"{prefix}_mean_hydro_generation": metrics.mean_hydro_generation,
        f"{prefix}_mean_thermal_generation": metrics.mean_thermal_generation,
        f"{prefix}_mean_peaker_generation": metrics.mean_peaker_generation,
        f"{prefix}_mean_thermal_cost": metrics.mean_thermal_cost,
        f"{prefix}_mean_spill": metrics.mean_spill,
        f"{prefix}_mean_evaporation": metrics.mean_evaporation,
        f"{prefix}_mean_ramp_cost": metrics.mean_ramp_cost,
        f"{prefix}_mean_storage_penalty": metrics.mean_storage_penalty,
        f"{prefix}_mean_terminal_cost": metrics.mean_terminal_cost,
        f"{prefix}_mean_terminal_storage": metrics.mean_terminal_storage,
        f"{prefix}_max_action_violation": metrics.max_action_violation,
        f"{prefix}_max_storage_violation": metrics.max_storage_violation,
    }
