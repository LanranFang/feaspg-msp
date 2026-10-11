"""Policy construction and graph transformations used by the study."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.policies import (
    GlobalGraphNeuralPolicy,
    MyopicPolicy,
    PathwiseMLPPolicy,
)

from .config import StudyProtocol


RESERVOIR_FIELDS = (
    "storage_capacity",
    "initial_storage",
    "terminal_target",
    "release_capacity",
    "generation_coefficient",
    "spill_cost",
    "terminal_water_value",
    "base_inflow",
    "inflow_amplitude",
    "inflow_phase",
    "head_base",
    "head_storage_gain",
    "head_downstream_loss",
    "head_release_loss",
    "efficiency_floor",
    "band1_amplitude",
    "band1_center",
    "band1_storage_shift",
    "band1_width",
    "band2_amplitude",
    "band2_center",
    "band2_storage_shift",
    "band2_width",
    "evaporation_rate",
    "flood_threshold",
    "low_storage_threshold",
)


def build_policy(name: str, config: ExperimentConfig, protocol: StudyProtocol):
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"myopic", "reference"}:
        return MyopicPolicy(config)
    if normalized in {"mlp", "pathwise_mlp", "projected_mlp_pathwise"}:
        return PathwiseMLPPolicy(
            config,
            width=protocol.mlp_width,
            depth=protocol.network_depth,
        )
    if normalized in {
        "graph",
        "global_graph",
        "projected_graph_neural_global",
    }:
        return GlobalGraphNeuralPolicy(config, width=protocol.graph_width)
    raise ValueError(f"Unknown study policy: {name}")


def copy_trainable_parameters(source, target) -> None:
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("Policies do not have matching trainable parameter names.")
    with torch.no_grad():
        for name, source_parameter in source_parameters.items():
            target_parameter = target_parameters[name]
            if source_parameter.shape != target_parameter.shape:
                raise ValueError(f"Parameter {name} has incompatible shapes.")
            target_parameter.copy_(source_parameter)


def perturb_parameters(policy, seed: int, scale: float = 0.02) -> None:
    """Move zero-output initializations away from policy kinks for diagnostics."""

    devices = []
    for parameter in policy.parameters():
        if parameter.is_cuda:
            devices = [parameter.device.index]
            break
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.add_(scale * torch.randn_like(parameter))


def initialize_channel_policy(
    policy,
    seed: int,
    normalized_bias: float = 0.0,
) -> None:
    """Create a common nondegenerate MLP start for gradient-channel studies."""

    if not hasattr(policy, "actor") or not isinstance(policy.actor[-1], nn.Linear):
        raise TypeError("The channel diagnostic initialization expects a pathwise MLP.")
    output = policy.actor[-1]
    devices = [output.weight.device.index] if output.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        scale = 0.08 / max(output.in_features, 1) ** 0.5
        with torch.no_grad():
            output.weight.normal_(mean=0.0, std=scale)
            output.bias.fill_(normalized_bias)
            output.bias.add_(0.01 * torch.randn_like(output.bias))


def permute_config(
    config: ExperimentConfig, permutation: np.ndarray
) -> ExperimentConfig:
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(config.n_reservoirs)):
        raise ValueError("permutation must contain every reservoir index once.")
    updates: dict[str, object] = {}
    for field_name in RESERVOIR_FIELDS:
        values = np.asarray(getattr(config, field_name), dtype=np.float64)
        updates[field_name] = tuple(float(value) for value in values[permutation])
    loadings = np.asarray(config.factor_loadings, dtype=np.float64)[permutation]
    updates["factor_loadings"] = tuple(
        tuple(float(value) for value in row) for row in loadings
    )
    routing = np.asarray(config.routing, dtype=np.float64)[
        np.ix_(permutation, permutation)
    ]
    updates["routing"] = tuple(
        tuple(float(value) for value in row) for row in routing
    )
    updates["downstream_index"] = _downstream_index(routing)
    return replace(config, **updates)


def permute_dataset(dataset: Dataset, permutation: np.ndarray) -> Dataset:
    permutation = np.asarray(permutation, dtype=np.int64)
    return Dataset(
        name=dataset.name,
        seed=dataset.seed,
        factors=dataset.factors.copy(),
        idiosyncratic=dataset.idiosyncratic[..., permutation].copy(),
        demand_noise=dataset.demand_noise.copy(),
        inflow=dataset.inflow[..., permutation].copy(),
        demand=dataset.demand.copy(),
        price=dataset.price.copy(),
    )


def perturb_routing(
    config: ExperimentConfig,
    severity: float,
    seed: int,
) -> ExperimentConfig:
    """Reweight and occasionally reroute existing acyclic links."""

    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1].")
    if severity == 0.0:
        return config
    rng = np.random.default_rng(seed)
    original = np.asarray(config.routing, dtype=np.float64)
    routing = np.zeros_like(original)
    for source in range(config.n_reservoirs - 1):
        targets = np.flatnonzero(original[source] > 0.0)
        if not len(targets):
            continue
        target = int(targets[0])
        if rng.random() < severity:
            target = int(rng.integers(source + 1, config.n_reservoirs))
        multiplier = float(np.exp(rng.normal(0.0, 0.30 * severity)))
        routing[source, target] = np.clip(
            original[source, targets[0]] * multiplier,
            0.60,
            0.99,
        )
    return replace(
        config,
        routing=tuple(tuple(float(value) for value in row) for row in routing),
        downstream_index=_downstream_index(routing),
    )


def _downstream_index(routing: np.ndarray) -> tuple[int, ...]:
    downstream: list[int] = []
    for row in routing:
        targets = np.flatnonzero(row > 0.0)
        downstream.append(int(targets[0]) if len(targets) else -1)
    return tuple(downstream)
