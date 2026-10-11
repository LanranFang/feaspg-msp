"""Reference, matched MLP, graph, and portfolio policies."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import ReservoirSystem


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class HydroPolicy(nn.Module):
    policy_name = "hydro_policy"

    def __init__(self, system: ReservoirSystem, feature_dim: int) -> None:
        super().__init__()
        self.system = system
        self.feature_dim = int(feature_dim)
        self.register_buffer(
            "storage_capacity", torch.as_tensor(system.storage_capacity, dtype=torch.float32)
        )
        self.register_buffer(
            "release_capacity", torch.as_tensor(system.release_capacity, dtype=torch.float32)
        )
        self.register_buffer(
            "release_target",
            torch.as_tensor(system.objective.release_target, dtype=torch.float32),
        )
        self.register_buffer(
            "guide_lower", torch.as_tensor(system.objective.guide_lower, dtype=torch.float32)
        )
        self.register_buffer(
            "guide_upper", torch.as_tensor(system.objective.guide_upper, dtype=torch.float32)
        )

    def propose(self, stage, storage, previous_release, hydrology_features):
        raise NotImplementedError

    def reference(self, stage: int, storage: torch.Tensor) -> torch.Tensor:
        period = self.system.period(stage)
        center = 0.5 * (self.guide_lower[period] + self.guide_upper[period])
        feedback = 0.20 * (storage - center)
        return self.release_target[period] + feedback


class SeasonalReferencePolicy(HydroPolicy):
    policy_name = "seasonal_reference"

    def propose(self, stage, storage, previous_release, hydrology_features):
        del previous_release, hydrology_features
        return self.reference(stage, storage)


class ProjectedMLPPolicy(HydroPolicy):
    policy_name = "algorithm1_matched_mlp"

    def __init__(
        self,
        system: ReservoirSystem,
        feature_dim: int,
        width: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__(system, feature_dim)
        input_dim = 2 * system.n_reservoirs + feature_dim
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layer = nn.Linear(current, width)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            current = width
        output = nn.Linear(current, system.n_reservoirs)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def propose(self, stage, storage, previous_release, hydrology_features):
        observation = torch.cat(
            (
                storage / self.storage_capacity,
                previous_release / self.release_capacity,
                hydrology_features,
            ),
            dim=-1,
        )
        residual = 0.25 * self.release_capacity * torch.tanh(self.network(observation))
        return self.reference(stage, storage) + residual


class ProjectedGraphPolicy(HydroPolicy):
    policy_name = "algorithm1_graph"

    def __init__(
        self,
        system: ReservoirSystem,
        feature_dim: int,
        width: int = 64,
        message_steps: int = 2,
    ) -> None:
        super().__init__(system, feature_dim)
        self.width = width
        self.message_steps = message_steps
        node_feature_dim = 7 + feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(message_steps))
        self.upstream_layers = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(message_steps)
        )
        self.downstream_layers = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(message_steps)
        )
        self.output = nn.Linear(width, 1)
        routing = np.asarray(system.routing, dtype=np.float32)
        upstream = routing.T.copy()
        downstream = routing.copy()
        upstream /= np.maximum(upstream.sum(axis=1, keepdims=True), 1.0)
        downstream /= np.maximum(downstream.sum(axis=1, keepdims=True), 1.0)
        self.register_buffer("upstream", torch.as_tensor(upstream))
        self.register_buffer("downstream", torch.as_tensor(downstream))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("tanh"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)

    def propose(self, stage, storage, previous_release, hydrology_features):
        period = self.system.period(stage)
        n, R = storage.shape
        target = self.release_target[period].expand(n, -1)
        lower = self.guide_lower[period].expand(n, -1)
        upper = self.guide_upper[period].expand(n, -1)
        hydro = hydrology_features[:, None, :].expand(-1, R, -1)
        features = torch.cat(
            (
                (storage / self.storage_capacity).unsqueeze(-1),
                (previous_release / self.release_capacity).unsqueeze(-1),
                (target / self.release_capacity).unsqueeze(-1),
                (lower / self.storage_capacity).unsqueeze(-1),
                (upper / self.storage_capacity).unsqueeze(-1),
                (self.release_capacity / self.storage_capacity)
                .view(1, R, 1)
                .expand(n, -1, -1),
                torch.full(
                    (n, R, 1),
                    float(stage + 1) / self.system.horizon,
                    dtype=storage.dtype,
                    device=storage.device,
                ),
                hydro,
            ),
            dim=-1,
        )
        hidden = self.encoder(features)
        for self_layer, upstream_layer, downstream_layer in zip(
            self.self_layers, self.upstream_layers, self.downstream_layers, strict=True
        ):
            from_upstream = torch.einsum("ij,bjw->biw", self.upstream, hidden)
            from_downstream = torch.einsum("ij,bjw->biw", self.downstream, hidden)
            hidden = torch.tanh(
                self_layer(hidden)
                + upstream_layer(from_upstream)
                + downstream_layer(from_downstream)
            )
        residual = (
            0.25
            * self.release_capacity
            * torch.tanh(self.output(hidden).squeeze(-1))
        )
        return self.reference(stage, storage) + residual


class ProjectedDeepSetsPolicy(HydroPolicy):
    """Permutation-equivariant portfolio policy for non-network CAMELS basins."""

    policy_name = "algorithm1_deepsets"

    def __init__(self, system: ReservoirSystem, feature_dim: int, width: int = 64) -> None:
        super().__init__(system, feature_dim)
        local_dim = 6 + feature_dim
        self.encoder = nn.Sequential(nn.Linear(local_dim, width), nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.decoder[-1].weight)

    def propose(self, stage, storage, previous_release, hydrology_features):
        period = self.system.period(stage)
        n, R = storage.shape
        hydro = hydrology_features[:, None, :].expand(-1, R, -1)
        local = torch.cat(
            (
                (storage / self.storage_capacity).unsqueeze(-1),
                (previous_release / self.release_capacity).unsqueeze(-1),
                (self.release_target[period] / self.release_capacity)
                .view(1, R, 1)
                .expand(n, -1, -1),
                (self.guide_lower[period] / self.storage_capacity)
                .view(1, R, 1)
                .expand(n, -1, -1),
                (self.guide_upper[period] / self.storage_capacity)
                .view(1, R, 1)
                .expand(n, -1, -1),
                (self.release_capacity / self.storage_capacity)
                .view(1, R, 1)
                .expand(n, -1, -1),
                hydro,
            ),
            dim=-1,
        )
        encoded = self.encoder(local)
        pooled = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        residual = 0.25 * self.release_capacity * torch.tanh(
            self.decoder(torch.cat((encoded, pooled), dim=-1)).squeeze(-1)
        )
        return self.reference(stage, storage) + residual


def matched_mlp_width(
    system: ReservoirSystem,
    feature_dim: int,
    graph_width: int = 64,
    depth: int = 2,
    maximum: int = 2048,
) -> tuple[int, int, int]:
    """Return the MLP width with parameter count nearest the graph policy."""

    graph_parameters = count_parameters(
        ProjectedGraphPolicy(system, feature_dim, width=graph_width)
    )
    input_dim = 2 * system.n_reservoirs + feature_dim

    def mlp_count(width: int) -> int:
        return int(
            (input_dim + 1) * width
            + max(depth - 1, 0) * (width + 1) * width
            + (width + 1) * system.n_reservoirs
        )

    width = min(range(1, maximum + 1), key=lambda value: abs(mlp_count(value) - graph_parameters))
    return width, mlp_count(width), graph_parameters
