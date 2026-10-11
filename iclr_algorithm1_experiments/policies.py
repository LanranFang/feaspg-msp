"""Parameter-matched policies and transferable graph baselines."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from efficiency_switching_hydro.policies import (
    GlobalGraphNeuralPolicy,
    PathwiseMLPPolicy,
    ReferencePolicy,
    count_parameters,
)

from .config import ICLRProtocol


def flat_mlp_parameter_count(config, width: int, depth: int = 2) -> int:
    input_dim = config.n_reservoirs * (2 + config.routing_delay) + 4
    total = (input_dim + 1) * width
    total += max(depth - 1, 0) * (width + 1) * width
    total += (width + 1) * config.n_reservoirs
    return int(total)


def parameter_matched_mlp_width(
    config,
    graph_width: int = 64,
    depth: int = 2,
    maximum_width: int = 1024,
) -> tuple[int, int, int]:
    """Return the flat width closest to the graph policy's parameter count."""

    graph_parameters = count_parameters(GlobalGraphNeuralPolicy(config, graph_width))
    widths = range(1, maximum_width + 1)
    width = min(
        widths,
        key=lambda candidate: abs(
            flat_mlp_parameter_count(config, candidate, depth) - graph_parameters
        ),
    )
    return width, flat_mlp_parameter_count(config, width, depth), graph_parameters


class GlobalDeepSetPolicy(ReferencePolicy):
    """Transferable node policy with global pooling but no routing messages."""

    policy_name = "projected_deepset_global"

    def __init__(self, config, width: int = 87) -> None:
        super().__init__(config)
        self.width = width
        node_feature_dim = 2 + config.routing_delay + 9 + 4
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.global_layers = nn.ModuleList(
            [nn.Linear(2 * width, width) for _ in range(3)]
        )
        self.output = nn.Linear(width, 1)

        capacity = np.asarray(config.storage_capacity, dtype=np.float32)
        release = np.asarray(config.release_capacity, dtype=np.float32)
        eta = np.asarray(config.generation_coefficient, dtype=np.float32)
        terminal_value = np.asarray(config.terminal_water_value, dtype=np.float32)
        attributes = np.stack(
            (
                eta / max(float(eta.mean()), 1.0e-12),
                capacity / max(float(capacity.mean()), 1.0e-12),
                release / capacity,
                terminal_value / max(float(terminal_value.mean()), 1.0e-12),
                np.asarray(config.band1_center, dtype=np.float32),
                np.asarray(config.band2_center, dtype=np.float32),
                np.asarray(config.band1_amplitude, dtype=np.float32),
                np.asarray(config.band2_amplitude, dtype=np.float32),
                (np.asarray(config.downstream_index) >= 0).astype(np.float32),
            ),
            axis=1,
        )
        self.register_buffer("capacity", torch.as_tensor(capacity))
        self.register_buffer("release_capacity", torch.as_tensor(release))
        self.register_buffer("attributes", torch.as_tensor(attributes))
        self.register_buffer(
            "queue_capacity",
            torch.as_tensor(config.queue_capacity().astype(np.float32)),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight,
                    gain=nn.init.calculate_gain("tanh"),
                )
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def node_features(self, stage, storage, previous_release, queue):
        batch_size = storage.shape[0]
        pieces = [
            (storage / self.capacity).unsqueeze(-1),
            (previous_release / self.release_capacity).unsqueeze(-1),
        ]
        for delay in range(self.config.routing_delay):
            scale = torch.clamp(self.queue_capacity[delay], min=1.0)
            pieces.append((queue[:, delay, :] / scale).unsqueeze(-1))
        attributes = self.attributes.unsqueeze(0).expand(batch_size, -1, -1)
        fraction = float(stage + 1) / self.config.T
        time = torch.tensor(
            (
                math.sin(2.0 * math.pi * fraction),
                math.cos(2.0 * math.pi * fraction),
                math.sin(4.0 * math.pi * fraction),
                math.cos(4.0 * math.pi * fraction),
            ),
            dtype=storage.dtype,
            device=storage.device,
        )
        time = time.view(1, 1, 4).expand(batch_size, self.config.n_reservoirs, -1)
        return torch.cat((*pieces, attributes, time), dim=-1)

    def propose(self, stage, storage, previous_release, queue):
        hidden = self.node_encoder(
            self.node_features(stage, storage, previous_release, queue)
        )
        for layer in self.global_layers:
            global_state = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            hidden = torch.tanh(layer(torch.cat((hidden, global_state), dim=-1)))
        residual_scale = 0.20 * float(np.mean(self.config.release_capacity))
        residual = residual_scale * self.output(hidden).squeeze(-1)
        return self.target_proposal(stage, storage) + residual


def parameter_matched_deepset_width(
    config,
    graph_width: int = 64,
    maximum_width: int = 512,
) -> tuple[int, int, int]:
    graph_parameters = count_parameters(GlobalGraphNeuralPolicy(config, graph_width))
    feature_dim = 2 + config.routing_delay + 9 + 4

    def deepset_count(width: int) -> int:
        return int(7 * width * width + (feature_dim + 6) * width + 1)

    width = min(
        range(1, maximum_width + 1),
        key=lambda candidate: abs(deepset_count(candidate) - graph_parameters),
    )
    return width, deepset_count(width), graph_parameters


def build_policy(name: str, config, protocol: ICLRProtocol):
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"matched_mlp", "projected_mlp_matched"}:
        width, _, _ = parameter_matched_mlp_width(
            config,
            graph_width=protocol.graph_width,
            depth=protocol.network_depth,
        )
        policy = PathwiseMLPPolicy(
            config,
            width=width,
            depth=protocol.network_depth,
        )
        policy.policy_name = "projected_mlp_matched"
        return policy
    if normalized in {"graph", "projected_graph_neural_global"}:
        return GlobalGraphNeuralPolicy(config, width=protocol.graph_width)
    if normalized in {"deepset", "projected_deepset_global"}:
        width, _, _ = parameter_matched_deepset_width(
            config,
            graph_width=protocol.graph_width,
        )
        return GlobalDeepSetPolicy(config, width=width)
    if normalized in {"shac", "projected_shac"}:
        width, _, _ = parameter_matched_mlp_width(
            config,
            graph_width=protocol.graph_width,
            depth=protocol.network_depth,
        )
        policy = PathwiseMLPPolicy(
            config,
            width=width,
            depth=protocol.network_depth,
        )
        policy.policy_name = "projected_shac"
        return policy
    raise ValueError(f"Unknown ICLR policy: {name}")


def copy_trainable_parameters(source: nn.Module, target: nn.Module) -> None:
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
