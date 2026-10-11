"""Projected policies for the efficiency-switching hydro experiment."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .policy_features import (
    flat_markov_observation_torch,
    reference_centered_proposal_torch,
)


class Policy(nn.Module):
    policy_name = "policy"

    def propose(self, stage, storage, previous_release, queue):
        raise NotImplementedError


class ReferencePolicy(Policy):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "reference_release",
            torch.as_tensor(config.reference_release(), dtype=torch.float32),
        )
        self.register_buffer(
            "reference_target",
            torch.as_tensor(config.planned_storage(), dtype=torch.float32),
        )

    def target_proposal(self, stage, storage):
        return self.reference_release[stage] + self.config.storage_feedback_gain * (
            storage - self.reference_target[stage]
        )


class MyopicPolicy(ReferencePolicy):
    policy_name = "myopic"

    def propose(self, stage, storage, previous_release, queue):
        return self.reference_release[stage].expand_as(storage)


def normalized_flat_state(config, storage, previous_release, queue):
    capacity = torch.as_tensor(
        config.storage_capacity, dtype=storage.dtype, device=storage.device
    )
    release = torch.as_tensor(
        config.release_capacity, dtype=storage.dtype, device=storage.device
    )
    pieces = [storage / capacity, previous_release / release]
    if config.routing_delay:
        queue_capacity = torch.as_tensor(
            config.queue_capacity(), dtype=storage.dtype, device=storage.device
        )
        pieces.append((queue / torch.clamp(queue_capacity, min=1.0)).flatten(1))
    return torch.cat(pieces, dim=-1)


class AffinePolicy(Policy):
    policy_name = "projected_affine"

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        state_dim = config.n_reservoirs * (2 + config.routing_delay)
        theta = torch.zeros(config.T, config.n_reservoirs, state_dim + 1)
        reference = config.reference_release()
        target = config.planned_storage()
        gain = config.storage_feedback_gain
        theta[:, :, 0] = torch.as_tensor(reference - gain * target)
        for reservoir, capacity in enumerate(config.storage_capacity):
            theta[:, reservoir, 1 + reservoir] = gain * capacity
        self.theta = nn.Parameter(theta)

    def propose(self, stage, storage, previous_release, queue):
        state = normalized_flat_state(
            self.config, storage, previous_release, queue
        )
        features = torch.cat((torch.ones_like(state[:, :1]), state), dim=-1)
        return features @ self.theta[stage].T


class PolynomialPolicy(Policy):
    policy_name = "projected_polynomial"

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.state_dim = config.n_reservoirs * (2 + config.routing_delay)
        row, column = torch.triu_indices(self.state_dim, self.state_dim)
        self.register_buffer("quadratic_row", row)
        self.register_buffer("quadratic_column", column)
        n_features = 1 + self.state_dim + row.numel()
        theta = torch.zeros(config.T, config.n_reservoirs, n_features)
        affine = AffinePolicy(config).theta.detach()
        theta[:, :, : 1 + self.state_dim] = affine
        self.theta = nn.Parameter(theta)

    def features(self, storage, previous_release, queue):
        state = normalized_flat_state(
            self.config, storage, previous_release, queue
        )
        quadratic = state[:, self.quadratic_row] * state[:, self.quadratic_column]
        return torch.cat((torch.ones_like(state[:, :1]), state, quadratic), dim=-1)

    def propose(self, stage, storage, previous_release, queue):
        return self.features(storage, previous_release, queue) @ self.theta[stage].T


class PathwiseMLPPolicy(Policy):
    """Algorithm 1 policy with the same flat actor shape used by SB3."""

    policy_name = "projected_mlp_pathwise"

    def __init__(
        self,
        config: ExperimentConfig,
        width: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.config = config
        input_dim = config.n_reservoirs * (2 + config.routing_delay) + 4
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layer = nn.Linear(current, width)
            nn.init.orthogonal_(layer.weight, gain=2.0**0.5)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.ReLU()))
            current = width
        output = nn.Linear(current, config.n_reservoirs)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.actor = nn.Sequential(*layers)

    def propose(self, stage, storage, previous_release, queue):
        observation = flat_markov_observation_torch(
            stage,
            storage,
            previous_release,
            queue,
            self.config,
        )
        normalized_action = torch.tanh(self.actor(observation))
        return reference_centered_proposal_torch(
            normalized_action, stage, self.config
        )


class GraphNeuralPolicy(ReferencePolicy):
    policy_name = "projected_graph_neural"

    def __init__(self, config: ExperimentConfig, width: int = 64) -> None:
        super().__init__(config)
        self.width = width
        node_feature_dim = 2 + config.routing_delay + 9 + 4
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.message_layers = nn.ModuleList(
            [nn.Linear(3 * width, width), nn.Linear(3 * width, width)]
        )
        self.output = nn.Linear(width, 1)
        self._reset_parameters()

        capacity = np.asarray(config.storage_capacity, dtype=np.float32)
        release = np.asarray(config.release_capacity, dtype=np.float32)
        eta = np.asarray(config.generation_coefficient, dtype=np.float32)
        terminal_value = np.asarray(config.terminal_water_value, dtype=np.float32)
        attributes = np.stack(
            (
                eta / max(float(eta.mean()), 1e-12),
                capacity / max(float(capacity.mean()), 1e-12),
                release / capacity,
                terminal_value / max(float(terminal_value.mean()), 1e-12),
                np.asarray(config.band1_center, dtype=np.float32),
                np.asarray(config.band2_center, dtype=np.float32),
                np.asarray(config.band1_amplitude, dtype=np.float32),
                np.asarray(config.band2_amplitude, dtype=np.float32),
                (np.asarray(config.downstream_index) >= 0).astype(np.float32),
            ),
            axis=1,
        )
        routing = np.asarray(config.routing, dtype=np.float32)
        incoming_norm = np.maximum(routing.sum(axis=0, keepdims=True), 1.0)
        outgoing_norm = np.maximum(routing.sum(axis=1, keepdims=True), 1.0)
        self.register_buffer("capacity", torch.as_tensor(capacity))
        self.register_buffer("release_capacity", torch.as_tensor(release))
        self.register_buffer("attributes", torch.as_tensor(attributes))
        self.register_buffer(
            "incoming_routing", torch.as_tensor(routing / incoming_norm)
        )
        self.register_buffer(
            "outgoing_routing", torch.as_tensor(routing / outgoing_norm)
        )
        self.register_buffer(
            "queue_capacity",
            torch.as_tensor(config.queue_capacity().astype(np.float32)),
        )

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("tanh")
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
        for layer in self.message_layers:
            incoming = torch.einsum(
                "sr,bsh->brh", self.incoming_routing, hidden
            )
            downstream = torch.einsum(
                "sr,brh->bsh", self.outgoing_routing, hidden
            )
            hidden = torch.tanh(layer(torch.cat((hidden, incoming, downstream), dim=-1)))
        residual_scale = 0.20 * float(np.mean(self.config.release_capacity))
        residual = residual_scale * self.output(hidden).squeeze(-1)
        return self.target_proposal(stage, storage) + residual


class GlobalGraphNeuralPolicy(GraphNeuralPolicy):
    """Graph policy with a network-wide summary for shared-capacity allocation."""

    policy_name = "projected_graph_neural_global"

    def __init__(self, config: ExperimentConfig, width: int = 64) -> None:
        super().__init__(config, width)
        self.message_layers = nn.ModuleList(
            [nn.Linear(4 * width, width) for _ in range(3)]
        )
        self._reset_parameters()

    def propose(self, stage, storage, previous_release, queue):
        hidden = self.node_encoder(
            self.node_features(stage, storage, previous_release, queue)
        )
        for layer in self.message_layers:
            incoming = torch.einsum(
                "sr,bsh->brh", self.incoming_routing, hidden
            )
            downstream = torch.einsum(
                "sr,brh->bsh", self.outgoing_routing, hidden
            )
            global_state = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            hidden = torch.tanh(
                layer(
                    torch.cat(
                        (hidden, incoming, downstream, global_state), dim=-1
                    )
                )
            )
        residual_scale = 0.20 * float(np.mean(self.config.release_capacity))
        residual = residual_scale * self.output(hidden).squeeze(-1)
        return self.target_proposal(stage, storage) + residual


def make_policy(name: str, config: ExperimentConfig) -> Policy:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "myopic":
        return MyopicPolicy(config)
    if normalized in {"projected_affine", "affine"}:
        return AffinePolicy(config)
    if normalized in {"projected_polynomial", "polynomial", "poly"}:
        return PolynomialPolicy(config)
    if normalized in {
        "projected_mlp_pathwise",
        "pathwise_mlp",
        "mlp_pathwise",
    }:
        return PathwiseMLPPolicy(config)
    if normalized in {
        "projected_graph_neural",
        "graph_neural",
        "neural",
    }:
        return GraphNeuralPolicy(config)
    if normalized in {
        "projected_graph_neural_global",
        "graph_neural_global",
        "global_neural",
    }:
        return GlobalGraphNeuralPolicy(config)
    raise ValueError(f"Unknown nonlinear hydro policy: {name}")


def count_parameters(policy: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in policy.parameters()))
