"""Shared flat observations and reference-centered action parameterization."""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import ExperimentConfig


def _stage_features(stage: int, horizon: int) -> tuple[float, float, float, float]:
    fraction = float(stage + 1) / horizon
    return (
        math.sin(2.0 * math.pi * fraction),
        math.cos(2.0 * math.pi * fraction),
        math.sin(4.0 * math.pi * fraction),
        math.cos(4.0 * math.pi * fraction),
    )


def flat_markov_observation_np(
    stage: int,
    storage: np.ndarray,
    previous_release: np.ndarray,
    queue: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    storage = np.asarray(storage)
    previous_release = np.asarray(previous_release)
    queue = np.asarray(queue)
    capacity = np.asarray(config.storage_capacity)
    release_capacity = np.asarray(config.release_capacity)
    pieces = [storage / capacity, previous_release / release_capacity]
    if config.routing_delay:
        queue_capacity = np.maximum(config.queue_capacity(), 1.0)
        pieces.append(
            (queue / queue_capacity).reshape(*storage.shape[:-1], -1)
        )
    time = np.asarray(_stage_features(stage, config.T), dtype=np.float64)
    time = np.broadcast_to(time, (*storage.shape[:-1], 4))
    return np.concatenate((*pieces, time), axis=-1).astype(
        np.float32, copy=False
    )


def flat_markov_observation_torch(
    stage: int,
    storage: torch.Tensor,
    previous_release: torch.Tensor,
    queue: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    capacity = torch.as_tensor(
        config.storage_capacity, dtype=storage.dtype, device=storage.device
    )
    release_capacity = torch.as_tensor(
        config.release_capacity, dtype=storage.dtype, device=storage.device
    )
    pieces = [storage / capacity, previous_release / release_capacity]
    if config.routing_delay:
        queue_capacity = torch.as_tensor(
            config.queue_capacity(), dtype=storage.dtype, device=storage.device
        )
        pieces.append((queue / torch.clamp(queue_capacity, min=1.0)).flatten(1))
    time = torch.tensor(
        _stage_features(stage, config.T),
        dtype=storage.dtype,
        device=storage.device,
    ).expand(storage.shape[0], -1)
    return torch.cat((*pieces, time), dim=-1)


def reference_centered_proposal_np(
    normalized_action: np.ndarray,
    stage: int,
    config: ExperimentConfig,
) -> np.ndarray:
    normalized = np.clip(
        np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0
    )
    release_capacity = np.asarray(config.release_capacity, dtype=np.float64)
    reference = np.asarray(config.reference_release()[stage], dtype=np.float64)
    return np.where(
        normalized < 0.0,
        reference * (normalized + 1.0),
        reference + normalized * (release_capacity - reference),
    )


def reference_centered_proposal_torch(
    normalized_action: torch.Tensor,
    stage: int,
    config: ExperimentConfig,
) -> torch.Tensor:
    normalized = torch.clamp(normalized_action, -1.0, 1.0)
    release_capacity = torch.as_tensor(
        config.release_capacity,
        dtype=normalized.dtype,
        device=normalized.device,
    )
    reference = torch.as_tensor(
        config.reference_release()[stage],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    return torch.where(
        normalized < 0.0,
        reference * (normalized + 1.0),
        reference + normalized * (release_capacity - reference),
    )
