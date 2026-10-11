"""Shared differentiable feasibility map for every policy family."""

from __future__ import annotations

import numpy as np
import torch

from .config import ReservoirSystem


def project_release(
    proposal: torch.Tensor,
    storage: torch.Tensor,
    system: ReservoirSystem,
    bisection_steps: int = 50,
) -> torch.Tensor:
    """Euclidean projection onto box, available-water, and shared limits."""

    capacity = torch.as_tensor(
        system.release_capacity, dtype=proposal.dtype, device=proposal.device
    )
    upper = torch.minimum(capacity, torch.clamp(storage, min=0.0))
    clipped = torch.minimum(torch.clamp(proposal, min=0.0), upper)
    if system.shared_release_capacity is None:
        return clipped
    shared = float(system.shared_release_capacity)
    needs_projection = clipped.sum(dim=-1) > shared
    low = torch.zeros_like(proposal[..., 0])
    high = torch.clamp(proposal.amax(dim=-1), min=0.0)
    for _ in range(bisection_steps):
        midpoint = (low + high) / 2.0
        candidate = torch.minimum(
            torch.clamp(proposal - midpoint.unsqueeze(-1), min=0.0), upper
        )
        too_large = candidate.sum(dim=-1) > shared
        low = torch.where(too_large, midpoint, low)
        high = torch.where(too_large, high, midpoint)
    projected = torch.minimum(
        torch.clamp(proposal - high.unsqueeze(-1), min=0.0), upper
    )
    return torch.where(needs_projection.unsqueeze(-1), projected, clipped)


def project_release_np(
    proposal: np.ndarray,
    storage: np.ndarray,
    system: ReservoirSystem,
    bisection_steps: int = 70,
) -> np.ndarray:
    proposal = np.asarray(proposal, dtype=np.float64)
    storage = np.asarray(storage, dtype=np.float64)
    capacity = np.asarray(system.release_capacity, dtype=np.float64)
    upper = np.minimum(capacity, np.maximum(storage, 0.0))
    clipped = np.minimum(np.maximum(proposal, 0.0), upper)
    if system.shared_release_capacity is None:
        return clipped
    shared = float(system.shared_release_capacity)
    needs_projection = clipped.sum(axis=-1) > shared
    low = np.zeros_like(proposal[..., 0])
    high = np.maximum(proposal.max(axis=-1), 0.0)
    for _ in range(bisection_steps):
        midpoint = (low + high) / 2.0
        candidate = np.minimum(
            np.maximum(proposal - midpoint[..., None], 0.0), upper
        )
        too_large = candidate.sum(axis=-1) > shared
        low = np.where(too_large, midpoint, low)
        high = np.where(too_large, high, midpoint)
    projected = np.minimum(
        np.maximum(proposal - high[..., None], 0.0), upper
    )
    return np.where(needs_projection[..., None], projected, clipped)


def proposal_violation(
    proposal: torch.Tensor, storage: torch.Tensor, system: ReservoirSystem
) -> torch.Tensor:
    """Per-path maximum violation before projection, for mechanism reporting."""

    capacity = torch.as_tensor(
        system.release_capacity, dtype=proposal.dtype, device=proposal.device
    )
    violations = torch.stack(
        (
            torch.relu(-proposal).amax(dim=-1),
            torch.relu(proposal - capacity).amax(dim=-1),
            torch.relu(proposal - storage).amax(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    if system.shared_release_capacity is not None:
        violations = torch.maximum(
            violations,
            torch.relu(proposal.sum(dim=-1) - float(system.shared_release_capacity)),
        )
    return violations

