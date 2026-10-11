"""Shared differentiable projection for every efficiency-switching policy."""

from __future__ import annotations

import numpy as np
import torch

from .config import ExperimentConfig


def action_bounds(
    storage: torch.Tensor,
    previous_release: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    release_capacity = torch.as_tensor(
        config.release_capacity, dtype=storage.dtype, device=storage.device
    )
    ramp = config.ramp_fraction * release_capacity
    lower = torch.clamp(previous_release - ramp, min=0.0)
    upper = torch.minimum(
        torch.minimum(release_capacity, torch.clamp(storage, min=0.0)),
        previous_release + ramp,
    )
    lower = torch.minimum(lower, upper)
    return lower, upper


def project_action(
    proposal: torch.Tensor,
    storage: torch.Tensor,
    previous_release: torch.Tensor,
    config: ExperimentConfig,
    bisection_steps: int = 50,
) -> torch.Tensor:
    lower, upper = action_bounds(storage, previous_release, config)
    weights = torch.as_tensor(
        config.generation_coefficient,
        dtype=proposal.dtype,
        device=proposal.device,
    )
    clipped = torch.minimum(torch.maximum(proposal, lower), upper)
    needs_projection = (
        clipped * weights
    ).sum(dim=-1) > config.shared_release_capacity
    # Bisection identifies the fixed active set. Differentiating through its
    # branch history does not give the projection Jacobian, so reconstruct the
    # multiplier from the KKT equality on that active set below.
    with torch.no_grad():
        detached_proposal = proposal.detach()
        detached_lower = lower.detach()
        detached_upper = upper.detach()
        low = torch.zeros_like(detached_proposal[..., 0])
        high = torch.clamp(
            ((detached_proposal - detached_lower) / weights).amax(dim=-1),
            min=0.0,
        )
        for _ in range(bisection_steps):
            midpoint = (low + high) / 2.0
            candidate = torch.minimum(
                torch.maximum(
                    detached_proposal - midpoint.unsqueeze(-1) * weights,
                    detached_lower,
                ),
                detached_upper,
            )
            too_large = (
                candidate * weights
            ).sum(dim=-1) > config.shared_release_capacity
            low = torch.where(too_large, midpoint, low)
            high = torch.where(too_large, high, midpoint)
        unconstrained = detached_proposal - high.unsqueeze(-1) * weights
        at_lower = unconstrained <= detached_lower
        at_upper = unconstrained >= detached_upper
        free = ~(at_lower | at_upper)

    free_weight = free.to(proposal.dtype) * weights
    lower_weight = at_lower.to(proposal.dtype) * weights
    upper_weight = at_upper.to(proposal.dtype) * weights
    denominator = (free_weight * weights).sum(dim=-1)
    numerator = (
        (free_weight * proposal).sum(dim=-1)
        + (lower_weight * lower).sum(dim=-1)
        + (upper_weight * upper).sum(dim=-1)
        - config.shared_release_capacity
    )
    implicit_multiplier = numerator / torch.clamp(denominator, min=1.0e-12)
    multiplier = torch.where(
        denominator > 1.0e-12,
        implicit_multiplier,
        high,
    )
    projected = torch.where(
        at_lower,
        lower,
        torch.where(
            at_upper,
            upper,
            proposal - multiplier.unsqueeze(-1) * weights,
        ),
    )
    numerical_projected = torch.minimum(
        torch.maximum(
            detached_proposal - high.unsqueeze(-1) * weights,
            detached_lower,
        ),
        detached_upper,
    )
    projected = numerical_projected + (projected - projected.detach())
    return torch.where(needs_projection.unsqueeze(-1), projected, clipped)


def action_bounds_np(
    storage: np.ndarray,
    previous_release: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    release = np.asarray(config.release_capacity, dtype=np.float64)
    ramp = config.ramp_fraction * release
    lower = np.maximum(previous_release - ramp, 0.0)
    upper = np.minimum(np.minimum(release, np.maximum(storage, 0.0)), previous_release + ramp)
    return np.minimum(lower, upper), upper


def project_action_np(
    proposal: np.ndarray,
    storage: np.ndarray,
    previous_release: np.ndarray,
    config: ExperimentConfig,
    bisection_steps: int = 70,
) -> np.ndarray:
    proposal = np.asarray(proposal, dtype=np.float64)
    storage = np.asarray(storage, dtype=np.float64)
    previous_release = np.asarray(previous_release, dtype=np.float64)
    lower, upper = action_bounds_np(storage, previous_release, config)
    weights = np.asarray(config.generation_coefficient, dtype=np.float64)
    clipped = np.minimum(np.maximum(proposal, lower), upper)
    needs_projection = (
        clipped * weights
    ).sum(axis=-1) > config.shared_release_capacity
    low = np.zeros_like(proposal[..., 0])
    high = np.maximum(((proposal - lower) / weights).max(axis=-1), 0.0)
    for _ in range(bisection_steps):
        midpoint = (low + high) / 2.0
        candidate = np.minimum(
            np.maximum(proposal - midpoint[..., None] * weights, lower), upper
        )
        too_large = (
            candidate * weights
        ).sum(axis=-1) > config.shared_release_capacity
        low = np.where(too_large, midpoint, low)
        high = np.where(too_large, high, midpoint)
    projected = np.minimum(
        np.maximum(proposal - high[..., None] * weights, lower), upper
    )
    return np.where(needs_projection[..., None], projected, clipped)
