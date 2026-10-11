"""Paired policy evaluation and preregistered operational metrics."""

from __future__ import annotations

import numpy as np
import torch

from .simulator import SimulationTrace, simulate


def cvar(values: np.ndarray, level: float = 0.95) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not 0.0 < level < 1.0:
        raise ValueError("CVaR needs a nonempty vector and level in (0, 1).")
    threshold = np.quantile(values, level)
    tail = values[values >= threshold]
    return float(tail.mean())


def trace_metrics(trace: SimulationTrace) -> dict[str, float]:
    costs = trace.total_cost.detach().cpu().numpy().astype(np.float64)
    return {
        "cost_mean": float(costs.mean()),
        "cost_std": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "cost_cvar95": cvar(costs),
        "shortage_mean": float(trace.shortage.sum(dim=(1, 2)).mean().cpu()),
        "spill_mean": float(trace.spill.sum(dim=(1, 2)).mean().cpu()),
        "low_storage_violation_mean": float(
            trace.low_storage.sum(dim=(1, 2)).mean().cpu()
        ),
        "flood_storage_violation_mean": float(
            trace.flood_storage.sum(dim=(1, 2)).mean().cpu()
        ),
        "terminal_cost_mean": float(trace.terminal_cost.mean().cpu()),
        "deficit_mean": float(trace.deficit.sum(dim=1).mean().cpu()),
        "max_projected_action_violation": float(trace.max_action_violation.cpu()),
        "max_storage_violation": float(trace.max_storage_violation.cpu()),
        "raw_proposal_violation_mean": float(
            trace.raw_proposal_violation.mean().cpu()
        ),
    }


@torch.no_grad()
def evaluate_policy(policy, dataset, system, n_regimes: int, device: str = "cpu"):
    policy.eval()
    trace = simulate(
        policy, dataset, system, n_regimes=n_regimes, device=device
    )
    return trace_metrics(trace), trace.total_cost.cpu().numpy().astype(np.float64)


def paired_bootstrap_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    repetitions: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("Paired costs must be aligned one-dimensional arrays.")
    difference = candidate - reference
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    chunk = 1000
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        means[start:stop] = difference[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "paired_mean_difference": float(difference.mean()),
        "paired_ci_lower": float(np.quantile(means, alpha / 2.0)),
        "paired_ci_upper": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "probability_candidate_better": float(np.mean(means < 0.0)),
    }

