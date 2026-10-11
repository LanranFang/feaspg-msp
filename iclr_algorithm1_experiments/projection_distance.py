"""Opt-in projection-distance loss for the existing full-path A1 simulator.

The penalty is a *sum* over stages and action coordinates, with a mean only
over trajectories. Both the proposal and the projected action stay on their
original computation graphs, including the moving-recourse dependence.
"""

from __future__ import annotations

import copy
import math
import operator
import random
import time
from contextlib import contextmanager
from collections.abc import Iterator

import numpy as np
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.simulator import Trace, simulate


ARMS = ("zero", "tenth", "fixed", "decay")
ACTIVE_PROJECTION_TOLERANCE = 1.0e-6
LAMBDA_REFERENCE_EPSILON = 1.0e-12


def projection_distance(trace: Trace) -> torch.Tensor:
    """Return ``0.5 sum_{t,r} (u-x)^2`` for each trajectory, without detach."""
    if trace.proposals.shape != trace.actions.shape or trace.actions.ndim != 3:
        raise ValueError("Proposals and actions must share shape [paths, stages, actions].")
    return 0.5 * (trace.proposals - trace.actions).square().sum(dim=(1, 2))


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}.")
    return value


def regularized_objective(trace: Trace, lambda_value: float) -> torch.Tensor:
    """Mean operating cost plus the frozen scalar penalty coefficient.

    The zero branch returns the original tensor verbatim, preserving the old
    objective and graph exactly rather than introducing a redundant multiply.
    """
    lambda_value = _nonnegative_finite(lambda_value, "lambda_value")
    if lambda_value == 0.0:
        return trace.mean_cost
    return trace.mean_cost + lambda_value * projection_distance(trace).mean()


def lambda_at_step(arm: str, lambda_ref: float, k: int) -> float:
    """Coefficient at zero-based optimizer update ``k`` (never epoch number)."""
    if arm not in ARMS:
        raise ValueError(f"Unknown regularization arm {arm!r}; expected one of {ARMS}.")
    lambda_ref = _nonnegative_finite(lambda_ref, "lambda_ref")
    if isinstance(k, bool):
        raise ValueError("k must be a nonnegative integer optimizer-update index.")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be a nonnegative integer optimizer-update index.") from exc
    if k < 0:
        raise ValueError("k must be a nonnegative integer optimizer-update index.")
    if arm == "zero":
        return 0.0
    if arm == "tenth":
        return 0.1 * lambda_ref
    if arm == "decay":
        return lambda_ref / math.sqrt(k + 1)
    return lambda_ref


def trace_metrics(
    trace: Trace,
    active_tolerance: float = ACTIVE_PROJECTION_TOLERANCE,
) -> dict[str, float]:
    """Detached operating and feasibility diagnostics.

    An active projection means the *action vector* at a path-stage pair moved:
    ``||u_t-x_t||_2 > active_tolerance``, in the simulator's action units. This
    measures correction, not the fraction of individual binding constraints.
    """
    active_tolerance = _nonnegative_finite(active_tolerance, "active_tolerance")
    with torch.no_grad():
        distance = projection_distance(trace)
        corrected = torch.linalg.vector_norm(trace.proposals - trace.actions, dim=-1)
        return {
            "mean_cost": float(trace.mean_cost.detach().cpu()),
            "proposal_distance": float(distance.mean().cpu()),
            "active_projection_fraction": float(
                (corrected > active_tolerance).to(torch.float64).mean().cpu()
            ),
            "active_projection_tolerance": active_tolerance,
            "max_action_violation": float(trace.max_action_violation.detach().cpu()),
            "max_storage_violation": float(trace.max_storage_violation.detach().cpu()),
        }


@contextmanager
def _preserve_rng() -> Iterator[None]:
    """Calibration must not advance the subsequent training randomness."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    # Avoid initializing CUDA just to save its RNG during a CPU experiment.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _parameter_gradient_norm(
    value: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> float:
    if not value.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        value, parameters, retain_graph=retain_graph, allow_unused=True
    )
    # Norm accumulation in double does not change the gradients' native dtype.
    squared_norm = sum(
        gradient.detach().to(torch.float64).square().sum()
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(float(squared_norm))


def estimate_lambda_ref(
    policy: torch.nn.Module,
    fit: Dataset,
    hydro: ExperimentConfig,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Estimate the advisor's initialization scale on every supplied fit path.

    The caller supplies fitting data only; no validation/test data are accepted
    as additional arguments. Compute the ratio of two medians of *per-path*
    Euclidean parameter-gradient norms, not the median of pathwise ratios.
    The result is a frozen scalar, not part of the optimization graph.

    An evaluation-mode copy prevents changes to the original parameters,
    buffers, gradients, or training mode. The current A1 MLP is deterministic
    and has no train/eval-dependent layers. Single-path simulations bound graph
    memory and keep every path's gradient separate; no clipping is applied.
    """
    if fit.n < 1:
        raise ValueError("At least one fitting trajectory is required.")
    started = time.perf_counter()
    cost_norms: list[float] = []
    distance_norms: list[float] = []
    with _preserve_rng(), torch.enable_grad():
        calibration_policy = copy.deepcopy(policy).to(device)
        calibration_policy.eval()
        parameters = tuple(p for p in calibration_policy.parameters() if p.requires_grad)
        if not parameters:
            raise ValueError("Policy has no trainable parameters for lambda calibration.")
        dtype = parameters[0].dtype
        for index in range(fit.n):
            path = fit.take(np.asarray([index], dtype=np.int64), name="lambda_reference_fit_path")
            trace = simulate(
                calibration_policy, path, hydro, device=device,
                dtype=dtype, gradient_mode="full",
            )
            distance = projection_distance(trace).sum()
            cost_norms.append(_parameter_gradient_norm(trace.mean_cost, parameters, retain_graph=True))
            distance_norms.append(_parameter_gradient_norm(distance, parameters, retain_graph=False))
    if not np.isfinite(cost_norms).all() or not np.isfinite(distance_norms).all():
        raise FloatingPointError("Non-finite per-path parameter-gradient norm in lambda calibration.")
    median_cost = float(np.median(cost_norms))
    median_distance = float(np.median(distance_norms))
    lambda_ref = median_cost / (median_distance + LAMBDA_REFERENCE_EPSILON)
    if not math.isfinite(lambda_ref):
        raise FloatingPointError("Non-finite lambda_ref in initialization calibration.")
    return {
        "lambda_ref": lambda_ref,
        "median_cost_gradient_norm": median_cost,
        "median_distance_gradient_norm": median_distance,
        "cost_gradient_norms": cost_norms,
        "distance_gradient_norms": distance_norms,
        "n_paths": int(fit.n),
        "fitting_data_name": fit.name,
        "fitting_data_seed": int(fit.seed),
        "gradient_parameter_count": sum(p.numel() for p in parameters),
        "gradient_dtype": str(dtype),
        "gradient_mode": "full",
        "normalization": "half_sum_over_stages_and_action_coordinates_per_path",
        "denominator_epsilon": LAMBDA_REFERENCE_EPSILON,
        "zero_cost_gradient_median": median_cost == 0.0,
        "zero_distance_gradient_median": median_distance == 0.0,
        "elapsed_seconds": time.perf_counter() - started,
    }
