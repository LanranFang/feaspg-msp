"""Gradient diagnostics for the two state-dependent Algorithm 1 channels."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.simulator import GRADIENT_MODES, simulate


def objective_gradient(
    policy,
    dataset: Dataset,
    config: ExperimentConfig,
    gradient_mode: str = "full",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[float, torch.Tensor]:
    policy.zero_grad(set_to_none=True)
    policy.train()
    loss = simulate(
        policy,
        dataset,
        config,
        device=device,
        dtype=dtype,
        gradient_mode=gradient_mode,
    ).mean_cost
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    pieces = [
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None
        else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    vector = torch.cat(pieces).detach().cpu().to(torch.float64)
    return float(loss.detach().cpu()), vector


def finite_difference_rows(
    policy,
    dataset: Dataset,
    config: ExperimentConfig,
    steps: tuple[float, ...],
    direction_seed: int,
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    policy.to(device=device, dtype=torch.float64)
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    base = parameters_to_vector(parameters).detach().clone()
    generator = torch.Generator(device=base.device)
    generator.manual_seed(direction_seed)
    direction = torch.randn(
        base.shape,
        generator=generator,
        dtype=base.dtype,
        device=base.device,
    )
    direction /= torch.clamp(torch.linalg.vector_norm(direction), min=1.0e-12)

    costs: dict[str, float] = {}
    gradients: dict[str, torch.Tensor] = {}
    for mode in GRADIENT_MODES:
        costs[mode], gradients[mode] = objective_gradient(
            policy,
            dataset,
            config,
            gradient_mode=mode,
            device=device,
            dtype=torch.float64,
        )
    direction_cpu = direction.detach().cpu()
    directional = {
        mode: float(gradient @ direction_cpu) for mode, gradient in gradients.items()
    }

    rows: list[dict[str, float | int | str]] = []
    try:
        for step in steps:
            with torch.no_grad():
                vector_to_parameters(base + step * direction, parameters)
                plus = float(
                    simulate(
                        policy,
                        dataset,
                        config,
                        device=device,
                        dtype=torch.float64,
                    ).mean_cost.cpu()
                )
                vector_to_parameters(base - step * direction, parameters)
                minus = float(
                    simulate(
                        policy,
                        dataset,
                        config,
                        device=device,
                        dtype=torch.float64,
                    ).mean_cost.cpu()
                )
            finite_difference = (plus - minus) / (2.0 * step)
            for mode in GRADIENT_MODES:
                autograd_value = directional[mode]
                absolute_error = abs(autograd_value - finite_difference)
                relative_error = absolute_error / max(
                    1.0,
                    abs(autograd_value),
                    abs(finite_difference),
                )
                rows.append(
                    {
                        "direction_seed": direction_seed,
                        "gradient_mode": mode,
                        "step": step,
                        "base_cost": costs[mode],
                        "autograd_directional_derivative": autograd_value,
                        "finite_difference_directional_derivative": finite_difference,
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                        "gradient_norm": float(torch.linalg.vector_norm(gradients[mode])),
                    }
                )
    finally:
        with torch.no_grad():
            vector_to_parameters(base, parameters)
    return rows


def gradient_comparison(
    policy,
    dataset: Dataset,
    config: ExperimentConfig,
    device: str = "cpu",
) -> list[dict[str, float | str]]:
    costs: dict[str, float] = {}
    gradients: dict[str, torch.Tensor] = {}
    for mode in GRADIENT_MODES:
        costs[mode], gradients[mode] = objective_gradient(
            policy,
            dataset,
            config,
            gradient_mode=mode,
            device=device,
            dtype=torch.float64,
        )
    full = gradients["full"]
    full_norm = float(torch.linalg.vector_norm(full))
    rows: list[dict[str, float | str]] = []
    for mode in GRADIENT_MODES:
        gradient = gradients[mode]
        difference = gradient - full
        norm = float(torch.linalg.vector_norm(gradient))
        difference_norm = float(torch.linalg.vector_norm(difference))
        denominator = max(full_norm * norm, 1.0e-24)
        cosine = float((full @ gradient) / denominator) if norm > 0.0 else float("nan")
        rows.append(
            {
                "gradient_mode": mode,
                "mean_cost": costs[mode],
                "gradient_norm": norm,
                "full_gradient_norm": full_norm,
                "difference_from_full_norm": difference_norm,
                "relative_difference_from_full": difference_norm / max(full_norm, 1.0e-12),
                "cosine_with_full": cosine,
            }
        )
    return rows


def nested_horizon_datasets(
    configs: list[ExperimentConfig],
    n: int,
    seed: int,
) -> dict[int, Dataset]:
    """Generate common standardized shock prefixes for all horizons."""

    if not configs:
        return {}
    max_horizon = max(config.T for config in configs)
    n_reservoirs = configs[0].n_reservoirs
    shock_factors = configs[0].shock_factors
    if any(
        config.n_reservoirs != n_reservoirs
        or config.shock_factors != shock_factors
        for config in configs
    ):
        raise ValueError("Nested horizon datasets require common state dimensions.")
    rng = np.random.default_rng(seed)
    standard_factors = np.clip(
        rng.standard_normal((n, max_horizon, shock_factors)), -4.0, 4.0
    )
    idiosyncratic_uniform = rng.uniform(
        0.0, 1.0, size=(n, max_horizon, n_reservoirs)
    )
    demand_standard = np.clip(
        rng.standard_normal((n, max_horizon)), -4.0, 4.0
    )
    datasets: dict[int, Dataset] = {}
    for config in configs:
        horizon = config.T
        factors = config.factor_std * standard_factors[:, :horizon]
        idiosyncratic = config.idiosyncratic_low + (
            config.idiosyncratic_high - config.idiosyncratic_low
        ) * idiosyncratic_uniform[:, :horizon]
        demand_noise = config.demand_noise_std * demand_standard[:, :horizon]
        inflow_factor = np.einsum(
            "ntf,rf->ntr", factors, np.asarray(config.factor_loadings)
        )
        demand_factor = np.einsum(
            "ntf,f->nt", factors, np.asarray(config.demand_factor_loadings)
        )
        inflow = np.maximum(
            0.0,
            config.inflow_mean()[None, :, :] + inflow_factor + idiosyncratic,
        )
        demand = np.maximum(
            0.0,
            config.demand_mean()[None, :] + demand_factor + demand_noise,
        )
        price = np.broadcast_to(config.price(), (n, horizon)).copy()
        datasets[horizon] = Dataset(
            name="fit_gradient_diagnostic",
            seed=seed,
            factors=factors.astype(np.float32),
            idiosyncratic=idiosyncratic.astype(np.float32),
            demand_noise=demand_noise.astype(np.float32),
            inflow=inflow.astype(np.float32),
            demand=demand.astype(np.float32),
            price=price.astype(np.float32),
        )
    return datasets


def geometric_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y) & (y > 0.0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])
