"""Validation-selected pathwise training for projected policies."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, TrainingConfig
from .data import Dataset
from .simulator import simulate


@dataclass(frozen=True)
class FitResult:
    policy_name: str
    train_seed: int
    selected_iteration: int
    best_validation_cost: float
    completed_iterations: int
    checkpoint_path: str
    history: list[dict[str, float | int | str]]


def unused_batches(
    n: int, iterations: int, batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    required = iterations * batch_size
    if required > n:
        raise ValueError(
            f"Training needs {required} unique paths but fit contains only {n}."
        )
    return rng.permutation(n)[:required].reshape(iterations, batch_size)


@torch.no_grad()
def validation_cost(policy, data, config, device) -> float:
    policy.eval()
    return float(simulate(policy, data, config, device=device).mean_cost.cpu())


def fit_checkpoint(
    policy,
    fit: Dataset,
    validation: Dataset,
    config: ExperimentConfig,
    training: TrainingConfig,
    train_seed: int,
    checkpoint_path: str | Path,
) -> FitResult:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    rng = np.random.default_rng(train_seed)
    policy.to(training.device)
    lr = {
        "projected_affine": training.lr_affine,
        "projected_polynomial": training.lr_polynomial,
        "projected_graph_neural": training.lr_neural,
        "projected_graph_neural_global": training.lr_neural,
        "projected_mlp_pathwise": training.lr_mlp,
    }[policy.policy_name]
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=lr, weight_decay=training.weight_decay
    )
    batches = unused_batches(fit.n, training.iterations, training.batch_size, rng)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_state = copy.deepcopy(policy.state_dict())
    best_iteration = 0
    best_validation = validation_cost(
        policy, validation, config, training.device
    )
    history: list[dict[str, float | int | str]] = [
        {
            "policy": policy.policy_name,
            "train_seed": train_seed,
            "iteration": 0,
            "fit_trajectories_seen": 0,
            "environment_steps": 0,
            "train_cost": float("nan"),
            "validation_cost": best_validation,
            "is_best": 1,
        }
    ]

    def save_best(completed: int) -> None:
        torch.save(
            {
                "policy_name": policy.policy_name,
                "profile": config.profile,
                "train_seed": train_seed,
                "selected_iteration": best_iteration,
                "best_validation_cost": best_validation,
                "completed_iterations": completed,
                "experiment_config": asdict(config),
                "training_config": asdict(training),
                "state_dict": best_state,
            },
            checkpoint_path,
        )

    save_best(0)
    for iteration in range(1, training.iterations + 1):
        policy.train()
        batch = fit.take(batches[iteration - 1]).to_torch(training.device)
        optimizer.zero_grad(set_to_none=True)
        loss = simulate(policy, batch, config, device=training.device).mean_cost
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), training.gradient_clip)
        optimizer.step()
        should_validate = (
            iteration % training.validation_every == 0
            or iteration == training.iterations
        )
        current_validation = float("nan")
        is_best = 0
        if should_validate:
            current_validation = validation_cost(
                policy, validation, config, training.device
            )
            if current_validation < best_validation:
                best_validation = current_validation
                best_iteration = iteration
                best_state = copy.deepcopy(policy.state_dict())
                save_best(iteration)
                is_best = 1
        history.append(
            {
                "policy": policy.policy_name,
                "train_seed": train_seed,
                "iteration": iteration,
                "fit_trajectories_seen": iteration * training.batch_size,
                "environment_steps": iteration * training.batch_size * config.T,
                "train_cost": float(loss.detach().cpu()),
                "validation_cost": current_validation,
                "is_best": is_best,
            }
        )
    policy.load_state_dict(best_state)
    policy.eval()
    save_best(training.iterations)
    return FitResult(
        policy_name=policy.policy_name,
        train_seed=train_seed,
        selected_iteration=best_iteration,
        best_validation_cost=best_validation,
        completed_iterations=training.iterations,
        checkpoint_path=str(checkpoint_path),
        history=history,
    )
