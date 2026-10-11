"""Direct pathwise (Algorithm 1) training with validation-only selection."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from .dataset import TrajectoryBatch
from .simulator import simulate


@dataclass(frozen=True)
class DirectTrainingConfig:
    iterations: int
    batch_size: int = 2
    validation_every: int = 10
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    gradient_clip: float = 10.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.iterations, self.batch_size, self.validation_every) < 1:
            raise ValueError("Training counts must be positive.")
        if min(self.learning_rate, self.gradient_clip) <= 0.0:
            raise ValueError("Learning rate and gradient clip must be positive.")


@dataclass
class DirectCandidate:
    policy: nn.Module
    train_seed: int
    learning_rate: float
    best_validation_cost: float
    selected_iteration: int
    fit_time_seconds: float
    environment_transitions: int
    gradient_updates: int
    history: list[dict[str, float | int | str]]
    checkpoint_path: str


@dataclass
class DirectSelection:
    selected: DirectCandidate
    candidates: list[DirectCandidate]
    total_tuning_time_seconds: float


def unused_path_batches(
    n_paths: int, iterations: int, batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Use nested fit paths evenly; no candidate receives extra trajectories."""

    needed = iterations * batch_size
    blocks = []
    while sum(len(block) for block in blocks) < needed:
        blocks.append(rng.permutation(n_paths))
    return np.concatenate(blocks)[:needed].reshape(iterations, batch_size)


@torch.no_grad()
def validation_cost(
    policy: nn.Module,
    validation: TrajectoryBatch,
    system,
    n_regimes: int,
    device: str,
) -> float:
    policy.eval()
    return float(
        simulate(
            policy,
            validation,
            system,
            n_regimes=n_regimes,
            device=device,
        ).mean_cost.cpu()
    )


def fit_direct_candidate(
    policy: nn.Module,
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system,
    n_regimes: int,
    config: DirectTrainingConfig,
    train_seed: int,
    checkpoint_path: str | Path,
    provenance: dict | None = None,
) -> DirectCandidate:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    rng = np.random.default_rng(train_seed)
    policy.to(config.device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batches = unused_path_batches(
        fit.n, config.iterations, config.batch_size, rng
    )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_state = copy.deepcopy(policy.state_dict())
    best_validation = validation_cost(
        policy, validation, system, n_regimes, config.device
    )
    best_iteration = 0
    history: list[dict[str, float | int | str]] = [
        {
            "iteration": 0,
            "train_seed": train_seed,
            "learning_rate": config.learning_rate,
            "fit_trajectories_seen": 0,
            "environment_transitions": 0,
            "gradient_updates": 0,
            "train_cost": float("nan"),
            "validation_cost": best_validation,
            "is_best": 1,
        }
    ]
    start = time.perf_counter()
    for iteration in range(1, config.iterations + 1):
        policy.train()
        batch = fit.take(batches[iteration - 1], name="fit_minibatch")
        trace = simulate(
            policy, batch, system, n_regimes=n_regimes, device=config.device
        )
        optimizer.zero_grad(set_to_none=True)
        trace.mean_cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
        optimizer.step()
        should_validate = iteration % config.validation_every == 0 or iteration == config.iterations
        current_validation = float("nan")
        is_best = 0
        if should_validate:
            current_validation = validation_cost(
                policy, validation, system, n_regimes, config.device
            )
            if current_validation < best_validation:
                best_validation = current_validation
                best_iteration = iteration
                best_state = copy.deepcopy(policy.state_dict())
                is_best = 1
        history.append(
            {
                "iteration": iteration,
                "train_seed": train_seed,
                "learning_rate": config.learning_rate,
                "fit_trajectories_seen": iteration * config.batch_size,
                "environment_transitions": iteration * config.batch_size * system.horizon,
                "gradient_updates": iteration,
                "train_cost": float(trace.mean_cost.detach().cpu()),
                "validation_cost": current_validation,
                "is_best": is_best,
            }
        )
    elapsed = time.perf_counter() - start
    policy.load_state_dict(best_state)
    policy.eval()
    payload = {
        "policy_name": getattr(policy, "policy_name", policy.__class__.__name__),
        "policy_class": policy.__class__.__name__,
        "train_seed": train_seed,
        "fit_seed": fit.seed,
        "validation_seed": validation.seed,
        "selected_iteration": best_iteration,
        "best_validation_cost": best_validation,
        "training_config": asdict(config),
        "system": system.as_jsonable(),
        "n_regimes": n_regimes,
        "state_dict": best_state,
        "provenance": {} if provenance is None else provenance,
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    return DirectCandidate(
        policy=policy,
        train_seed=train_seed,
        learning_rate=config.learning_rate,
        best_validation_cost=best_validation,
        selected_iteration=best_iteration,
        fit_time_seconds=elapsed,
        environment_transitions=config.iterations * config.batch_size * system.horizon,
        gradient_updates=config.iterations,
        history=history,
        checkpoint_path=str(checkpoint_path),
    )


def tune_direct(
    policy_builder: Callable[[], nn.Module],
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system,
    n_regimes: int,
    base_config: DirectTrainingConfig,
    train_seed: int,
    learning_rates: tuple[float, ...],
    checkpoint_dir: str | Path,
    provenance: dict | None = None,
) -> DirectSelection:
    if not learning_rates:
        raise ValueError("At least one learning rate is required.")
    checkpoint_dir = Path(checkpoint_dir)
    candidates = []
    for index, learning_rate in enumerate(learning_rates):
        torch.manual_seed(train_seed)
        policy = policy_builder()
        config = DirectTrainingConfig(
            **{**asdict(base_config), "learning_rate": float(learning_rate)}
        )
        candidates.append(
            fit_direct_candidate(
                policy,
                fit,
                validation,
                system,
                n_regimes,
                config,
                train_seed,
                checkpoint_dir / f"candidate_{index}_seed{train_seed}.pt",
                provenance,
            )
        )
    selected = min(candidates, key=lambda candidate: candidate.best_validation_cost)
    return DirectSelection(
        selected=selected,
        candidates=candidates,
        total_tuning_time_seconds=sum(candidate.fit_time_seconds for candidate in candidates),
    )

