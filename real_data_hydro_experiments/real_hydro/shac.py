"""Projected short-horizon actor-critic on the shared reservoir simulator."""

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
from .simulator import SimulationTrace, simulate
from .train import unused_path_batches


@dataclass(frozen=True)
class SHACConfig:
    iterations: int
    batch_size: int = 2
    validation_every: int = 10
    window: int = 8
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 1.0e-3
    critic_epochs: int = 2
    critic_tau: float = 0.05
    cost_scale: float = 1_000.0
    weight_decay: float = 1.0e-6
    gradient_clip: float = 10.0
    critic_width: int = 256
    critic_depth: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.batch_size,
            self.validation_every,
            self.window,
            self.critic_epochs,
            self.critic_width,
            self.critic_depth,
        )
        if min(positive) < 1 or min(
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.cost_scale,
            self.gradient_clip,
        ) <= 0.0:
            raise ValueError("SHAC sizes/rates must be positive.")
        if not 0.0 < self.critic_tau <= 1.0:
            raise ValueError("critic_tau must lie in (0, 1].")


class CostCritic(nn.Module):
    def __init__(self, system, feature_dim: int, width: int, depth: int) -> None:
        super().__init__()
        input_dim = 2 * system.n_reservoirs + feature_dim + 2
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layer = nn.Linear(current, width)
            nn.init.orthogonal_(layer.weight, gain=2.0**0.5)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.ReLU()))
            current = width
        output = nn.Linear(current, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


@dataclass
class SHACCandidate:
    policy: nn.Module
    critic: CostCritic
    train_seed: int
    actor_learning_rate: float
    best_validation_cost: float
    selected_iteration: int
    fit_time_seconds: float
    actor_gradient_updates: int
    critic_gradient_updates: int
    environment_transitions: int
    history: list[dict[str, float | int | str]]
    checkpoint_path: str


@dataclass
class SHACSelection:
    selected: SHACCandidate
    candidates: list[SHACCandidate]
    total_tuning_time_seconds: float


def critic_observation(
    stage: int,
    storage: torch.Tensor,
    previous_release: torch.Tensor,
    features: torch.Tensor,
    system,
) -> torch.Tensor:
    capacity = torch.as_tensor(
        system.storage_capacity, dtype=storage.dtype, device=storage.device
    )
    release = torch.as_tensor(
        system.release_capacity, dtype=storage.dtype, device=storage.device
    )
    angle = 2.0 * np.pi * system.period(stage) / system.periods
    season = torch.tensor(
        (np.sin(angle), np.cos(angle)), dtype=storage.dtype, device=storage.device
    ).expand(storage.shape[0], -1)
    return torch.cat(
        (storage / capacity, previous_release / release, features, season), dim=-1
    )


def critic_observations(trace: SimulationTrace) -> torch.Tensor:
    return torch.stack(
        [
            critic_observation(
                stage,
                trace.storage_before_action[:, stage],
                trace.previous_release_before_action[:, stage],
                trace.policy_features[:, stage],
                trace.system,
            )
            for stage in range(trace.system.horizon)
        ],
        dim=1,
    )


def realized_cost_to_go(trace: SimulationTrace) -> torch.Tensor:
    future = trace.terminal_cost
    values = []
    for stage in range(trace.system.horizon - 1, -1, -1):
        future = trace.stage_cost[:, stage] + future
        values.append(future)
    return torch.stack(list(reversed(values)), dim=1)


def actor_objective(
    trace: SimulationTrace,
    target_critic: CostCritic,
    window: int,
    cost_scale: float,
) -> torch.Tensor:
    objective = trace.total_cost
    for boundary in range(window, trace.system.horizon, window):
        observation = critic_observation(
            boundary,
            trace.storage[:, boundary - 1],
            trace.actions[:, boundary - 1],
            trace.policy_features[:, boundary],
            trace.system,
        )
        objective = objective + float(cost_scale) * target_critic(observation)
    return objective.mean()


@torch.no_grad()
def _validation(policy, dataset, system, n_regimes, device) -> float:
    policy.eval()
    return float(
        simulate(
            policy, dataset, system, n_regimes=n_regimes, device=device
        ).mean_cost.cpu()
    )


def fit_shac_candidate(
    policy: nn.Module,
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system,
    n_regimes: int,
    config: SHACConfig,
    train_seed: int,
    checkpoint_path: str | Path,
    provenance: dict | None = None,
) -> SHACCandidate:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    rng = np.random.default_rng(train_seed)
    policy.to(config.device)
    policy.policy_name = "projected_shac"
    critic = CostCritic(
        system, n_regimes, config.critic_width, config.critic_depth
    ).to(config.device)
    target_critic = copy.deepcopy(critic).to(config.device)
    target_critic.eval()
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.actor_learning_rate,
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.critic_learning_rate,
        weight_decay=config.weight_decay,
    )
    batches = unused_path_batches(fit.n, config.iterations, config.batch_size, rng)
    best_policy = copy.deepcopy(policy.state_dict())
    best_critic = copy.deepcopy(critic.state_dict())
    best_validation = _validation(
        policy, validation, system, n_regimes, config.device
    )
    best_iteration = 0
    history = []
    start = time.perf_counter()
    for iteration in range(1, config.iterations + 1):
        batch = fit.take(batches[iteration - 1], name="shac_minibatch")
        policy.train()
        critic.train()
        trace = simulate(
            policy,
            batch,
            system,
            n_regimes=n_regimes,
            device=config.device,
            detach_state_every=config.window,
        )
        actor_optimizer.zero_grad(set_to_none=True)
        loss = actor_objective(trace, target_critic, config.window, config.cost_scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
        actor_optimizer.step()

        observations = critic_observations(trace).detach().flatten(0, 1)
        targets = (realized_cost_to_go(trace) / config.cost_scale).detach().flatten()
        critic_loss_value = float("nan")
        for _ in range(config.critic_epochs):
            critic_optimizer.zero_grad(set_to_none=True)
            prediction = critic(observations)
            critic_loss = torch.nn.functional.mse_loss(prediction, targets)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
            critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach().cpu())
        with torch.no_grad():
            for target, source in zip(
                target_critic.parameters(), critic.parameters(), strict=True
            ):
                target.mul_(1.0 - config.critic_tau).add_(
                    config.critic_tau * source
                )

        should_validate = iteration % config.validation_every == 0 or iteration == config.iterations
        validation_value = float("nan")
        is_best = 0
        if should_validate:
            validation_value = _validation(
                policy, validation, system, n_regimes, config.device
            )
            if validation_value < best_validation:
                best_validation = validation_value
                best_iteration = iteration
                best_policy = copy.deepcopy(policy.state_dict())
                best_critic = copy.deepcopy(critic.state_dict())
                is_best = 1
        history.append(
            {
                "iteration": iteration,
                "train_seed": train_seed,
                "actor_loss": float(loss.detach().cpu()),
                "critic_loss": critic_loss_value,
                "train_cost": float(trace.mean_cost.detach().cpu()),
                "validation_cost": validation_value,
                "is_best": is_best,
                "environment_transitions": iteration * config.batch_size * system.horizon,
            }
        )
    elapsed = time.perf_counter() - start
    policy.load_state_dict(best_policy)
    critic.load_state_dict(best_critic)
    policy.eval()
    critic.eval()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "policy_name": "projected_shac",
            "train_seed": train_seed,
            "fit_seed": fit.seed,
            "validation_seed": validation.seed,
            "selected_iteration": best_iteration,
            "best_validation_cost": best_validation,
            "shac_config": asdict(config),
            "system": system.as_jsonable(),
            "n_regimes": n_regimes,
            "policy_state_dict": best_policy,
            "critic_state_dict": best_critic,
            "provenance": {} if provenance is None else provenance,
        },
        temporary,
    )
    temporary.replace(checkpoint_path)
    return SHACCandidate(
        policy=policy,
        critic=critic,
        train_seed=train_seed,
        actor_learning_rate=config.actor_learning_rate,
        best_validation_cost=best_validation,
        selected_iteration=best_iteration,
        fit_time_seconds=elapsed,
        actor_gradient_updates=config.iterations,
        critic_gradient_updates=config.iterations * config.critic_epochs,
        environment_transitions=config.iterations * config.batch_size * system.horizon,
        history=history,
        checkpoint_path=str(checkpoint_path),
    )


def tune_shac(
    policy_builder: Callable[[], nn.Module],
    fit,
    validation,
    system,
    n_regimes,
    base_config: SHACConfig,
    train_seed: int,
    learning_rates: tuple[float, ...],
    checkpoint_dir: str | Path,
    provenance: dict | None = None,
) -> SHACSelection:
    candidates = []
    for index, learning_rate in enumerate(learning_rates):
        torch.manual_seed(train_seed)
        config = SHACConfig(
            **{**asdict(base_config), "actor_learning_rate": learning_rate}
        )
        candidates.append(
            fit_shac_candidate(
                policy_builder(),
                fit,
                validation,
                system,
                n_regimes,
                config,
                train_seed,
                Path(checkpoint_dir) / f"candidate_{index}_seed{train_seed}.pt",
                provenance,
            )
        )
    selected = min(candidates, key=lambda item: item.best_validation_cost)
    return SHACSelection(
        selected=selected,
        candidates=candidates,
        total_tuning_time_seconds=sum(item.fit_time_seconds for item in candidates),
    )

