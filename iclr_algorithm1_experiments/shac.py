"""Projected short-horizon actor-critic for the shared hydro simulator."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.policy_features import flat_markov_observation_torch
from efficiency_switching_hydro.simulator import Trace, simulate
from efficiency_switching_hydro.train import unused_batches


@dataclass(frozen=True)
class SHACConfig:
    iterations: int
    batch_size: int
    validation_every: int
    window: int
    actor_learning_rate: float
    critic_learning_rate: float
    critic_epochs: int
    critic_tau: float
    cost_scale: float
    weight_decay: float
    gradient_clip: float
    critic_width: int
    critic_depth: int
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
        if any(value < 1 for value in positive):
            raise ValueError("SHAC iteration and architecture settings must be positive.")
        if min(
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.cost_scale,
        ) <= 0.0:
            raise ValueError("SHAC rates and cost scale must be positive.")
        if not 0.0 < self.critic_tau <= 1.0:
            raise ValueError("critic_tau must lie in (0, 1].")


class CostCritic(nn.Module):
    def __init__(self, config, width: int, depth: int) -> None:
        super().__init__()
        input_dim = config.n_reservoirs * (2 + config.routing_delay) + 4
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
class SHACCandidateFit:
    policy: nn.Module
    critic: CostCritic
    actor_learning_rate: float
    best_validation_cost: float
    selected_iteration: int
    fit_time_seconds: float
    actor_gradient_updates: int
    critic_gradient_updates: int
    environment_transitions: int
    deployment_parameters: int
    training_parameters: int
    history: list[dict[str, float | int | str]]
    checkpoint_path: str


@dataclass
class SHACSelectedFit:
    policy: nn.Module
    critic: CostCritic
    actor_learning_rate: float
    best_validation_cost: float
    selected_iteration: int
    selected_fit_time_seconds: float
    total_tuning_time_seconds: float
    candidates: list[SHACCandidateFit]
    checkpoint_path: str


def critic_observations(trace: Trace, config) -> torch.Tensor:
    observations = []
    for stage in range(config.T):
        observations.append(
            flat_markov_observation_torch(
                stage,
                trace.storage_before_action[:, stage],
                trace.previous_release_before_action[:, stage],
                trace.queue_before_action[:, stage],
                config,
            )
        )
    return torch.stack(observations, dim=1)


def realized_cost_to_go(trace: Trace) -> torch.Tensor:
    future = trace.terminal_cost
    values = []
    for stage in range(trace.stage_cost.shape[1] - 1, -1, -1):
        future = trace.stage_cost[:, stage] + future
        values.append(future)
    return torch.stack(list(reversed(values)), dim=1)


def shac_actor_objective(
    trace: Trace,
    target_critic: CostCritic,
    config,
    window: int,
    cost_scale: float,
) -> torch.Tensor:
    path_objective = trace.stage_cost.sum(dim=1) + trace.terminal_cost
    for boundary in range(window, config.T, window):
        observation = flat_markov_observation_torch(
            boundary,
            trace.storage[:, boundary - 1],
            trace.actions[:, boundary - 1],
            trace.queues[:, boundary - 1],
            config,
        )
        path_objective = path_objective + cost_scale * target_critic(observation)
    return path_objective.mean()


@torch.no_grad()
def validation_cost(policy, validation: Dataset, config, device: str) -> float:
    policy.eval()
    return float(simulate(policy, validation, config, device=device).mean_cost.cpu())


def fit_shac_candidate(
    policy: nn.Module,
    fit: Dataset,
    validation: Dataset,
    config,
    shac: SHACConfig,
    train_seed: int,
    checkpoint_path: str | Path,
) -> SHACCandidateFit:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    rng = np.random.default_rng(train_seed)
    policy.to(shac.device)
    policy.policy_name = "projected_shac"
    critic = CostCritic(config, shac.critic_width, shac.critic_depth).to(shac.device)
    target_critic = copy.deepcopy(critic).to(shac.device)
    target_critic.eval()
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=shac.actor_learning_rate,
        weight_decay=shac.weight_decay,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=shac.critic_learning_rate,
        weight_decay=shac.weight_decay,
    )
    batches = unused_batches(fit.n, shac.iterations, shac.batch_size, rng)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_policy_state = copy.deepcopy(policy.state_dict())
    best_critic_state = copy.deepcopy(critic.state_dict())
    best_iteration = 0
    best_validation = validation_cost(policy, validation, config, shac.device)
    history: list[dict[str, float | int | str]] = [
        {
            "policy": "projected_shac",
            "train_seed": train_seed,
            "iteration": 0,
            "fit_trajectories_seen": 0,
            "environment_transitions": 0,
            "actor_gradient_updates": 0,
            "critic_gradient_updates": 0,
            "train_cost": float("nan"),
            "actor_loss": float("nan"),
            "critic_loss": float("nan"),
            "validation_cost": best_validation,
            "is_best": 1,
        }
    ]

    start = time.perf_counter()
    for iteration in range(1, shac.iterations + 1):
        policy.train()
        critic.train()
        batch = fit.take(batches[iteration - 1], name="fit_batch")
        trace = simulate(
            policy,
            batch,
            config,
            device=shac.device,
            detach_state_every=shac.window,
        )

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = shac_actor_objective(
            trace,
            target_critic,
            config,
            shac.window,
            shac.cost_scale,
        )
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), shac.gradient_clip)
        actor_optimizer.step()

        observations = critic_observations(trace, config).detach()
        targets = (realized_cost_to_go(trace) / shac.cost_scale).detach()
        flat_observations = observations.flatten(0, 1)
        flat_targets = targets.flatten()
        critic_loss_value = float("nan")
        for _ in range(shac.critic_epochs):
            critic_optimizer.zero_grad(set_to_none=True)
            prediction = critic(flat_observations)
            critic_loss = torch.nn.functional.mse_loss(prediction, flat_targets)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), shac.gradient_clip)
            critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach().cpu())

        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_critic.parameters(), critic.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - shac.critic_tau)
                target_parameter.add_(shac.critic_tau * parameter)

        should_validate = (
            iteration % shac.validation_every == 0 or iteration == shac.iterations
        )
        current_validation = float("nan")
        is_best = 0
        if should_validate:
            current_validation = validation_cost(
                policy,
                validation,
                config,
                shac.device,
            )
            if current_validation < best_validation:
                best_validation = current_validation
                best_iteration = iteration
                best_policy_state = copy.deepcopy(policy.state_dict())
                best_critic_state = copy.deepcopy(critic.state_dict())
                is_best = 1
        history.append(
            {
                "policy": "projected_shac",
                "train_seed": train_seed,
                "iteration": iteration,
                "fit_trajectories_seen": iteration * shac.batch_size,
                "environment_transitions": iteration * shac.batch_size * config.T,
                "actor_gradient_updates": iteration,
                "critic_gradient_updates": iteration * shac.critic_epochs,
                "train_cost": float(trace.mean_cost.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": critic_loss_value,
                "validation_cost": current_validation,
                "is_best": is_best,
            }
        )

    fit_time = time.perf_counter() - start
    policy.load_state_dict(best_policy_state)
    critic.load_state_dict(best_critic_state)
    policy.eval()
    critic.eval()
    deployment_parameters = sum(parameter.numel() for parameter in policy.parameters())
    critic_parameters = sum(parameter.numel() for parameter in critic.parameters())
    training_parameters = deployment_parameters + 2 * critic_parameters
    torch.save(
        {
            "policy_name": "projected_shac",
            "train_seed": train_seed,
            "selected_iteration": best_iteration,
            "best_validation_cost": best_validation,
            "shac_config": asdict(shac),
            "experiment_config": asdict(config),
            "policy_state_dict": best_policy_state,
            "critic_state_dict": best_critic_state,
            "deployment_parameters": deployment_parameters,
            "training_parameters": training_parameters,
        },
        checkpoint_path,
    )
    return SHACCandidateFit(
        policy=policy,
        critic=critic,
        actor_learning_rate=shac.actor_learning_rate,
        best_validation_cost=best_validation,
        selected_iteration=best_iteration,
        fit_time_seconds=fit_time,
        actor_gradient_updates=shac.iterations,
        critic_gradient_updates=shac.iterations * shac.critic_epochs,
        environment_transitions=shac.iterations * shac.batch_size * config.T,
        deployment_parameters=deployment_parameters,
        training_parameters=training_parameters,
        history=history,
        checkpoint_path=str(checkpoint_path),
    )


def tune_shac(
    policy_builder: Callable[[], nn.Module],
    fit: Dataset,
    validation: Dataset,
    config,
    base_config: SHACConfig,
    train_seed: int,
    learning_rates: tuple[float, ...],
    checkpoint_dir: Path,
) -> SHACSelectedFit:
    candidates = []
    for index, learning_rate in enumerate(learning_rates):
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        policy = policy_builder()
        candidate_config = SHACConfig(
            **{
                **asdict(base_config),
                "actor_learning_rate": learning_rate,
            }
        )
        rate = f"{learning_rate:.8g}".replace(".", "p")
        candidates.append(
            fit_shac_candidate(
                policy,
                fit,
                validation,
                config,
                candidate_config,
                train_seed,
                checkpoint_dir
                / f"projected_shac_candidate{index}_lr{rate}_seed{train_seed}.pt",
            )
        )
    selected = min(candidates, key=lambda item: item.best_validation_cost)
    selected_path = checkpoint_dir / f"projected_shac_selected_seed{train_seed}.pt"
    payload = torch.load(selected.checkpoint_path, map_location="cpu", weights_only=False)
    payload["candidate_learning_rates"] = list(learning_rates)
    payload["total_tuning_time_seconds"] = sum(
        candidate.fit_time_seconds for candidate in candidates
    )
    torch.save(payload, selected_path)
    return SHACSelectedFit(
        policy=selected.policy,
        critic=selected.critic,
        actor_learning_rate=selected.actor_learning_rate,
        best_validation_cost=selected.best_validation_cost,
        selected_iteration=selected.selected_iteration,
        selected_fit_time_seconds=selected.fit_time_seconds,
        total_tuning_time_seconds=sum(
            candidate.fit_time_seconds for candidate in candidates
        ),
        candidates=candidates,
        checkpoint_path=str(selected_path),
    )
