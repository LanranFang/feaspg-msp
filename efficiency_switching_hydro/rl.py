"""Stable-Baselines3 comparators with validation-selected checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from .config import ExperimentConfig, RL_ALGORITHMS, RLConfig
from .data import Dataset
from .evaluation import evaluate_with_costs
from .rl_env import (
    HydroTrajectoryEnv,
    normalized_action_to_proposal,
    rl_observation,
)


ALGORITHM_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}


def canonical_algorithm(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized.startswith("rl_"):
        normalized = normalized[3:]
    if normalized not in RL_ALGORITHMS:
        raise ValueError(
            f"Unknown RL algorithm {name!r}; choose from {', '.join(RL_ALGORITHMS)}."
        )
    return normalized


class SB3PolicyAdapter:
    """Expose a deterministic SB3 actor through the shared simulator interface."""

    def __init__(
        self,
        model: BaseAlgorithm,
        algorithm: str,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.algorithm = canonical_algorithm(algorithm)
        self.config = config
        self.policy_name = f"rl_{self.algorithm}"

    def eval(self) -> "SB3PolicyAdapter":
        self.model.policy.set_training_mode(False)
        return self

    def propose(
        self,
        stage: int,
        storage: torch.Tensor,
        previous_release: torch.Tensor,
        queue: torch.Tensor,
    ) -> torch.Tensor:
        observation = rl_observation(
            stage,
            storage.detach().cpu().numpy(),
            previous_release.detach().cpu().numpy(),
            queue.detach().cpu().numpy(),
            self.config,
        )
        normalized_action, _ = self.model.predict(
            observation, deterministic=True
        )
        proposal = normalized_action_to_proposal(
            normalized_action, stage, self.config
        )
        return torch.as_tensor(
            proposal, dtype=storage.dtype, device=storage.device
        )

    def close(self) -> None:
        environment = self.model.get_env()
        if environment is not None:
            environment.close()


def _unique_parameter_count(parameters) -> int:
    seen: set[int] = set()
    count = 0
    for parameter in parameters:
        identifier = id(parameter)
        if identifier not in seen:
            seen.add(identifier)
            count += parameter.numel()
    return int(count)


def model_parameter_counts(
    model: BaseAlgorithm, algorithm: str
) -> tuple[int, int]:
    """Return deployment actor parameters and all train-time policy parameters."""

    algorithm = canonical_algorithm(algorithm)
    training = _unique_parameter_count(model.policy.parameters())
    if algorithm == "ppo":
        actor_parameters = list(model.policy.features_extractor.parameters())
        actor_parameters.extend(model.policy.mlp_extractor.policy_net.parameters())
        actor_parameters.extend(model.policy.action_net.parameters())
        actor_parameters.append(model.policy.log_std)
        deployment = _unique_parameter_count(actor_parameters)
    else:
        deployment = _unique_parameter_count(model.actor.parameters())
    return deployment, training


def _largest_divisor_at_most(value: int, target: int) -> int:
    for candidate in range(min(value, target), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def build_model(
    algorithm: str,
    environment: HydroTrajectoryEnv,
    config: ExperimentConfig,
    rl_config: RLConfig,
    n_fit_episodes: int,
    train_seed: int,
) -> BaseAlgorithm:
    """Build a standard SB3 model without using validation information."""

    algorithm = canonical_algorithm(algorithm)
    layers = [rl_config.network_width] * rl_config.network_depth
    total_steps = n_fit_episodes * config.T
    common: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": environment,
        "gamma": rl_config.gamma,
        "seed": train_seed,
        "device": rl_config.device,
        "verbose": 0,
    }
    if algorithm == "ppo":
        rollout_episodes = _largest_divisor_at_most(
            n_fit_episodes, rl_config.ppo_rollout_episodes
        )
        n_steps = rollout_episodes * config.T
        batch_size = _largest_divisor_at_most(
            n_steps, min(n_steps, rl_config.ppo_batch_size)
        )
        return PPO(
            **common,
            learning_rate=rl_config.ppo_learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=rl_config.ppo_epochs,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={"net_arch": {"pi": layers, "vf": layers}},
        )

    learning_starts = min(
        rl_config.learning_starts, max(1, total_steps // 10)
    )
    batch_size = min(
        rl_config.off_policy_batch_size,
        max(1, total_steps - learning_starts),
    )
    off_policy = {
        **common,
        "buffer_size": min(rl_config.replay_buffer_size, total_steps),
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "train_freq": (1, "step"),
        "gradient_steps": rl_config.gradient_steps,
        "n_steps": min(rl_config.off_policy_n_steps, config.T),
        "policy_kwargs": {"net_arch": layers},
    }
    if algorithm == "sac":
        return SAC(
            **off_policy,
            learning_rate=rl_config.sac_learning_rate,
            ent_coef="auto",
        )
    action_noise = NormalActionNoise(
        mean=np.zeros(config.n_reservoirs, dtype=np.float32),
        sigma=rl_config.td3_action_noise
        * np.ones(config.n_reservoirs, dtype=np.float32),
    )
    return TD3(
        **off_policy,
        learning_rate=rl_config.td3_learning_rate,
        action_noise=action_noise,
    )


class ValidationCallback(BaseCallback):
    """Select checkpoints only with the independent validation trajectories."""

    def __init__(
        self,
        algorithm: str,
        validation: Dataset,
        config: ExperimentConfig,
        checkpoint_path: Path,
        validation_every_steps: int,
        initial_validation_cost: float,
        history: list[dict[str, float | int | str]],
        evaluation_device: str,
    ) -> None:
        super().__init__(verbose=0)
        self.algorithm = canonical_algorithm(algorithm)
        self.validation = validation
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.validation_every_steps = validation_every_steps
        self.best_validation_cost = initial_validation_cost
        self.selected_step = 0
        self.history = history
        self.evaluation_device = evaluation_device
        self.next_validation_step = validation_every_steps
        self.last_validation_step = 0
        self.recent_episode_costs: list[float] = []

    def _record_validation(self) -> None:
        step = int(self.model.num_timesteps)
        if step == self.last_validation_step:
            return
        was_training = self.model.policy.training
        adapter = SB3PolicyAdapter(self.model, self.algorithm, self.config)
        metrics, _ = evaluate_with_costs(
            adapter,
            self.validation,
            self.config,
            self.evaluation_device,
        )
        if was_training:
            self.model.policy.set_training_mode(True)
        validation_cost = metrics.mean_cost
        is_best = int(validation_cost < self.best_validation_cost)
        if is_best:
            self.best_validation_cost = validation_cost
            self.selected_step = step
            self.model.save(self.checkpoint_path)
        train_cost = (
            float(np.mean(self.recent_episode_costs))
            if self.recent_episode_costs
            else float("nan")
        )
        self.recent_episode_costs.clear()
        self.history.append(
            {
                "policy": f"rl_{self.algorithm}",
                "train_seed": int(self.model.seed),
                "iteration": step // self.config.T,
                "fit_trajectories_seen": step // self.config.T,
                "environment_steps": step,
                "train_cost": train_cost,
                "validation_cost": validation_cost,
                "is_best": is_best,
            }
        )
        self.last_validation_step = step

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_cost" in info:
                self.recent_episode_costs.append(float(info["episode_cost"]))
        if self.model.num_timesteps >= self.next_validation_step:
            self._record_validation()
            while self.next_validation_step <= self.model.num_timesteps:
                self.next_validation_step += self.validation_every_steps
        return True

    def record_final(self) -> None:
        self._record_validation()


@dataclass(frozen=True)
class RLFitResult:
    policy: SB3PolicyAdapter
    policy_name: str
    train_seed: int
    selected_iteration: int
    best_validation_cost: float
    completed_episodes: int
    environment_steps: int
    gradient_updates: int
    actor_gradient_updates: int
    critic_gradient_updates: int
    deployment_parameters: int
    training_parameters: int
    checkpoint_path: str
    history: list[dict[str, float | int | str]]


class _OptimizerStepCounter:
    """Count actual optimizer steps without changing the SB3 training loop."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.value = 0
        original_step = optimizer.step

        def counted_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.value += 1
            return result

        optimizer.step = counted_step


def _gradient_step_counters(
    model: BaseAlgorithm,
    algorithm: str,
) -> tuple[_OptimizerStepCounter, _OptimizerStepCounter]:
    if algorithm == "ppo":
        joint = _OptimizerStepCounter(model.policy.optimizer)
        return joint, joint
    return (
        _OptimizerStepCounter(model.actor.optimizer),
        _OptimizerStepCounter(model.critic.optimizer),
    )


def fit_checkpoint(
    algorithm: str,
    fit: Dataset,
    validation: Dataset,
    config: ExperimentConfig,
    rl_config: RLConfig,
    train_seed: int,
    checkpoint_path: str | Path,
    evaluation_device: str = "cpu",
) -> RLFitResult:
    """Fit one RL seed using each fit trajectory exactly once."""

    algorithm = canonical_algorithm(algorithm)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".zip":
        checkpoint_path = checkpoint_path.with_suffix(".zip")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    environment = HydroTrajectoryEnv(
        fit, config, rl_config.reward_scale, train_seed
    )
    model = build_model(
        algorithm, environment, config, rl_config, fit.n, train_seed
    )
    actor_steps, critic_steps = _gradient_step_counters(model, algorithm)
    initial_adapter = SB3PolicyAdapter(model, algorithm, config)
    initial_metrics, _ = evaluate_with_costs(
        initial_adapter, validation, config, evaluation_device
    )
    initial_validation = initial_metrics.mean_cost
    model.save(checkpoint_path)
    history: list[dict[str, float | int | str]] = [
        {
            "policy": f"rl_{algorithm}",
            "train_seed": train_seed,
            "iteration": 0,
            "fit_trajectories_seen": 0,
            "environment_steps": 0,
            "train_cost": float("nan"),
            "validation_cost": initial_validation,
            "is_best": 1,
        }
    ]
    callback = ValidationCallback(
        algorithm=algorithm,
        validation=validation,
        config=config,
        checkpoint_path=checkpoint_path,
        validation_every_steps=rl_config.validation_every_episodes * config.T,
        initial_validation_cost=initial_validation,
        history=history,
        evaluation_device=evaluation_device,
    )
    total_steps = fit.n * config.T
    model.learn(total_timesteps=total_steps, callback=callback)
    callback.record_final()

    if environment.transitions_taken != total_steps:
        raise RuntimeError(
            "RL trainer did not consume the prescribed environment-step budget."
        )
    if environment.episodes_completed != fit.n:
        raise RuntimeError("RL trainer did not complete exactly n_fit episodes.")
    completed_paths = environment.completed_path_indices
    if len(set(completed_paths)) != fit.n:
        raise RuntimeError("An RL fit trajectory was reused before all paths were consumed.")

    deployment_parameters, training_parameters = model_parameter_counts(
        model, algorithm
    )
    gradient_updates = int(getattr(model, "_n_updates", 0))
    metadata_path = checkpoint_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "algorithm": algorithm,
                "policy_name": f"rl_{algorithm}",
                "profile": config.profile,
                "train_seed": train_seed,
                "selected_iteration": callback.selected_step // config.T,
                "selected_environment_step": callback.selected_step,
                "best_validation_cost": callback.best_validation_cost,
                "completed_episodes": fit.n,
                "environment_steps": total_steps,
                "gradient_updates": gradient_updates,
                "actor_gradient_updates": actor_steps.value,
                "critic_gradient_updates": critic_steps.value,
                "deployment_parameters": deployment_parameters,
                "training_parameters": training_parameters,
                "experiment_config": asdict(config),
                "rl_config": asdict(rl_config),
            },
            handle,
            indent=2,
        )

    model.get_env().close()
    loaded = ALGORITHM_CLASSES[algorithm].load(
        checkpoint_path, device=rl_config.device
    )
    policy = SB3PolicyAdapter(loaded, algorithm, config)
    return RLFitResult(
        policy=policy,
        policy_name=f"rl_{algorithm}",
        train_seed=train_seed,
        selected_iteration=callback.selected_step // config.T,
        best_validation_cost=callback.best_validation_cost,
        completed_episodes=fit.n,
        environment_steps=total_steps,
        gradient_updates=gradient_updates,
        actor_gradient_updates=actor_steps.value,
        critic_gradient_updates=critic_steps.value,
        deployment_parameters=deployment_parameters,
        training_parameters=training_parameters,
        checkpoint_path=str(checkpoint_path),
        history=history,
    )


def load_checkpoint(
    algorithm: str,
    checkpoint_path: str | Path,
    config: ExperimentConfig,
    device: str = "auto",
) -> tuple[SB3PolicyAdapter, dict[str, Any]]:
    """Load a fitted actor and its reproducibility metadata."""

    algorithm = canonical_algorithm(algorithm)
    checkpoint_path = Path(checkpoint_path)
    model = ALGORITHM_CLASSES[algorithm].load(checkpoint_path, device=device)
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["algorithm"] != algorithm or metadata["profile"] != config.profile:
        raise ValueError(f"Checkpoint {checkpoint_path} is incompatible with this run.")
    return SB3PolicyAdapter(model, algorithm, config), metadata
