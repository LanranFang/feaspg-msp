"""PPO/SAC comparators using the common projected action and information state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from .dataset import TrajectoryBatch
from .evaluation import evaluate_policy
from .projection import project_release_np


@dataclass(frozen=True)
class RLConfig:
    algorithm: str
    learning_rate: float = 3.0e-4
    reward_scale: float = 1_000.0
    gamma: float = 1.0
    validation_every_episodes: int = 50
    network_width: int = 256
    network_depth: int = 2
    ppo_rollout_episodes: int = 10
    ppo_epochs: int = 10
    ppo_batch_size: int = 256
    replay_buffer_size: int = 100_000
    learning_starts: int = 1_000
    off_policy_batch_size: int = 256
    off_policy_gradient_steps: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "sac"}:
            raise ValueError("RL algorithm must be ppo or sac.")
        counts = (
            self.validation_every_episodes,
            self.network_width,
            self.network_depth,
            self.ppo_rollout_episodes,
            self.ppo_epochs,
            self.ppo_batch_size,
            self.replay_buffer_size,
            self.off_policy_batch_size,
            self.off_policy_gradient_steps,
        )
        if min(counts) < 1 or self.learning_starts < 0:
            raise ValueError("RL counts must be positive (learning_starts may be zero).")
        if min(self.learning_rate, self.reward_scale, self.gamma) <= 0.0:
            raise ValueError("RL rate/scale/gamma must be positive.")


@dataclass
class RLCandidate:
    policy: "SB3PolicyAdapter"
    algorithm: str
    train_seed: int
    learning_rate: float
    best_validation_cost: float
    selected_timesteps: int
    environment_transitions: int
    fit_time_seconds: float
    history: list[dict[str, float | int | str]]
    checkpoint_path: str

    def close(self) -> None:
        self.policy.close()


@dataclass
class RLSelection:
    selected: RLCandidate
    candidates: list[RLCandidate]
    total_tuning_time_seconds: float


def normalized_observation(
    stage: int,
    storage: np.ndarray,
    previous_release: np.ndarray,
    regime_features: np.ndarray,
    system,
) -> np.ndarray:
    angle = 2.0 * np.pi * system.period(stage) / system.periods
    return np.concatenate(
        (
            np.asarray(storage) / np.asarray(system.storage_capacity),
            np.asarray(previous_release) / np.asarray(system.release_capacity),
            np.asarray(regime_features),
            np.asarray((np.sin(angle), np.cos(angle))),
        ),
        axis=-1,
    ).astype(np.float32)


def _thermal_cost_np(net_demand: float, objective) -> tuple[float, float, float]:
    remaining = max(float(net_demand), 0.0)
    cost = 0.0
    generation = 0.0
    for capacity, marginal_cost in zip(
        objective.thermal_block_capacity,
        objective.thermal_block_cost,
        strict=True,
    ):
        used = min(remaining, float(capacity))
        cost += used * float(marginal_cost)
        generation += used
        remaining -= used
    return cost + remaining * float(objective.deficit_cost), generation, remaining


def reservoir_step_np(
    system,
    stage: int,
    storage: np.ndarray,
    previous_release: np.ndarray,
    proposal: np.ndarray,
    inflow: np.ndarray,
    demand: float | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """NumPy mirror of one stage in :func:`real_hydro.simulator.simulate`."""

    storage = np.asarray(storage, dtype=np.float64)
    previous_release = np.asarray(previous_release, dtype=np.float64)
    action = project_release_np(
        np.asarray(proposal, dtype=np.float64)[None], storage[None], system
    )[0]
    raw_storage = (
        storage - action + np.asarray(inflow, dtype=np.float64)
        + action @ np.asarray(system.routing, dtype=np.float64)
    )
    capacity = np.asarray(system.storage_capacity, dtype=np.float64)
    spill = np.maximum(raw_storage - capacity, 0.0)
    next_storage = np.minimum(raw_storage, capacity)
    objective = system.objective
    period = system.period(stage)
    target = np.asarray(objective.release_target[period], dtype=np.float64)
    lower = np.asarray(objective.guide_lower[period], dtype=np.float64)
    upper = np.asarray(objective.guide_upper[period], dtype=np.float64)
    shortage = np.maximum(target - action, 0.0)
    excess = np.maximum(action - target, 0.0)
    low = np.maximum(lower - next_storage, 0.0)
    flood = np.maximum(next_storage - upper, 0.0)
    ramp = np.abs(action - previous_release)
    cost = float(
        shortage @ np.asarray(objective.shortage_cost)
        + excess @ np.asarray(objective.excess_release_cost)
        + low @ np.asarray(objective.low_storage_cost)
        + flood @ np.asarray(objective.flood_cost)
        + spill @ np.asarray(objective.spill_cost)
        + ramp @ np.asarray(objective.ramp_cost)
    )
    thermal = deficit = 0.0
    if objective.kind == "hydrothermal":
        realized_demand = (
            float(objective.seasonal_demand[period]) if demand is None else float(demand)
        )
        hydro = float(action @ np.asarray(objective.generation_coefficient))
        thermal_cost, thermal, deficit = _thermal_cost_np(realized_demand - hydro, objective)
        cost += thermal_cost
    components = {
        "stage_cost": cost,
        "shortage": float(shortage.sum()),
        "spill": float(spill.sum()),
        "thermal": float(thermal),
        "deficit": float(deficit),
    }
    return next_storage, action, cost, components


def terminal_cost_np(system, storage: np.ndarray) -> float:
    objective = system.objective
    storage = np.asarray(storage, dtype=np.float64)
    target = np.asarray(objective.terminal_target, dtype=np.float64)
    return float(
        np.maximum(target - storage, 0.0)
        @ np.asarray(objective.terminal_shortfall_cost)
        + np.maximum(storage - target, 0.0)
        @ np.asarray(objective.terminal_excess_cost)
    )


class ReservoirTrainingEnv(gym.Env):
    """Finite fit-pool environment with permutation-without-replacement resets."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: TrajectoryBatch,
        system,
        n_regimes: int,
        reward_scale: float,
        seed: int,
    ) -> None:
        super().__init__()
        if dataset.horizon != system.horizon:
            raise ValueError("Training paths and system horizon do not match.")
        self.dataset = dataset
        self.system = system
        self.n_regimes = int(n_regimes)
        self.reward_scale = float(reward_scale)
        self._rng = np.random.default_rng(seed)
        self._permutation = np.empty(0, dtype=np.int64)
        self._cursor = 0
        observation_dim = 2 * system.n_reservoirs + n_regimes + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(system.n_reservoirs,),
            dtype=np.float32,
        )
        self._features = dataset.policy_features(n_regimes)
        self._path = 0
        self._stage = 0
        self._storage = np.asarray(system.initial_storage, dtype=np.float64)
        self._previous = np.zeros(system.n_reservoirs, dtype=np.float64)

    def _next_path(self) -> int:
        if self._cursor >= len(self._permutation):
            self._permutation = self._rng.permutation(self.dataset.n)
            self._cursor = 0
        path = int(self._permutation[self._cursor])
        self._cursor += 1
        return path

    def _observation(self) -> np.ndarray:
        stage = min(self._stage, self.system.horizon - 1)
        return normalized_observation(
            stage,
            self._storage,
            self._previous,
            self._features[self._path, stage],
            self.system,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._permutation = np.empty(0, dtype=np.int64)
            self._cursor = 0
        self._path = self._next_path()
        self._stage = 0
        self._storage = np.asarray(self.system.initial_storage, dtype=np.float64).copy()
        self._previous = np.zeros(self.system.n_reservoirs, dtype=np.float64)
        return self._observation(), {"path_index": self._path}

    def step(self, normalized_action):
        normalized_action = np.clip(np.asarray(normalized_action), -1.0, 1.0)
        proposal = 0.5 * (normalized_action + 1.0) * np.asarray(
            self.system.release_capacity
        )
        demand = (
            None
            if self.dataset.demand is None
            else float(self.dataset.demand[self._path, self._stage])
        )
        self._storage, action, cost, info = reservoir_step_np(
            self.system,
            self._stage,
            self._storage,
            self._previous,
            proposal,
            self.dataset.inflow[self._path, self._stage],
            demand,
        )
        self._previous = action
        self._stage += 1
        terminated = self._stage == self.system.horizon
        if terminated:
            terminal = terminal_cost_np(self.system, self._storage)
            cost += terminal
            info["terminal_cost"] = terminal
        info["path_index"] = self._path
        info["projected_action"] = action.astype(np.float32)
        reward = -cost / self.reward_scale
        return self._observation(), float(reward), terminated, False, info


class SB3PolicyAdapter(nn.Module):
    """Expose a Stable-Baselines model through the shared simulator interface."""

    def __init__(self, model, env: ReservoirTrainingEnv, algorithm: str) -> None:
        super().__init__()
        self.model = model
        self.training_env = env
        self.system = env.system
        self.n_regimes = env.n_regimes
        self.policy_name = f"rl_{algorithm}"

    def propose(self, stage, storage, previous_release, hydrology_features):
        storage_np = storage.detach().cpu().numpy()
        previous_np = previous_release.detach().cpu().numpy()
        feature_np = hydrology_features.detach().cpu().numpy()
        observations = np.stack(
            [
                normalized_observation(
                    stage,
                    storage_np[index],
                    previous_np[index],
                    feature_np[index],
                    self.system,
                )
                for index in range(len(storage_np))
            ]
        )
        normalized, _ = self.model.predict(observations, deterministic=True)
        proposal = 0.5 * (np.asarray(normalized) + 1.0) * np.asarray(
            self.system.release_capacity
        )
        return torch.as_tensor(proposal, dtype=storage.dtype, device=storage.device)

    @property
    def deployment_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.policy.parameters())

    def close(self) -> None:
        self.training_env.close()


def _algorithm_class(name: str):
    if name == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if name == "sac":
        from stable_baselines3 import SAC

        return SAC
    raise ValueError(name)


def _build_model(config: RLConfig, env, seed: int, total_timesteps: int):
    common = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        policy_kwargs={"net_arch": [config.network_width] * config.network_depth},
        verbose=0,
        seed=seed,
        device=config.device,
    )
    if config.algorithm == "ppo":
        rollout_episodes = min(config.ppo_rollout_episodes, env.dataset.n)
        n_steps = rollout_episodes * env.system.horizon
        if total_timesteps % n_steps:
            # Preserve the exact interaction budget for small smoke fixtures.
            divisors = [value for value in range(1, rollout_episodes + 1) if env.dataset.n % value == 0]
            n_steps = max(divisors) * env.system.horizon
        batch_size = min(config.ppo_batch_size, n_steps)
        while n_steps % batch_size:
            batch_size -= 1
        return _algorithm_class("ppo")(
            **common,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=config.ppo_epochs,
            gae_lambda=1.0,
        ), n_steps
    return _algorithm_class("sac")(
        **common,
        buffer_size=max(config.replay_buffer_size, total_timesteps),
        learning_starts=min(config.learning_starts, max(total_timesteps - 1, 0)),
        batch_size=min(config.off_policy_batch_size, max(total_timesteps, 1)),
        train_freq=1,
        gradient_steps=config.off_policy_gradient_steps,
    ), max(config.validation_every_episodes * env.system.horizon, 1)


def fit_rl_candidate(
    fit,
    validation,
    system,
    n_regimes: int,
    config: RLConfig,
    train_seed: int,
    checkpoint_path: str | Path,
    provenance: dict | None = None,
) -> RLCandidate:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    env = ReservoirTrainingEnv(
        fit, system, n_regimes, config.reward_scale, train_seed
    )
    total_timesteps = fit.n * system.horizon
    model, native_chunk = _build_model(config, env, train_seed, total_timesteps)
    adapter = SB3PolicyAdapter(model, env, config.algorithm)
    best_metrics, _ = evaluate_policy(adapter, validation, system, n_regimes)
    best_validation = best_metrics["cost_mean"]
    best_timesteps = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_base = checkpoint_path.parent / (checkpoint_path.stem + "_best")
    model.save(best_base)
    history = []
    validation_chunk = config.validation_every_episodes * system.horizon
    chunk = native_chunk if config.algorithm == "ppo" else validation_chunk
    if total_timesteps % chunk:
        chunk = system.horizon
    start = time.perf_counter()
    trained = 0
    while trained < total_timesteps:
        requested = min(chunk, total_timesteps - trained)
        model.learn(total_timesteps=requested, reset_num_timesteps=False, progress_bar=False)
        actual = int(model.num_timesteps)
        trained = actual
        metrics, _ = evaluate_policy(adapter, validation, system, n_regimes)
        value = metrics["cost_mean"]
        is_best = 0
        if value < best_validation:
            best_validation = value
            best_timesteps = actual
            model.save(best_base)
            is_best = 1
        history.append(
            {
                "algorithm": config.algorithm,
                "train_seed": train_seed,
                "environment_transitions": actual,
                "validation_cost": value,
                "is_best": is_best,
            }
        )
        if actual >= total_timesteps:
            break
    elapsed = time.perf_counter() - start
    loaded = _algorithm_class(config.algorithm).load(
        str(best_base) + ".zip", env=env, device=config.device
    )
    adapter.model = loaded
    checkpoint_path = checkpoint_path.with_suffix(".zip")
    loaded.save(checkpoint_path)
    metadata = checkpoint_path.with_suffix(".metadata.pt")
    torch.save(
        {
            "algorithm": config.algorithm,
            "train_seed": train_seed,
            "fit_seed": fit.seed,
            "validation_seed": validation.seed,
            "best_validation_cost": best_validation,
            "selected_timesteps": best_timesteps,
            "environment_transitions": int(model.num_timesteps),
            "rl_config": asdict(config),
            "system": system.as_jsonable(),
            "n_regimes": n_regimes,
            "provenance": {} if provenance is None else provenance,
        },
        metadata,
    )
    best_zip = Path(str(best_base) + ".zip")
    best_zip.unlink(missing_ok=True)
    return RLCandidate(
        policy=adapter,
        algorithm=config.algorithm,
        train_seed=train_seed,
        learning_rate=config.learning_rate,
        best_validation_cost=best_validation,
        selected_timesteps=best_timesteps,
        environment_transitions=int(model.num_timesteps),
        fit_time_seconds=elapsed,
        history=history,
        checkpoint_path=str(checkpoint_path),
    )


def tune_rl(
    algorithm,
    fit,
    validation,
    system,
    n_regimes,
    base_config,
    train_seed,
    learning_rates,
    checkpoint_dir,
    provenance=None,
) -> RLSelection:
    candidates = []
    for index, learning_rate in enumerate(learning_rates):
        config = RLConfig(
            **{
                **asdict(base_config),
                "algorithm": algorithm,
                "learning_rate": learning_rate,
            }
        )
        candidates.append(
            fit_rl_candidate(
                fit,
                validation,
                system,
                n_regimes,
                config,
                train_seed,
                Path(checkpoint_dir) / f"candidate_{index}_seed{train_seed}.zip",
                provenance,
            )
        )
    selected_index = min(
        range(len(candidates)), key=lambda index: candidates[index].best_validation_cost
    )
    selected = candidates[selected_index]
    return RLSelection(
        selected=selected,
        candidates=candidates,
        total_tuning_time_seconds=sum(item.fit_time_seconds for item in candidates),
    )

