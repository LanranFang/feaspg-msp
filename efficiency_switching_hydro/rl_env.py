"""Gymnasium interface for model-free training on fixed hydro trajectories."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .config import ExperimentConfig
from .data import Dataset
from .physics import transition_np
from .policy_features import (
    flat_markov_observation_np,
    reference_centered_proposal_np,
)
from .projection import action_bounds_np, project_action_np


def rl_observation(
    stage: int,
    storage: np.ndarray,
    previous_release: np.ndarray,
    queue: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Return the normalized Markov state available before the stage shock."""

    return flat_markov_observation_np(
        stage, storage, previous_release, queue, config
    )


def normalized_action_to_proposal(
    action: np.ndarray, stage: int, config: ExperimentConfig
) -> np.ndarray:
    """Map a Box action around the reference schedule onto all possible releases."""

    return reference_centered_proposal_np(action, stage, config)


class HydroTrajectoryEnv(gym.Env[np.ndarray, np.ndarray]):
    """Finite-horizon environment that consumes fit trajectories in fixed order."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: Dataset,
        config: ExperimentConfig,
        reward_scale: float,
        seed: int,
    ) -> None:
        super().__init__()
        if dataset.n < 1:
            raise ValueError("The RL fitting dataset must contain at least one path.")
        if reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        self.dataset = dataset
        self.config = config
        self.reward_scale = float(reward_scale)
        self._seed = int(seed)
        self._order = np.random.default_rng(self._seed).permutation(dataset.n)
        self._episode_cursor = 0
        self.episodes_started = 0
        self.episodes_completed = 0
        self.transitions_taken = 0
        self.path_indices_used: list[int] = []
        self.completed_path_indices: list[int] = []

        state_dim = config.n_reservoirs * (2 + config.routing_delay)
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (np.zeros(state_dim, dtype=np.float32), -np.ones(4, dtype=np.float32))
            ),
            high=np.ones(state_dim + 4, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.n_reservoirs,),
            dtype=np.float32,
        )
        self._storage: np.ndarray | None = None
        self._previous_release: np.ndarray | None = None
        self._queue: np.ndarray | None = None
        self._stage = 0
        self._path_index = 0
        self._episode_cost = 0.0

    def _reset_order(self, seed: int) -> None:
        self._seed = int(seed)
        self._order = np.random.default_rng(self._seed).permutation(self.dataset.n)
        self._episode_cursor = 0
        self.episodes_started = 0
        self.episodes_completed = 0
        self.transitions_taken = 0
        self.path_indices_used.clear()
        self.completed_path_indices.clear()

    def _observation(self) -> np.ndarray:
        if (
            self._storage is None
            or self._previous_release is None
            or self._queue is None
        ):
            raise RuntimeError("reset() must be called before requesting an observation.")
        stage = min(self._stage, self.config.T - 1)
        return rl_observation(
            stage,
            self._storage,
            self._previous_release,
            self._queue,
            self.config,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._reset_order(seed)
        requested_path = None if options is None else options.get("path_index")
        if requested_path is None:
            self._path_index = int(
                self._order[self._episode_cursor % self.dataset.n]
            )
            self._episode_cursor += 1
        else:
            self._path_index = int(requested_path)
            if not 0 <= self._path_index < self.dataset.n:
                raise IndexError("path_index is outside the dataset.")
        self._storage = np.asarray(
            self.config.initial_storage, dtype=np.float64
        ).copy()
        self._previous_release = np.zeros(
            self.config.n_reservoirs, dtype=np.float64
        )
        self._queue = np.zeros(
            (self.config.routing_delay, self.config.n_reservoirs),
            dtype=np.float64,
        )
        self._stage = 0
        self._episode_cost = 0.0
        self.episodes_started += 1
        self.path_indices_used.append(self._path_index)
        return self._observation(), {"path_index": self._path_index}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if (
            self._storage is None
            or self._previous_release is None
            or self._queue is None
        ):
            raise RuntimeError("reset() must be called before step().")
        if self._stage >= self.config.T:
            raise RuntimeError("The episode has ended; call reset() before step().")

        storage_before = self._storage.copy()
        previous_before = self._previous_release.copy()
        stage = self._stage
        proposal = normalized_action_to_proposal(action, stage, self.config)
        projected = project_action_np(
            proposal, storage_before, previous_before, self.config
        )
        lower, upper = action_bounds_np(
            storage_before, previous_before, self.config
        )
        self._storage, self._previous_release, self._queue, components = transition_np(
            storage_before,
            previous_before,
            self._queue,
            projected,
            self.dataset.inflow[self._path_index, stage],
            float(self.dataset.demand[self._path_index, stage]),
            float(self.dataset.price[self._path_index, stage]),
            self.config,
        )
        stage_cost = float(
            components["thermal"]
            + components["spill"]
            + components["ramp"]
            + components["storage_penalty"]
        )
        self._episode_cost += stage_cost
        self._stage += 1
        self.transitions_taken += 1
        terminated = self._stage == self.config.T
        terminal_cost = 0.0
        if terminated:
            terminal_water = self._storage + (
                self._queue.sum(axis=0) if self.config.routing_delay else 0.0
            )
            shortfall = np.maximum(
                np.asarray(self.config.terminal_target) - terminal_water, 0.0
            )
            terminal_cost = float(
                np.asarray(self.config.terminal_water_value) @ shortfall
            )
            self._episode_cost += terminal_cost
            self.episodes_completed += 1
            self.completed_path_indices.append(self._path_index)

        weights = np.asarray(self.config.generation_coefficient)
        action_violation = max(
            float(np.max(lower - projected)),
            float(np.max(projected - upper)),
            float(weights @ projected - self.config.shared_release_capacity),
            0.0,
        )
        cost = stage_cost + terminal_cost
        info = {
            "path_index": self._path_index,
            "stage": stage,
            "proposal": proposal.astype(np.float32),
            "projected_action": projected.astype(np.float32),
            "stage_cost": stage_cost,
            "terminal_cost": terminal_cost,
            "action_violation": action_violation,
        }
        if terminated:
            info["episode_cost"] = self._episode_cost
        return self._observation(), -cost / self.reward_scale, terminated, False, info
