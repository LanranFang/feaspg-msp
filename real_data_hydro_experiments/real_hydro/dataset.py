"""In-memory trajectory interfaces shared by every policy family."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class TrajectoryBatch:
    """Independent paths with pre-innovation hydrology information.

    ``hydrology_state[:, t]`` and ``hydrology_regime[:, t]`` are computed only
    from observations strictly before ``inflow[:, t]``.  This invariant is the
    core information-timing check for all methods.
    """

    name: str
    seed: int
    inflow: np.ndarray
    hydrology_state: np.ndarray
    hydrology_regime: np.ndarray
    calendar: np.ndarray
    demand: np.ndarray | None = None
    source_path_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        inflow = np.asarray(self.inflow)
        state = np.asarray(self.hydrology_state)
        regime = np.asarray(self.hydrology_regime)
        calendar = np.asarray(self.calendar)
        if inflow.ndim != 3:
            raise ValueError("inflow must have shape [path, stage, reservoir].")
        if state.ndim != 3 or state.shape[:2] != inflow.shape[:2]:
            raise ValueError("hydrology_state must align with path and stage.")
        if regime.shape != inflow.shape[:2]:
            raise ValueError("hydrology_regime must have shape [path, stage].")
        if calendar.shape != (inflow.shape[1],):
            raise ValueError("calendar must have one period index per stage.")
        if not np.issubdtype(regime.dtype, np.integer) or np.any(regime < 0):
            raise ValueError("hydrology regimes must be nonnegative integers.")
        if np.any(~np.isfinite(inflow)) or np.any(inflow < 0.0):
            raise ValueError("Generated inflow must be finite and nonnegative.")
        if np.any(~np.isfinite(state)):
            raise ValueError("Hydrology state must be finite.")
        if self.demand is not None:
            demand = np.asarray(self.demand)
            if demand.shape != inflow.shape[:2]:
                raise ValueError("demand must have shape [path, stage].")
            if np.any(~np.isfinite(demand)) or np.any(demand < 0.0):
                raise ValueError("Demand must be finite and nonnegative.")
        if self.source_path_id is not None and np.asarray(self.source_path_id).shape != (
            inflow.shape[0],
        ):
            raise ValueError("source_path_id must have one value per path.")

    @property
    def n(self) -> int:
        return int(self.inflow.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.inflow.shape[1])

    @property
    def n_reservoirs(self) -> int:
        return int(self.inflow.shape[2])

    @property
    def n_regimes(self) -> int:
        return int(self.hydrology_regime.max(initial=0)) + 1

    def take(self, indices: np.ndarray | slice, name: str = "batch") -> "TrajectoryBatch":
        ids = None if self.source_path_id is None else self.source_path_id[indices]
        return replace(
            self,
            name=name,
            inflow=self.inflow[indices],
            hydrology_state=self.hydrology_state[indices],
            hydrology_regime=self.hydrology_regime[indices],
            demand=None if self.demand is None else self.demand[indices],
            source_path_id=ids,
        )

    def with_demand(self, demand: np.ndarray) -> "TrajectoryBatch":
        return replace(self, demand=np.asarray(demand, dtype=np.float32))

    def policy_features(self, n_regimes: int | None = None) -> np.ndarray:
        """Return the preregistered finite hydrology-state representation."""

        k = self.n_regimes if n_regimes is None else int(n_regimes)
        if k <= int(self.hydrology_regime.max(initial=0)):
            raise ValueError("n_regimes does not cover every observed regime.")
        # Calendar is already known from the stage index and encoded by every
        # policy's seasonal targets.  Returning only the regime prevents a
        # short smoke horizon from accidentally redefining the annual period.
        return np.eye(k, dtype=np.float32)[self.hydrology_regime]

    def to_torch(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        n_regimes: int | None = None,
    ) -> dict[str, torch.Tensor]:
        def tensor(values: np.ndarray, *, tensor_dtype: torch.dtype) -> torch.Tensor:
            array = np.asarray(values)
            # Prepared historical batches are intentionally read-only.  PyTorch
            # warns that sharing such NumPy storage is undefined even when the
            # simulator itself never mutates it, so copy only that uncommon
            # case while retaining zero-copy conversion for generated pools.
            if not array.flags.writeable:
                array = array.copy()
            return torch.as_tensor(array, dtype=tensor_dtype, device=device)

        result = {
            "inflow": tensor(self.inflow, tensor_dtype=dtype),
            "hydrology_state": tensor(
                self.hydrology_state, tensor_dtype=dtype
            ),
            "hydrology_regime": tensor(
                self.hydrology_regime, tensor_dtype=torch.long
            ),
            "policy_features": tensor(
                self.policy_features(n_regimes), tensor_dtype=dtype
            ),
        }
        if self.demand is not None:
            result["demand"] = tensor(self.demand, tensor_dtype=dtype)
        return result


@dataclass(frozen=True)
class TrajectorySplits:
    fit: TrajectoryBatch
    validation: TrajectoryBatch
    test: TrajectoryBatch
    calibration: TrajectoryBatch
    evaluation: TrajectoryBatch

    def as_dict(self) -> dict[str, TrajectoryBatch]:
        return {
            "fit": self.fit,
            "validation": self.validation,
            "test": self.test,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
        }

    def validate_independence(self) -> None:
        seeds = [dataset.seed for dataset in self.as_dict().values()]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Every generated split must use a distinct seed.")
