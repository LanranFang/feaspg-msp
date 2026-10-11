"""Frozen, serializable configuration for the real-hydrology benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataSeeds:
    """Independent random-number interfaces used by every benchmark."""

    fit: int = 61_101
    validation: int = 61_202
    test: int = 61_303
    calibration: int = 61_404
    evaluation: int = 61_505

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if len(values) != len(set(values)):
            raise ValueError("Every data interface must use a distinct seed.")


@dataclass(frozen=True)
class FormalProtocol:
    """Preregistered sample sizes and optimization budgets."""

    fit_budgets: tuple[int, ...] = (100, 250, 500, 1000, 2500)
    n_validation: int = 500
    n_test: int = 4096
    n_calibration: int = 500
    n_evaluation: int = 2048
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    data_seeds: DataSeeds = field(default_factory=DataSeeds)
    batch_size: int = 2
    max_validation_checks: int = 50
    par_max_order: int = 3
    par_burn_in_years: int = 10
    hydrology_regimes: int = 6
    sddp_noise_scenarios: int = 5
    sddp_iterations: int = 80
    decision_before_innovation: bool = True
    smoke_mode: bool = False

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.fit_budgets))) != self.fit_budgets:
            raise ValueError("fit_budgets must be unique and increasing.")
        if not self.smoke_mode and self.fit_budgets[-1] != 2500:
            raise ValueError("The frozen master fit pool contains 2,500 paths.")
        if (not self.smoke_mode and len(self.train_seeds) != 5) or len(
            set(self.train_seeds)
        ) != len(self.train_seeds):
            raise ValueError("Formal comparisons require five unique training seeds.")
        positive = (
            *self.fit_budgets,
            self.n_validation,
            self.n_test,
            self.n_calibration,
            self.n_evaluation,
            self.batch_size,
            self.max_validation_checks,
            self.par_burn_in_years,
            self.hydrology_regimes,
            self.sddp_noise_scenarios,
            self.sddp_iterations,
        )
        if min(positive) < 1 or self.par_max_order < 0:
            raise ValueError("Protocol sizes must be positive and PAR order nonnegative.")
        if not self.decision_before_innovation:
            raise ValueError("The preregistered information pattern is hazard-decision.")

    @classmethod
    def smoke(cls) -> "FormalProtocol":
        """Tiny protocol that preserves all interfaces and leakage checks."""

        return cls(
            fit_budgets=(4, 8, 16, 32),
            n_validation=8,
            n_test=16,
            n_calibration=8,
            n_evaluation=8,
            train_seeds=(0,),
            batch_size=2,
            max_validation_checks=3,
            par_max_order=1,
            par_burn_in_years=1,
            hydrology_regimes=2,
            sddp_noise_scenarios=2,
            sddp_iterations=2,
            smoke_mode=True,
        )

    def fit_prefix(self, n_fit: int) -> slice:
        if n_fit not in self.fit_budgets:
            raise ValueError(f"Unknown fit budget {n_fit}.")
        return slice(0, n_fit)


@dataclass(frozen=True)
class ObjectiveConfig:
    """Classic convex stage and terminal objective.

    Seasonal arrays use ``periods`` rows and one column per reservoir.  The
    multipurpose objective uses release and guide targets.  The hydrothermal
    objective uses generation coefficients, thermal blocks, and demand.
    """

    kind: str
    release_target: tuple[tuple[float, ...], ...]
    guide_lower: tuple[tuple[float, ...], ...]
    guide_upper: tuple[tuple[float, ...], ...]
    terminal_target: tuple[float, ...]
    shortage_cost: tuple[float, ...]
    excess_release_cost: tuple[float, ...]
    low_storage_cost: tuple[float, ...]
    flood_cost: tuple[float, ...]
    spill_cost: tuple[float, ...]
    ramp_cost: tuple[float, ...]
    terminal_shortfall_cost: tuple[float, ...]
    terminal_excess_cost: tuple[float, ...]
    generation_coefficient: tuple[float, ...] = ()
    seasonal_demand: tuple[float, ...] = ()
    thermal_block_capacity: tuple[float, ...] = ()
    thermal_block_cost: tuple[float, ...] = ()
    deficit_cost: float = 0.0

    def validate(self, n_reservoirs: int, periods: int) -> None:
        if self.kind not in {"multipurpose", "hydrothermal"}:
            raise ValueError("Objective kind must be multipurpose or hydrothermal.")
        matrices = (self.release_target, self.guide_lower, self.guide_upper)
        if any(len(matrix) != periods for matrix in matrices):
            raise ValueError("Every seasonal target must have one row per period.")
        if any(len(row) != n_reservoirs for matrix in matrices for row in matrix):
            raise ValueError("Seasonal target width must equal n_reservoirs.")
        vectors = (
            self.terminal_target,
            self.shortage_cost,
            self.excess_release_cost,
            self.low_storage_cost,
            self.flood_cost,
            self.spill_cost,
            self.ramp_cost,
            self.terminal_shortfall_cost,
            self.terminal_excess_cost,
        )
        if any(len(values) != n_reservoirs for values in vectors):
            raise ValueError("Every objective vector must equal n_reservoirs.")
        if np.any(np.asarray(self.guide_lower) > np.asarray(self.guide_upper)):
            raise ValueError("Lower guide curves cannot exceed upper guide curves.")
        cost_vectors = vectors[1:]
        if any(np.any(np.asarray(values) < 0.0) for values in cost_vectors):
            raise ValueError("Penalty coefficients must be nonnegative.")
        if self.kind == "hydrothermal":
            if len(self.generation_coefficient) != n_reservoirs:
                raise ValueError("Hydrothermal generation coefficients are required.")
            if len(self.seasonal_demand) != periods:
                raise ValueError("Hydrothermal demand needs one value per period.")
            if not self.thermal_block_capacity:
                raise ValueError("At least one thermal block is required.")
            if len(self.thermal_block_capacity) != len(self.thermal_block_cost):
                raise ValueError("Thermal block capacities and costs must match.")
            if min(self.thermal_block_capacity) <= 0.0:
                raise ValueError("Thermal block capacities must be positive.")
            if min((*self.thermal_block_cost, self.deficit_cost)) < 0.0:
                raise ValueError("Generation costs must be nonnegative.")


@dataclass(frozen=True)
class ReservoirSystem:
    """Physical system and the train-calibrated classic control objective."""

    benchmark: str
    case_id: str
    horizon: int
    periods: int
    step_unit: str
    reservoir_ids: tuple[str, ...]
    storage_capacity: tuple[float, ...]
    release_capacity: tuple[float, ...]
    initial_storage: tuple[float, ...]
    routing: tuple[tuple[float, ...], ...]
    objective: ObjectiveConfig
    shared_release_capacity: float | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def n_reservoirs(self) -> int:
        return len(self.reservoir_ids)

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.periods < 1:
            raise ValueError("Horizon and period count must be positive.")
        if self.step_unit not in {"hm3", "mcm", "mwh_month", "mw_week"}:
            raise ValueError(f"Unsupported storage/action unit: {self.step_unit}")
        R = self.n_reservoirs
        vectors = (
            self.storage_capacity,
            self.release_capacity,
            self.initial_storage,
            self.routing,
        )
        if R < 1 or any(len(values) != R for values in vectors):
            raise ValueError("Every physical vector must match reservoir_ids.")
        if any(len(row) != R for row in self.routing):
            raise ValueError("routing must be a square source-to-target matrix.")
        capacity = np.asarray(self.storage_capacity, dtype=np.float64)
        release = np.asarray(self.release_capacity, dtype=np.float64)
        initial = np.asarray(self.initial_storage, dtype=np.float64)
        routing = np.asarray(self.routing, dtype=np.float64)
        if np.any(capacity <= 0.0) or np.any(release <= 0.0):
            raise ValueError("Storage and release capacities must be positive.")
        if np.any(initial < 0.0) or np.any(initial > capacity):
            raise ValueError("Initial storage must be feasible.")
        if np.any(routing < 0.0) or np.any(routing > 1.0):
            raise ValueError("Routing fractions must lie in [0, 1].")
        if np.any(np.diag(routing) != 0.0) or np.any(routing.sum(axis=1) > 1.0 + 1e-12):
            raise ValueError("Routing cannot self-loop or create water.")
        if self.shared_release_capacity is not None and self.shared_release_capacity <= 0:
            raise ValueError("A shared release capacity must be positive.")
        self.objective.validate(R, self.periods)

    def period(self, stage: int) -> int:
        if not 0 <= stage < self.horizon:
            raise IndexError(stage)
        return stage % self.periods

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def repeated_rows(row: tuple[float, ...], periods: int) -> tuple[tuple[float, ...], ...]:
    """Build an immutable seasonal matrix without sharing mutable arrays."""

    return tuple(tuple(float(value) for value in row) for _ in range(periods))
