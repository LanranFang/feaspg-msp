"""Training-only periodic autoregressive hydrology and path generation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .config import DataSeeds, FormalProtocol
from .dataset import TrajectoryBatch, TrajectorySplits


def _stable_scale(values: np.ndarray) -> float:
    scale = float(np.nanstd(values, ddof=1))
    return scale if np.isfinite(scale) and scale > 1.0e-8 else 1.0


def _kmeans(values: np.ndarray, n_clusters: int) -> np.ndarray:
    """Deterministic small k-means with farthest-point initialization."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or n_clusters < 1:
        raise ValueError("k-means requires a nonempty matrix and positive k.")
    effective = min(n_clusters, values.shape[0])
    centers = [values[np.square(values - values.mean(axis=0)).sum(axis=1).argmin()]]
    for _ in range(1, effective):
        distance = np.min(
            np.stack([np.square(values - center).sum(axis=1) for center in centers]),
            axis=0,
        )
        centers.append(values[int(distance.argmax())])
    result = np.stack(centers)
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    for _ in range(100):
        new_labels = np.square(values[:, None] - result[None]).sum(axis=2).argmin(axis=1)
        updated = result.copy()
        for cluster in range(effective):
            members = values[new_labels == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(updated, result):
            break
        labels, result = new_labels, updated
    if effective < n_clusters:
        result = np.concatenate(
            (result, np.repeat(result[-1:], n_clusters - effective, axis=0)), axis=0
        )
    return result


@dataclass(frozen=True)
class PeriodicARModel:
    """Diagonal PAR dynamics with joint calendar-conditioned residual vectors.

    Cross-site dependence is retained by sampling an entire residual row.  The
    lag coefficients are site-specific so the number of parameters remains
    identifiable for the short public ONS record.
    """

    periods: int
    order: int
    transform: str
    annual_component: bool
    seasonal_mean: np.ndarray
    seasonal_scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    annual_coefficient: np.ndarray
    residual_pools: tuple[np.ndarray, ...]
    standardized_history: np.ndarray
    history_calendar: np.ndarray
    regime_centers: np.ndarray
    training_fingerprint: str
    aic_by_order: tuple[float, ...]

    @property
    def n_series(self) -> int:
        return int(self.seasonal_mean.shape[1])

    @property
    def n_regimes(self) -> int:
        return int(self.regime_centers.shape[1])

    @property
    def state_dimension(self) -> int:
        return self.n_series * (self.order + int(self.annual_component))

    def _inverse(self, standardized: np.ndarray, period: int) -> np.ndarray:
        transformed = (
            self.seasonal_mean[period] + self.seasonal_scale[period] * standardized
        )
        if self.transform == "log1p":
            transformed = np.expm1(np.clip(transformed, -30.0, 30.0))
        return np.maximum(transformed, 0.0)

    def _state_vector(self, history: list[np.ndarray]) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for lag in range(1, self.order + 1):
            pieces.append(history[-lag])
        if self.annual_component:
            pieces.append(np.mean(history[-self.periods :], axis=0))
        if not pieces:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(pieces)

    def _regime(self, period: int, state: np.ndarray) -> int:
        centers = self.regime_centers[period]
        return int(np.square(centers - state[None]).sum(axis=1).argmin())

    def sample(
        self,
        name: str,
        n: int,
        horizon: int,
        seed: int,
        burn_in_years: int = 10,
        start_period: int = 0,
    ) -> TrajectoryBatch:
        if n < 1 or horizon < 1 or burn_in_years < 0:
            raise ValueError("Path count/horizon must be positive and burn-in nonnegative.")
        if not 0 <= start_period < self.periods:
            raise ValueError("start_period is outside the seasonal calendar.")
        rng = np.random.default_rng(seed)
        burn = burn_in_years * self.periods
        burn_start = (start_period - burn) % self.periods
        minimum_history = max(self.order, self.periods if self.annual_component else 1)
        candidates = np.flatnonzero(
            (self.history_calendar == burn_start)
            & (np.arange(len(self.history_calendar)) >= minimum_history)
        )
        candidates = np.asarray(
            [
                index
                for index in candidates
                if _has_contiguous_prehistory(
                    self.history_calendar,
                    int(index),
                    minimum_history,
                    self.periods,
                )
                if np.all(
                    np.isfinite(
                        self.standardized_history[index - minimum_history : index]
                    )
                )
            ],
            dtype=np.int64,
        )
        if not len(candidates):
            raise ValueError("Training history has no aligned state for path initialization.")

        inflow = np.empty((n, horizon, self.n_series), dtype=np.float32)
        states = np.empty((n, horizon, self.state_dimension), dtype=np.float32)
        regimes = np.empty((n, horizon), dtype=np.int16)
        source_ids = rng.choice(candidates, size=n, replace=True)
        for path, initial_index in enumerate(source_ids):
            history = [
                row.copy()
                for row in self.standardized_history[
                    initial_index - minimum_history : initial_index
                ]
            ]
            if len(history) < minimum_history:
                raise RuntimeError("Insufficient initialization history.")
            for simulated_stage in range(burn + horizon):
                period = (burn_start + simulated_stage) % self.periods
                state = self._state_vector(history)
                residual_pool = self.residual_pools[period]
                residual = residual_pool[rng.integers(0, len(residual_pool))]
                current = self.intercept[period].copy()
                for lag in range(self.order):
                    current += self.coefficients[period, :, lag] * history[-lag - 1]
                if self.annual_component:
                    annual = np.mean(history[-self.periods :], axis=0)
                    current += self.annual_coefficient[period] * annual
                current += residual
                if simulated_stage >= burn:
                    retained = simulated_stage - burn
                    states[path, retained] = state
                    regimes[path, retained] = self._regime(period, state)
                    inflow[path, retained] = self._inverse(current, period)
                history.append(current)
                if len(history) > minimum_history:
                    history.pop(0)
        calendar = (start_period + np.arange(horizon)) % self.periods
        return TrajectoryBatch(
            name=name,
            seed=seed,
            inflow=inflow,
            hydrology_state=states,
            hydrology_regime=regimes,
            calendar=calendar.astype(np.int16),
            source_path_id=source_ids.astype(np.int32),
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        residual_lengths = np.asarray([len(pool) for pool in self.residual_pools])
        residuals = np.concatenate(self.residual_pools, axis=0)
        np.savez_compressed(
            directory / "periodic_ar_arrays.npz",
            seasonal_mean=self.seasonal_mean,
            seasonal_scale=self.seasonal_scale,
            intercept=self.intercept,
            coefficients=self.coefficients,
            annual_coefficient=self.annual_coefficient,
            residuals=residuals,
            residual_lengths=residual_lengths,
            standardized_history=self.standardized_history,
            history_calendar=self.history_calendar,
            regime_centers=self.regime_centers,
        )
        metadata = {
            "periods": self.periods,
            "order": self.order,
            "transform": self.transform,
            "annual_component": self.annual_component,
            "training_fingerprint": self.training_fingerprint,
            "aic_by_order": list(self.aic_by_order),
        }
        (directory / "periodic_ar_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PeriodicARModel":
        directory = Path(directory)
        metadata = json.loads((directory / "periodic_ar_metadata.json").read_text())
        arrays = np.load(directory / "periodic_ar_arrays.npz")
        lengths = arrays["residual_lengths"].astype(int)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        pools = tuple(
            arrays["residuals"][offsets[index] : offsets[index + 1]].copy()
            for index in range(len(lengths))
        )
        return cls(
            periods=int(metadata["periods"]),
            order=int(metadata["order"]),
            transform=str(metadata["transform"]),
            annual_component=bool(metadata["annual_component"]),
            seasonal_mean=arrays["seasonal_mean"],
            seasonal_scale=arrays["seasonal_scale"],
            intercept=arrays["intercept"],
            coefficients=arrays["coefficients"],
            annual_coefficient=arrays["annual_coefficient"],
            residual_pools=pools,
            standardized_history=arrays["standardized_history"],
            history_calendar=arrays["history_calendar"],
            regime_centers=arrays["regime_centers"],
            training_fingerprint=str(metadata["training_fingerprint"]),
            aic_by_order=tuple(float(value) for value in metadata["aic_by_order"]),
        )


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "level":
        return values.copy()
    if transform == "log1p":
        if np.any(values < 0.0):
            raise ValueError("log1p inflow transform requires nonnegative observations.")
        return np.log1p(values)
    raise ValueError("transform must be 'level' or 'log1p'.")


def _seasonal_standardize(
    transformed: np.ndarray, calendar: np.ndarray, periods: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_series = transformed.shape[1]
    means = np.empty((periods, n_series), dtype=np.float64)
    scales = np.empty_like(means)
    standardized = np.empty_like(transformed)
    for period in range(periods):
        mask = calendar == period
        if mask.sum() < 3:
            raise ValueError(f"Period {period} has fewer than three fit observations.")
        means[period] = np.nanmean(transformed[mask], axis=0)
        scales[period] = np.asarray(
            [_stable_scale(transformed[mask, series]) for series in range(n_series)]
        )
        standardized[mask] = (transformed[mask] - means[period]) / scales[period]
    return standardized, means, scales


def _has_contiguous_prehistory(
    calendar: np.ndarray, index: int, length: int, periods: int
) -> bool:
    """Return whether prior rows are consecutive seasonal model periods.

    Public operational records can lose an entire week or month.  Once those
    rows are omitted, finite observations on either side must not become an
    artificial autoregressive transition.  Calendar wraparound remains valid.
    """

    if length < 0 or index < length:
        return False
    current = int(calendar[index])
    return all(
        int(calendar[index - lag]) == (current - lag) % periods
        for lag in range(1, length + 1)
    )


def _fit_order(
    standardized: np.ndarray,
    calendar: np.ndarray,
    periods: int,
    order: int,
    annual_component: bool,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], float]:
    n_times, n_series = standardized.shape
    intercept = np.zeros((periods, n_series), dtype=np.float64)
    coefficients = np.zeros((periods, n_series, order), dtype=np.float64)
    annual_coefficient = np.zeros((periods, n_series), dtype=np.float64)
    residual_pools: list[np.ndarray] = []
    minimum = max(order, periods if annual_component else 0)
    total_squared_error = 0.0
    scalar_observations = 0
    parameter_count = 0
    for period in range(periods):
        candidate_rows = [
            index
            for index in range(minimum, n_times)
            if calendar[index] == period
            and _has_contiguous_prehistory(calendar, index, minimum, periods)
            and np.all(np.isfinite(standardized[index]))
            and all(np.all(np.isfinite(standardized[index - lag])) for lag in range(1, order + 1))
            and (
                not annual_component
                or np.all(np.isfinite(standardized[index - periods : index]))
            )
        ]
        if len(candidate_rows) < max(4, order + int(annual_component) + 2):
            raise ValueError(f"Period {period} has too few complete rows for PAR({order}).")
        residual_matrix = np.empty((len(candidate_rows), n_series), dtype=np.float64)
        for series in range(n_series):
            design_rows = []
            targets = []
            for index in candidate_rows:
                row = [1.0]
                row.extend(standardized[index - lag, series] for lag in range(1, order + 1))
                if annual_component:
                    row.append(float(np.mean(standardized[index - periods : index, series])))
                design_rows.append(row)
                targets.append(standardized[index, series])
            design = np.asarray(design_rows, dtype=np.float64)
            target = np.asarray(targets, dtype=np.float64)
            penalty = ridge * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
            intercept[period, series] = beta[0]
            if order:
                coefficients[period, series] = beta[1 : order + 1]
            if annual_component:
                annual_coefficient[period, series] = beta[-1]
            residual_matrix[:, series] = target - design @ beta
            parameter_count += len(beta)
        residual_matrix -= residual_matrix.mean(axis=0, keepdims=True)
        residual_pools.append(residual_matrix)
        total_squared_error += float(np.square(residual_matrix).sum())
        scalar_observations += residual_matrix.size
    variance = max(total_squared_error / scalar_observations, 1.0e-12)
    aic = scalar_observations * np.log(variance) + 2.0 * parameter_count
    return intercept, coefficients, annual_coefficient, tuple(residual_pools), float(aic)


def _historical_states(
    standardized: np.ndarray,
    calendar: np.ndarray,
    periods: int,
    order: int,
    annual_component: bool,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = max(order, periods if annual_component else 0)
    states = []
    state_periods = []
    for index in range(minimum, len(standardized)):
        if not np.all(np.isfinite(standardized[index])):
            continue
        if not _has_contiguous_prehistory(calendar, index, minimum, periods):
            continue
        pieces = [standardized[index - lag] for lag in range(1, order + 1)]
        if annual_component:
            window = standardized[index - periods : index]
            if not np.all(np.isfinite(window)):
                continue
            pieces.append(window.mean(axis=0))
        state = np.concatenate(pieces) if pieces else np.zeros(0)
        if np.all(np.isfinite(state)):
            states.append(state)
            state_periods.append(calendar[index])
    return np.asarray(states), np.asarray(state_periods, dtype=np.int16)


def fit_periodic_ar(
    values: np.ndarray,
    calendar: np.ndarray,
    periods: int,
    *,
    transform: str = "log1p",
    max_order: int = 3,
    annual_component: bool = False,
    regime_count: int = 6,
    ridge: float = 1.0e-5,
) -> PeriodicARModel:
    """Fit a PAR model using only the supplied chronological fit block."""

    values = np.asarray(values, dtype=np.float64)
    calendar = np.asarray(calendar, dtype=np.int16)
    if values.ndim != 2 or calendar.shape != (len(values),):
        raise ValueError("values must be [time, series] with an aligned calendar.")
    if periods < 2 or max_order < 0 or regime_count < 1:
        raise ValueError("Invalid period/order/regime configuration.")
    if np.any((calendar < 0) | (calendar >= periods)):
        raise ValueError("calendar values are outside [0, periods).")
    if np.any(np.isinf(values)):
        raise ValueError("Training observations cannot contain infinities.")
    transformed = _transform(values, transform)
    standardized, seasonal_mean, seasonal_scale = _seasonal_standardize(
        transformed, calendar, periods
    )
    fits = []
    errors: list[Exception] = []
    for order in range(max_order + 1):
        try:
            fits.append(
                (
                    order,
                    *_fit_order(
                        standardized,
                        calendar,
                        periods,
                        order,
                        annual_component,
                        ridge,
                    ),
                )
            )
        except ValueError as error:
            errors.append(error)
    if not fits:
        raise ValueError(f"No estimable PAR order: {errors[-1] if errors else 'unknown'}")
    selected = min(fits, key=lambda item: item[-1])
    order, intercept, coefficients, annual_coefficient, residual_pools, _ = selected
    states, state_periods = _historical_states(
        standardized, calendar, periods, order, annual_component
    )
    state_dimension = values.shape[1] * (order + int(annual_component))
    centers = np.empty((periods, regime_count, state_dimension), dtype=np.float64)
    for period in range(periods):
        period_states = states[state_periods == period]
        if not len(period_states):
            raise ValueError(f"No hydrology states available for period {period}.")
        centers[period] = _kmeans(period_states, regime_count)
    fingerprint_input = np.ascontiguousarray(values).tobytes() + np.ascontiguousarray(
        calendar
    ).tobytes()
    aic_lookup = {item[0]: item[-1] for item in fits}
    return PeriodicARModel(
        periods=periods,
        order=order,
        transform=transform,
        annual_component=annual_component,
        seasonal_mean=seasonal_mean,
        seasonal_scale=seasonal_scale,
        intercept=intercept,
        coefficients=coefficients,
        annual_coefficient=annual_coefficient,
        residual_pools=residual_pools,
        standardized_history=standardized,
        history_calendar=calendar,
        regime_centers=centers,
        training_fingerprint=sha256(fingerprint_input).hexdigest(),
        aic_by_order=tuple(float(aic_lookup.get(order, np.inf)) for order in range(max_order + 1)),
    )


def generate_splits(
    model: PeriodicARModel,
    protocol: FormalProtocol,
    horizon: int,
    *,
    start_period: int = 0,
    seeds: DataSeeds | None = None,
) -> TrajectorySplits:
    """Generate independent pools; fit budgets are nested prefixes by design."""

    seeds = protocol.data_seeds if seeds is None else seeds
    specs = {
        "fit": (protocol.fit_budgets[-1], seeds.fit),
        "validation": (protocol.n_validation, seeds.validation),
        "test": (protocol.n_test, seeds.test),
        "calibration": (protocol.n_calibration, seeds.calibration),
        "evaluation": (protocol.n_evaluation, seeds.evaluation),
    }
    batches = {
        name: model.sample(
            name,
            count,
            horizon,
            seed,
            burn_in_years=protocol.par_burn_in_years,
            start_period=start_period,
        )
        for name, (count, seed) in specs.items()
    }
    result = TrajectorySplits(**batches)
    result.validate_independence()
    return result


def observed_trajectory_batch(
    model: PeriodicARModel,
    name: str,
    values: np.ndarray,
    calendar: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    *,
    seed: int = 0,
    demand: np.ndarray | None = None,
) -> TrajectoryBatch:
    """Encode locked historical windows without exposing current inflow.

    ``values`` must include the observable history preceding every requested
    start.  Features at time ``t`` use rows strictly before ``t``; perturbing
    the current or future observation therefore cannot change the action state.
    """

    values = np.asarray(values, dtype=np.float64)
    calendar = np.asarray(calendar, dtype=np.int16)
    starts = np.asarray(starts, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != model.n_series:
        raise ValueError("Historical values must have shape [time, model series].")
    if calendar.shape != (len(values),):
        raise ValueError("Historical calendar does not align with values.")
    if np.any((calendar < 0) | (calendar >= model.periods)):
        raise ValueError("Historical calendar is out of range.")
    if len(values) > 1 and not np.all(
        calendar[1:] == (calendar[:-1] + 1) % model.periods
    ):
        raise ValueError("Historical rows must follow a contiguous seasonal calendar.")
    if horizon < 1 or starts.ndim != 1 or not len(starts):
        raise ValueError("At least one positive-horizon historical window is required.")
    minimum = max(model.order, model.periods if model.annual_component else 0)
    if np.any(starts < minimum) or np.any(starts + horizon > len(values)):
        raise ValueError("Historical windows lack required prehistory or exceed the data.")
    transformed = _transform(values, model.transform)
    standardized = np.empty_like(transformed)
    for period in range(model.periods):
        mask = calendar == period
        standardized[mask] = (
            transformed[mask] - model.seasonal_mean[period]
        ) / model.seasonal_scale[period]
    inflow = np.empty((len(starts), horizon, model.n_series), dtype=np.float32)
    state = np.empty(
        (len(starts), horizon, model.state_dimension), dtype=np.float32
    )
    regime = np.empty((len(starts), horizon), dtype=np.int16)
    historical_demand = None
    if demand is not None:
        demand = np.asarray(demand, dtype=np.float64)
        if demand.shape != (len(values),):
            raise ValueError("Historical demand must align with the full value history.")
        historical_demand = np.empty((len(starts), horizon), dtype=np.float32)
    for path, start in enumerate(starts):
        for stage in range(horizon):
            index = int(start + stage)
            pieces = [standardized[index - lag] for lag in range(1, model.order + 1)]
            if model.annual_component:
                pieces.append(standardized[index - model.periods : index].mean(axis=0))
            current_state = np.concatenate(pieces) if pieces else np.zeros(0)
            current_inflow = values[index]
            if not np.all(np.isfinite(current_state)) or not np.all(
                np.isfinite(current_inflow)
            ):
                raise ValueError("Historical window or its observable prehistory is incomplete.")
            state[path, stage] = current_state
            regime[path, stage] = model._regime(int(calendar[index]), current_state)
            inflow[path, stage] = current_inflow
            if historical_demand is not None:
                if not np.isfinite(demand[index]) or demand[index] < 0.0:
                    raise ValueError("Historical demand must be finite and nonnegative.")
                historical_demand[path, stage] = demand[index]
    window_calendar = calendar[starts[0] : starts[0] + horizon]
    if any(
        not np.array_equal(
            window_calendar, calendar[start : start + horizon]
        )
        for start in starts[1:]
    ):
        raise ValueError("All historical windows must share the same seasonal alignment.")
    return TrajectoryBatch(
        name=name,
        seed=seed,
        inflow=inflow,
        hydrology_state=state,
        hydrology_regime=regime,
        calendar=window_calendar.copy(),
        demand=historical_demand,
        source_path_id=starts.astype(np.int32),
    )
