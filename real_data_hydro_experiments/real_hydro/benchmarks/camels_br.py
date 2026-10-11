"""Leakage-safe CAMELS-BR semi-synthetic reservoir benchmark builder.

CAMELS-BR supplies catchment discharge, not dam physics.  This module therefore
builds a *parallel portfolio of stylized reservoirs* driven by real hydrology;
it must never be described as a real cascade or a graph benchmark.  Reservoir
sizes, the multipurpose objective, and every normalizer are derived solely from
the September 1980--August 2005 fit block (water years 1981--2005).

The chronological matrices returned here are the input contract for
``fit_periodic_ar`` and historical replay.  Later-period missing observations
remain explicit NaNs: this layer neither interpolates them nor uses them to
alter the fitted control problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import ObjectiveConfig, ReservoirSystem
from ..data.camels_br import (
    CAMELS_BR_FILES,
    CAMELS_BR_RECORD_ID,
    DataValidationError,
    HistoricalWaterYearSplit,
)


CAMELS_BR_HORIZON = 48
CAMELS_BR_PERIODS = 12
CAMELS_BR_FORMAL_BASINS = 12
CAMELS_BR_HISTORICAL_SPLIT = HistoricalWaterYearSplit()

STORAGE_MONTHS = 6.0
RELEASE_MEAN_MULTIPLIER = 1.5
INITIAL_STORAGE_FRACTION = 0.64
TERMINAL_TARGET_FRACTION = 0.62
GUIDE_QUANTILES = (0.20, 0.80)

# Fixed dimensionless priorities are divided by each basin's fit-period mean
# monthly inflow.  Consequently a one-normalizer deviation has the same weight
# for a small and a large catchment, without consulting validation/test data.
CLASSIC_MULTIPURPOSE_MULTIPLIERS: Mapping[str, float] = MappingProxyType(
    {
        "shortage_cost": 1.00,
        "excess_release_cost": 0.10,
        "low_storage_cost": 0.50,
        "flood_cost": 1.00,
        "spill_cost": 2.00,
        "ramp_cost": 0.05,
        "terminal_shortfall_cost": 2.00,
        "terminal_excess_cost": 0.10,
    }
)


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _normalise_gauge_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise DataValidationError(f"Invalid CAMELS-BR gauge_id: {value!r}")
    return text.zfill(8)


def _matrix_tuple(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _vector_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _water_year(months: pd.DatetimeIndex) -> np.ndarray:
    return np.where(months.month >= 9, months.year + 1, months.year).astype(np.int16)


def _water_calendar(months: pd.DatetimeIndex) -> np.ndarray:
    """Return September=0, ..., August=11."""

    return ((months.month - 9) % CAMELS_BR_PERIODS).astype(np.int16)


@dataclass(frozen=True)
class HistoricalSplitArrays:
    """A complete monthly calendar grid for one chronological history split.

    ``inflow`` may contain NaNs outside the fit block.  They are retained as
    quality flags, and :meth:`contiguous_paths` drops any 48-month window that
    touches one rather than silently filling it.
    """

    name: str
    water_year_range: tuple[int, int]
    reservoir_ids: tuple[str, ...]
    months: np.ndarray
    water_year: np.ndarray
    calendar: np.ndarray
    inflow: np.ndarray

    def __post_init__(self) -> None:
        months = _readonly(self.months, "datetime64[ns]")
        water_year = _readonly(self.water_year, np.int16)
        calendar = _readonly(self.calendar, np.int16)
        inflow = _readonly(self.inflow, np.float64)
        expected = (self.water_year_range[1] - self.water_year_range[0] + 1) * 12
        if self.water_year_range[0] > self.water_year_range[1]:
            raise ValueError("A historical split needs an ordered water-year range.")
        if months.shape != (expected,):
            raise ValueError("Historical split does not contain its full monthly grid.")
        if water_year.shape != (expected,) or calendar.shape != (expected,):
            raise ValueError("Historical labels do not align with the monthly grid.")
        if inflow.shape != (expected, len(self.reservoir_ids)):
            raise ValueError("Historical inflow width does not match reservoir_ids.")
        if not np.array_equal(calendar, np.tile(np.arange(12), expected // 12)):
            raise ValueError("Historical calendar must start in September at period 0.")
        if np.any(np.isinf(inflow)) or np.any(inflow[np.isfinite(inflow)] < 0.0):
            raise ValueError("Historical inflow must be nonnegative or missing.")
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "water_year", water_year)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "inflow", inflow)

    @property
    def missing_cells(self) -> int:
        return int(np.isnan(self.inflow).sum())

    @property
    def complete_joint_water_years(self) -> int:
        complete = np.isfinite(self.inflow).all(axis=1).reshape(-1, 12).all(axis=1)
        return int(complete.sum())

    def contiguous_paths(
        self, *, horizon: int = CAMELS_BR_HORIZON, stride: int = 12
    ) -> np.ndarray:
        """Return complete chronological windows without filling missing data.

        A stride of 12 produces water-year-aligned rolling four-year paths for
        the frozen 48-month control horizon.
        """

        if horizon < 1 or horizon % 12:
            raise ValueError("Historical path horizon must be a positive whole year.")
        if stride < 1 or stride % 12:
            raise ValueError("Historical path stride must be a positive whole year.")
        candidates = [
            self.inflow[start : start + horizon]
            for start in range(0, len(self.inflow) - horizon + 1, stride)
            if np.isfinite(self.inflow[start : start + horizon]).all()
        ]
        if not candidates:
            return _readonly(
                np.empty((0, horizon, len(self.reservoir_ids))), np.float64
            )
        return _readonly(np.stack(candidates), np.float64)


@dataclass(frozen=True)
class CamelsBRBenchmark:
    """Built control system plus stochastic-fit and historical data contracts."""

    system: ReservoirSystem
    fit_matrix: np.ndarray
    fit_calendar: np.ndarray
    historical_splits: Mapping[str, HistoricalSplitArrays]
    basin_audit: pd.DataFrame
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        fit_matrix = _readonly(self.fit_matrix, np.float64)
        fit_calendar = _readonly(self.fit_calendar, np.int16)
        splits = MappingProxyType(dict(self.historical_splits))
        required = {"fit", "generator_validation", "locked_real_test"}
        if set(splits) != required:
            raise ValueError(f"Historical splits must be exactly {sorted(required)}.")
        if fit_matrix.shape != splits["fit"].inflow.shape:
            raise ValueError("fit_matrix does not match the historical fit split.")
        if fit_calendar.shape != splits["fit"].calendar.shape:
            raise ValueError("fit_calendar does not match the historical fit split.")
        if not np.array_equal(fit_matrix, splits["fit"].inflow):
            raise ValueError("fit_matrix must be the unchanged fit history.")
        if not np.array_equal(fit_calendar, splits["fit"].calendar):
            raise ValueError("fit_calendar must be the September-based fit calendar.")
        if np.any(~np.isfinite(fit_matrix)):
            raise ValueError("The stochastic-process fit matrix must be complete.")
        object.__setattr__(self, "fit_matrix", fit_matrix)
        object.__setattr__(self, "fit_calendar", fit_calendar)
        object.__setattr__(self, "historical_splits", splits)
        object.__setattr__(self, "basin_audit", self.basin_audit.copy(deep=True))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def split(self, name: str) -> HistoricalSplitArrays:
        try:
            return self.historical_splits[name]
        except KeyError as exc:
            raise KeyError(f"Unknown CAMELS-BR historical split: {name}") from exc


def _prepare_monthly(monthly_inflow: pd.DataFrame) -> pd.DataFrame:
    required = {"gauge_id", "month", "inflow_hm3"}
    missing = required.difference(monthly_inflow.columns)
    if missing:
        raise DataValidationError(f"Monthly inflow lacks columns: {sorted(missing)}")
    if monthly_inflow.empty:
        raise DataValidationError("Monthly inflow cannot be empty.")

    frame = monthly_inflow.copy()
    frame["gauge_id"] = frame["gauge_id"].map(_normalise_gauge_id)
    parsed = pd.to_datetime(frame["month"], errors="coerce")
    if parsed.isna().any():
        raise DataValidationError("Monthly inflow contains invalid month values.")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    frame["month"] = parsed.dt.to_period("M").dt.to_timestamp()
    if frame.duplicated(["gauge_id", "month"]).any():
        duplicate = frame.loc[
            frame.duplicated(["gauge_id", "month"], keep=False),
            ["gauge_id", "month"],
        ].iloc[0]
        raise DataValidationError(
            "Monthly inflow repeats gauge/month "
            f"{duplicate['gauge_id']} {duplicate['month']:%Y-%m}."
        )

    frame["inflow_hm3"] = pd.to_numeric(frame["inflow_hm3"], errors="coerce")
    inflow_values = frame["inflow_hm3"].to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    finite = frame["inflow_hm3"].notna()
    if np.isinf(inflow_values).any():
        raise DataValidationError("Monthly inflow contains infinite values.")
    if frame.loc[finite, "inflow_hm3"].lt(0.0).any():
        raise DataValidationError("Monthly inflow volumes cannot be negative.")

    expected_water_year = np.where(
        frame["month"].dt.month >= 9,
        frame["month"].dt.year + 1,
        frame["month"].dt.year,
    )
    expected_water_month = ((frame["month"].dt.month - 9) % 12) + 1
    for column, expected in (
        ("water_year", expected_water_year),
        ("water_month", expected_water_month),
    ):
        if column in frame:
            supplied = pd.to_numeric(frame[column], errors="coerce")
            if supplied.isna().any() or not np.array_equal(
                supplied.to_numpy(dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
            ):
                raise DataValidationError(
                    f"{column} is inconsistent with the September--August calendar."
                )
    frame["water_year"] = expected_water_year.astype(np.int16)
    frame["water_month"] = expected_water_month.astype(np.int8)
    return frame.sort_values(["month", "gauge_id"], kind="stable").reset_index(drop=True)


def _prepare_calibration(
    calibration: pd.DataFrame,
    *,
    basin_ids: Sequence[str | int] | None,
    n_basins: int | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    required = {
        "gauge_id",
        "fit_water_year_start",
        "fit_water_year_end",
        "fit_months",
        "mean_monthly_inflow_hm3",
        "storage_capacity_hm3",
        "release_capacity_hm3_per_month",
        "initial_storage_hm3",
        "terminal_target_hm3",
        "storage_months",
        "release_mean_multiplier",
    }
    missing = required.difference(calibration.columns)
    if missing:
        raise DataValidationError(f"Calibration lacks columns: {sorted(missing)}")
    if calibration.empty:
        raise DataValidationError("Calibration cannot be empty.")

    result = calibration.copy()
    result["gauge_id"] = result["gauge_id"].map(_normalise_gauge_id)
    if result["gauge_id"].duplicated().any():
        raise DataValidationError("Calibration repeats a gauge_id.")
    if basin_ids is None:
        identifiers = tuple(result["gauge_id"])
    else:
        identifiers = tuple(_normalise_gauge_id(value) for value in basin_ids)
        if len(identifiers) != len(set(identifiers)):
            raise DataValidationError("basin_ids contains duplicates.")
    if set(identifiers) != set(result["gauge_id"]):
        raise DataValidationError(
            "basin_ids must identify exactly the rows in the training calibration."
        )
    if n_basins is not None and len(identifiers) != n_basins:
        raise DataValidationError(
            f"CAMELS-BR case requires {n_basins} selected non-nested basins; "
            f"received {len(identifiers)}."
        )
    result = result.set_index("gauge_id").loc[list(identifiers)].reset_index()

    numeric = required - {"gauge_id"}
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(numeric)].isna().any().any() or np.isinf(
        result[list(numeric)].to_numpy(dtype=float)
    ).any():
        raise DataValidationError("Calibration values must all be finite numeric values.")
    return result, identifiers


def _historical_arrays(
    monthly: pd.DataFrame, basin_ids: tuple[str, ...]
) -> dict[str, HistoricalSplitArrays]:
    observed = set(monthly["gauge_id"])
    if observed != set(basin_ids):
        raise DataValidationError(
            "Monthly long table must contain exactly the calibrated selected basins."
        )
    pivot = monthly.pivot(index="month", columns="gauge_id", values="inflow_hm3")
    pivot = pivot.reindex(columns=list(basin_ids))
    output: dict[str, HistoricalSplitArrays] = {}
    for name, water_year_range in (
        ("fit", CAMELS_BR_HISTORICAL_SPLIT.fit),
        (
            "generator_validation",
            CAMELS_BR_HISTORICAL_SPLIT.generator_validation,
        ),
        ("locked_real_test", CAMELS_BR_HISTORICAL_SPLIT.locked_real_test),
    ):
        start, end = water_year_range
        months = pd.date_range(
            f"{start - 1:04d}-09-01", f"{end:04d}-08-01", freq="MS"
        )
        output[name] = HistoricalSplitArrays(
            name=name,
            water_year_range=water_year_range,
            reservoir_ids=basin_ids,
            months=months.to_numpy(dtype="datetime64[ns]"),
            water_year=_water_year(months),
            calendar=_water_calendar(months),
            inflow=pivot.reindex(months).to_numpy(dtype=np.float64),
        )
    return output


def _validate_train_calibration(
    calibration: pd.DataFrame, fit: HistoricalSplitArrays
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fit.missing_cells:
        missing_by_basin = np.isnan(fit.inflow).sum(axis=0)
        detail = ", ".join(
            f"{identifier}={int(count)}"
            for identifier, count in zip(fit.reservoir_ids, missing_by_basin, strict=True)
            if count
        )
        raise DataValidationError(
            "The 1981--2005 fit matrix must contain 25 complete water years; "
            f"missing monthly cells: {detail}. No interpolation is allowed here."
        )
    fit_mean = fit.inflow.mean(axis=0)
    if np.any(~np.isfinite(fit_mean)) or np.any(fit_mean <= 0.0):
        raise DataValidationError("Every basin needs positive mean fit-period inflow.")

    expected_months = len(fit.months)
    expected = {
        "fit_water_year_start": np.full(len(fit_mean), 1981.0),
        "fit_water_year_end": np.full(len(fit_mean), 2005.0),
        "fit_months": np.full(len(fit_mean), float(expected_months)),
        "mean_monthly_inflow_hm3": fit_mean,
        "storage_months": np.full(len(fit_mean), STORAGE_MONTHS),
        "release_mean_multiplier": np.full(
            len(fit_mean), RELEASE_MEAN_MULTIPLIER
        ),
        "storage_capacity_hm3": STORAGE_MONTHS * fit_mean,
        "release_capacity_hm3_per_month": RELEASE_MEAN_MULTIPLIER * fit_mean,
        "initial_storage_hm3": (
            INITIAL_STORAGE_FRACTION * STORAGE_MONTHS * fit_mean
        ),
        "terminal_target_hm3": (
            TERMINAL_TARGET_FRACTION * STORAGE_MONTHS * fit_mean
        ),
    }
    for column, reference in expected.items():
        supplied = calibration[column].to_numpy(dtype=np.float64)
        if not np.allclose(supplied, reference, rtol=1.0e-9, atol=1.0e-9):
            raise DataValidationError(
                f"Calibration column {column} is not the frozen training-only "
                "CAMELS-BR rule. Recompute it from water years 1981--2005."
            )
    return (
        fit_mean,
        expected["storage_capacity_hm3"],
        expected["release_capacity_hm3_per_month"],
        expected["initial_storage_hm3"],
        expected["terminal_target_hm3"],
    )


def _train_only_objective(
    fit: HistoricalSplitArrays,
    *,
    fit_mean: np.ndarray,
    storage_capacity: np.ndarray,
    release_capacity: np.ndarray,
    initial_storage: np.ndarray,
    terminal_target: np.ndarray,
) -> ObjectiveConfig:
    n_years = len(fit.inflow) // CAMELS_BR_PERIODS
    by_year = fit.inflow.reshape(n_years, CAMELS_BR_PERIODS, -1)
    release_target = np.median(by_year, axis=0)
    release_target = np.minimum(release_target, release_capacity[None, :])

    # Empirical guide bands are post-transition storage quantiles under the
    # seasonal median release rule.  The action is bounded by opening storage,
    # matching the simulator's decision-before-current-inflow convention.
    reference_storage = np.empty_like(by_year)
    for year in range(n_years):
        storage = initial_storage.copy()
        for period in range(CAMELS_BR_PERIODS):
            release = np.minimum(release_target[period], storage)
            storage = np.minimum(
                storage - release + by_year[year, period], storage_capacity
            )
            reference_storage[year, period] = storage
    guide_lower, guide_upper = np.quantile(
        reference_storage, GUIDE_QUANTILES, axis=0
    )

    normalized_costs = {
        name: multiplier / fit_mean
        for name, multiplier in CLASSIC_MULTIPURPOSE_MULTIPLIERS.items()
    }
    return ObjectiveConfig(
        kind="multipurpose",
        release_target=_matrix_tuple(release_target),
        guide_lower=_matrix_tuple(guide_lower),
        guide_upper=_matrix_tuple(guide_upper),
        terminal_target=_vector_tuple(terminal_target),
        shortage_cost=_vector_tuple(normalized_costs["shortage_cost"]),
        excess_release_cost=_vector_tuple(
            normalized_costs["excess_release_cost"]
        ),
        low_storage_cost=_vector_tuple(normalized_costs["low_storage_cost"]),
        flood_cost=_vector_tuple(normalized_costs["flood_cost"]),
        spill_cost=_vector_tuple(normalized_costs["spill_cost"]),
        ramp_cost=_vector_tuple(normalized_costs["ramp_cost"]),
        terminal_shortfall_cost=_vector_tuple(
            normalized_costs["terminal_shortfall_cost"]
        ),
        terminal_excess_cost=_vector_tuple(
            normalized_costs["terminal_excess_cost"]
        ),
    )


def _fit_fingerprint(
    fit_matrix: np.ndarray, fit_calendar: np.ndarray, basin_ids: tuple[str, ...]
) -> str:
    payload = "\0".join(basin_ids).encode("ascii")
    payload += np.ascontiguousarray(fit_calendar, dtype=np.int16).tobytes()
    payload += np.ascontiguousarray(fit_matrix, dtype=np.float64).tobytes()
    return sha256(payload).hexdigest()


def build_camels_br_benchmark(
    monthly_inflow: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    basin_ids: Sequence[str | int] | None = None,
    n_basins: int | None = CAMELS_BR_FORMAL_BASINS,
) -> CamelsBRBenchmark:
    """Build the frozen CAMELS-BR parallel semi-synthetic benchmark.

    Parameters
    ----------
    monthly_inflow:
        Long table produced by :func:`build_monthly_inflow` for the already
        selected, mutually non-nested catchments.  The formal table spans the
        three fixed water-year windows; post-fit gaps remain missing.
    calibration:
        Output of :func:`calibrate_semisynthetic_reservoirs` using exactly water
        years 1981--2005 and the frozen 6/1.5/0.64/0.62 sizing rules.  Every
        value is independently checked against ``monthly_inflow``.
    basin_ids:
        Optional explicit reservoir order.  It must contain exactly the
        calibrated basins.  Otherwise calibration row order is retained.
    n_basins:
        The formal case requires 12.  Small offline fixtures may explicitly
        pass a smaller number; ``None`` accepts the supplied count.
    """

    if n_basins is not None and n_basins < 1:
        raise ValueError("n_basins must be positive or None.")
    monthly = _prepare_monthly(monthly_inflow)
    calibrated, identifiers = _prepare_calibration(
        calibration, basin_ids=basin_ids, n_basins=n_basins
    )
    historical = _historical_arrays(monthly, identifiers)
    fit = historical["fit"]
    (
        fit_mean,
        storage_capacity,
        release_capacity,
        initial_storage,
        terminal_target,
    ) = _validate_train_calibration(calibrated, fit)
    objective = _train_only_objective(
        fit,
        fit_mean=fit_mean,
        storage_capacity=storage_capacity,
        release_capacity=release_capacity,
        initial_storage=initial_storage,
        terminal_target=terminal_target,
    )

    zero_routing = np.zeros((len(identifiers), len(identifiers)), dtype=np.float64)
    system = ReservoirSystem(
        benchmark="camels_br_v1.2",
        case_id=f"camels_br_parallel_{len(identifiers)}_basins",
        horizon=CAMELS_BR_HORIZON,
        periods=CAMELS_BR_PERIODS,
        step_unit="hm3",
        reservoir_ids=identifiers,
        storage_capacity=_vector_tuple(storage_capacity),
        release_capacity=_vector_tuple(release_capacity),
        initial_storage=_vector_tuple(initial_storage),
        routing=_matrix_tuple(zero_routing),
        objective=objective,
        metadata=(
            ("benchmark_semantics", "real_hydrology_semisynthetic_reservoirs"),
            ("semisynthetic_reservoirs", "true"),
            ("graph_topology", "none"),
            ("routing", "parallel_zero"),
            ("source", f"CAMELS-BR_v1.2_Zenodo_{CAMELS_BR_RECORD_ID}"),
            ("fit_water_years", "1981-2005"),
            ("calendar_period_0", "September"),
        ),
    )

    missing_by_split = {
        name: np.isnan(block.inflow).sum(axis=0).astype(int)
        for name, block in historical.items()
    }
    basin_audit = calibrated.copy()
    basin_audit["computed_fit_mean_hm3"] = fit_mean
    basin_audit["objective_volume_scale_hm3"] = fit_mean
    for name, counts in missing_by_split.items():
        basin_audit[f"{name}_missing_months"] = counts
    basin_audit["semisynthetic_reservoir"] = True
    basin_audit["graph_node"] = False

    manifest = tuple(
        (key, spec.filename, spec.md5) for key, spec in sorted(CAMELS_BR_FILES.items())
    )
    audit: dict[str, object] = {
        "source_record_id": CAMELS_BR_RECORD_ID,
        "source_manifest": manifest,
        "benchmark_semantics": "real hydrology with semi-synthetic reservoirs",
        "semisynthetic_reservoirs": True,
        "has_graph": False,
        "routing_assumption": "independent parallel reservoirs; zero routing",
        "non_nested_input_contract": True,
        "non_nested_verified_by_builder": False,
        "n_basins": len(identifiers),
        "basin_ids": identifiers,
        "horizon": CAMELS_BR_HORIZON,
        "periods": CAMELS_BR_PERIODS,
        "calendar_period_0": "September",
        "fit_water_years": CAMELS_BR_HISTORICAL_SPLIT.fit,
        "generator_validation_water_years": (
            CAMELS_BR_HISTORICAL_SPLIT.generator_validation
        ),
        "locked_real_test_water_years": (
            CAMELS_BR_HISTORICAL_SPLIT.locked_real_test
        ),
        "capacity_rule": "6 * fit mean monthly inflow",
        "release_capacity_rule": "1.5 * fit mean monthly inflow",
        "initial_storage_rule": "0.64 * semi-synthetic capacity",
        "terminal_target_rule": "0.62 * semi-synthetic capacity",
        "release_target_rule": "fit-only calendar-month median, capacity clipped",
        "guide_curve_rule": (
            "fit-only 0.20/0.80 post-transition storage quantiles under the "
            "seasonal median release rule"
        ),
        "objective_normalizer_rule": "fit-only mean monthly inflow by basin",
        "post_fit_missing_policy": "preserve NaN; never interpolate",
        "split_missing_cells": tuple(
            (name, block.missing_cells) for name, block in historical.items()
        ),
        "split_complete_joint_water_years": tuple(
            (name, block.complete_joint_water_years)
            for name, block in historical.items()
        ),
        "fit_fingerprint_sha256": _fit_fingerprint(
            fit.inflow, fit.calendar, identifiers
        ),
        "train_only_audit_passed": True,
    }
    return CamelsBRBenchmark(
        system=system,
        fit_matrix=fit.inflow,
        fit_calendar=fit.calendar,
        historical_splits=historical,
        basin_audit=basin_audit,
        audit=audit,
    )


__all__ = [
    "CAMELS_BR_FORMAL_BASINS",
    "CAMELS_BR_HISTORICAL_SPLIT",
    "CAMELS_BR_HORIZON",
    "CAMELS_BR_PERIODS",
    "CLASSIC_MULTIPURPOSE_MULTIPLIERS",
    "CamelsBRBenchmark",
    "GUIDE_QUANTILES",
    "HistoricalSplitArrays",
    "build_camels_br_benchmark",
]
