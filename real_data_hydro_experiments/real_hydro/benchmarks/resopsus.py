"""Leakage-safe benchmark construction for the two ResOpsUS v2 cases.

The data layer returns a long weekly table.  This module turns that table into
the immutable physical/control configuration used by the common simulator and
the chronological arrays consumed by :func:`real_hydro.stochastic.fit_periodic_ar`.

Only *local* inflow is exposed to the stochastic model.  In the Ross--Diablo
case, Ross release is therefore routed by ``ReservoirSystem.routing`` exactly
once; it is never folded into Diablo's exogenous inflow a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import ObjectiveConfig, ReservoirSystem
from ..data.resopsus import (
    CaseSpec,
    DataIntegrityError,
    MassBalanceAudit,
    audit_mass_balance,
    calibrate_training_limits,
    get_case,
)


HORIZON = 52
PERIODS = 52
GUIDE_LOWER_QUANTILE = 0.10
GUIDE_UPPER_QUANTILE = 0.90
RELEASE_TARGET_QUANTILE = 0.50
STORAGE_CAPACITY_QUANTILE = 0.999
RELEASE_CAPACITY_QUANTILE = 0.995

# Costs apply to normalized deviations, making the fixed multipliers comparable
# between Lucky Peak and the much larger Ross reservoir.  The normalizers below
# are the training-calibrated physical limits, never held-out observations.
_PENALTY_MULTIPLIERS: Mapping[str, float] = {
    "shortage": 1.0,
    "excess_release": 0.10,
    "low_storage": 0.50,
    "flood": 1.0,
    "spill": 2.0,
    "ramp": 0.05,
    "terminal_shortfall": 2.0,
    "terminal_excess": 0.10,
}

_MODEL_COLUMNS = (
    "reported_inflow_mcm",
    "local_inflow_mcm",
    "model_local_inflow_mcm",
    "routed_inflow_mcm",
    "storage_mcm",
    "release_mcm",
    "reference_capacity_mcm",
)


class ResOpsBenchmarkError(DataIntegrityError):
    """Raised when a weekly table cannot define the frozen benchmark."""


def _readonly_array(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HistoricalWindow:
    """One complete, non-overlapping 52-week historical replay window."""

    split: str
    water_year: int
    week_end: np.ndarray
    calendar: np.ndarray
    inflow: np.ndarray
    observed_storage: np.ndarray
    observed_release: np.ndarray
    opening_storage: tuple[float, ...]

    def __post_init__(self) -> None:
        week_end = _readonly_array(self.week_end, "datetime64[ns]")
        calendar = _readonly_array(self.calendar, np.int16)
        inflow = _readonly_array(self.inflow, np.float64)
        storage = _readonly_array(self.observed_storage, np.float64)
        release = _readonly_array(self.observed_release, np.float64)
        if week_end.shape != (PERIODS,) or calendar.shape != (PERIODS,):
            raise ValueError("Historical dates/calendar must contain exactly 52 weeks.")
        if inflow.ndim != 2 or inflow.shape[0] != PERIODS:
            raise ValueError("Historical inflow must have shape [52, reservoir].")
        if storage.shape != inflow.shape or release.shape != inflow.shape:
            raise ValueError("Historical storage/release must align with inflow.")
        if len(self.opening_storage) != inflow.shape[1]:
            raise ValueError("opening_storage must have one value per reservoir.")
        if not np.array_equal(calendar, np.arange(PERIODS, dtype=np.int16)):
            raise ValueError("A historical window must contain calendar periods 0..51.")
        if np.any(~np.isfinite(inflow)) or np.any(inflow < 0.0):
            raise ValueError("Historical model inflow must be finite and nonnegative.")
        if np.any(np.diff(week_end).astype("timedelta64[D]").astype(int) != 7):
            raise ValueError("A historical window must contain consecutive weeks.")
        object.__setattr__(self, "week_end", week_end)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "inflow", inflow)
        object.__setattr__(self, "observed_storage", storage)
        object.__setattr__(self, "observed_release", release)
        object.__setattr__(
            self,
            "opening_storage",
            tuple(float(value) for value in self.opening_storage),
        )

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.week_end[0]) - pd.Timedelta(days=6)

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.week_end[-1])


@dataclass(frozen=True)
class ResOpsBenchmarkAudit:
    """Physical and preprocessing diagnostics shipped with every bundle."""

    raw_mass_balance: MassBalanceAudit
    projected_mass_balance: MassBalanceAudit
    alignment_by_split: pd.DataFrame
    calibration_week_ends: tuple[pd.Timestamp, ...]
    calibration_splits: tuple[str, ...] = ("train",)

    @property
    def mass_balance(self) -> MassBalanceAudit:
        """The unprojected local-inflow audit used for data-quality reporting."""

        return self.raw_mass_balance


@dataclass(frozen=True)
class ResOpsBenchmark:
    """A ResOpsUS system plus generator fit data and locked historical replay."""

    system: ReservoirSystem
    fit_inflow: np.ndarray
    fit_calendar: np.ndarray
    fit_week_end: np.ndarray
    historical_windows: tuple[HistoricalWindow, ...]
    limits: pd.DataFrame
    audit: ResOpsBenchmarkAudit

    def __post_init__(self) -> None:
        inflow = _readonly_array(self.fit_inflow, np.float64)
        calendar = _readonly_array(self.fit_calendar, np.int16)
        week_end = _readonly_array(self.fit_week_end, "datetime64[ns]")
        if inflow.ndim != 2 or inflow.shape[1] != self.system.n_reservoirs:
            raise ValueError("fit_inflow must have shape [time, system reservoir].")
        if calendar.shape != (len(inflow),) or week_end.shape != (len(inflow),):
            raise ValueError("Fit calendar/dates must align with fit_inflow.")
        if np.any(~np.isfinite(inflow)) or np.any(inflow < 0.0):
            raise ValueError("Fit inflow must be finite and nonnegative.")
        if np.any((calendar < 0) | (calendar >= self.system.periods)):
            raise ValueError("Fit calendar lies outside the system period range.")
        object.__setattr__(self, "fit_inflow", inflow)
        object.__setattr__(self, "fit_calendar", calendar)
        object.__setattr__(self, "fit_week_end", week_end)

    @property
    def fit_local_inflow(self) -> np.ndarray:
        """Explicit alias documenting the semantics of ``fit_inflow``."""

        return self.fit_inflow

    @property
    def calendar(self) -> np.ndarray:
        """Alias used by stochastic-process fitting call sites."""

        return self.fit_calendar

    def historical_by_split(self, split: str) -> tuple[HistoricalWindow, ...]:
        return tuple(window for window in self.historical_windows if window.split == split)


@dataclass(frozen=True)
class _AlignedPanel:
    """Internal common-date matrices in the reservoir order of ``CaseSpec``."""

    long: pd.DataFrame
    week_end: pd.DatetimeIndex
    week_start: pd.DatetimeIndex
    split: np.ndarray
    water_year: np.ndarray
    calendar: np.ndarray
    complete: np.ndarray
    matrices: Mapping[str, np.ndarray]


def _hydrologic_calendar(
    week_start: pd.Series, week_end: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return October water year and a stable 0..52 week coordinate.

    The midpoint assigns boundary-crossing Monday--Sunday observations to one
    water year without duplicating them.  Days are binned in seven-day blocks
    after October 1.  A long water year's overflow coordinate 52 is deliberately
    omitted from the fixed 52-period benchmark rather than shifting all later
    seasons.
    """

    midpoint = week_start + (week_end - week_start) / 2
    water_year = midpoint.dt.year + midpoint.dt.month.ge(10).astype(int)
    starts = pd.to_datetime((water_year - 1).astype(str) + "-10-01")
    calendar = ((midpoint.dt.normalize() - starts).dt.days // 7).to_numpy(dtype=int)
    years = water_year.to_numpy(dtype=int)
    if np.any((calendar < 0) | (calendar > PERIODS)):
        raise ResOpsBenchmarkError("Weekly dates cannot be mapped to a water-year calendar.")
    return years, calendar


def _validate_split_dates(data: pd.DataFrame, spec: CaseSpec) -> None:
    windows = {window.name: window for window in spec.splits}
    labels = set(data["split"].dropna().astype(str).unique())
    unknown = sorted(labels - set(windows))
    if unknown:
        raise ResOpsBenchmarkError(f"Unknown split labels: {unknown}.")
    for label, window in windows.items():
        rows = data.loc[data["split"].eq(label)]
        invalid = (rows["week_start"] < pd.Timestamp(window.start)) | (
            rows["week_end"] > pd.Timestamp(window.end)
        )
        if invalid.any():
            example = rows.loc[invalid, "week_end"].iloc[0]
            raise ResOpsBenchmarkError(
                f"Week ending {example:%Y-%m-%d} is mislabeled as {label!r}."
            )


def _validate_routing_semantics(panel: _AlignedPanel, spec: CaseSpec) -> None:
    ids = spec.dam_ids
    reported = panel.matrices["reported_inflow_mcm"]
    local = panel.matrices["local_inflow_mcm"]
    model_local = panel.matrices["model_local_inflow_mcm"]
    routed = panel.matrices["routed_inflow_mcm"]
    release = panel.matrices["release_mcm"]

    finite = np.isfinite(local) & np.isfinite(model_local)
    expected_model = np.maximum(local, 0.0)
    if not np.allclose(model_local[finite], expected_model[finite], rtol=1e-7, atol=1e-9):
        raise ResOpsBenchmarkError(
            "model_local_inflow_mcm must be the nonnegative projection of raw local inflow."
        )
    finite_model = model_local[np.isfinite(model_local)]
    if np.any(finite_model < 0.0):
        raise ResOpsBenchmarkError("Model local inflow cannot be negative.")

    if spec.inflow_semantics == "direct_local":
        valid = np.isfinite(reported) & np.isfinite(local)
        if not np.allclose(reported[valid], local[valid], rtol=1e-7, atol=1e-9):
            raise ResOpsBenchmarkError(
                "Direct-local cases require local inflow to equal the prepared "
                "reported net-local series."
            )
        lookup = {dam_id: index for index, dam_id in enumerate(ids)}
        expected_routed = np.zeros_like(routed)
        for source, target in spec.routing_edges:
            expected_routed[:, lookup[target]] += release[:, lookup[source]]
        valid = np.isfinite(routed) & np.isfinite(expected_routed)
        if not np.allclose(
            routed[valid], expected_routed[valid], rtol=1e-7, atol=1e-9
        ):
            raise ResOpsBenchmarkError(
                "Direct-local routed inflow must equal observed upstream release "
                "applied once along each frozen edge."
            )
        return

    if spec.inflow_semantics != "cumulative_natural" or ids != (41, 42):
        raise ResOpsBenchmarkError(f"Unsupported ResOpsUS routing case {spec.name!r}.")
    ross, diablo = 0, 1
    comparisons = (
        (local[:, ross], reported[:, ross], "Ross local inflow must equal q41."),
        (
            local[:, diablo],
            reported[:, diablo] - reported[:, ross],
            "Diablo local inflow must equal q42 - q41.",
        ),
        (
            routed[:, diablo],
            release[:, ross],
            "Diablo routed inflow must equal Ross release.",
        ),
        (routed[:, ross], np.zeros(len(routed)), "Ross cannot have routed inflow."),
    )
    for observed, expected, message in comparisons:
        valid = np.isfinite(observed) & np.isfinite(expected)
        if not np.allclose(observed[valid], expected[valid], rtol=1e-7, atol=1e-9):
            raise ResOpsBenchmarkError(message)


def _aligned_panel(weekly_case: pd.DataFrame, spec: CaseSpec) -> _AlignedPanel:
    required = {
        "dam_id",
        "reservoir_name",
        "week_start",
        "week_end",
        "split",
        *_MODEL_COLUMNS,
    }
    missing = sorted(required - set(weekly_case.columns))
    if missing:
        raise ResOpsBenchmarkError(f"Prepared weekly case is missing columns: {missing}.")
    data = weekly_case.copy()
    numeric_ids = pd.to_numeric(data["dam_id"], errors="coerce")
    if numeric_ids.isna().any() or not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ResOpsBenchmarkError("Prepared weekly case contains an invalid DAM_ID.")
    data["dam_id"] = numeric_ids.astype(int)
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data["week_end"] = pd.to_datetime(data["week_end"], errors="coerce")
    if data[["week_start", "week_end"]].isna().any(axis=None):
        raise ResOpsBenchmarkError("Prepared weekly case contains invalid dates.")
    if data.duplicated(["dam_id", "week_end"]).any():
        raise ResOpsBenchmarkError("Prepared weekly case contains duplicate dam/week rows.")
    ids = tuple(int(value) for value in pd.unique(data["dam_id"]))
    if set(ids) != set(spec.dam_ids):
        raise ResOpsBenchmarkError(
            f"Case {spec.name!r} requires DAM_ID values {spec.dam_ids}, got {ids}."
        )
    if "case_name" in data:
        case_names = set(data["case_name"].dropna().astype(str).unique())
        if case_names and case_names != {spec.name}:
            raise ResOpsBenchmarkError(
                f"Prepared case_name values {sorted(case_names)} do not match {spec.name!r}."
            )
    duration = (data["week_end"] - data["week_start"]).dt.days
    if not duration.eq(6).all() or not data["week_end"].dt.dayofweek.eq(6).all():
        raise ResOpsBenchmarkError(
            "Every observation must be a complete Monday--Sunday week."
        )
    _validate_split_dates(data, spec)
    for column in _MODEL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    group = data.groupby("week_end", sort=True)
    inconsistent_split = group["split"].nunique(dropna=False).gt(1)
    inconsistent_start = group["week_start"].nunique(dropna=False).gt(1)
    if inconsistent_split.any() or inconsistent_start.any():
        raise ResOpsBenchmarkError("Reservoir rows disagree on week bounds or split.")
    week_end = pd.DatetimeIndex(sorted(data["week_end"].unique()))
    metadata = (
        data.drop_duplicates("week_end")
        .set_index("week_end")
        .reindex(week_end)
    )
    matrices: dict[str, np.ndarray] = {}
    for column in _MODEL_COLUMNS:
        wide = data.pivot(index="week_end", columns="dam_id", values=column)
        wide = wide.reindex(index=week_end, columns=spec.dam_ids)
        matrices[column] = wide.to_numpy(dtype=np.float64)
        if np.isinf(matrices[column]).any():
            raise ResOpsBenchmarkError(f"Column {column} contains an infinite value.")
    for column in ("storage_mcm", "release_mcm", "reference_capacity_mcm"):
        finite = matrices[column][np.isfinite(matrices[column])]
        if np.any(finite < 0.0):
            raise ResOpsBenchmarkError(f"Column {column} cannot contain negative values.")
    reference = matrices["reference_capacity_mcm"]
    for column, reservoir in enumerate(spec.reservoirs):
        observed = reference[:, column]
        valid = np.isfinite(observed)
        if not np.allclose(
            observed[valid], reservoir.reference_capacity_mcm, rtol=0.0, atol=1e-9
        ):
            raise ResOpsBenchmarkError(
                f"DAM_ID {reservoir.dam_id} reference capacity does not match "
                "the frozen case specification."
            )
    complete = np.ones(len(week_end), dtype=bool)
    for matrix in matrices.values():
        complete &= np.all(np.isfinite(matrix), axis=1)
    week_start = pd.DatetimeIndex(metadata["week_start"])
    water_year, calendar = _hydrologic_calendar(
        pd.Series(week_start), pd.Series(week_end)
    )
    panel = _AlignedPanel(
        long=data,
        week_end=week_end,
        week_start=week_start,
        split=(
            metadata["split"]
            .astype("string")
            .fillna("__unassigned__")
            .to_numpy(dtype=str)
        ),
        water_year=water_year,
        calendar=calendar,
        complete=complete,
        matrices=matrices,
    )
    _validate_routing_semantics(panel, spec)
    return panel


def _routing_matrix(spec: CaseSpec) -> tuple[tuple[float, ...], ...]:
    lookup = {dam_id: index for index, dam_id in enumerate(spec.dam_ids)}
    routing = np.zeros((len(lookup), len(lookup)), dtype=np.float64)
    for source, target in spec.routing_edges:
        if source not in lookup or target not in lookup:
            raise ResOpsBenchmarkError(f"Routing edge {(source, target)} leaves the case.")
        if routing[lookup[source], lookup[target]] != 0.0:
            raise ResOpsBenchmarkError(f"Duplicate routing edge {(source, target)}.")
        routing[lookup[source], lookup[target]] = 1.0
    return tuple(tuple(float(value) for value in row) for row in routing)


def _seasonal_training_targets(
    panel: _AlignedPanel,
    fit_mask: np.ndarray,
    storage_capacity: np.ndarray,
    release_capacity: np.ndarray,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    storage = panel.matrices["storage_mcm"]
    release = panel.matrices["release_mcm"]
    guide_lower: list[tuple[float, ...]] = []
    guide_upper: list[tuple[float, ...]] = []
    release_target: list[tuple[float, ...]] = []
    for period in range(PERIODS):
        rows = fit_mask & (panel.calendar == period)
        if not rows.any():
            raise ResOpsBenchmarkError(
                f"Training data contain no complete observation for period {period}."
            )
        lower = np.quantile(storage[rows], GUIDE_LOWER_QUANTILE, axis=0)
        upper = np.quantile(storage[rows], GUIDE_UPPER_QUANTILE, axis=0)
        target = np.quantile(release[rows], RELEASE_TARGET_QUANTILE, axis=0)
        lower = np.clip(lower, 0.0, storage_capacity)
        upper = np.clip(upper, lower, storage_capacity)
        target = np.clip(target, 0.0, release_capacity)
        guide_lower.append(tuple(float(value) for value in lower))
        guide_upper.append(tuple(float(value) for value in upper))
        release_target.append(tuple(float(value) for value in target))
    return tuple(release_target), tuple(guide_lower), tuple(guide_upper)


def _boundary_targets(
    panel: _AlignedPanel,
    fit_mask: np.ndarray,
    storage_capacity: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    storage = panel.matrices["storage_mcm"]
    period_zero_indices = np.flatnonzero(fit_mask & (panel.calendar == 0))
    opening_rows: list[np.ndarray] = []
    date_lookup = {date: index for index, date in enumerate(panel.week_end)}
    for index in period_zero_indices:
        previous = date_lookup.get(panel.week_end[index] - pd.Timedelta(days=7))
        if previous is not None and panel.complete[previous] and panel.split[previous] == "train":
            opening_rows.append(storage[previous])
    terminal_rows = storage[fit_mask & (panel.calendar == PERIODS - 1)]
    if not len(terminal_rows):
        raise ResOpsBenchmarkError("Training data contain no period-51 storage target.")
    if opening_rows:
        initial = np.median(np.stack(opening_rows), axis=0)
    else:
        # A fixture or truncated source may lack the immediately preceding week.
        # Period-51 storage is the closest leakage-safe cyclic boundary proxy.
        initial = np.median(terminal_rows, axis=0)
    terminal = np.median(terminal_rows, axis=0)
    initial = np.clip(initial, 0.0, storage_capacity)
    terminal = np.clip(terminal, 0.0, storage_capacity)
    return (
        tuple(float(value) for value in initial),
        tuple(float(value) for value in terminal),
    )


def _objective(
    release_target: tuple[tuple[float, ...], ...],
    guide_lower: tuple[tuple[float, ...], ...],
    guide_upper: tuple[tuple[float, ...], ...],
    terminal_target: tuple[float, ...],
    storage_capacity: np.ndarray,
    release_capacity: np.ndarray,
) -> ObjectiveConfig:
    storage_scale = 1.0 / storage_capacity
    release_scale = 1.0 / release_capacity

    def scaled(multiplier: str, scale: np.ndarray) -> tuple[float, ...]:
        return tuple(float(_PENALTY_MULTIPLIERS[multiplier] * value) for value in scale)

    return ObjectiveConfig(
        kind="multipurpose",
        release_target=release_target,
        guide_lower=guide_lower,
        guide_upper=guide_upper,
        terminal_target=terminal_target,
        shortage_cost=scaled("shortage", release_scale),
        excess_release_cost=scaled("excess_release", release_scale),
        low_storage_cost=scaled("low_storage", storage_scale),
        flood_cost=scaled("flood", storage_scale),
        spill_cost=scaled("spill", storage_scale),
        ramp_cost=scaled("ramp", release_scale),
        terminal_shortfall_cost=scaled("terminal_shortfall", storage_scale),
        terminal_excess_cost=scaled("terminal_excess", storage_scale),
    )


def _historical_windows(
    panel: _AlignedPanel,
    system: ReservoirSystem,
) -> tuple[HistoricalWindow, ...]:
    windows: list[HistoricalWindow] = []
    storage = panel.matrices["storage_mcm"]
    release = panel.matrices["release_mcm"]
    inflow = panel.matrices["model_local_inflow_mcm"]
    date_lookup = {date: index for index, date in enumerate(panel.week_end)}
    for split in ("validation", "test"):
        split_mask = (
            panel.complete
            & (panel.calendar < PERIODS)
            & np.asarray(panel.split == split, dtype=bool)
        )
        for water_year in sorted(set(panel.water_year[split_mask])):
            indices = np.flatnonzero(split_mask & (panel.water_year == water_year))
            if len(indices) != PERIODS:
                continue
            order = np.argsort(panel.calendar[indices], kind="stable")
            indices = indices[order]
            if not np.array_equal(panel.calendar[indices], np.arange(PERIODS)):
                continue
            dates = panel.week_end[indices]
            gaps = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int)
            if np.any(gaps != 7):
                continue
            previous = date_lookup.get(dates[0] - pd.Timedelta(days=7))
            if previous is not None and panel.complete[previous]:
                opening = storage[previous]
            else:
                opening = np.asarray(system.initial_storage, dtype=np.float64)
            windows.append(
                HistoricalWindow(
                    split=split,
                    water_year=int(water_year),
                    week_end=dates.to_numpy(),
                    calendar=panel.calendar[indices],
                    inflow=inflow[indices],
                    observed_storage=storage[indices],
                    observed_release=release[indices],
                    opening_storage=tuple(float(value) for value in opening),
                )
            )
    for split in ("validation", "test"):
        if not any(window.split == split for window in windows):
            raise ResOpsBenchmarkError(
                f"Split {split!r} contains no complete aligned 52-week window."
            )
    return tuple(windows)


def _alignment_audit(
    panel: _AlignedPanel, windows: tuple[HistoricalWindow, ...]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        in_split = np.asarray(panel.split == split, dtype=bool)
        complete = in_split & panel.complete
        calendar_ready = complete & (panel.calendar < PERIODS)
        rows.append(
            {
                "split": split,
                "input_weeks": int(in_split.sum()),
                "complete_reservoir_weeks": int(complete.sum()),
                "fixed_calendar_weeks": int(calendar_ready.sum()),
                "dropped_incomplete_weeks": int((in_split & ~panel.complete).sum()),
                "dropped_overflow_weeks": int((complete & (panel.calendar == PERIODS)).sum()),
                "complete_historical_windows": sum(
                    window.split == split for window in windows
                ),
            }
        )
    return pd.DataFrame(rows)


def build_resopsus_benchmark(
    weekly_case: pd.DataFrame,
    case: str | CaseSpec,
    *,
    benchmark_name: str = "resopsus_v2",
    source_metadata: tuple[tuple[str, str], ...] = (),
) -> ResOpsBenchmark:
    """Build the frozen Lucky Peak or Ross--Diablo weekly benchmark.

    Parameters
    ----------
    weekly_case:
        Output of :func:`real_hydro.data.resopsus.prepare_weekly_case`.  Missing
        or unaligned reservoir weeks are audited and omitted jointly.
    case:
        ``"lucky_peak"`` or ``"ross_diablo"``.

    Returns
    -------
    ResOpsBenchmark
        ``fit_inflow`` is a chronological ``[week, reservoir]`` matrix of local
        inflow only; ``fit_calendar`` is its October-based 0..51 coordinate.
        Validation/test replays contain only complete, non-overlapping water
        years.  Every objective statistic and physical fitted limit uses rows
        labelled ``train`` exclusively.
    """

    if isinstance(case, str) and case not in {"lucky_peak", "ross_diablo"}:
        raise KeyError("ResOps benchmark case must be 'lucky_peak' or 'ross_diablo'.")
    spec = get_case(case)
    panel = _aligned_panel(weekly_case, spec)
    fit_mask = (
        panel.complete
        & (panel.calendar < PERIODS)
        & np.asarray(panel.split == "train", dtype=bool)
    )
    if not fit_mask.any():
        raise ResOpsBenchmarkError("No complete training weeks are available.")

    fit_dates = panel.week_end[fit_mask]
    fit_long = panel.long.loc[
        panel.long["week_end"].isin(fit_dates) & panel.long["split"].eq("train")
    ].copy()
    limits = calibrate_training_limits(
        fit_long,
        spec,
        storage_quantile=STORAGE_CAPACITY_QUANTILE,
        release_quantile=RELEASE_CAPACITY_QUANTILE,
    )
    limits_by_id = limits.set_index("dam_id").reindex(spec.dam_ids)
    storage_capacity = limits_by_id["model_storage_capacity_mcm"].to_numpy(float)
    release_capacity = limits_by_id["release_capacity_mcm_per_week"].to_numpy(float)
    if np.any(~np.isfinite(storage_capacity)) or np.any(storage_capacity <= 0.0):
        raise ResOpsBenchmarkError("Training-calibrated storage limits must be positive.")
    if np.any(~np.isfinite(release_capacity)) or np.any(release_capacity <= 0.0):
        raise ResOpsBenchmarkError("Training-calibrated release limits must be positive.")

    release_target, guide_lower, guide_upper = _seasonal_training_targets(
        panel, fit_mask, storage_capacity, release_capacity
    )
    initial_storage, terminal_target = _boundary_targets(
        panel, fit_mask, storage_capacity
    )
    objective = _objective(
        release_target,
        guide_lower,
        guide_upper,
        terminal_target,
        storage_capacity,
        release_capacity,
    )
    system = ReservoirSystem(
        benchmark=str(benchmark_name),
        case_id=spec.name,
        horizon=HORIZON,
        periods=PERIODS,
        step_unit="mcm",
        reservoir_ids=tuple(str(dam_id) for dam_id in spec.dam_ids),
        storage_capacity=tuple(float(value) for value in storage_capacity),
        release_capacity=tuple(float(value) for value in release_capacity),
        initial_storage=initial_storage,
        routing=_routing_matrix(spec),
        objective=objective,
        metadata=(
            *(
                source_metadata
                if source_metadata
                else (("source", "ResOpsUS v2; Zenodo 6612040"),)
            ),
            ("calibration_split", "train only"),
            ("calendar", "52 October-based midpoint weeks; overflow week excluded"),
            ("inflow", "local weekly MCM; system routing applies upstream release once"),
            ("guide_curves", "train q10/q90 by calendar period"),
            ("release_target", "train median by calendar period"),
        ),
    )
    historical = _historical_windows(panel, system)
    raw_audit = audit_mass_balance(panel.long, use_projected_local_inflow=False)
    projected_audit = audit_mass_balance(panel.long, use_projected_local_inflow=True)
    audit = ResOpsBenchmarkAudit(
        raw_mass_balance=raw_audit,
        projected_mass_balance=projected_audit,
        alignment_by_split=_alignment_audit(panel, historical),
        calibration_week_ends=tuple(pd.Timestamp(value) for value in fit_dates),
    )
    return ResOpsBenchmark(
        system=system,
        fit_inflow=panel.matrices["model_local_inflow_mcm"][fit_mask],
        fit_calendar=panel.calendar[fit_mask],
        fit_week_end=fit_dates.to_numpy(),
        historical_windows=historical,
        limits=limits,
        audit=audit,
    )


__all__ = [
    "GUIDE_LOWER_QUANTILE",
    "GUIDE_UPPER_QUANTILE",
    "HORIZON",
    "HistoricalWindow",
    "PERIODS",
    "RELEASE_CAPACITY_QUANTILE",
    "RELEASE_TARGET_QUANTILE",
    "ResOpsBenchmark",
    "ResOpsBenchmarkAudit",
    "ResOpsBenchmarkError",
    "STORAGE_CAPACITY_QUANTILE",
    "build_resopsus_benchmark",
]
