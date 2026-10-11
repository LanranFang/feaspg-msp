"""Pure constructors for locked, observed historical replay batches.

These functions only encode provider observations with a stochastic model that
was fitted elsewhere.  They never fit, update, select, or otherwise expose the
locked paths to a training interface.  Hydrology features are delegated to
``observed_trajectory_batch``, whose stage-t state uses observations strictly
before stage t.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .benchmarks.camels_br import HistoricalSplitArrays
from .benchmarks.resopsus import HistoricalWindow
from .dataset import TrajectoryBatch
from .stochastic import PeriodicARModel, observed_trajectory_batch


ONS_HISTORY_START = pd.Timestamp("2000-01-01")
ONS_HISTORY_END = pd.Timestamp("2025-12-01")
ONS_LOCKED_START = pd.Timestamp("2022-01-01")
ONS_LOCKED_END = pd.Timestamp("2025-12-01")
ONS_HORIZON = 48
RESOPS_HORIZON = 52
CAMELS_HORIZON = 48
CAMELS_STRIDE = 12


def _require_model_width(
    model: PeriodicARModel, *, periods: int, width: int, source: str
) -> None:
    if not isinstance(model, PeriodicARModel):
        raise TypeError("model must be a fitted PeriodicARModel")
    if model.periods != periods:
        raise ValueError(
            f"{source} needs a {periods}-period model; got {model.periods}"
        )
    if model.n_series != width:
        raise ValueError(
            f"{source} series width {width} differs from model width "
            f"{model.n_series}"
        )


def _month_start(values: pd.Series, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise ValueError(f"invalid dates in {column}: {examples}")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.to_period("M").dt.to_timestamp()


def build_ons_locked_replay(
    historical: pd.DataFrame,
    model: PeriodicARModel,
    *,
    reservoir_ids: Sequence[str] | None = None,
    name: str = "ons_locked_2022_2025",
    seed: int = 0,
) -> TrajectoryBatch:
    """Encode the single January-2022--December-2025 ONS replay path.

    ``historical`` is the complete long table returned by the ONS benchmark.
    Its earlier rows supply observable prehistory only.  ``load_mwh`` is passed
    through as path-specific realized demand rather than replaced by the
    training-period seasonal demand curve.
    """

    if not isinstance(historical, pd.DataFrame):
        raise TypeError("historical must be a pandas DataFrame")
    required = {"time", "unit_id", "inflow_mwh", "load_mwh", "split"}
    missing = sorted(required.difference(historical.columns))
    if missing:
        raise KeyError(f"ONS historical table is missing columns: {missing}")
    frame = historical.loc[:, sorted(required)].copy()
    frame["time"] = _month_start(frame["time"], "historical.time")
    frame["unit_id"] = frame["unit_id"].astype("string").str.strip()
    frame["split"] = frame["split"].astype("string").str.strip()
    frame["inflow_mwh"] = pd.to_numeric(frame["inflow_mwh"], errors="coerce")
    frame["load_mwh"] = pd.to_numeric(frame["load_mwh"], errors="coerce")
    if frame["unit_id"].isna().any() or frame["unit_id"].eq("").any():
        raise ValueError("ONS historical table contains an empty unit_id")
    if frame.duplicated(["time", "unit_id"]).any():
        raise ValueError("ONS historical table contains duplicate unit/month rows")

    if reservoir_ids is None:
        ordered_ids = tuple(str(value) for value in frame["unit_id"].drop_duplicates())
    else:
        ordered_ids = tuple(str(value).strip() for value in reservoir_ids)
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("reservoir_ids must be nonempty and unique")
    observed_ids = set(frame["unit_id"].astype(str))
    if observed_ids != set(ordered_ids):
        raise ValueError("reservoir_ids must identify exactly the ONS long-table series")
    _require_model_width(
        model, periods=12, width=len(ordered_ids), source="ONS replay"
    )

    months = pd.date_range(ONS_HISTORY_START, ONS_HISTORY_END, freq="MS")
    observed_months = pd.DatetimeIndex(frame["time"].drop_duplicates().sort_values())
    if not observed_months.equals(months):
        raise ValueError(
            "ONS replay needs the complete monthly grid from 2000-01 through 2025-12"
        )
    inflow = (
        frame.pivot(index="time", columns="unit_id", values="inflow_mwh")
        .reindex(index=months, columns=list(ordered_ids))
        .to_numpy(dtype=np.float64)
    )
    if np.any(~np.isfinite(inflow)) or np.any(inflow < 0.0):
        raise ValueError("ONS historical inflow must be complete and nonnegative")

    demand_variants = frame.groupby("time", sort=True)["load_mwh"].nunique(
        dropna=False
    )
    if demand_variants.ne(1).any():
        raise ValueError("ONS load_mwh must be identical across units in each month")
    demand = (
        frame.groupby("time", sort=True)["load_mwh"]
        .first()
        .reindex(months)
        .to_numpy(dtype=np.float64)
    )
    if np.any(~np.isfinite(demand)) or np.any(demand < 0.0):
        raise ValueError("ONS historical demand must be complete and nonnegative")

    split_variants = frame.groupby("time", sort=True)["split"].nunique(dropna=False)
    if split_variants.ne(1).any():
        raise ValueError("ONS split label must be identical across units in each month")
    split_by_month = frame.groupby("time", sort=True)["split"].first().reindex(months)
    locked = months.to_series(index=months).between(ONS_LOCKED_START, ONS_LOCKED_END)
    if not split_by_month.loc[locked].eq("historical_test").all():
        raise ValueError("ONS 2022-01--2025-12 rows must be labelled historical_test")
    if int(locked.sum()) != ONS_HORIZON:
        raise RuntimeError("ONS locked interval is not 48 months")

    calendar = (months.month.to_numpy(dtype=np.int16) - 1).astype(np.int16)
    start = int(months.get_loc(ONS_LOCKED_START))
    return observed_trajectory_batch(
        model,
        name,
        inflow,
        calendar,
        np.asarray([start], dtype=np.int64),
        ONS_HORIZON,
        seed=seed,
        demand=demand,
    )


def build_resops_locked_replay(
    windows: Sequence[HistoricalWindow],
    model: PeriodicARModel,
    *,
    name: str = "resops_locked_test",
    seed: int = 0,
) -> TrajectoryBatch:
    """Use the first test water year as prehistory and replay all later years."""

    ordered = tuple(windows)
    if len(ordered) < 2:
        raise ValueError(
            "ResOps replay needs one prehistory water year and at least one path"
        )
    if not all(isinstance(window, HistoricalWindow) for window in ordered):
        raise TypeError("windows must contain HistoricalWindow instances")
    if any(window.split != "test" for window in ordered):
        raise ValueError("ResOps locked replay accepts only test windows")
    years = np.asarray([window.water_year for window in ordered], dtype=np.int64)
    if not np.array_equal(years, np.sort(years)) or np.any(np.diff(years) != 1):
        raise ValueError(
            "ResOps test windows must be sorted consecutive water years"
        )
    width = ordered[0].inflow.shape[1]
    if any(window.inflow.shape[1] != width for window in ordered):
        raise ValueError("ResOps windows do not share one reservoir width")
    _require_model_width(
        model, periods=RESOPS_HORIZON, width=width, source="ResOps replay"
    )

    week_end = np.concatenate([window.week_end for window in ordered])
    gaps = np.diff(week_end).astype("timedelta64[D]").astype(np.int64)
    # The benchmark's fixed 52-period water-year calendar deliberately drops
    # its occasional overflow week (calendar coordinate 52).  Consequently a
    # valid boundary can be 14 rather than 7 days apart; treating the overflow
    # observation as a 53rd model stage would shift every later season and no
    # longer match the fitted system.  Larger or intra-window gaps remain an
    # error (HistoricalWindow already enforces seven-day internal spacing).
    if np.any(~np.isin(gaps, (7, 14))):
        raise ValueError(
            "ResOps water-year windows must form the fixed 52-week timeline "
            "with at most one audited overflow week between years"
        )
    values = np.concatenate([window.inflow for window in ordered], axis=0)
    calendar = np.concatenate([window.calendar for window in ordered])
    starts = np.arange(1, len(ordered), dtype=np.int64) * RESOPS_HORIZON
    return observed_trajectory_batch(
        model,
        name,
        values,
        calendar,
        starts,
        RESOPS_HORIZON,
        seed=seed,
    )


def build_camels_locked_replay(
    split: HistoricalSplitArrays,
    model: PeriodicARModel,
    *,
    name: str = "camels_br_locked_real_test",
    seed: int = 0,
) -> TrajectoryBatch:
    """Encode all complete 48-month, stride-12 CAMELS locked-test paths.

    The first water year is initialization history and can never become an
    evaluation path.  Windows or their observable prehistory containing a
    missing value are omitted without imputation.
    """

    if not isinstance(split, HistoricalSplitArrays):
        raise TypeError("split must be HistoricalSplitArrays")
    if split.name != "locked_real_test":
        raise ValueError("CAMELS replay accepts only the locked_real_test split")
    width = len(split.reservoir_ids)
    _require_model_width(
        model, periods=12, width=width, source="CAMELS-BR replay"
    )
    minimum = max(model.order, model.periods if model.annual_component else 0)
    if minimum > CAMELS_STRIDE:
        raise ValueError("CAMELS model needs more than the locked first-year prehistory")
    if len(split.inflow) < CAMELS_STRIDE + CAMELS_HORIZON:
        raise ValueError(
            "CAMELS replay needs one prehistory year plus a complete 48-month path"
        )
    if not np.isfinite(split.inflow[:CAMELS_STRIDE]).all():
        raise ValueError("CAMELS first locked water year must be complete prehistory")

    starts = []
    for start in range(
        CAMELS_STRIDE,
        len(split.inflow) - CAMELS_HORIZON + 1,
        CAMELS_STRIDE,
    ):
        window = split.inflow[start : start + CAMELS_HORIZON]
        prehistory = split.inflow[start - minimum : start]
        if np.isfinite(window).all() and np.isfinite(prehistory).all():
            starts.append(start)
    if not starts:
        raise ValueError("CAMELS locked split has no complete replay window")
    return observed_trajectory_batch(
        model,
        name,
        split.inflow,
        split.calendar,
        np.asarray(starts, dtype=np.int64),
        CAMELS_HORIZON,
        seed=seed,
    )


__all__ = [
    "CAMELS_HORIZON",
    "CAMELS_STRIDE",
    "ONS_HORIZON",
    "RESOPS_HORIZON",
    "build_camels_locked_replay",
    "build_ons_locked_replay",
    "build_resops_locked_replay",
]
