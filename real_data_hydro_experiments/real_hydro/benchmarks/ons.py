"""Build a leakage-safe NEWAVE-style benchmark from normalized ONS tables.

The builder works in monthly energy units.  Reservoir ENA is integrated from
daily average MW to MWh, useful storage is converted to equivalent stored energy,
and subsystem load is converted from monthly average MW to MWh.  Every parameter
of :class:`~real_hydro.config.ReservoirSystem` is calibrated from 2000--2014 or
from static plant attributes whose commissioning date is no later than 2014.

The ONS installed-generation table is a current snapshot rather than a historical
capacity series.  It is nevertheless preferable to inventing thermal capacity:
the formal path matches training-period CVU plants to thermal units and converts
their effective MW to monthly MWh.  A demand-scaled proxy exists only behind the
explicit ``thermal_capacity_proxy=True`` sensitivity flag.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import ObjectiveConfig, ReservoirSystem
from ..data.ons import assign_temporal_split


TRAIN_START = pd.Timestamp("2000-01-01")
TRAIN_END = pd.Timestamp("2014-12-31")
HISTORY_START = pd.Timestamp("2000-01-01")
HISTORY_END = pd.Timestamp("2025-12-31")
PERIODS = 12
EXPECTED_TRAIN_MONTHS = 15 * PERIODS
THERMAL_FUZZY_SCORE_MIN = 0.92
THERMAL_FUZZY_MARGIN_MIN = 0.10

# 1 hm3 passing at 1 m3/s takes 1e6 / 3600 hours.
MWH_PER_HM3_PER_MW_PER_M3S = 1_000_000.0 / 3_600.0


@dataclass(frozen=True)
class ONSBenchmarkAudit:
    """Tables that make selection and every proxy visible."""

    reservoir_capacity: pd.DataFrame
    aggregation_selection: pd.DataFrame
    monthly_coverage: pd.DataFrame
    reservoir_component_coverage: pd.DataFrame
    monthly_load: pd.DataFrame
    thermal_plants: pd.DataFrame
    thermal_blocks: pd.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "reservoir_capacity",
            "aggregation_selection",
            "monthly_coverage",
            "reservoir_component_coverage",
            "monthly_load",
            "thermal_plants",
            "thermal_blocks",
        ):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))

    @property
    def ree_selection(self) -> pd.DataFrame:
        """Compatibility alias for explicit ``aggregation_level='ree'`` cases."""

        return self.aggregation_selection


@dataclass(frozen=True)
class ONSBenchmark:
    """Fully constructed ONS case plus the historical evidence used to build it."""

    system: ReservoirSystem
    fit_inflow: np.ndarray
    calendar: np.ndarray
    fit_times: pd.DatetimeIndex
    historical: pd.DataFrame
    audit: ONSBenchmarkAudit

    def __post_init__(self) -> None:
        inflow = np.array(self.fit_inflow, dtype=np.float64, copy=True)
        calendar = np.array(self.calendar, dtype=np.int16, copy=True)
        if inflow.shape != (EXPECTED_TRAIN_MONTHS, self.system.n_reservoirs):
            raise ValueError("fit_inflow must contain 180 months by selected aggregate")
        if calendar.shape != (EXPECTED_TRAIN_MONTHS,):
            raise ValueError("calendar must contain one value per fit month")
        if len(self.fit_times) != EXPECTED_TRAIN_MONTHS:
            raise ValueError("fit_times must cover 2000-01 through 2014-12")
        if np.any(~np.isfinite(inflow)) or np.any(inflow < 0.0):
            raise ValueError("fit_inflow must be finite and nonnegative")
        if not np.array_equal(calendar, np.arange(EXPECTED_TRAIN_MONTHS) % PERIODS):
            raise ValueError("calendar must be zero-based January--December")
        inflow.setflags(write=False)
        calendar.setflags(write=False)
        object.__setattr__(self, "fit_inflow", inflow)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "historical", self.historical.copy(deep=True))

    @property
    def selected_aggregates(self) -> tuple[str, ...]:
        return self.system.reservoir_ids

    @property
    def selected_rees(self) -> tuple[str, ...]:
        """Compatibility alias retained for explicit REE sensitivity cases."""

        return self.selected_aggregates

    @property
    def fit_matrix(self) -> np.ndarray:
        """Alias used by monthly benchmark adapters."""

        return self.fit_inflow

    @property
    def fit_calendar(self) -> np.ndarray:
        """Alias shared with the other stochastic-process fit contracts."""

        return self.calendar

    @property
    def historical_split_table(self) -> pd.DataFrame:
        return self.historical


def _require(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table} must be a pandas DataFrame")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{table} is missing required columns: {missing}")


def _dates(values: pd.Series, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise ValueError(f"invalid dates in {column!r}: {examples}")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _numbers(values: pd.Series, column: str) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce").astype(float)
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise ValueError(f"invalid numbers in {column!r}: {examples}")
    return parsed


def _strings(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().replace("", pd.NA)


def _month_start(values: pd.Series) -> pd.Series:
    return values.dt.to_period("M").dt.to_timestamp()


def _within_history(times: pd.Series) -> pd.Series:
    return times.between(HISTORY_START, HISTORY_END)


def _within_train(times: pd.Series) -> pd.Series:
    return times.between(TRAIN_START, TRAIN_END)


def aggregate_monthly_ena(
    ena: pd.DataFrame, *, aggregation_level: str = "subsystem"
) -> pd.DataFrame:
    """Integrate daily gross ENA to an auditable aggregate/month MWh table.

    Gross rather than storable ENA is used because storable ENA embeds spill
    effects.  A month is usable when every calendar day has at least one
    published numeric contribution for the aggregate.  ONS reservoir-level ENA
    contains structural blanks, so requiring every listed reservoir value would
    discard otherwise complete official subsystem histories.  Component-level
    missingness is retained in the audit columns rather than silently labelled
    complete.  Reservoir sums produced here are coverage diagnostics only in a
    formal subsystem case; model inflow comes from the separately published ONS
    subsystem aggregate.  REE aggregation remains an explicit
    sensitivity/fixture mode.
    """

    if aggregation_level not in {"subsystem", "ree"}:
        raise ValueError("aggregation_level must be 'subsystem' or 'ree'")
    assignment_column = (
        "subsystem_id" if aggregation_level == "subsystem" else "ree_name"
    )
    required = ["time", "unit_id", assignment_column, "ena_gross_mw_mean"]
    _require(ena, required, "ENA")
    frame = ena.loc[:, required].copy()
    frame["time"] = _dates(frame["time"], "ENA.time")
    frame["unit_id"] = _strings(frame["unit_id"])
    frame["aggregation_id"] = _strings(frame[assignment_column])
    frame["ena_gross_mw_mean"] = _numbers(
        frame["ena_gross_mw_mean"], "ENA.ena_gross_mw_mean"
    )
    frame = frame.loc[_within_history(frame["time"])].copy()
    missing_assignment = frame["aggregation_id"].isna()
    if missing_assignment.any():
        label = "subsystem_id" if aggregation_level == "subsystem" else "ree_name"
        train_missing = missing_assignment & _within_train(frame["time"])
        if train_missing.any():
            raise ValueError(f"training ENA contains an empty {label}")
        frame = frame.loc[~missing_assignment].copy()
    if frame["unit_id"].isna().any():
        raise ValueError("ENA contains an empty unit_id")
    if (frame["ena_gross_mw_mean"].dropna() < 0.0).any():
        raise ValueError("gross ENA cannot be negative")
    if frame.duplicated(["unit_id", "time"]).any():
        raise ValueError("ENA contains duplicate planning-reservoir/day rows")

    daily = (
        frame.groupby(["aggregation_id", "time"], sort=True, dropna=False)
        .agg(
            ena_gross_mw_mean=(
                "ena_gross_mw_mean",
                lambda values: values.sum(min_count=1),
            ),
            listed_units=("unit_id", "nunique"),
            valid_units=("ena_gross_mw_mean", "count"),
        )
        .reset_index()
    )
    daily["all_values_present"] = daily["listed_units"].eq(daily["valid_units"])
    daily["daily_ena_mwh"] = daily["ena_gross_mw_mean"] * 24.0
    daily["month"] = _month_start(daily["time"])

    monthly = (
        daily.groupby(["aggregation_id", "month"], sort=True, dropna=False)
        .agg(
            observed_inflow_mwh=(
                "daily_ena_mwh",
                lambda values: values.sum(min_count=1),
            ),
            observed_days=("ena_gross_mw_mean", "count"),
            minimum_daily_units=("valid_units", "min"),
            maximum_daily_units=("listed_units", "max"),
            listed_component_rows=("listed_units", "sum"),
            valid_component_rows=("valid_units", "sum"),
            all_values_present=("all_values_present", "all"),
        )
        .reset_index()
        .rename(columns={"month": "time"})
    )
    monthly["expected_days"] = monthly["time"].dt.days_in_month.astype(int)
    monthly["missing_component_rows"] = (
        monthly["listed_component_rows"] - monthly["valid_component_rows"]
    )
    monthly["component_value_coverage"] = (
        monthly["valid_component_rows"] / monthly["listed_component_rows"]
    )
    monthly["is_complete_month"] = monthly["observed_days"].eq(
        monthly["expected_days"]
    )
    monthly["inflow_mwh"] = monthly["observed_inflow_mwh"].where(
        monthly["is_complete_month"]
    )
    monthly["aggregation_level"] = aggregation_level
    monthly["split"] = assign_temporal_split(monthly["time"])
    return monthly.sort_values(["time", "aggregation_id"]).reset_index(drop=True)


def aggregate_ree_monthly_ena(ena: pd.DataFrame) -> pd.DataFrame:
    """Explicit REE-mode compatibility wrapper used by synthetic fixtures."""

    result = aggregate_monthly_ena(ena, aggregation_level="ree")
    return result.rename(columns={"aggregation_id": "ree_name"})


def aggregate_official_subsystem_monthly_ena(
    subsystem_ena: pd.DataFrame,
) -> pd.DataFrame:
    """Integrate the official N/NE/S/SE daily aggregate without imputation."""

    required = ["time", "unit_id", "ena_gross_mw_mean"]
    _require(subsystem_ena, required, "official subsystem ENA")
    frame = subsystem_ena.loc[:, required].copy()
    frame["time"] = _dates(frame["time"], "subsystem_ENA.time")
    frame["aggregation_id"] = _strings(frame["unit_id"])
    frame["ena_gross_mw_mean"] = _numbers(
        frame["ena_gross_mw_mean"], "subsystem_ENA.ena_gross_mw_mean"
    )
    frame = frame.loc[_within_history(frame["time"])].copy()
    expected_subsystems = {"N", "NE", "S", "SE"}
    unexpected = set(frame["aggregation_id"].dropna()).difference(
        expected_subsystems
    )
    if frame.empty or frame["aggregation_id"].isna().any() or unexpected:
        raise ValueError(
            "official subsystem ENA must use only N/NE/S/SE; "
            f"unexpected={sorted(unexpected)}"
        )
    if frame["ena_gross_mw_mean"].isna().any() or (
        frame["ena_gross_mw_mean"] < 0.0
    ).any():
        raise ValueError("official subsystem gross ENA must be finite and nonnegative")
    if frame.duplicated(["aggregation_id", "time"]).any():
        raise ValueError("official subsystem ENA has duplicate subsystem/day rows")
    daily_sets = frame.groupby("time", sort=True)["aggregation_id"].agg(
        lambda values: frozenset(values)
    )
    incomplete_days = daily_sets.ne(frozenset(expected_subsystems))
    if incomplete_days.any():
        examples = [
            str(value.date()) for value in daily_sets.index[incomplete_days][:3]
        ]
        raise ValueError(
            "official subsystem ENA date does not contain exactly N/NE/S/SE: "
            f"{examples}"
        )

    frame["daily_ena_mwh"] = frame["ena_gross_mw_mean"] * 24.0
    frame["month"] = _month_start(frame["time"])
    monthly = (
        frame.groupby(["aggregation_id", "month"], sort=True)
        .agg(
            observed_inflow_mwh=("daily_ena_mwh", "sum"),
            observed_days=("ena_gross_mw_mean", "count"),
        )
        .reset_index()
        .rename(columns={"month": "time"})
    )
    monthly["expected_days"] = monthly["time"].dt.days_in_month.astype(int)
    monthly["minimum_daily_units"] = 1
    monthly["maximum_daily_units"] = 1
    monthly["listed_component_rows"] = monthly["observed_days"]
    monthly["valid_component_rows"] = monthly["observed_days"]
    monthly["missing_component_rows"] = 0
    monthly["component_value_coverage"] = 1.0
    monthly["all_values_present"] = True
    monthly["is_complete_month"] = monthly["observed_days"].eq(
        monthly["expected_days"]
    )
    monthly["inflow_mwh"] = monthly["observed_inflow_mwh"].where(
        monthly["is_complete_month"]
    )
    monthly["aggregation_level"] = "subsystem"
    monthly["inflow_source"] = "ons_official_subsystem_ena_daily"
    monthly["split"] = assign_temporal_split(monthly["time"])
    return monthly.sort_values(["time", "aggregation_id"]).reset_index(drop=True)


def aggregate_monthly_load(load: pd.DataFrame) -> pd.DataFrame:
    """Convert subsystem monthly-average MW to total system MWh."""

    required = ["time", "unit_id", "load_mw_mean"]
    _require(load, required, "load")
    frame = load.loc[:, required].copy()
    frame["time"] = _month_start(_dates(frame["time"], "load.time"))
    frame["unit_id"] = _strings(frame["unit_id"])
    frame["load_mw_mean"] = _numbers(frame["load_mw_mean"], "load.load_mw_mean")
    frame = frame.loc[_within_history(frame["time"])].copy()
    if frame["unit_id"].isna().any():
        raise ValueError("load contains an empty subsystem identifier")
    if (frame["load_mw_mean"].dropna() < 0.0).any():
        raise ValueError("load cannot be negative")
    if frame.duplicated(["unit_id", "time"]).any():
        raise ValueError("load contains duplicate subsystem/month rows")
    frame["load_mwh"] = (
        frame["load_mw_mean"] * frame["time"].dt.days_in_month * 24.0
    )
    monthly = (
        frame.groupby("time", sort=True)
        .agg(
            load_mwh=("load_mwh", lambda values: values.sum(min_count=1)),
            subsystem_count=("unit_id", "nunique"),
            valid_subsystem_count=("load_mwh", "count"),
        )
        .reset_index()
    )
    all_months = pd.DataFrame(
        {"time": pd.date_range(HISTORY_START, HISTORY_END, freq="MS")}
    )
    monthly = all_months.merge(monthly, how="left", on="time", validate="one_to_one")
    monthly["split"] = assign_temporal_split(monthly["time"])
    training_rows = frame.loc[_within_train(frame["time"])]
    expected_subsystems = int(training_rows["unit_id"].nunique())
    monthly["expected_subsystem_count"] = expected_subsystems
    monthly["is_complete_month"] = (
        monthly["load_mwh"].notna()
        & monthly["subsystem_count"].eq(expected_subsystems)
        & monthly["valid_subsystem_count"].eq(expected_subsystems)
    )
    return monthly


def _training_ree_mapping(ena: pd.DataFrame) -> pd.DataFrame:
    frame = ena.loc[:, ["time", "unit_id", "ree_name"]].copy()
    frame["time"] = _dates(frame["time"], "ENA.time")
    frame["unit_id"] = _strings(frame["unit_id"])
    frame["ree_name"] = _strings(frame["ree_name"])
    frame = frame.loc[_within_train(frame["time"])].dropna()
    summary = (
        frame.groupby("unit_id", sort=True)
        .agg(
            mapped_ree_count=("ree_name", "nunique"),
            mapped_ree=("ree_name", "first"),
            first_train_ena_date=("time", "min"),
            last_train_ena_date=("time", "max"),
        )
        .reset_index()
        .rename(columns={"unit_id": "planning_id"})
    )
    summary.loc[summary["mapped_ree_count"].ne(1), "mapped_ree"] = pd.NA
    return summary


def static_energy_capacity_audit(
    reservoir_static: pd.DataFrame,
    ena: pd.DataFrame,
    *,
    aggregation_level: str = "subsystem",
) -> pd.DataFrame:
    """Compute training-identifiable equivalent energy capacity per reservoir."""

    if aggregation_level not in {"subsystem", "ree"}:
        raise ValueError("aggregation_level must be 'subsystem' or 'ree'")
    required = ["unit_id", "storage_capacity_hm3", "productivity_65pct_mw_per_m3s"]
    if aggregation_level == "subsystem":
        required.append("subsystem_id")
    else:
        required.append("planning_id")
    _require(reservoir_static, required, "reservoir_static")
    frame = reservoir_static.copy()
    frame["reservoir_id"] = _strings(frame["unit_id"])
    frame["planning_id"] = (
        _strings(frame["planning_id"])
        if "planning_id" in frame
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    frame["storage_capacity_hm3"] = _numbers(
        frame["storage_capacity_hm3"], "static.storage_capacity_hm3"
    )
    frame["productivity_65pct_mw_per_m3s"] = _numbers(
        frame["productivity_65pct_mw_per_m3s"],
        "static.productivity_65pct_mw_per_m3s",
    )
    if "commissioned_on" in frame:
        frame["commissioned_on"] = _dates(
            frame["commissioned_on"], "static.commissioned_on"
        )
    else:
        frame["commissioned_on"] = pd.NaT
    if aggregation_level == "subsystem":
        frame["static_assignment"] = _strings(frame["subsystem_id"])
        frame["aggregation_id"] = frame["static_assignment"]
        frame["assignment_source"] = "static.subsystem_id"
        frame["mapped_ree_count"] = np.nan
        frame["first_train_ena_date"] = pd.NaT
        frame["last_train_ena_date"] = pd.NaT
        if "planning_id" in reservoir_static and "unit_id" in ena:
            history = ena.loc[:, ["time", "unit_id"]].copy()
            history["time"] = _dates(history["time"], "ENA.time")
            history["planning_id"] = _strings(history["unit_id"])
            history = history.loc[_within_train(history["time"])]
            history = (
                history.groupby("planning_id", sort=True)
                .agg(
                    first_train_ena_date=("time", "min"),
                    last_train_ena_date=("time", "max"),
                )
                .reset_index()
            )
            frame = frame.drop(
                columns=["first_train_ena_date", "last_train_ena_date"]
            ).merge(history, how="left", on="planning_id", validate="many_to_one")
        frame["mapped_ree"] = pd.NA
        frame["ree_mapping_mismatch"] = False
    else:
        frame["static_assignment"] = (
            _strings(frame["ree_name"])
            if "ree_name" in frame
            else pd.Series(pd.NA, index=frame.index, dtype="string")
        )
        mapping = _training_ree_mapping(ena)
        frame = frame.merge(
            mapping, how="left", on="planning_id", validate="many_to_one"
        )
        frame["aggregation_id"] = frame["mapped_ree"]
        frame["assignment_source"] = "training_ena.ree_name"
        frame["ree_mapping_mismatch"] = (
            frame["static_assignment"].notna()
            & frame["mapped_ree"].notna()
            & frame["static_assignment"].ne(frame["mapped_ree"])
        )
    frame["energy_capacity_mwh"] = (
        frame["storage_capacity_hm3"]
        * frame["productivity_65pct_mw_per_m3s"]
        * MWH_PER_HM3_PER_MW_PER_M3S
    )
    frame["aggregation_level"] = aggregation_level

    reasons: list[str] = []
    eligible: list[bool] = []
    for row in frame.itertuples(index=False):
        row_reasons = []
        if aggregation_level == "ree":
            if pd.isna(row.planning_id):
                row_reasons.append("planning_id_missing")
            if pd.isna(row.aggregation_id):
                if getattr(row, "mapped_ree_count", np.nan) > 1:
                    row_reasons.append("ambiguous_train_ree_mapping")
                else:
                    row_reasons.append("no_train_ena_mapping")
        elif pd.isna(row.aggregation_id):
            row_reasons.append("static_subsystem_missing")
        if pd.notna(row.commissioned_on) and row.commissioned_on > TRAIN_END:
            row_reasons.append("commissioned_after_train")
        if not np.isfinite(row.storage_capacity_hm3) or row.storage_capacity_hm3 <= 0:
            row_reasons.append("storage_capacity_not_identifiable")
        productivity = row.productivity_65pct_mw_per_m3s
        if not np.isfinite(productivity) or productivity <= 0:
            row_reasons.append("productivity_not_identifiable")
        reasons.append(";".join(row_reasons))
        eligible.append(not row_reasons)
    frame["eligible_capacity"] = eligible
    frame["exclusion_reason"] = reasons
    frame["capacity_formula_factor"] = MWH_PER_HM3_PER_MW_PER_M3S
    columns = [
        "reservoir_id",
        "planning_id",
        "aggregation_level",
        "aggregation_id",
        "assignment_source",
        "static_assignment",
        "mapped_ree",
        "ree_mapping_mismatch",
        "commissioned_on",
        "first_train_ena_date",
        "last_train_ena_date",
        "storage_capacity_hm3",
        "productivity_65pct_mw_per_m3s",
        "capacity_formula_factor",
        "energy_capacity_mwh",
        "eligible_capacity",
        "exclusion_reason",
    ]
    for optional in ("unit_name", "plant_name"):
        if optional in frame:
            columns.insert(2, optional)
    return frame.loc[:, columns].sort_values(
        ["aggregation_id", "planning_id", "reservoir_id"], na_position="last"
    ).reset_index(drop=True)


def _canonical_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", ascii_text.lower())
    return " ".join(tokens) or None


def _canonical_ceg(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    token = re.sub(r"\s+", "", str(value)).upper()
    return token or None


def _ceg_plant_code(value: object) -> str | None:
    canonical = _canonical_ceg(value)
    if canonical is None:
        return None
    return re.sub(r"\.\d+$", "", canonical)


def _ceg_registration(value: object) -> str | None:
    """Return the plant registration, ignoring fuel and generating-unit codes."""

    plant_code = _ceg_plant_code(value)
    if plant_code is None:
        return None
    match = re.search(r"(?:^|\.)([A-Z]{2})\.([0-9]{6}-[0-9])$", plant_code)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return plant_code


def _thermal_alias_name(value: object) -> str | None:
    """Normalize documented legacy ONS abbreviations without fuzzy guessing."""

    canonical = _canonical_name(value)
    if canonical is None:
        return None
    if re.fullmatch(r"p medici(?: [a-z0-9]+)?", canonical):
        return "presidente medici"
    if re.fullmatch(r"st cruz(?: [0-9]+)?", canonical):
        return "santa cruz"
    if re.fullmatch(r"maua b[0-9]+[a-z]*", canonical):
        return "maua"
    if re.fullmatch(r"maua bloco [0-9]+", canonical):
        return "maua"
    return canonical


def _planning_ids(values: pd.Series) -> pd.Series:
    identifiers = _strings(values)
    numeric = pd.to_numeric(identifiers, errors="coerce")
    integral = numeric.notna() & numeric.mod(1.0).eq(0.0)
    result = identifiers.copy()
    result.loc[integral] = numeric.loc[integral].astype("int64").astype("string")
    return result


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _training_cvu_plants(cvu: pd.DataFrame) -> pd.DataFrame:
    required = ["time", "unit_id", "cvu_brl_per_mwh"]
    _require(cvu, required, "CVU")
    frame = cvu.copy()
    frame["time"] = _dates(frame["time"], "CVU.time")
    if "time_end" in frame:
        frame["time_end"] = _dates(frame["time_end"], "CVU.time_end")
    else:
        frame["time_end"] = frame["time"]
    frame["unit_id"] = _planning_ids(frame["unit_id"])
    frame["cvu_brl_per_mwh"] = _numbers(
        frame["cvu_brl_per_mwh"], "CVU.cvu_brl_per_mwh"
    )
    if "unit_name" in frame:
        frame["unit_name"] = _strings(frame["unit_name"])
    else:
        frame["unit_name"] = pd.NA
    if "subsystem_id" in frame:
        frame["subsystem_id"] = _strings(frame["subsystem_id"])
    else:
        frame["subsystem_id"] = pd.NA
    if "ceg" in frame:
        frame["ceg"] = _strings(frame["ceg"])
    else:
        frame["ceg"] = pd.NA
    frame = frame.loc[
        _within_train(frame["time"]) & frame["time_end"].le(TRAIN_END)
    ].copy()
    frame = frame.dropna(subset=["cvu_brl_per_mwh"])
    if frame.empty:
        raise ValueError("CVU has no observations wholly inside the training period")
    if frame["unit_id"].isna().any():
        raise ValueError("training CVU contains an empty plant identifier")
    if (frame["cvu_brl_per_mwh"].dropna() < 0.0).any():
        raise ValueError("CVU cannot be negative")
    terminal_catalog_date = frame["time"].max()
    frame["canonical_name"] = frame["unit_name"].map(_canonical_name)
    frame["alias_name"] = frame["unit_name"].map(_thermal_alias_name)
    frame["canonical_ceg"] = frame["ceg"].map(_canonical_ceg)
    frame["ceg_registration"] = frame["ceg"].map(_ceg_registration)
    plants = (
        frame.groupby("unit_id", sort=True, dropna=False)
        .agg(
            unit_name=("unit_name", "first"),
            subsystem_id=("subsystem_id", "first"),
            canonical_name=("canonical_name", "first"),
            alias_name=("alias_name", "first"),
            canonical_ceg=("canonical_ceg", "first"),
            ceg_registration=("ceg_registration", "first"),
            median_train_cvu_brl_per_mwh=("cvu_brl_per_mwh", "median"),
            train_cvu_observations=("cvu_brl_per_mwh", "count"),
            first_train_cvu_date=("time", "min"),
            last_train_cvu_date=("time", "max"),
        )
        .reset_index()
    )
    plants["terminal_catalog_date"] = terminal_catalog_date
    plants["population_eligible"] = plants["last_train_cvu_date"].eq(
        terminal_catalog_date
    )
    plants["population_rule"] = "present_on_latest_training_cvu_date"
    plants["population_exclusion_reason"] = np.where(
        plants["population_eligible"],
        "",
        "absent_from_terminal_training_cvu_catalog",
    )
    return plants


def _normalized_generation_capacity(generation_capacity: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(generation_capacity, pd.DataFrame):
        raise TypeError("generation_capacity must be a pandas DataFrame")
    capacity_col = _first_column(
        generation_capacity,
        ("capacity_mw", "effective_capacity_mw", "val_potenciaefetiva"),
    )
    type_col = _first_column(
        generation_capacity, ("plant_type", "source_type", "nom_tipousina")
    )
    subsystem_col = _first_column(
        generation_capacity, ("subsystem_id", "id_subsistema")
    )
    name_col = _first_column(
        generation_capacity, ("plant_name", "unit_name", "nom_usina")
    )
    ceg_col = _first_column(generation_capacity, ("ceg",))
    if capacity_col is None or subsystem_col is None:
        raise KeyError(
            "generation_capacity needs capacity_mw/effective_capacity_mw and "
            "subsystem_id"
        )
    if type_col is None and "is_thermal" not in generation_capacity:
        raise KeyError("generation_capacity needs plant_type or is_thermal")
    if name_col is None and ceg_col is None:
        raise KeyError("generation_capacity needs a plant name or CEG")
    if "capacity_reconstruction_as_of" in generation_capacity:
        reconstruction_dates = _dates(
            generation_capacity["capacity_reconstruction_as_of"],
            "capacity.capacity_reconstruction_as_of",
        ).dropna()
        if reconstruction_dates.gt(TRAIN_END).any():
            raise ValueError(
                "generation-capacity reconstruction must not use an as-of date "
                "after the training period"
            )

    frame = pd.DataFrame(index=generation_capacity.index)
    frame["capacity_mw"] = _numbers(
        generation_capacity[capacity_col], f"capacity.{capacity_col}"
    )
    frame["subsystem_id"] = _strings(generation_capacity[subsystem_col])
    frame["plant_name"] = (
        _strings(generation_capacity[name_col]) if name_col else pd.NA
    )
    frame["ceg"] = _strings(generation_capacity[ceg_col]) if ceg_col else pd.NA
    if type_col:
        plant_type = generation_capacity[type_col].map(_canonical_name)
        frame["is_thermal"] = plant_type.fillna("").str.contains("termica|thermal")
        frame["plant_type"] = generation_capacity[type_col]
    else:
        frame["is_thermal"] = generation_capacity["is_thermal"].astype(bool)
        frame["plant_type"] = "thermal"

    commissioning_columns = [
        column
        for column in (
            "operation_on",
            "commissioning_on",
            "commissioned_on",
            "operation_start",
            "dat_entradaoperacao",
        )
        if column in generation_capacity
    ]
    deactivated_col = _first_column(
        generation_capacity, ("deactivated_on", "dat_desativacao")
    )
    commissioned = pd.Series(pd.NaT, index=generation_capacity.index)
    for column in commissioning_columns:
        parsed = _dates(generation_capacity[column], f"capacity.{column}")
        commissioned = commissioned.combine_first(parsed)
    frame["commissioned_on"] = commissioned
    frame["deactivated_on"] = (
        _dates(generation_capacity[deactivated_col], f"capacity.{deactivated_col}")
        if deactivated_col
        else pd.NaT
    )
    frame["canonical_name"] = frame["plant_name"].map(_canonical_name)
    frame["alias_name"] = frame["plant_name"].map(_thermal_alias_name)
    frame["canonical_ceg"] = frame["ceg"].map(_canonical_ceg)
    frame["ceg_plant_code"] = frame["ceg"].map(_ceg_plant_code)
    frame["ceg_registration"] = frame["ceg"].map(_ceg_registration)
    frame = frame.loc[
        frame["is_thermal"]
        & frame["capacity_mw"].gt(0.0)
        & (
            frame["commissioned_on"].isna()
            | frame["commissioned_on"].le(TRAIN_END)
        )
        & (
            frame["deactivated_on"].isna()
            | frame["deactivated_on"].ge(TRAIN_START)
        )
    ].copy()
    return frame.reset_index(drop=True)


def _normalized_thermal_crosswalk(
    thermal_crosswalk: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "planning_id",
        "crosswalk_ceg",
        "crosswalk_ceg_registration",
        "crosswalk_plant_name",
        "crosswalk_subsystem_id",
        "crosswalk_first_observed_on",
        "crosswalk_last_observed_on",
        "crosswalk_source_years",
    ]
    if thermal_crosswalk is None:
        return pd.DataFrame(columns=columns)
    if not isinstance(thermal_crosswalk, pd.DataFrame):
        raise TypeError("thermal_crosswalk must be a pandas DataFrame")
    planning_col = _first_column(
        thermal_crosswalk,
        ("planning_id", "unit_id", "cod_usinaplanejamento"),
    )
    ceg_col = _first_column(thermal_crosswalk, ("ceg", "crosswalk_ceg"))
    if planning_col is None or ceg_col is None:
        raise KeyError("thermal_crosswalk needs planning_id and ceg")

    frame = pd.DataFrame(index=thermal_crosswalk.index)
    frame["planning_id"] = _planning_ids(thermal_crosswalk[planning_col])
    frame["crosswalk_ceg"] = thermal_crosswalk[ceg_col].map(_canonical_ceg)
    frame["crosswalk_ceg_registration"] = thermal_crosswalk[ceg_col].map(
        _ceg_registration
    )
    name_col = _first_column(
        thermal_crosswalk, ("plant_name", "unit_name", "nom_usina")
    )
    subsystem_col = _first_column(
        thermal_crosswalk, ("subsystem_id", "id_subsistema")
    )
    frame["crosswalk_plant_name"] = (
        _strings(thermal_crosswalk[name_col]) if name_col else pd.NA
    )
    frame["crosswalk_subsystem_id"] = (
        _strings(thermal_crosswalk[subsystem_col]) if subsystem_col else pd.NA
    )
    first_col = _first_column(
        thermal_crosswalk, ("first_observed_on", "time", "din_instante")
    )
    last_col = _first_column(
        thermal_crosswalk, ("last_observed_on", "time", "din_instante")
    )
    frame["crosswalk_first_observed_on"] = (
        _dates(thermal_crosswalk[first_col], f"crosswalk.{first_col}")
        if first_col
        else pd.NaT
    )
    frame["crosswalk_last_observed_on"] = (
        _dates(thermal_crosswalk[last_col], f"crosswalk.{last_col}")
        if last_col
        else pd.NaT
    )
    source_years_col = _first_column(thermal_crosswalk, ("source_years",))
    frame["crosswalk_source_years"] = (
        _strings(thermal_crosswalk[source_years_col])
        if source_years_col
        else pd.NA
    )
    frame = frame.loc[
        frame["crosswalk_first_observed_on"].isna()
        | frame["crosswalk_first_observed_on"].le(TRAIN_END)
    ].copy()
    if frame.empty:
        raise ValueError("thermal_crosswalk has no identity observed by 2014-12-31")
    if frame["planning_id"].isna().any() or frame[
        "crosswalk_ceg_registration"
    ].isna().any():
        raise ValueError("thermal_crosswalk contains an empty planning_id or CEG")
    if frame["crosswalk_last_observed_on"].gt(TRAIN_END).any():
        raise ValueError(
            "thermal_crosswalk identity must be reconstructed only from "
            "observations through 2014-12-31"
        )

    summary = (
        frame.groupby("planning_id", sort=True)
        .agg(
            crosswalk_ceg=("crosswalk_ceg", "first"),
            crosswalk_ceg_count=("crosswalk_ceg_registration", "nunique"),
            crosswalk_ceg_registration=(
                "crosswalk_ceg_registration",
                "first",
            ),
            crosswalk_plant_name=("crosswalk_plant_name", "first"),
            crosswalk_subsystem_id=("crosswalk_subsystem_id", "first"),
            crosswalk_first_observed_on=("crosswalk_first_observed_on", "min"),
            crosswalk_last_observed_on=("crosswalk_last_observed_on", "max"),
            crosswalk_source_years=("crosswalk_source_years", "first"),
        )
        .reset_index()
    )
    ambiguous = summary["crosswalk_ceg_count"].ne(1)
    if ambiguous.any():
        ids = summary.loc[ambiguous, "planning_id"].head(5).tolist()
        raise ValueError(f"ambiguous planning_id-to-CEG crosswalk: {ids}")
    return summary.loc[:, columns]


def _capacity_sites(capacity: pd.DataFrame) -> pd.DataFrame:
    frame = capacity.copy()
    has_registration = frame["ceg_registration"].notna()
    frame["capacity_match_key"] = np.where(
        has_registration,
        "ceg_registration:" + frame["ceg_registration"].fillna(""),
        "name:"
        + frame["subsystem_id"].fillna("")
        + ":"
        + frame["canonical_name"].fillna(""),
    )
    sites = (
        frame.groupby("capacity_match_key", sort=True)
        .agg(
            rated_capacity_mw=("capacity_mw", "sum"),
            capacity_unit_count=("capacity_mw", "size"),
            capacity_subsystem_id=("subsystem_id", "first"),
            capacity_subsystem_count=("subsystem_id", "nunique"),
            capacity_plant_name=("plant_name", "first"),
            capacity_plant_names=(
                "plant_name",
                lambda values: tuple(sorted(set(values.dropna().astype(str)))),
            ),
            capacity_canonical_names=(
                "canonical_name",
                lambda values: tuple(sorted(set(values.dropna().astype(str)))),
            ),
            capacity_alias_names=(
                "alias_name",
                lambda values: tuple(sorted(set(values.dropna().astype(str)))),
            ),
            capacity_ceg_registration=("ceg_registration", "first"),
            capacity_ceg_registration_count=("ceg_registration", "nunique"),
        )
        .reset_index()
    )
    invalid = sites["capacity_subsystem_count"].ne(1) | sites[
        "capacity_ceg_registration_count"
    ].gt(1)
    if invalid.any():
        keys = sites.loc[invalid, "capacity_match_key"].head(5).tolist()
        raise ValueError(f"ambiguous installed-capacity plant identities: {keys}")
    return sites


def _unique_site_index(
    sites: pd.DataFrame, names_column: str
) -> dict[tuple[str, str], str]:
    candidates: dict[tuple[str, str], set[str]] = {}
    for row in sites.itertuples(index=False):
        subsystem = row.capacity_subsystem_id
        names = getattr(row, names_column)
        if pd.isna(subsystem):
            continue
        for name in names:
            candidates.setdefault((name, str(subsystem)), set()).add(
                row.capacity_match_key
            )
    return {
        name: next(iter(keys))
        for name, keys in candidates.items()
        if len(keys) == 1
    }


def _match_cvu_to_capacity(
    plants: pd.DataFrame,
    generation_capacity: pd.DataFrame,
    thermal_crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    capacity = _normalized_generation_capacity(generation_capacity)
    sites = _capacity_sites(capacity)
    site_by_key = sites.set_index("capacity_match_key")
    registration_index = {
        str(row.capacity_ceg_registration): row.capacity_match_key
        for row in sites.itertuples(index=False)
        if pd.notna(row.capacity_ceg_registration)
    }
    exact_name_index = _unique_site_index(sites, "capacity_canonical_names")
    alias_name_index = _unique_site_index(sites, "capacity_alias_names")
    crosswalk = _normalized_thermal_crosswalk(thermal_crosswalk)
    audit = plants.merge(
        crosswalk, how="left", left_on="unit_id", right_on="planning_id"
    ).drop(columns="planning_id")

    methods: list[str] = []
    keys: list[str | None] = []
    details: list[str | None] = []
    fuzzy_scores: list[float] = []
    runner_up_scores: list[float] = []
    for row in audit.itertuples(index=False):
        key: str | None = None
        method = "unmatched"
        detail: str | None = None
        best_score = np.nan
        runner_up = np.nan
        crosswalk_registration = row.crosswalk_ceg_registration
        if (
            pd.notna(crosswalk_registration)
            and str(crosswalk_registration) in registration_index
        ):
            key = registration_index[str(crosswalk_registration)]
            method = "planning_id_to_ceg_registration"
            detail = str(row.crosswalk_ceg)
        elif (
            pd.notna(row.ceg_registration)
            and str(row.ceg_registration) in registration_index
        ):
            key = registration_index[str(row.ceg_registration)]
            method = "cvu_ceg_registration"
            detail = str(row.canonical_ceg)
        else:
            name_key = (row.canonical_name, row.subsystem_id)
            if (
                row.canonical_name is not None
                and pd.notna(row.subsystem_id)
                and name_key in exact_name_index
            ):
                key = exact_name_index[name_key]
                method = "exact_name_subsystem"
                detail = row.canonical_name
            else:
                alias_key = (row.alias_name, row.subsystem_id)
                if (
                    row.alias_name is not None
                    and pd.notna(row.subsystem_id)
                    and alias_key in alias_name_index
                ):
                    key = alias_name_index[alias_key]
                    method = "unique_alias_subsystem"
                    detail = row.alias_name

        if key is None and row.canonical_name is not None and pd.notna(
            row.subsystem_id
        ):
            candidates: list[tuple[float, str, str]] = []
            in_subsystem = sites.loc[
                sites["capacity_subsystem_id"].eq(row.subsystem_id)
            ]
            for site in in_subsystem.itertuples(index=False):
                scored_names = [
                    (
                        SequenceMatcher(None, row.canonical_name, candidate).ratio(),
                        candidate,
                    )
                    for candidate in site.capacity_canonical_names
                ]
                if scored_names:
                    score, candidate = max(scored_names)
                    candidates.append((score, site.capacity_match_key, candidate))
            candidates.sort(reverse=True)
            if candidates:
                best_score, best_key, best_name = candidates[0]
                runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
                if (
                    best_score >= THERMAL_FUZZY_SCORE_MIN
                    and best_score - runner_up >= THERMAL_FUZZY_MARGIN_MIN
                ):
                    key = best_key
                    method = "unique_high_confidence_fuzzy_name_subsystem"
                    detail = best_name

        methods.append(method)
        keys.append(key)
        details.append(detail)
        fuzzy_scores.append(best_score)
        runner_up_scores.append(runner_up)

    audit["capacity_match_method"] = methods
    audit["capacity_match_key"] = keys
    audit["capacity_match_detail"] = details
    audit["fuzzy_match_score"] = fuzzy_scores
    audit["fuzzy_runner_up_score"] = runner_up_scores
    audit["fuzzy_score_margin"] = (
        audit["fuzzy_match_score"] - audit["fuzzy_runner_up_score"]
    )
    audit["rated_capacity_mw"] = audit["capacity_match_key"].map(
        site_by_key["rated_capacity_mw"]
    )
    audit["matched_capacity_plant_name"] = audit["capacity_match_key"].map(
        site_by_key["capacity_plant_name"]
    )
    audit["matched_capacity_unit_count"] = audit["capacity_match_key"].map(
        site_by_key["capacity_unit_count"]
    )
    audit["capacity_matched"] = audit["rated_capacity_mw"].gt(0.0)
    return audit


def _collapse_merit_blocks(
    matched: pd.DataFrame, max_thermal_blocks: int | None
) -> pd.DataFrame:
    plants = (
        matched.loc[matched["capacity_matched"] & matched["population_eligible"]]
        .groupby("capacity_match_key", sort=True)
        .agg(
            marginal_cost_brl_per_mwh=(
                "median_train_cvu_brl_per_mwh",
                "median",
            ),
            rated_capacity_mw=("rated_capacity_mw", "first"),
            mapped_cvu_ids=("unit_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["marginal_cost_brl_per_mwh", "capacity_match_key"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if plants.empty:
        return plants
    if max_thermal_blocks is None or len(plants) <= max_thermal_blocks:
        plants["block"] = np.arange(len(plants), dtype=int)
        return plants
    if max_thermal_blocks < 1:
        raise ValueError("max_thermal_blocks must be positive or None")

    pieces = []
    partitions = np.array_split(np.arange(len(plants)), max_thermal_blocks)
    for block, indices in enumerate(partitions):
        group = plants.iloc[indices]
        capacity = float(group["rated_capacity_mw"].sum())
        cost = float(
            np.average(
                group["marginal_cost_brl_per_mwh"],
                weights=group["rated_capacity_mw"],
            )
        )
        pieces.append(
            {
                "capacity_match_key": ";".join(group["capacity_match_key"]),
                "marginal_cost_brl_per_mwh": cost,
                "rated_capacity_mw": capacity,
                "mapped_cvu_ids": int(group["mapped_cvu_ids"].sum()),
                "block": block,
            }
        )
    return pd.DataFrame(pieces)


def build_cvu_merit_order(
    cvu: pd.DataFrame,
    *,
    generation_capacity: pd.DataFrame | None,
    thermal_crosswalk: pd.DataFrame | None = None,
    training_peak_load_mwh: float,
    mean_training_month_hours: float,
    thermal_capacity_proxy: bool = False,
    max_thermal_blocks: int | None = None,
    minimum_capacity_match_fraction: float = 0.80,
) -> tuple[tuple[float, ...], tuple[float, ...], pd.DataFrame, pd.DataFrame]:
    """Create MWh blocks and return plant/block audits.

    The proxy path is intentionally impossible to enter by omission: callers
    must set ``thermal_capacity_proxy=True`` explicitly.
    """

    if not 0.0 <= minimum_capacity_match_fraction <= 1.0:
        raise ValueError("minimum_capacity_match_fraction must lie in [0, 1]")
    if max_thermal_blocks is not None and (
        isinstance(max_thermal_blocks, bool)
        or not isinstance(max_thermal_blocks, (int, np.integer))
        or max_thermal_blocks < 1
    ):
        raise ValueError("max_thermal_blocks must be a positive integer or None")
    if not np.isfinite(training_peak_load_mwh) or training_peak_load_mwh <= 0.0:
        raise ValueError("training_peak_load_mwh must be positive and finite")
    if not np.isfinite(mean_training_month_hours) or mean_training_month_hours <= 0:
        raise ValueError("mean_training_month_hours must be positive and finite")
    plants = _training_cvu_plants(cvu)
    eligible_count = int(plants["population_eligible"].sum())
    if eligible_count < 1:
        raise ValueError("CVU terminal training catalog contains no plants")
    capacity_source = "ons_effective_capacity_snapshot"
    if generation_capacity is None:
        if not thermal_capacity_proxy:
            raise ValueError(
                "formal ONS benchmark requires generation_capacity; set "
                "thermal_capacity_proxy=True only for a sensitivity run or fixture"
            )
        plants["capacity_match_method"] = np.where(
            plants["population_eligible"],
            "explicit_training_load_proxy",
            "excluded_historical_population",
        )
        plants["capacity_match_key"] = np.where(
            plants["population_eligible"],
            "proxy:" + plants["unit_id"].astype(str),
            None,
        )
        plants["rated_capacity_mw"] = np.nan
        plants["capacity_matched"] = False
        plants["capacity_match_fraction"] = 0.0
        plants["required_capacity_match_fraction"] = (
            minimum_capacity_match_fraction
        )
        eligible_plants = plants.loc[plants["population_eligible"]].sort_values(
            ["median_train_cvu_brl_per_mwh", "unit_id"], kind="stable"
        ).reset_index(drop=True)
        n_blocks = len(eligible_plants) if max_thermal_blocks is None else min(
            len(eligible_plants), max_thermal_blocks
        )
        groups = np.array_split(np.arange(len(eligible_plants)), n_blocks)
        rows = []
        for block, indices in enumerate(groups):
            group = eligible_plants.iloc[indices]
            rows.append(
                {
                    "capacity_match_key": ";".join(group["capacity_match_key"]),
                    "marginal_cost_brl_per_mwh": float(
                        group["median_train_cvu_brl_per_mwh"].median()
                    ),
                    "rated_capacity_mw": np.nan,
                    "mapped_cvu_ids": len(group),
                    "block": block,
                    "capacity_mwh": (
                        training_peak_load_mwh * len(group) / len(eligible_plants)
                    ),
                }
            )
        blocks = pd.DataFrame(rows)
        capacity_source = "explicit_training_peak_proxy_by_plant_count"
    else:
        plants = _match_cvu_to_capacity(
            plants, generation_capacity, thermal_crosswalk=thermal_crosswalk
        )
        eligible = plants["population_eligible"]
        matched_count = int(plants.loc[eligible, "capacity_matched"].sum())
        match_fraction = float(matched_count / eligible_count)
        plants["capacity_match_fraction"] = match_fraction
        plants["required_capacity_match_fraction"] = (
            minimum_capacity_match_fraction
        )
        if match_fraction < minimum_capacity_match_fraction:
            if not thermal_capacity_proxy:
                raise ValueError(
                    "terminal training CVU-to-capacity match fraction "
                    f"{match_fraction:.3f} is below required "
                    f"{minimum_capacity_match_fraction:.3f} "
                    f"({matched_count}/{eligible_count} plants)"
                )
            return build_cvu_merit_order(
                cvu,
                generation_capacity=None,
                thermal_crosswalk=thermal_crosswalk,
                training_peak_load_mwh=training_peak_load_mwh,
                mean_training_month_hours=mean_training_month_hours,
                thermal_capacity_proxy=True,
                max_thermal_blocks=max_thermal_blocks,
                minimum_capacity_match_fraction=minimum_capacity_match_fraction,
            )
        blocks = _collapse_merit_blocks(plants, max_thermal_blocks)
        if blocks.empty:
            if not thermal_capacity_proxy:
                raise ValueError(
                    "no training CVU plant matched an eligible thermal capacity row"
                )
            return build_cvu_merit_order(
                cvu,
                generation_capacity=None,
                thermal_crosswalk=thermal_crosswalk,
                training_peak_load_mwh=training_peak_load_mwh,
                mean_training_month_hours=mean_training_month_hours,
                thermal_capacity_proxy=True,
                max_thermal_blocks=max_thermal_blocks,
                minimum_capacity_match_fraction=minimum_capacity_match_fraction,
            )
        blocks["capacity_mwh"] = (
            blocks["rated_capacity_mw"] * mean_training_month_hours
        )

    blocks["capacity_source"] = capacity_source
    blocks["reference_month_hours"] = mean_training_month_hours
    if (
        generation_capacity is not None
        and "capacity_reconstruction_as_of" in generation_capacity
        and capacity_source == "ons_effective_capacity_snapshot"
    ):
        as_of = _dates(
            generation_capacity["capacity_reconstruction_as_of"],
            "capacity.capacity_reconstruction_as_of",
        ).max()
        blocks["capacity_reconstruction_as_of"] = as_of
    else:
        blocks["capacity_reconstruction_as_of"] = pd.NaT
    blocks["capacity_snapshot_limitation"] = (
        "current_snapshot_with_training_date_reconstruction"
        if capacity_source == "ons_effective_capacity_snapshot"
        else "explicit_sensitivity_proxy"
    )
    blocks["capacity_match_fraction"] = float(
        plants["capacity_match_fraction"].iloc[0]
    )
    blocks["historical_training_cvu_plants"] = len(plants)
    blocks["terminal_catalog_cvu_plants"] = eligible_count
    blocks["matched_terminal_cvu_plants"] = int(
        plants.loc[
            plants["population_eligible"], "capacity_matched"
        ].sum()
    )
    blocks["terminal_cvu_catalog_date"] = plants["terminal_catalog_date"].max()
    blocks["thermal_population_rule"] = plants["population_rule"].iloc[0]
    crosswalk_present = plants.loc[
        plants["population_eligible"], "crosswalk_ceg_registration"
    ].notna() if "crosswalk_ceg_registration" in plants else pd.Series(False)
    blocks["training_crosswalk_coverage_fraction"] = float(
        crosswalk_present.sum() / eligible_count
    )
    blocks["required_capacity_match_fraction"] = (
        minimum_capacity_match_fraction
    )
    blocks = blocks.sort_values(
        ["marginal_cost_brl_per_mwh", "block"], kind="stable"
    ).reset_index(drop=True)
    blocks["block"] = np.arange(len(blocks), dtype=int)
    capacities = tuple(float(value) for value in blocks["capacity_mwh"])
    costs = tuple(float(value) for value in blocks["marginal_cost_brl_per_mwh"])
    if not capacities or min(capacities) <= 0.0:
        raise ValueError("thermal merit order has no positive-capacity block")
    return capacities, costs, plants.reset_index(drop=True), blocks


def _aggregation_selection_audit(
    monthly_ena: pd.DataFrame,
    reservoir_audit: pd.DataFrame,
    max_aggregates: int,
    aggregation_level: str,
) -> pd.DataFrame:
    train = monthly_ena.loc[_within_train(monthly_ena["time"])].copy()
    coverage = (
        train.groupby("aggregation_id", sort=True)
        .agg(
            train_months_present=("time", "nunique"),
            complete_train_months=("is_complete_month", "sum"),
            first_train_month=("time", "min"),
            last_train_month=("time", "max"),
            mean_train_inflow_mwh=("inflow_mwh", "mean"),
            release_capacity_mwh=(
                "inflow_mwh",
                lambda values: values.dropna().quantile(0.995),
            ),
        )
        .reset_index()
    )
    capacity = (
        reservoir_audit.loc[reservoir_audit["eligible_capacity"]]
        .groupby("aggregation_id", sort=True)
        .agg(
            energy_capacity_mwh=("energy_capacity_mwh", "sum"),
            identifiable_reservoirs=("reservoir_id", "nunique"),
        )
        .reset_index()
    )
    audit = coverage.merge(capacity, how="outer", on="aggregation_id")
    audit["aggregation_level"] = aggregation_level
    audit["required_train_months"] = EXPECTED_TRAIN_MONTHS
    audit["complete_training_record"] = (
        audit["train_months_present"].eq(EXPECTED_TRAIN_MONTHS)
        & audit["complete_train_months"].eq(EXPECTED_TRAIN_MONTHS)
        & audit["first_train_month"].eq(TRAIN_START)
        & audit["last_train_month"].eq(pd.Timestamp("2014-12-01"))
    )
    audit["capacity_identifiable"] = (
        audit["energy_capacity_mwh"].gt(0.0)
        & audit["identifiable_reservoirs"].gt(0)
    )
    audit["release_identifiable"] = audit["release_capacity_mwh"].gt(0.0)
    audit["eligible"] = (
        audit["complete_training_record"]
        & audit["capacity_identifiable"]
        & audit["release_identifiable"]
    )

    def reason(row: pd.Series) -> str:
        reasons = []
        if not bool(row["complete_training_record"]):
            reasons.append("incomplete_2000_2014_ena")
        if not bool(row["capacity_identifiable"]):
            reasons.append("energy_capacity_not_identifiable")
        if not bool(row["release_identifiable"]):
            reasons.append("release_capacity_not_identifiable")
        return ";".join(reasons)

    audit["exclusion_reason"] = audit.apply(reason, axis=1)
    if aggregation_level == "subsystem":
        subsystem_order = {name: rank for rank, name in enumerate(("SE", "S", "NE", "N"))}
        audit["_order"] = audit["aggregation_id"].map(subsystem_order).fillna(999)
        audit = audit.sort_values(
            ["eligible", "_order", "aggregation_id"],
            ascending=[False, True, True],
            na_position="last",
            kind="stable",
        ).drop(columns="_order").reset_index(drop=True)
    else:
        audit = audit.sort_values(
            ["eligible", "energy_capacity_mwh", "aggregation_id"],
            ascending=[False, False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    audit["selection_rank"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    eligible_indices = audit.index[audit["eligible"]]
    audit.loc[eligible_indices, "selection_rank"] = np.arange(
        1, len(eligible_indices) + 1
    )
    audit["selected"] = audit["eligible"] & audit["selection_rank"].le(
        max_aggregates
    )
    return audit.sort_values("aggregation_id").reset_index(drop=True)


def _selected_history(
    monthly_ena: pd.DataFrame,
    monthly_load: pd.DataFrame,
    selected_aggregates: tuple[str, ...],
    aggregation_level: str,
) -> pd.DataFrame:
    all_months = pd.date_range(HISTORY_START, HISTORY_END, freq="MS")
    grid = pd.MultiIndex.from_product(
        [all_months, selected_aggregates], names=["time", "unit_id"]
    ).to_frame(index=False)
    source = monthly_ena.rename(columns={"aggregation_id": "unit_id"})
    columns = [
        "time",
        "unit_id",
        "inflow_mwh",
        "observed_inflow_mwh",
        "observed_days",
        "expected_days",
        "minimum_daily_units",
        "maximum_daily_units",
        "is_complete_month",
    ]
    history = grid.merge(
        source.loc[source["unit_id"].isin(selected_aggregates), columns],
        how="left",
        on=["time", "unit_id"],
        validate="one_to_one",
    )
    history["expected_days"] = history["time"].dt.days_in_month
    history["observed_days"] = history["observed_days"].fillna(0).astype(int)
    history["is_complete_month"] = (
        history["is_complete_month"].astype("boolean").fillna(False).astype(bool)
    )
    history = history.merge(
        monthly_load.loc[:, ["time", "load_mwh", "subsystem_count"]],
        how="left",
        on="time",
        validate="many_to_one",
    )
    history.insert(0, "dataset", f"ons_newave_{aggregation_level}")
    history["aggregation_level"] = aggregation_level
    history["split"] = assign_temporal_split(history["time"])
    rank = {name: index for index, name in enumerate(selected_aggregates)}
    history["_rank"] = history["unit_id"].map(rank)
    return history.sort_values(["time", "_rank"]).drop(columns="_rank").reset_index(
        drop=True
    )


def build_ons_benchmark(
    reservoir_static: pd.DataFrame,
    ena: pd.DataFrame,
    load: pd.DataFrame,
    cvu: pd.DataFrame,
    *,
    subsystem_ena: pd.DataFrame | None = None,
    generation_capacity: pd.DataFrame | None = None,
    thermal_crosswalk: pd.DataFrame | None = None,
    aggregation_level: str = "subsystem",
    max_aggregates: int = 6,
    max_rees: int | None = None,
    horizon: int = 48,
    initial_storage_fraction: float = 0.65,
    release_capacity_quantile: float = 0.995,
    deficit_cost_brl_per_mwh: float = 10_000.0,
    terminal_water_value: float | None = None,
    thermal_capacity_proxy: bool = False,
    max_thermal_blocks: int | None = None,
    minimum_thermal_capacity_match_fraction: float = 0.80,
) -> ONSBenchmark:
    """Construct the frozen ONS hydrothermal benchmark.

    Validation, calibration, and historical-test rows remain in ``historical``
    but cannot affect aggregate selection, storage/release limits, seasonal
    demand, CVU costs, or terminal value.  Formal runs use electrical
    ``subsystem`` aggregation.  ``ree`` is retained only as an explicit
    sensitivity/fixture mode because ONS reservoir REE labels are absent in the
    fit-period raw files.  Formal subsystem runs must pass the official ONS
    daily subsystem aggregate through ``subsystem_ena``; reservoir-row sums are
    retained only as a missing-component diagnostic.
    """

    if aggregation_level not in {"subsystem", "ree"}:
        raise ValueError("aggregation_level must be 'subsystem' or 'ree'")
    if max_rees is not None:
        if aggregation_level != "ree":
            raise ValueError("max_rees is only valid with aggregation_level='ree'")
        if max_aggregates != 6:
            raise ValueError("pass only one of max_aggregates and max_rees")
        max_aggregates = max_rees
    if (
        isinstance(max_aggregates, bool)
        or not isinstance(max_aggregates, (int, np.integer))
        or not 1 <= int(max_aggregates) <= 6
    ):
        raise ValueError("max_aggregates must be between 1 and 6")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, np.integer))
        or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer")
    if not 0.0 < initial_storage_fraction <= 1.0:
        raise ValueError("initial_storage_fraction must lie in (0, 1]")
    if not 0.5 <= release_capacity_quantile <= 1.0:
        raise ValueError("release_capacity_quantile must lie in [0.5, 1]")
    if not np.isfinite(deficit_cost_brl_per_mwh):
        raise ValueError("deficit cost must be finite")

    if aggregation_level == "subsystem":
        if subsystem_ena is None:
            raise ValueError(
                "formal subsystem benchmark requires official subsystem_ena; "
                "reservoir ENA sums are coverage diagnostics only"
            )
        monthly_ena = aggregate_official_subsystem_monthly_ena(subsystem_ena)
        reservoir_component_coverage = aggregate_monthly_ena(
            ena, aggregation_level="subsystem"
        ).rename(
            columns={
                "observed_inflow_mwh": "observed_partial_reservoir_sum_mwh",
                "inflow_mwh": "partial_reservoir_sum_mwh",
                "is_complete_month": "is_aggregate_calendar_complete_month",
            }
        )
        reservoir_component_coverage["used_as_benchmark_inflow"] = False
        reservoir_component_coverage["audit_warning"] = (
            "missing_reservoir_components_are_not_zero_and_sum_is_not_model_input"
        )
    else:
        monthly_ena = aggregate_monthly_ena(ena, aggregation_level="ree")
        reservoir_component_coverage = pd.DataFrame()
    reservoir_audit = static_energy_capacity_audit(
        reservoir_static, ena, aggregation_level=aggregation_level
    )
    selection_audit = _aggregation_selection_audit(
        monthly_ena,
        reservoir_audit,
        int(max_aggregates),
        aggregation_level,
    )
    selected_table = selection_audit.loc[selection_audit["selected"]].sort_values(
        "selection_rank"
    )
    if selected_table.empty:
        reasons = selection_audit.loc[
            :, ["aggregation_id", "exclusion_reason"]
        ].to_dict("records")
        raise ValueError(f"no eligible ONS {aggregation_level}: {reasons}")
    selected_aggregates = tuple(
        str(value) for value in selected_table["aggregation_id"]
    )

    monthly_load = aggregate_monthly_load(load)
    training_load = monthly_load.loc[_within_train(monthly_load["time"])].copy()
    if len(training_load) != EXPECTED_TRAIN_MONTHS or not training_load[
        "is_complete_month"
    ].all():
        raise ValueError("load must contain every month from 2000-01 through 2014-12")
    if not np.all(np.isfinite(training_load["load_mwh"])):
        raise ValueError("training load must be finite")
    seasonal_demand = (
        training_load.assign(month=training_load["time"].dt.month)
        .groupby("month")["load_mwh"]
        .mean()
        .reindex(range(1, 13))
    )
    training_peak_load = float(training_load["load_mwh"].max())
    mean_training_month_hours = float(
        (training_load["time"].dt.days_in_month * 24.0).mean()
    )
    thermal_capacity, thermal_cost, thermal_plants, thermal_blocks = (
        build_cvu_merit_order(
            cvu,
            generation_capacity=generation_capacity,
            thermal_crosswalk=thermal_crosswalk,
            training_peak_load_mwh=training_peak_load,
            mean_training_month_hours=mean_training_month_hours,
            thermal_capacity_proxy=thermal_capacity_proxy,
            max_thermal_blocks=max_thermal_blocks,
            minimum_capacity_match_fraction=(
                minimum_thermal_capacity_match_fraction
            ),
        )
    )
    if deficit_cost_brl_per_mwh <= max(thermal_cost):
        raise ValueError("deficit cost must exceed every thermal marginal cost")

    history = _selected_history(
        monthly_ena,
        monthly_load,
        selected_aggregates,
        aggregation_level,
    )
    fit_times = pd.date_range(TRAIN_START, "2014-12-01", freq="MS")
    fit = (
        history.loc[history["split"].eq("train")]
        .pivot(index="time", columns="unit_id", values="inflow_mwh")
        .reindex(index=fit_times, columns=selected_aggregates)
    )
    if fit.isna().any().any():
        raise RuntimeError(
            f"selected {aggregation_level} unexpectedly has an incomplete training month"
        )
    fit_inflow = fit.to_numpy(dtype=np.float64)
    calendar = (fit_times.month.to_numpy(dtype=np.int16) - 1).astype(np.int16)

    # Recompute the requested quantile here so the public parameter is honored;
    # selection used q99.5 only to establish positive identifiability.
    training_selected = monthly_ena.loc[
        _within_train(monthly_ena["time"])
        & monthly_ena["aggregation_id"].isin(selected_aggregates)
    ]
    release_by_aggregate = training_selected.groupby("aggregation_id")[
        "inflow_mwh"
    ].quantile(release_capacity_quantile)
    capacity_by_aggregate = selected_table.set_index("aggregation_id")[
        "energy_capacity_mwh"
    ]
    storage_capacity = tuple(
        float(capacity_by_aggregate[name]) for name in selected_aggregates
    )
    release_capacity = tuple(
        float(release_by_aggregate[name]) for name in selected_aggregates
    )
    initial_storage = tuple(
        initial_storage_fraction * capacity for capacity in storage_capacity
    )
    terminal_target = initial_storage

    seasonal_release = (
        training_selected.assign(month=training_selected["time"].dt.month)
        .pivot_table(
            index="month",
            columns="aggregation_id",
            values="inflow_mwh",
            aggfunc="mean",
        )
        .reindex(index=range(1, 13), columns=selected_aggregates)
    )
    if seasonal_release.isna().any().any():
        raise RuntimeError(
            f"selected {aggregation_level} has an incomplete seasonal release target"
        )
    release_target = tuple(
        tuple(float(value) for value in seasonal_release.loc[month])
        for month in range(1, 13)
    )
    guide_lower = tuple(
        tuple(0.0 for _ in selected_aggregates) for _ in range(PERIODS)
    )
    guide_upper = tuple(storage_capacity for _ in range(PERIODS))
    zeros = tuple(0.0 for _ in selected_aggregates)
    if terminal_water_value is None:
        terminal_water_value = float(max(thermal_cost))
    if terminal_water_value < 0.0:
        raise ValueError("terminal_water_value must be nonnegative")

    objective = ObjectiveConfig(
        kind="hydrothermal",
        release_target=release_target,
        guide_lower=guide_lower,
        guide_upper=guide_upper,
        terminal_target=terminal_target,
        shortage_cost=zeros,
        excess_release_cost=zeros,
        low_storage_cost=zeros,
        flood_cost=zeros,
        spill_cost=zeros,
        ramp_cost=zeros,
        terminal_shortfall_cost=tuple(
            float(terminal_water_value) for _ in selected_aggregates
        ),
        terminal_excess_cost=zeros,
        generation_coefficient=tuple(1.0 for _ in selected_aggregates),
        seasonal_demand=tuple(float(value) for value in seasonal_demand),
        thermal_block_capacity=thermal_capacity,
        thermal_block_cost=thermal_cost,
        deficit_cost=float(deficit_cost_brl_per_mwh),
    )
    routing = tuple(
        tuple(0.0 for _ in selected_aggregates) for _ in selected_aggregates
    )
    system = ReservoirSystem(
        benchmark="ons_newave",
        case_id=f"ons_{aggregation_level}_{len(selected_aggregates)}",
        horizon=int(horizon),
        periods=PERIODS,
        step_unit="mwh_month",
        reservoir_ids=selected_aggregates,
        storage_capacity=storage_capacity,
        release_capacity=release_capacity,
        initial_storage=initial_storage,
        routing=routing,
        objective=objective,
        metadata=(
            ("calibration_window", "2000-01/2014-12"),
            ("aggregation_level", aggregation_level),
            (
                "aggregation_assignment",
                "historical_ena.subsystem_id+static.subsystem_id"
                if aggregation_level == "subsystem"
                else "historical_training_ena.ree_name",
            ),
            ("ena_quantity", "gross_ena_mwh"),
            (
                "ena_source",
                "ons_official_subsystem_ena_daily"
                if aggregation_level == "subsystem"
                else "ons_reservoir_ena_daily_explicit_ree_sensitivity",
            ),
            (
                "storage_energy_formula",
                "usable_hm3*productivity65*277.7777777778",
            ),
            ("release_capacity", f"training_q{release_capacity_quantile:g}_ena"),
            (
                "thermal_capacity_source",
                str(thermal_blocks["capacity_source"].iloc[0]),
            ),
            (
                "thermal_capacity_match_fraction",
                f"{thermal_blocks['capacity_match_fraction'].iloc[0]:.6f}",
            ),
            (
                "thermal_population_rule",
                str(thermal_blocks["thermal_population_rule"].iloc[0]),
            ),
            (
                "thermal_terminal_catalog_date",
                str(
                    pd.Timestamp(
                        thermal_blocks["terminal_cvu_catalog_date"].iloc[0]
                    ).date()
                ),
            ),
            ("decision_timing", "decision_before_current_ena"),
        ),
    )
    audit = ONSBenchmarkAudit(
        reservoir_capacity=reservoir_audit,
        aggregation_selection=selection_audit,
        monthly_coverage=monthly_ena,
        reservoir_component_coverage=reservoir_component_coverage,
        monthly_load=monthly_load,
        thermal_plants=thermal_plants,
        thermal_blocks=thermal_blocks,
    )
    return ONSBenchmark(
        system=system,
        fit_inflow=fit_inflow,
        calendar=calendar,
        fit_times=fit_times,
        historical=history,
        audit=audit,
    )


__all__ = [
    "EXPECTED_TRAIN_MONTHS",
    "HISTORY_END",
    "HISTORY_START",
    "MWH_PER_HM3_PER_MW_PER_M3S",
    "ONSBenchmark",
    "ONSBenchmarkAudit",
    "TRAIN_END",
    "TRAIN_START",
    "aggregate_monthly_ena",
    "aggregate_monthly_load",
    "aggregate_official_subsystem_monthly_ena",
    "aggregate_ree_monthly_ena",
    "build_cvu_merit_order",
    "build_ons_benchmark",
    "static_energy_capacity_audit",
]
