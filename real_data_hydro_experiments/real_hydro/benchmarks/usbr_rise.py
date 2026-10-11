"""Leakage-safe USBR RISE benchmark wrapper for the Aspinall cascade.

The common physical/control construction is deliberately delegated to
``build_resopsus_benchmark``.  This module adds only source-specific checks
that are meaningful for the Upper Colorado Hydrologic Database representation:

* the prepared local series must be the directly observed local/side inflow
  less observed evaporation;
* policy-independent routed inflow must contain each observed upstream release
  exactly once; and
* downstream total-inflow records are compared with upstream release plus the
  directly observed side inflow.

All summaries are functions only of the prepared weekly source table.  No
training checkpoint, policy evaluation, or method result is accepted by this
API, and none of the summaries is fed back into benchmark calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
import pandas as pd

from ..data.usbr_rise import ASPINALL_CASE
from .resopsus import (
    ResOpsBenchmark,
    ResOpsBenchmarkAudit,
    ResOpsBenchmarkError,
    build_resopsus_benchmark,
)


BENCHMARK_NAME = "rise_uc_hdb_v1"

RISE_SOURCE_METADATA: tuple[tuple[str, str], ...] = (
    ("source", "USBR RISE; Upper Colorado Hydrologic Database"),
    ("source_url", "https://data.usbr.gov/"),
    ("catalog_records", "2249/2386/2278"),
    ("source_snapshot", "content-addressed RISE JSON:API pages"),
    ("benchmark_semantics", "real three-reservoir Aspinall cascade"),
    (
        "inflow_source_fields",
        "direct local/side inflow net of evaporation; downstream total inflow audit-only",
    ),
)

_SOURCE_COLUMNS = (
    "source_local_inflow_mcm",
    "source_evaporation_mcm",
    "net_local_inflow_mcm",
)


class RiseBenchmarkError(ResOpsBenchmarkError):
    """Raised when a prepared RISE table violates frozen source semantics."""


def _copied_frame(value: pd.DataFrame) -> pd.DataFrame:
    return value.copy(deep=True).reset_index(drop=True)


@dataclass(frozen=True)
class RiseBenchmarkAudit(ResOpsBenchmarkAudit):
    """Common benchmark diagnostics plus RISE-only source summaries."""

    source_fields_by_split: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_inflow_reconstruction: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover_by_split: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        # Match the existing audit convention (DataFrames are public artifacts)
        # while preventing a caller's input frame from aliasing the audit object.
        object.__setattr__(
            self, "source_fields_by_split", _copied_frame(self.source_fields_by_split)
        )
        object.__setattr__(
            self,
            "total_inflow_reconstruction",
            _copied_frame(self.total_inflow_reconstruction),
        )
        object.__setattr__(
            self, "turnover_by_split", _copied_frame(self.turnover_by_split)
        )


@dataclass(frozen=True)
class USBRRiseBenchmark(ResOpsBenchmark):
    """A common :class:`ResOpsBenchmark` carrying RISE-specific audits."""

    audit: RiseBenchmarkAudit


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise RiseBenchmarkError(
            f"Prepared Aspinall weekly table is missing RISE fields: {missing}."
        )


def _allclose_or_raise(
    observed: np.ndarray,
    expected: np.ndarray,
    message: str,
) -> None:
    valid = np.isfinite(observed) & np.isfinite(expected)
    if not valid.all():
        raise RiseBenchmarkError(f"{message} Values must be finite.")
    if not np.allclose(observed, expected, rtol=1.0e-7, atol=1.0e-9):
        maximum = float(np.max(np.abs(observed - expected)))
        raise RiseBenchmarkError(f"{message} Maximum absolute error is {maximum:.6g} MCM.")


def _normalized_source_table(weekly_case: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dam_id",
        "reservoir_name",
        "week_end",
        "split",
        "reported_inflow_mcm",
        "local_inflow_mcm",
        "model_local_inflow_mcm",
        "routed_inflow_mcm",
        "storage_mcm",
        "release_mcm",
        "reference_capacity_mcm",
        *_SOURCE_COLUMNS,
        "source_total_inflow_mcm",
    }
    _require_columns(weekly_case, required)
    data = weekly_case.copy()
    data["week_end"] = pd.to_datetime(data["week_end"], errors="coerce")
    numeric_ids = pd.to_numeric(data["dam_id"], errors="coerce")
    if data["week_end"].isna().any() or numeric_ids.isna().any():
        raise RiseBenchmarkError("Aspinall source audit contains invalid dates or IDs.")
    if not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise RiseBenchmarkError("Aspinall source audit contains a nonintegral ID.")
    data["dam_id"] = numeric_ids.astype(int)
    if set(data["dam_id"].unique()) != set(ASPINALL_CASE.dam_ids):
        raise RiseBenchmarkError("Aspinall source audit must contain exactly three records.")
    if data.duplicated(["dam_id", "week_end"]).any():
        raise RiseBenchmarkError("Aspinall source audit contains duplicate reservoir weeks.")

    numeric_columns = {
        "reported_inflow_mcm",
        "local_inflow_mcm",
        "model_local_inflow_mcm",
        "routed_inflow_mcm",
        "storage_mcm",
        "release_mcm",
        "reference_capacity_mcm",
        *_SOURCE_COLUMNS,
        "source_total_inflow_mcm",
    }
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in (*_SOURCE_COLUMNS, "reported_inflow_mcm", "local_inflow_mcm"):
        if not np.isfinite(data[column].to_numpy(float)).all():
            raise RiseBenchmarkError(f"RISE source field {column} must be finite.")
    if (data["source_evaporation_mcm"] < 0.0).any():
        raise RiseBenchmarkError("RISE source evaporation cannot be negative.")

    _allclose_or_raise(
        data["net_local_inflow_mcm"].to_numpy(float),
        (
            data["source_local_inflow_mcm"]
            - data["source_evaporation_mcm"]
        ).to_numpy(float),
        "Net local inflow must equal direct source local inflow minus evaporation.",
    )
    _allclose_or_raise(
        data["reported_inflow_mcm"].to_numpy(float),
        data["net_local_inflow_mcm"].to_numpy(float),
        "Prepared reported inflow must retain the RISE net-local source field.",
    )
    _allclose_or_raise(
        data["local_inflow_mcm"].to_numpy(float),
        data["reported_inflow_mcm"].to_numpy(float),
        "Aspinall is a direct-local case; local and prepared reported inflow must agree.",
    )

    lookup = {dam_id: position for position, dam_id in enumerate(ASPINALL_CASE.dam_ids)}
    expected_routed = np.zeros(len(data), dtype=np.float64)
    release = data.set_index(["dam_id", "week_end"])["release_mcm"]
    for source, target in ASPINALL_CASE.routing_edges:
        rows = data["dam_id"].eq(target)
        expected_routed[rows.to_numpy()] += np.asarray(
            [release.get((source, date), np.nan) for date in data.loc[rows, "week_end"]],
            dtype=np.float64,
        )
    # ``lookup`` is also an explicit assertion that every frozen edge endpoint
    # belongs to the three-reservoir ordering used by the common builder.
    if any(
        source not in lookup or target not in lookup
        for source, target in ASPINALL_CASE.routing_edges
    ):
        raise RiseBenchmarkError("A frozen Aspinall routing edge leaves the case.")
    _allclose_or_raise(
        data["routed_inflow_mcm"].to_numpy(float),
        expected_routed,
        "Routed inflow must apply each observed upstream release exactly once.",
    )

    data["split_label"] = data["split"].astype("string").fillna("unassigned")
    return data.sort_values(["week_end", "dam_id"], kind="stable").reset_index(
        drop=True
    )


def _source_field_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, dam_id), values in data.groupby(
        ["split_label", "dam_id"], sort=True, dropna=False
    ):
        identity_error = values["net_local_inflow_mcm"] - (
            values["source_local_inflow_mcm"]
            - values["source_evaporation_mcm"]
        )
        rows.append(
            {
                "split": str(split),
                "dam_id": int(dam_id),
                "reservoir_name": str(values["reservoir_name"].iloc[0]),
                "weeks": int(len(values)),
                "source_local_inflow_sum_mcm": float(
                    values["source_local_inflow_mcm"].sum()
                ),
                "source_evaporation_sum_mcm": float(
                    values["source_evaporation_mcm"].sum()
                ),
                "net_local_inflow_sum_mcm": float(
                    values["net_local_inflow_mcm"].sum()
                ),
                "negative_net_local_weeks": int(
                    values["net_local_inflow_mcm"].lt(0.0).sum()
                ),
                "max_abs_source_identity_error_mcm": float(
                    identity_error.abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _total_inflow_reconstruction(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, target in ASPINALL_CASE.routing_edges:
        upstream = data.loc[
            data["dam_id"].eq(source), ["week_end", "release_mcm"]
        ].rename(columns={"release_mcm": "upstream_release_mcm"})
        downstream = data.loc[data["dam_id"].eq(target)].merge(
            upstream, on="week_end", how="left", validate="one_to_one"
        )
        reconstructed = (
            downstream["source_local_inflow_mcm"]
            + downstream["upstream_release_mcm"]
        )
        downstream = downstream.assign(
            reconstructed_total_inflow_mcm=reconstructed,
            reconstruction_residual_mcm=(
                downstream["source_total_inflow_mcm"] - reconstructed
            ),
        )
        for split, values in downstream.groupby("split_label", sort=True):
            valid = np.isfinite(
                values[
                    [
                        "source_total_inflow_mcm",
                        "reconstructed_total_inflow_mcm",
                        "reconstruction_residual_mcm",
                    ]
                ].to_numpy(float)
            ).all(axis=1)
            complete = values.loc[valid]
            if complete.empty:
                raise RiseBenchmarkError(
                    f"Downstream record {target} has no finite total-inflow "
                    f"audit weeks in split {split}."
                )
            residual = complete["reconstruction_residual_mcm"].to_numpy(float)
            absolute = np.abs(residual)
            observed = complete["source_total_inflow_mcm"].to_numpy(float)
            scale = max(float(np.mean(np.abs(observed))), 1.0e-12)
            rows.append(
                {
                    "split": str(split),
                    "source_dam_id": int(source),
                    "target_dam_id": int(target),
                    "weeks": int(len(values)),
                    "finite_audit_weeks": int(valid.sum()),
                    "missing_audit_weeks": int((~valid).sum()),
                    "observed_total_inflow_sum_mcm": float(observed.sum()),
                    "reconstructed_total_inflow_sum_mcm": float(
                        complete["reconstructed_total_inflow_mcm"].sum()
                    ),
                    "residual_mean_mcm": float(residual.mean()),
                    "residual_median_mcm": float(np.median(residual)),
                    "residual_mae_mcm": float(absolute.mean()),
                    "residual_rmse_mcm": float(np.sqrt(np.mean(residual**2))),
                    "residual_p95_abs_mcm": float(np.quantile(absolute, 0.95)),
                    "residual_max_abs_mcm": float(absolute.max()),
                    "relative_mae": float(absolute.mean() / scale),
                }
            )
    return pd.DataFrame(rows)


def _turnover_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, dam_id), values in data.groupby(
        ["split_label", "dam_id"], sort=True, dropna=False
    ):
        capacity_values = values["reference_capacity_mcm"].to_numpy(float)
        if not np.isfinite(capacity_values).all() or np.any(capacity_values <= 0.0):
            raise RiseBenchmarkError("Turnover audit requires positive frozen capacity.")
        capacity = float(capacity_values[0])
        if not np.allclose(capacity_values, capacity, rtol=0.0, atol=1.0e-9):
            raise RiseBenchmarkError("Frozen reference capacity changes within a split.")
        annualizer = 52.0 / float(len(values))

        def turnover(column: str) -> float:
            numbers = values[column].to_numpy(float)
            if not np.isfinite(numbers).all():
                raise RiseBenchmarkError(
                    f"Turnover source {column} contains a nonfinite value."
                )
            return float(numbers.sum() * annualizer / capacity)

        rows.append(
            {
                "split": str(split),
                "dam_id": int(dam_id),
                "reservoir_name": str(values["reservoir_name"].iloc[0]),
                "weeks": int(len(values)),
                "reference_capacity_mcm": capacity,
                "gross_local_turnover_per_52_weeks": turnover(
                    "source_local_inflow_mcm"
                ),
                "net_local_turnover_per_52_weeks": turnover(
                    "net_local_inflow_mcm"
                ),
                "routed_turnover_per_52_weeks": turnover("routed_inflow_mcm"),
                "release_turnover_per_52_weeks": turnover("release_mcm"),
                "median_storage_fraction_capacity": float(
                    values["storage_mcm"].median() / capacity
                ),
            }
        )
    return pd.DataFrame(rows)


def build_usbr_rise_benchmark(weekly_case: pd.DataFrame) -> USBRRiseBenchmark:
    """Build the frozen Aspinall benchmark and source-only RISE QA bundle."""

    source = _normalized_source_table(weekly_case)
    source_summary = _source_field_summary(source)
    reconstruction = _total_inflow_reconstruction(source)
    turnover = _turnover_summary(source)
    try:
        base = build_resopsus_benchmark(
            weekly_case,
            ASPINALL_CASE,
            benchmark_name=BENCHMARK_NAME,
            source_metadata=RISE_SOURCE_METADATA,
        )
    except ResOpsBenchmarkError as error:
        raise RiseBenchmarkError(str(error)) from error

    common = base.audit
    system = replace(
        base.system,
        metadata=(
            *base.system.metadata,
            (
                "available_water",
                "hazard-decision; opening storage plus same-stage upstream release",
            ),
            ("availability_semantics", "same_stage_upstream_release"),
            ("runtime_adapter", "aspinall_coupled_active_faces_v1"),
        ),
    )
    audit = RiseBenchmarkAudit(
        raw_mass_balance=common.raw_mass_balance,
        projected_mass_balance=common.projected_mass_balance,
        alignment_by_split=common.alignment_by_split,
        calibration_week_ends=common.calibration_week_ends,
        calibration_splits=common.calibration_splits,
        source_fields_by_split=source_summary,
        total_inflow_reconstruction=reconstruction,
        turnover_by_split=turnover,
    )
    return USBRRiseBenchmark(
        system=system,
        fit_inflow=base.fit_inflow,
        fit_calendar=base.fit_calendar,
        fit_week_end=base.fit_week_end,
        historical_windows=base.historical_windows,
        limits=base.limits,
        audit=audit,
    )


# A case-oriented alias is convenient for preparation/runner call sites while
# retaining the source-specific public name above.
build_aspinall_benchmark = build_usbr_rise_benchmark


__all__ = [
    "BENCHMARK_NAME",
    "RISE_SOURCE_METADATA",
    "RiseBenchmarkAudit",
    "RiseBenchmarkError",
    "USBRRiseBenchmark",
    "build_aspinall_benchmark",
    "build_usbr_rise_benchmark",
]
