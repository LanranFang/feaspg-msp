"""Frozen USBR RISE access for the three-reservoir Aspinall cascade.

The network API is used only by :func:`download_aspinall_snapshot`.  Training
and audit code consume the resulting content-addressed JSON pages, so a later
RISE revision cannot silently change an experiment.  The primary control model
uses daily storage, local inflow, total release, and evaporation volumes.  Two
downstream total-inflow series are retained only to audit routing semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .resopsus import (
    CaseSpec,
    DataIntegrityError,
    ReservoirSpec,
    SplitWindow,
    assign_case_splits,
)


RISE_API_ROOT = "https://data.usbr.gov/rise/api"
RISE_ACCEPT = "application/vnd.api+json"
RISE_TIMEZONE = "America/Denver"
RISE_ITEMS_PER_PAGE = 10_000
SNAPSHOT_SCHEMA_VERSION = 1

# Exact SI conversion used by Reclamation and NIST-style unit tables.
ACRE_FOOT_TO_MCM = 0.00123348183754752
ACRE_FOOT_PER_DAY_TO_CMS = ACRE_FOOT_TO_MCM * 1_000_000.0 / 86_400.0

ASPINALL_RAW_START = date(1980, 9, 1)
ASPINALL_RAW_END = date(2019, 9, 30)

BLUE_MESA = ReservoirSpec(
    dam_id=2249,
    name="Blue Mesa",
    reference_capacity_mcm=940_800.0 * ACRE_FOOT_TO_MCM,
)
MORROW_POINT = ReservoirSpec(
    dam_id=2386,
    name="Morrow Point",
    reference_capacity_mcm=117_190.0 * ACRE_FOOT_TO_MCM,
)
CRYSTAL = ReservoirSpec(
    dam_id=2278,
    name="Crystal",
    reference_capacity_mcm=26_000.0 * ACRE_FOOT_TO_MCM,
)

ASPINALL_CASE = CaseSpec(
    name="aspinall_3",
    reservoirs=(BLUE_MESA, MORROW_POINT, CRYSTAL),
    splits=(
        SplitWindow("train", date(1980, 10, 1), date(2008, 9, 30)),
        SplitWindow("validation", date(2008, 10, 1), date(2013, 9, 30)),
        SplitWindow("test", date(2013, 10, 1), date(2019, 9, 30)),
    ),
    routing_edges=((2249, 2386), (2386, 2278)),
    inflow_semantics="direct_local",
)


class RiseDataError(RuntimeError):
    """Raised when a RISE response or frozen snapshot violates the protocol."""


@dataclass(frozen=True)
class RiseSeriesSpec:
    """One pinned RISE catalog item used by the Aspinall snapshot."""

    reservoir_record_id: int
    reservoir_name: str
    item_id: int
    field: str
    parameter_name: str
    unit: str = "af"
    role: str = "primary"

    def __post_init__(self) -> None:
        if self.reservoir_record_id not in ASPINALL_CASE.dam_ids:
            raise ValueError("RISE series reservoir is outside the Aspinall case.")
        if self.unit != "af":
            raise ValueError("The frozen primary Aspinall pipeline uses native AF series.")
        if self.role not in {"primary", "routing_audit"}:
            raise ValueError("RISE series role must be primary or routing_audit.")


ASPINALL_SERIES: tuple[RiseSeriesSpec, ...] = (
    RiseSeriesSpec(2249, "Blue Mesa", 76, "storage_af", "Lake/Reservoir Storage"),
    RiseSeriesSpec(2249, "Blue Mesa", 4283, "local_inflow_af", "Lake/Reservoir Inflow"),
    RiseSeriesSpec(2249, "Blue Mesa", 4349, "release_af", "Lake/Reservoir Release - Total"),
    RiseSeriesSpec(2249, "Blue Mesa", 79, "evaporation_af", "Lake/Reservoir Evaporation"),
    RiseSeriesSpec(2386, "Morrow Point", 592, "storage_af", "Lake/Reservoir Storage"),
    RiseSeriesSpec(2386, "Morrow Point", 4293, "local_inflow_af", "Lake/Reservoir Side Inflow"),
    RiseSeriesSpec(2386, "Morrow Point", 4350, "release_af", "Lake/Reservoir Release - Total"),
    RiseSeriesSpec(2386, "Morrow Point", 595, "evaporation_af", "Lake/Reservoir Evaporation"),
    RiseSeriesSpec(2278, "Crystal", 274, "storage_af", "Lake/Reservoir Storage"),
    RiseSeriesSpec(2278, "Crystal", 4294, "local_inflow_af", "Lake/Reservoir Side Inflow"),
    RiseSeriesSpec(2278, "Crystal", 4351, "release_af", "Lake/Reservoir Release - Total"),
    RiseSeriesSpec(2278, "Crystal", 277, "evaporation_af", "Lake/Reservoir Evaporation"),
    RiseSeriesSpec(
        2386,
        "Morrow Point",
        4284,
        "total_inflow_af",
        "Lake/Reservoir Inflow",
        role="routing_audit",
    ),
    RiseSeriesSpec(
        2278,
        "Crystal",
        4285,
        "total_inflow_af",
        "Lake/Reservoir Inflow",
        role="routing_audit",
    ),
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _utc_bound(value: date, *, start: bool) -> str:
    local = datetime.combine(value, time.min, tzinfo=ZoneInfo(RISE_TIMEZONE))
    if start:
        local -= timedelta(seconds=1)
    return local.astimezone(timezone.utc).isoformat()


def result_query_url(
    item_id: int,
    start: date,
    end: date,
    *,
    page: int,
    items_per_page: int = RISE_ITEMS_PER_PAGE,
) -> str:
    """Construct an exact inclusive-local-date RISE result query."""

    if start > end:
        raise ValueError("RISE query start must not exceed end.")
    if page < 1 or items_per_page < 1:
        raise ValueError("RISE pagination values must be positive.")
    parameters = (
        ("itemId", int(item_id)),
        ("dateTime[after]", _utc_bound(start, start=True)),
        ("dateTime[before]", _utc_bound(end + timedelta(days=1), start=False)),
        ("itemsPerPage", int(items_per_page)),
        ("page", int(page)),
    )
    return f"{RISE_API_ROOT}/result?{urlencode(parameters)}"


def _fetch(url: str, *, timeout_seconds: float) -> tuple[bytes, Mapping[str, str]]:
    request = Request(
        url,
        headers={
            "Accept": RISE_ACCEPT,
            "User-Agent": "real-hydro-experiments/1.0",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
        headers = {str(key): str(value) for key, value in response.headers.items()}
    return payload, headers


def _decode_json_api(payload: bytes, *, source: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RiseDataError(f"{source} is not valid JSON.") from error
    if not isinstance(value, Mapping) or not isinstance(value.get("data"), list):
        raise RiseDataError(f"{source} is not a RISE JSON:API collection.")
    return value


def download_aspinall_snapshot(
    destination: str | Path,
    *,
    start: date = ASPINALL_RAW_START,
    end: date = ASPINALL_RAW_END,
    series: Iterable[RiseSeriesSpec] = ASPINALL_SERIES,
    timeout_seconds: float = 120.0,
) -> Path:
    """Explicitly download one immutable, content-addressed RISE snapshot.

    A complete existing snapshot is verified and reused.  New downloads are
    assembled in a sibling temporary directory and renamed only after every
    page and catalog record has been hashed and counted.
    """

    destination = Path(destination)
    if destination.exists():
        verify_aspinall_snapshot(destination)
        return destination / "snapshot_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = tuple(series)
    if not selected or len({item.item_id for item in selected}) != len(selected):
        raise ValueError("RISE snapshot series must be nonempty with unique item IDs.")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    records: list[dict[str, Any]] = []
    try:
        for spec in selected:
            item_directory = temporary / f"item_{spec.item_id}"
            item_directory.mkdir(parents=True, exist_ok=False)

            catalog_url = f"{RISE_API_ROOT}/catalog-item/{spec.item_id}"
            catalog_payload, catalog_headers = _fetch(
                catalog_url, timeout_seconds=timeout_seconds
            )
            catalog_path = item_directory / "catalog_item.json"
            _atomic_bytes(catalog_path, catalog_payload)

            pages: list[dict[str, Any]] = []
            page = 1
            total_items: int | None = None
            while True:
                url = result_query_url(spec.item_id, start, end, page=page)
                payload, headers = _fetch(url, timeout_seconds=timeout_seconds)
                decoded = _decode_json_api(payload, source=url)
                meta = decoded.get("meta")
                if not isinstance(meta, Mapping):
                    raise RiseDataError(f"RISE result page {url} has no metadata.")
                observed_total = int(meta.get("totalItems", -1))
                if observed_total < 0:
                    raise RiseDataError(f"RISE result page {url} has invalid totalItems.")
                if total_items is None:
                    total_items = observed_total
                elif total_items != observed_total:
                    raise RiseDataError("RISE totalItems changed during the snapshot.")
                page_payload = decoded["data"]
                if not isinstance(page_payload, list):
                    raise RiseDataError(f"RISE result page {url} has invalid data.")
                page_path = item_directory / f"results_page_{page:04d}.json"
                _atomic_bytes(page_path, payload)
                pages.append(
                    {
                        "page": page,
                        "url": url,
                        "file": str(page_path.relative_to(temporary)),
                        "bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                        "records": len(page_payload),
                        "headers": {
                            key: value
                            for key, value in headers.items()
                            if key.casefold() in {"content-type", "etag", "last-modified"}
                        },
                    }
                )
                if sum(int(item["records"]) for item in pages) >= total_items:
                    break
                if not page_payload:
                    raise RiseDataError("RISE pagination ended before totalItems.")
                page += 1

            if total_items is None or sum(item["records"] for item in pages) != total_items:
                raise RiseDataError("RISE page counts do not equal totalItems.")
            records.append(
                {
                    "series": asdict(spec),
                    "catalog": {
                        "url": catalog_url,
                        "file": str(catalog_path.relative_to(temporary)),
                        "bytes": len(catalog_payload),
                        "sha256": _sha256_bytes(catalog_payload),
                        "headers": {
                            key: value
                            for key, value in catalog_headers.items()
                            if key.casefold() in {"content-type", "etag", "last-modified"}
                        },
                    },
                    "total_records": total_items,
                    "pages": pages,
                }
            )

        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "provider": "USBR RISE",
            "api_root": RISE_API_ROOT,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "local_timezone": RISE_TIMEZONE,
            "inclusive_start": start.isoformat(),
            "inclusive_end": end.isoformat(),
            "series": records,
        }
        _atomic_bytes(temporary / "snapshot_manifest.json", _canonical_json(manifest))
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_aspinall_snapshot(destination)
    return destination / "snapshot_manifest.json"


def _load_manifest(snapshot: Path) -> Mapping[str, Any]:
    path = snapshot / "snapshot_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RiseDataError(f"Invalid Aspinall snapshot manifest: {path}.") from error
    if not isinstance(value, Mapping):
        raise RiseDataError("Aspinall snapshot manifest must be an object.")
    return value


def verify_aspinall_snapshot(snapshot: str | Path) -> Mapping[str, Any]:
    """Verify hashes, page counts, item identities, and the frozen date range."""

    snapshot = Path(snapshot)
    manifest = _load_manifest(snapshot)
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise RiseDataError("Unsupported Aspinall snapshot schema.")
    try:
        inclusive_start = date.fromisoformat(str(manifest["inclusive_start"]))
        inclusive_end = date.fromisoformat(str(manifest["inclusive_end"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RiseDataError("Aspinall snapshot has invalid inclusive dates.") from error
    if (
        inclusive_start != ASPINALL_RAW_START
        or inclusive_end != ASPINALL_RAW_END
    ):
        raise RiseDataError(
            "Aspinall snapshot date range differs from the frozen interval "
            f"{ASPINALL_RAW_START} through {ASPINALL_RAW_END}."
        )
    raw_series = manifest.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        raise RiseDataError("Aspinall snapshot contains no series.")
    observed_ids: list[int] = []
    expected_series = {item.item_id: asdict(item) for item in ASPINALL_SERIES}
    for record in raw_series:
        if not isinstance(record, Mapping) or not isinstance(record.get("series"), Mapping):
            raise RiseDataError("Malformed Aspinall series record.")
        item_id = int(record["series"]["item_id"])
        observed_ids.append(item_id)
        if dict(record["series"]) != expected_series.get(item_id):
            raise RiseDataError(
                f"Item {item_id} metadata differs from the frozen series definition."
            )
        files = [record.get("catalog"), *record.get("pages", [])]
        for descriptor in files:
            if not isinstance(descriptor, Mapping):
                raise RiseDataError(f"Item {item_id} has a malformed file descriptor.")
            relative = Path(str(descriptor.get("file")))
            if relative.is_absolute() or ".." in relative.parts:
                raise RiseDataError("Snapshot file path escapes its root.")
            path = snapshot / relative
            if not path.is_file() or path.stat().st_size != int(descriptor.get("bytes", -1)):
                raise RiseDataError(f"Snapshot file is missing or truncated: {path}.")
            if _sha256_file(path) != str(descriptor.get("sha256")):
                raise RiseDataError(f"Snapshot SHA-256 mismatch: {path}.")
        pages = record.get("pages")
        if not isinstance(pages, list) or sum(int(page["records"]) for page in pages) != int(
            record.get("total_records", -1)
        ):
            raise RiseDataError(f"Item {item_id} has inconsistent page counts.")
    if len(observed_ids) != len(set(observed_ids)):
        raise RiseDataError("Aspinall snapshot repeats a RISE item ID.")
    expected = {item.item_id for item in ASPINALL_SERIES}
    if set(observed_ids) != expected:
        raise RiseDataError(
            f"Aspinall snapshot item IDs differ from the frozen set: {set(observed_ids)}."
        )
    return manifest


def _series_values(
    snapshot: Path,
    record: Mapping[str, Any],
    start: date,
    end: date,
    *,
    allow_null_results: bool = False,
) -> pd.Series:
    spec = record["series"]
    item_id = int(spec["item_id"])
    rows: list[tuple[pd.Timestamp, float]] = []
    for descriptor in record["pages"]:
        path = snapshot / str(descriptor["file"])
        decoded = _decode_json_api(path.read_bytes(), source=str(path))
        for entry in decoded["data"]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("attributes"), Mapping):
                raise RiseDataError(f"Item {item_id} contains a malformed result.")
            attributes = entry["attributes"]
            if int(attributes.get("itemId", -1)) != item_id:
                raise RiseDataError(f"Item {item_id} page contains another item ID.")
            result_attributes = attributes.get("resultAttributes")
            if not isinstance(result_attributes, Mapping) or result_attributes.get(
                "resultType"
            ) != "observed":
                raise RiseDataError(f"Item {item_id} contains a non-observed result.")
            raw_value = attributes.get("result")
            if raw_value is None and allow_null_results:
                value = np.nan
            else:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise RiseDataError(
                        f"Item {item_id} contains a nonnumeric result."
                    ) from error
                if not math.isfinite(value):
                    raise RiseDataError(f"Item {item_id} contains a nonfinite result.")
            timestamp = pd.Timestamp(attributes.get("dateTime"))
            if timestamp.tzinfo is None:
                raise RiseDataError(f"Item {item_id} contains a timezone-naive timestamp.")
            local_date = timestamp.tz_convert(RISE_TIMEZONE).date()
            if start <= local_date <= end:
                rows.append((pd.Timestamp(local_date), value))
    frame = pd.DataFrame(rows, columns=("date", "value"))
    if frame["date"].duplicated().any():
        raise RiseDataError(f"Item {item_id} contains duplicate local dates.")
    result = frame.set_index("date")["value"].sort_index()
    expected = pd.date_range(start, end, freq="D")
    result = result.reindex(expected)
    if result.isna().any() and not allow_null_results:
        missing = result.index[result.isna()][:3].strftime("%Y-%m-%d").tolist()
        raise RiseDataError(f"Item {item_id} has missing dates, e.g. {missing}.")
    result.name = str(spec["field"])
    return result


def load_aspinall_daily(snapshot: str | Path) -> pd.DataFrame:
    """Load the verified raw pages as a normalized daily long table.

    The simulator has no explicit evaporation input.  We therefore expose
    ``local_inflow`` as the observed local/side volume net of observed
    evaporation, while preserving both source fields for audit.  This is fixed
    before optimization and keeps the transition in physical volume units.
    """

    snapshot = Path(snapshot)
    manifest = verify_aspinall_snapshot(snapshot)
    start = date.fromisoformat(str(manifest["inclusive_start"]))
    end = date.fromisoformat(str(manifest["inclusive_end"]))
    by_reservoir: dict[int, dict[str, pd.Series]] = {
        item.dam_id: {} for item in ASPINALL_CASE.reservoirs
    }
    for record in manifest["series"]:
        spec = record["series"]
        reservoir_id = int(spec["reservoir_record_id"])
        by_reservoir[reservoir_id][str(spec["field"])] = _series_values(
            snapshot,
            record,
            start,
            end,
            allow_null_results=str(spec["role"]) == "routing_audit",
        )

    frames: list[pd.DataFrame] = []
    required = {"storage_af", "local_inflow_af", "release_af", "evaporation_af"}
    names = {item.dam_id: item.name for item in ASPINALL_CASE.reservoirs}
    for reservoir_id in ASPINALL_CASE.dam_ids:
        fields = by_reservoir[reservoir_id]
        missing = sorted(required - set(fields))
        if missing:
            raise RiseDataError(
                f"Reservoir record {reservoir_id} lacks primary fields {missing}."
            )
        frame = pd.concat(fields.values(), axis=1).reset_index(names="date")
        frame.insert(1, "dam_id", reservoir_id)
        frame.insert(2, "reservoir_name", names[reservoir_id])
        frame["source_local_inflow_mcm"] = (
            frame["local_inflow_af"] * ACRE_FOOT_TO_MCM
        )
        frame["source_evaporation_mcm"] = (
            frame["evaporation_af"] * ACRE_FOOT_TO_MCM
        )
        frame["net_local_inflow_mcm"] = (
            frame["source_local_inflow_mcm"]
            - frame["source_evaporation_mcm"]
        )
        frame["storage_mcm"] = frame["storage_af"] * ACRE_FOOT_TO_MCM
        frame["inflow_m3s"] = (
            frame["net_local_inflow_mcm"] * 1_000_000.0 / 86_400.0
        )
        frame["outflow_m3s"] = frame["release_af"] * ACRE_FOOT_PER_DAY_TO_CMS
        frame["evaporation_mcm"] = 0.0  # already netted from model-local inflow
        frame["elevation_m"] = np.nan
        if "total_inflow_af" in frame:
            frame["source_total_inflow_mcm"] = (
                frame["total_inflow_af"] * ACRE_FOOT_TO_MCM
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "dam_id"], kind="stable", ignore_index=True
    )


def _weekly_source_volumes(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    source_columns = [
        "source_local_inflow_mcm",
        "source_evaporation_mcm",
        "net_local_inflow_mcm",
        "source_total_inflow_mcm",
    ]
    for reservoir_id, values in daily.groupby("dam_id", sort=False):
        data = values.set_index("date").sort_index()
        available = [column for column in source_columns if column in data]
        weekly = data[available].resample("W-SUN").sum(min_count=7)
        weekly["source_valid_days"] = data.resample("W-SUN").size()
        weekly = (
            weekly.loc[weekly["source_valid_days"].eq(7)]
            .rename_axis("week_end")
            .reset_index()
        )
        weekly.insert(1, "dam_id", int(reservoir_id))
        rows.append(weekly)
    return pd.concat(rows, ignore_index=True)


def prepare_aspinall_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the all-RISE daily panel and route each release exactly once."""

    from .resopsus import daily_to_weekly  # avoid a circular public re-export

    present = set(int(value) for value in pd.unique(daily["dam_id"]))
    if present != set(ASPINALL_CASE.dam_ids):
        raise DataIntegrityError("Aspinall daily input does not contain all three records.")
    frames = [
        daily_to_weekly(daily.loc[daily["dam_id"].eq(reservoir_id)])
        for reservoir_id in ASPINALL_CASE.dam_ids
    ]
    weekly = pd.concat(frames, ignore_index=True).rename(
        columns={"inflow_mcm": "reported_inflow_mcm", "outflow_mcm": "release_mcm"}
    )
    weekly = weekly.merge(
        _weekly_source_volumes(daily),
        on=["dam_id", "week_end"],
        how="left",
        validate="one_to_one",
    )
    weekly["case_name"] = ASPINALL_CASE.name
    capacities = {
        item.dam_id: item.reference_capacity_mcm for item in ASPINALL_CASE.reservoirs
    }
    weekly["reference_capacity_mcm"] = weekly["dam_id"].map(capacities).astype(float)
    weekly["local_inflow_mcm"] = weekly["reported_inflow_mcm"]
    weekly["routed_inflow_mcm"] = 0.0
    keyed_release = weekly.set_index(["dam_id", "week_end"])["release_mcm"]
    for source, target in ASPINALL_CASE.routing_edges:
        target_rows = weekly["dam_id"].eq(target)
        dates = weekly.loc[target_rows, "week_end"]
        source_release = pd.Series(
            [keyed_release.get((source, value), np.nan) for value in dates],
            index=weekly.index[target_rows],
            dtype=float,
        )
        weekly.loc[target_rows, "routed_inflow_mcm"] += source_release
    weekly["negative_local_inflow"] = weekly["local_inflow_mcm"].lt(0).fillna(False)
    weekly["model_local_inflow_mcm"] = weekly["local_inflow_mcm"].clip(lower=0.0)
    return assign_case_splits(weekly, ASPINALL_CASE).sort_values(
        ["week_end", "dam_id"], kind="stable", ignore_index=True
    )


def aspinall_source_balance_audit(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return source-field storage and routing residuals without model repairs."""

    required = {
        "dam_id",
        "week_end",
        "storage_mcm",
        "release_mcm",
        "source_local_inflow_mcm",
        "source_evaporation_mcm",
        "routed_inflow_mcm",
    }
    missing = sorted(required - set(weekly.columns))
    if missing:
        raise DataIntegrityError(f"Aspinall weekly audit lacks columns {missing}.")
    result = weekly.copy().sort_values(["dam_id", "week_end"], kind="stable")
    grouped = result.groupby("dam_id", sort=False)
    result["previous_storage_mcm"] = grouped["storage_mcm"].shift(1)
    result["days_since_previous"] = grouped["week_end"].diff().dt.days
    result["observed_storage_delta_mcm"] = (
        result["storage_mcm"] - result["previous_storage_mcm"]
    )
    result["source_balance_delta_mcm"] = (
        result["source_local_inflow_mcm"]
        + result["routed_inflow_mcm"]
        - result["release_mcm"]
        - result["source_evaporation_mcm"]
    )
    valid = result["days_since_previous"].eq(7)
    result["source_balance_residual_mcm"] = np.where(
        valid,
        result["observed_storage_delta_mcm"] - result["source_balance_delta_mcm"],
        np.nan,
    )
    result["release_exceeds_opening_storage"] = (
        valid & result["release_mcm"].gt(result["previous_storage_mcm"])
    )
    result["release_excess_over_opening_mcm"] = np.where(
        result["release_exceeds_opening_storage"],
        result["release_mcm"] - result["previous_storage_mcm"],
        0.0,
    )
    return result


__all__ = [
    "ACRE_FOOT_PER_DAY_TO_CMS",
    "ACRE_FOOT_TO_MCM",
    "ASPINALL_CASE",
    "ASPINALL_RAW_END",
    "ASPINALL_RAW_START",
    "ASPINALL_SERIES",
    "BLUE_MESA",
    "CRYSTAL",
    "MORROW_POINT",
    "RISE_API_ROOT",
    "RISE_TIMEZONE",
    "RiseDataError",
    "RiseSeriesSpec",
    "aspinall_source_balance_audit",
    "download_aspinall_snapshot",
    "load_aspinall_daily",
    "prepare_aspinall_weekly",
    "result_query_url",
    "verify_aspinall_snapshot",
]
