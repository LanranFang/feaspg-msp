"""Pinned ResOpsUS v2 access and preprocessing for real-data hydro cases.

The module deliberately treats the Zenodo ZIP as the cache artifact.  It never
extracts the complete archive: callers can either read selected reservoir CSVs
directly from the ZIP or explicitly extract only the requested members.

All temporal calibration is performed after a chronological split.  In
particular, :func:`calibrate_training_limits` ignores validation and test rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import md5
import os
from numbers import Integral
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterable, Mapping
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


ZENODO_RECORD_ID = 6_612_040
ZENODO_VERSION_DOI = "10.5281/zenodo.6612040"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.5367382"
RESOPSUS_ARCHIVE_NAME = "ResOpsUS_v2.zip"
RESOPSUS_ARCHIVE_URL = (
    "https://zenodo.org/api/records/6612040/files/ResOpsUS%202.zip/content"
)
RESOPSUS_ARCHIVE_MD5 = "d0684cbacf6196c246c73b858ab5b752"
RESOPSUS_ARCHIVE_SIZE_BYTES = 307_542_558

# One daily mean cubic metre per second is 86,400 m3/day = 0.0864 MCM/day.
CMS_DAY_TO_MCM = 0.0864
WEEK_DAYS = 7
DEFAULT_MIN_VALID_DAYS = 6

_TIME_SERIES_PREFIX = "ResOpsUS/time_series_all"
_SOURCE_COLUMNS = (
    "date",
    "storage",
    "inflow",
    "outflow",
    "elevation",
    "evaporation",
)


class ResOpsUSError(RuntimeError):
    """Base class for deterministic ResOpsUS data errors."""


class ChecksumMismatchError(ResOpsUSError):
    """Raised when an archive does not match its pinned checksum or size."""


class ArchiveMemberMissingError(ResOpsUSError):
    """Raised when a requested reservoir is absent from the archive."""


class DataIntegrityError(ResOpsUSError):
    """Raised when a selected CSV violates the expected daily-data schema."""


@dataclass(frozen=True)
class ReservoirSpec:
    """Static information needed by a selected experiment case."""

    dam_id: int
    name: str
    reference_capacity_mcm: float


@dataclass(frozen=True)
class SplitWindow:
    """Inclusive daily bounds for one chronological data split."""

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Invalid {self.name!r} split: {self.start} > {self.end}.")


@dataclass(frozen=True)
class CaseSpec:
    """A reproducible ResOpsUS case and its leakage-free date windows."""

    name: str
    reservoirs: tuple[ReservoirSpec, ...]
    splits: tuple[SplitWindow, ...]
    routing_edges: tuple[tuple[int, int], ...] = ()
    inflow_semantics: str = "direct_local"

    def __post_init__(self) -> None:
        ids = tuple(item.dam_id for item in self.reservoirs)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Case {self.name!r} contains duplicate DAM_ID values.")
        if self.inflow_semantics not in {"direct_local", "cumulative_natural"}:
            raise ValueError(
                "inflow_semantics must be 'direct_local' or "
                "'cumulative_natural'."
            )
        known_ids = set(ids)
        if any(source not in known_ids or target not in known_ids for source, target in self.routing_edges):
            raise ValueError(f"Case {self.name!r} contains a routing edge outside the case.")
        ordered = sorted(self.splits, key=lambda item: item.start)
        for left, right in zip(ordered, ordered[1:]):
            if left.end >= right.start:
                raise ValueError(
                    f"Case {self.name!r} has overlapping splits "
                    f"{left.name!r} and {right.name!r}."
                )

    @property
    def dam_ids(self) -> tuple[int, ...]:
        return tuple(item.dam_id for item in self.reservoirs)


@dataclass(frozen=True)
class MassBalanceAudit:
    """Per-week residuals and reservoir-level robust summaries."""

    per_week: pd.DataFrame
    summary: pd.DataFrame


LUCKY_PEAK = ReservoirSpec(391, "Lucky Peak", 378.7)
ROSS = ReservoirSpec(41, "Ross", 1_791.9)
DIABLO = ReservoirSpec(42, "Diablo", 111.0)

_COMMON_VALIDATION = SplitWindow(
    "validation", date(2008, 10, 1), date(2013, 9, 30)
)
_COMMON_TEST = SplitWindow("test", date(2013, 10, 1), date(2019, 9, 30))

RESOPSUS_CASES: Mapping[str, CaseSpec] = MappingProxyType(
    {
        "lucky_peak": CaseSpec(
            name="lucky_peak",
            reservoirs=(LUCKY_PEAK,),
            splits=(
                SplitWindow("train", date(1984, 10, 1), date(2008, 9, 30)),
                _COMMON_VALIDATION,
                _COMMON_TEST,
            ),
        ),
        "ross_diablo": CaseSpec(
            name="ross_diablo",
            reservoirs=(ROSS, DIABLO),
            splits=(
                SplitWindow("train", date(1990, 10, 1), date(2008, 9, 30)),
                _COMMON_VALIDATION,
                _COMMON_TEST,
            ),
            routing_edges=((41, 42),),
            inflow_semantics="cumulative_natural",
        ),
    }
)


def default_cache_dir() -> Path:
    """Return the user cache directory without creating it."""

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return root / "real_data_hydro_experiments" / "resopsus"


def file_md5(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a lowercase MD5 digest without loading the archive into memory."""

    digest = md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(
    path: str | Path,
    *,
    expected_md5: str = RESOPSUS_ARCHIVE_MD5,
    expected_size_bytes: int | None = None,
) -> Path:
    """Verify a cached archive and return its normalized :class:`Path`.

    ``expected_size_bytes`` is optional for fixture archives.  The production
    downloader supplies the pinned Zenodo size in addition to the checksum.
    """

    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if expected_size_bytes is not None and archive.stat().st_size != expected_size_bytes:
        raise ChecksumMismatchError(
            f"Unexpected size for {archive}: got {archive.stat().st_size}, "
            f"expected {expected_size_bytes}."
        )
    observed = file_md5(archive)
    if observed.casefold() != expected_md5.casefold():
        raise ChecksumMismatchError(
            f"MD5 mismatch for {archive}: got {observed}, expected {expected_md5}."
        )
    return archive


def download_archive(
    cache_dir: str | Path | None = None,
    *,
    url: str = RESOPSUS_ARCHIVE_URL,
    archive_name: str = RESOPSUS_ARCHIVE_NAME,
    expected_md5: str = RESOPSUS_ARCHIVE_MD5,
    expected_size_bytes: int | None = RESOPSUS_ARCHIVE_SIZE_BYTES,
    timeout_seconds: float = 120.0,
    opener: Callable[..., BinaryIO] | None = None,
) -> Path:
    """Download the pinned ZIP once, atomically, and reuse a valid cache.

    A corrupt pre-existing cache is never silently overwritten.  This makes a
    provenance problem explicit and avoids turning a transient network failure
    into an apparently valid experiment input.
    """

    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / archive_name
    if destination.exists():
        return verify_archive(
            destination,
            expected_md5=expected_md5,
            expected_size_bytes=expected_size_bytes,
        )

    open_url = opener or urlopen
    request = Request(url, headers={"User-Agent": "real-hydro-experiments/1.0"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{archive_name}.", suffix=".part", dir=root, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with open_url(request, timeout=timeout_seconds) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        verify_archive(
            temporary_path,
            expected_md5=expected_md5,
            expected_size_bytes=expected_size_bytes,
        )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _reservoir_member(dam_id: int) -> str:
    if isinstance(dam_id, bool) or not isinstance(dam_id, Integral):
        raise TypeError(f"DAM_ID must be an integer, got {dam_id!r}.")
    if int(dam_id) <= 0:
        raise ValueError(f"DAM_ID must be a positive integer, got {dam_id!r}.")
    return f"{_TIME_SERIES_PREFIX}/ResOpsUS_{int(dam_id)}.csv"


def _checked_members(zip_file: ZipFile, dam_ids: Iterable[int]) -> dict[int, str]:
    requested_list: list[int] = []
    for value in dam_ids:
        _reservoir_member(value)
        requested_list.append(int(value))
    requested = tuple(dict.fromkeys(requested_list))
    if not requested:
        raise ValueError("At least one DAM_ID is required.")
    available = set(zip_file.namelist())
    members = {dam_id: _reservoir_member(dam_id) for dam_id in requested}
    missing = [dam_id for dam_id, member in members.items() if member not in available]
    if missing:
        raise ArchiveMemberMissingError(
            f"ResOpsUS archive lacks requested DAM_ID values: {missing}."
        )
    return members


def extract_selected_reservoirs(
    archive_path: str | Path,
    destination_dir: str | Path,
    dam_ids: Iterable[int],
    *,
    expected_md5: str = RESOPSUS_ARCHIVE_MD5,
    overwrite: bool = False,
) -> dict[int, Path]:
    """Extract only requested per-reservoir CSVs, never the full archive."""

    archive = verify_archive(archive_path, expected_md5=expected_md5)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        members = _checked_members(zip_file, dam_ids)
        targets = {
            dam_id: destination / f"ResOpsUS_{dam_id}.csv" for dam_id in members
        }
        existing = [path for path in targets.values() if path.exists() and not overwrite]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite selected extraction(s): "
                + ", ".join(str(path) for path in existing)
            )
        for dam_id, member in members.items():
            target = targets[dam_id]
            with zip_file.open(member) as source, tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{target.name}.", suffix=".part", dir=destination,
                delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    shutil.copyfileobj(source, temporary)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    os.replace(temporary_path, target)
                finally:
                    temporary_path.unlink(missing_ok=True)
    return targets


def _normalize_daily(frame: pd.DataFrame, dam_id: int) -> pd.DataFrame:
    missing = sorted(set(_SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise DataIntegrityError(f"DAM_ID {dam_id} is missing columns: {missing}.")
    result = frame.loc[:, _SOURCE_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise DataIntegrityError(f"DAM_ID {dam_id} contains an invalid date.")
    for column in _SOURCE_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values("date", kind="stable").reset_index(drop=True)
    if result["date"].duplicated().any():
        duplicates = result.loc[result["date"].duplicated(False), "date"]
        sample = ", ".join(item.strftime("%Y-%m-%d") for item in duplicates.head(3))
        raise DataIntegrityError(
            f"DAM_ID {dam_id} contains duplicate daily dates (for example {sample})."
        )
    result = result.rename(
        columns={
            "storage": "storage_mcm",
            "inflow": "inflow_m3s",
            "outflow": "outflow_m3s",
            "elevation": "elevation_m",
            "evaporation": "evaporation_mcm",
        }
    )
    result.insert(1, "dam_id", int(dam_id))
    names = {item.dam_id: item.name for case in RESOPSUS_CASES.values() for item in case.reservoirs}
    result.insert(2, "reservoir_name", names.get(int(dam_id), f"DAM_ID {dam_id}"))
    return result


def load_daily_reservoirs(
    archive_path: str | Path,
    dam_ids: Iterable[int],
    *,
    expected_md5: str = RESOPSUS_ARCHIVE_MD5,
) -> pd.DataFrame:
    """Read selected daily CSVs from the ZIP and return one normalized long table."""

    archive = verify_archive(archive_path, expected_md5=expected_md5)
    frames: list[pd.DataFrame] = []
    with ZipFile(archive) as zip_file:
        members = _checked_members(zip_file, dam_ids)
        for dam_id, member in members.items():
            with zip_file.open(member) as stream:
                raw = pd.read_csv(stream, na_values=("NA", "NaN", ""))
            frames.append(_normalize_daily(raw, dam_id))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["dam_id", "date"], kind="stable", ignore_index=True
    )


def daily_to_weekly(
    daily: pd.DataFrame,
    *,
    min_valid_days: int = DEFAULT_MIN_VALID_DAYS,
) -> pd.DataFrame:
    """Aggregate one reservoir's daily table to Sunday-ending weekly MCM.

    Flow variables are daily means in m3/s.  For a retained week their mean is
    multiplied by seven days and ``0.0864 MCM / (m3/s day)``.  Consequently a
    single missing day (six valid values) does not bias the weekly volume low.
    Each variable is masked independently when it has fewer than
    ``min_valid_days`` observations.  Partial boundary weeks are discarded.
    """

    if not 1 <= min_valid_days <= WEEK_DAYS:
        raise ValueError("min_valid_days must lie in [1, 7].")
    required = {
        "date",
        "dam_id",
        "storage_mcm",
        "inflow_m3s",
        "outflow_m3s",
        "elevation_m",
        "evaporation_mcm",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise DataIntegrityError(f"Daily input is missing columns: {missing}.")
    dam_ids = pd.unique(daily["dam_id"])
    if len(dam_ids) != 1:
        raise ValueError("daily_to_weekly expects exactly one DAM_ID.")
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date", kind="stable")
    if data["date"].duplicated().any():
        raise DataIntegrityError("Daily input contains duplicate dates.")
    indexed = data.set_index("date")
    weekly = pd.DataFrame(index=indexed.resample("W-SUN").size().index)
    weekly.index.name = "week_end"
    weekly["observed_days"] = indexed.resample("W-SUN").size().astype(int)

    source_to_output = {
        "storage_mcm": ("storage_mcm", "last", 1.0),
        "inflow_m3s": ("inflow_mcm", "mean", WEEK_DAYS * CMS_DAY_TO_MCM),
        "outflow_m3s": ("outflow_mcm", "mean", WEEK_DAYS * CMS_DAY_TO_MCM),
        "elevation_m": ("elevation_m", "last", 1.0),
        "evaporation_mcm": ("evaporation_mcm", "mean", float(WEEK_DAYS)),
    }
    for source, (output, aggregation, multiplier) in source_to_output.items():
        resampler = indexed[source].resample("W-SUN")
        counts = resampler.count().astype(int)
        values = getattr(resampler, aggregation)() * multiplier
        weekly[f"{output}_valid_days"] = counts
        weekly[output] = values.where(counts >= min_valid_days)

    weekly = weekly.reset_index()
    weekly.insert(0, "week_start", weekly["week_end"] - pd.Timedelta(days=6))
    coverage_start = data["date"].min().normalize()
    coverage_end = data["date"].max().normalize()
    full_boundary_week = (weekly["week_start"] >= coverage_start) & (
        weekly["week_end"] <= coverage_end
    )
    weekly = weekly.loc[full_boundary_week].reset_index(drop=True)
    weekly.insert(2, "dam_id", int(dam_ids[0]))
    reservoir_name = (
        str(data["reservoir_name"].iloc[0])
        if "reservoir_name" in data.columns
        else f"DAM_ID {int(dam_ids[0])}"
    )
    weekly.insert(3, "reservoir_name", reservoir_name)
    return weekly


def assign_case_splits(
    weekly: pd.DataFrame,
    case: str | CaseSpec,
    *,
    drop_unassigned: bool = True,
) -> pd.DataFrame:
    """Assign only weeks wholly contained in a pre-registered date window."""

    spec = get_case(case)
    required = {"week_start", "week_end"}
    missing = sorted(required - set(weekly.columns))
    if missing:
        raise DataIntegrityError(f"Weekly input is missing columns: {missing}.")
    result = weekly.copy()
    result["week_start"] = pd.to_datetime(result["week_start"], errors="raise")
    result["week_end"] = pd.to_datetime(result["week_end"], errors="raise")
    split = pd.Series(pd.NA, index=result.index, dtype="string")
    for window in spec.splits:
        mask = (result["week_start"] >= pd.Timestamp(window.start)) & (
            result["week_end"] <= pd.Timestamp(window.end)
        )
        if split.loc[mask].notna().any():
            raise DataIntegrityError(f"A week maps to multiple splits in {spec.name}.")
        split.loc[mask] = window.name
    result["split"] = split
    if drop_unassigned:
        result = result.loc[result["split"].notna()].reset_index(drop=True)
    return result


def get_case(case: str | CaseSpec) -> CaseSpec:
    """Resolve a public case name while preserving support for test fixtures."""

    if isinstance(case, CaseSpec):
        return case
    try:
        return RESOPSUS_CASES[case]
    except KeyError as error:
        raise KeyError(
            f"Unknown ResOpsUS case {case!r}; choose from {sorted(RESOPSUS_CASES)}."
        ) from error


def prepare_weekly_case(
    daily: pd.DataFrame,
    case: str | CaseSpec,
    *,
    min_valid_days: int = DEFAULT_MIN_VALID_DAYS,
    drop_unassigned: bool = True,
) -> pd.DataFrame:
    """Create a model-ready weekly long table for Lucky Peak or Ross--Diablo.

    ``reported_inflow_mcm`` retains the source field.  For Ross--Diablo the
    hydrologically incremental inflow is explicitly
    ``Diablo.inflow - Ross.inflow``; Ross release is a same-week routed arrival.
    ``model_local_inflow_mcm`` is the nonnegative projection used by a physical
    simulator, while the unprojected value remains available for auditing.
    """

    spec = get_case(case)
    present = set(int(value) for value in pd.unique(daily["dam_id"]))
    absent = sorted(set(spec.dam_ids) - present)
    if absent:
        raise DataIntegrityError(f"Case {spec.name!r} lacks DAM_ID values {absent}.")
    frames = [
        daily_to_weekly(
            daily.loc[daily["dam_id"] == reservoir.dam_id],
            min_valid_days=min_valid_days,
        )
        for reservoir in spec.reservoirs
    ]
    weekly = pd.concat(frames, ignore_index=True)
    weekly = weekly.rename(
        columns={"inflow_mcm": "reported_inflow_mcm", "outflow_mcm": "release_mcm"}
    )
    weekly["case_name"] = spec.name
    capacity = {item.dam_id: item.reference_capacity_mcm for item in spec.reservoirs}
    weekly["reference_capacity_mcm"] = weekly["dam_id"].map(capacity).astype(float)
    weekly["local_inflow_mcm"] = weekly["reported_inflow_mcm"]
    weekly["routed_inflow_mcm"] = 0.0

    if spec.inflow_semantics == "cumulative_natural":
        if spec.dam_ids != (41, 42) or spec.routing_edges != ((41, 42),):
            raise DataIntegrityError(
                "The frozen cumulative-natural preprocessing is defined only "
                "for Ross--Diablo."
            )
        keyed = weekly.set_index(["dam_id", "week_end"])
        ross_inflow = keyed.loc[41, "reported_inflow_mcm"]
        ross_release = keyed.loc[41, "release_mcm"]
        is_diablo = weekly["dam_id"].eq(42)
        diablo_weeks = weekly.loc[is_diablo, "week_end"]
        weekly.loc[is_diablo, "local_inflow_mcm"] = (
            weekly.loc[is_diablo, "reported_inflow_mcm"].to_numpy()
            - diablo_weeks.map(ross_inflow).to_numpy()
        )
        weekly.loc[is_diablo, "routed_inflow_mcm"] = diablo_weeks.map(
            ross_release
        ).to_numpy()

    weekly["negative_local_inflow"] = weekly["local_inflow_mcm"].lt(0).fillna(False)
    weekly["model_local_inflow_mcm"] = weekly["local_inflow_mcm"].clip(lower=0.0)
    weekly = assign_case_splits(
        weekly, spec, drop_unassigned=drop_unassigned
    ).sort_values(["week_end", "dam_id"], kind="stable", ignore_index=True)
    return weekly


def load_weekly_case(
    archive_path: str | Path,
    case: str | CaseSpec,
    *,
    expected_md5: str = RESOPSUS_ARCHIVE_MD5,
    min_valid_days: int = DEFAULT_MIN_VALID_DAYS,
    drop_unassigned: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: selected ZIP reads followed by weekly preparation."""

    spec = get_case(case)
    daily = load_daily_reservoirs(
        archive_path, spec.dam_ids, expected_md5=expected_md5
    )
    return prepare_weekly_case(
        daily,
        spec,
        min_valid_days=min_valid_days,
        drop_unassigned=drop_unassigned,
    )


def calibrate_training_limits(
    weekly_case: pd.DataFrame,
    case: str | CaseSpec,
    *,
    storage_quantile: float = 0.999,
    release_quantile: float = 0.995,
) -> pd.DataFrame:
    """Calibrate capacities using training rows only.

    The model storage capacity is the larger of the static reference capacity
    and the selected training-storage quantile.  The release/action capacity is
    the requested training outflow quantile.  Held-out extreme values therefore
    cannot change either limit.
    """

    if not 0.0 < storage_quantile <= 1.0:
        raise ValueError("storage_quantile must lie in (0, 1].")
    if not 0.0 < release_quantile <= 1.0:
        raise ValueError("release_quantile must lie in (0, 1].")
    required = {"dam_id", "week_end", "split", "storage_mcm", "release_mcm"}
    missing = sorted(required - set(weekly_case.columns))
    if missing:
        raise DataIntegrityError(f"Weekly case is missing columns: {missing}.")
    spec = get_case(case)
    train = weekly_case.loc[weekly_case["split"].eq("train")].copy()
    rows: list[dict[str, object]] = []
    for reservoir in spec.reservoirs:
        values = train.loc[train["dam_id"].eq(reservoir.dam_id)]
        storage = values["storage_mcm"].dropna()
        release = values["release_mcm"].dropna()
        if storage.empty or release.empty:
            raise DataIntegrityError(
                f"No complete training storage/release values for DAM_ID "
                f"{reservoir.dam_id}."
            )
        training_storage = float(storage.quantile(storage_quantile))
        rows.append(
            {
                "dam_id": reservoir.dam_id,
                "reservoir_name": reservoir.name,
                "reference_capacity_mcm": reservoir.reference_capacity_mcm,
                "training_storage_quantile": storage_quantile,
                "training_storage_capacity_mcm": training_storage,
                "model_storage_capacity_mcm": max(
                    reservoir.reference_capacity_mcm, training_storage
                ),
                "release_quantile": release_quantile,
                "release_capacity_mcm_per_week": float(
                    release.quantile(release_quantile)
                ),
                "n_training_storage_weeks": int(storage.size),
                "n_training_release_weeks": int(release.size),
                "training_start": pd.Timestamp(values["week_end"].min()),
                "training_end": pd.Timestamp(values["week_end"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("dam_id", ignore_index=True)


def audit_mass_balance(
    weekly_case: pd.DataFrame,
    *,
    use_projected_local_inflow: bool = False,
) -> MassBalanceAudit:
    """Audit weekly storage changes using raw or nonnegative local inflow.

    Missing evaporation is treated as zero and identified in the per-week
    output.  Transitions separated by anything other than seven days are not
    scored, so gaps and split boundaries cannot masquerade as balance errors.
    """

    inflow_column = (
        "model_local_inflow_mcm"
        if use_projected_local_inflow
        else "local_inflow_mcm"
    )
    required = {
        "dam_id",
        "reservoir_name",
        "week_end",
        "storage_mcm",
        "release_mcm",
        "routed_inflow_mcm",
        inflow_column,
    }
    missing = sorted(required - set(weekly_case.columns))
    if missing:
        raise DataIntegrityError(f"Weekly case is missing columns: {missing}.")
    audit = weekly_case.copy().sort_values(
        ["dam_id", "week_end"], kind="stable", ignore_index=True
    )
    audit["week_end"] = pd.to_datetime(audit["week_end"], errors="raise")
    groups = audit.groupby("dam_id", sort=False)
    audit["previous_storage_mcm"] = groups["storage_mcm"].shift(1)
    audit["days_since_previous_week"] = groups["week_end"].diff().dt.days
    audit["observed_storage_delta_mcm"] = (
        audit["storage_mcm"] - audit["previous_storage_mcm"]
    )
    if "evaporation_mcm" in audit:
        audit["evaporation_assumed_zero"] = audit["evaporation_mcm"].isna()
        evaporation = audit["evaporation_mcm"].fillna(0.0)
    else:
        audit["evaporation_assumed_zero"] = True
        evaporation = 0.0
    audit["modeled_storage_delta_mcm"] = (
        audit[inflow_column]
        + audit["routed_inflow_mcm"]
        - audit["release_mcm"]
        - evaporation
    )
    valid = (
        audit["days_since_previous_week"].eq(WEEK_DAYS)
        & audit["observed_storage_delta_mcm"].notna()
        & audit["modeled_storage_delta_mcm"].notna()
    )
    audit["balance_residual_mcm"] = np.nan
    audit.loc[valid, "balance_residual_mcm"] = (
        audit.loc[valid, "observed_storage_delta_mcm"]
        - audit.loc[valid, "modeled_storage_delta_mcm"]
    )
    audit["balance_valid"] = valid

    summaries: list[dict[str, object]] = []
    for dam_id, values in audit.loc[valid].groupby("dam_id", sort=True):
        residual = values["balance_residual_mcm"].astype(float)
        absolute = residual.abs()
        scale = float(values["modeled_storage_delta_mcm"].abs().median())
        summaries.append(
            {
                "dam_id": int(dam_id),
                "reservoir_name": str(values["reservoir_name"].iloc[0]),
                "n_valid_transitions": int(residual.size),
                "mean_residual_mcm": float(residual.mean()),
                "median_absolute_residual_mcm": float(absolute.median()),
                "p90_absolute_residual_mcm": float(absolute.quantile(0.90)),
                "p95_absolute_residual_mcm": float(absolute.quantile(0.95)),
                "p99_absolute_residual_mcm": float(absolute.quantile(0.99)),
                "rmse_residual_mcm": float(np.sqrt(np.mean(np.square(residual)))),
                "median_absolute_modeled_delta_mcm": scale,
                "normalized_median_absolute_residual": (
                    float(absolute.median()) / scale if scale > 0.0 else np.nan
                ),
                "projected_local_inflow": bool(use_projected_local_inflow),
            }
        )
    summary_columns = (
        "dam_id",
        "reservoir_name",
        "n_valid_transitions",
        "mean_residual_mcm",
        "median_absolute_residual_mcm",
        "p90_absolute_residual_mcm",
        "p95_absolute_residual_mcm",
        "p99_absolute_residual_mcm",
        "rmse_residual_mcm",
        "median_absolute_modeled_delta_mcm",
        "normalized_median_absolute_residual",
        "projected_local_inflow",
    )
    summary = pd.DataFrame(summaries, columns=summary_columns)
    return MassBalanceAudit(per_week=audit, summary=summary)


__all__ = [
    "ArchiveMemberMissingError",
    "CMS_DAY_TO_MCM",
    "CaseSpec",
    "ChecksumMismatchError",
    "DEFAULT_MIN_VALID_DAYS",
    "DIABLO",
    "DataIntegrityError",
    "LUCKY_PEAK",
    "MassBalanceAudit",
    "RESOPSUS_ARCHIVE_MD5",
    "RESOPSUS_ARCHIVE_NAME",
    "RESOPSUS_ARCHIVE_SIZE_BYTES",
    "RESOPSUS_ARCHIVE_URL",
    "RESOPSUS_CASES",
    "ROSS",
    "ReservoirSpec",
    "ResOpsUSError",
    "SplitWindow",
    "ZENODO_CONCEPT_DOI",
    "ZENODO_RECORD_ID",
    "ZENODO_VERSION_DOI",
    "assign_case_splits",
    "audit_mass_balance",
    "calibrate_training_limits",
    "daily_to_weekly",
    "default_cache_dir",
    "download_archive",
    "extract_selected_reservoirs",
    "file_md5",
    "get_case",
    "load_daily_reservoirs",
    "load_weekly_case",
    "prepare_weekly_case",
    "verify_archive",
]
