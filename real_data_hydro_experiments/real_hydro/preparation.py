"""Build and persist leakage-safe real-hydrology benchmark inputs.

Only this module reads raw provider files.  Training consumes the compact,
content-addressed case directories written here, which makes restarts cheap and
prevents a revised upstream CSV from silently changing an in-progress run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import ObjectiveConfig, ReservoirSystem
from .dataset import TrajectoryBatch
from .stochastic import PeriodicARModel, fit_periodic_ar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

CASE_IDS = (
    "ons_subsystems",
    "resops_lucky_peak",
    "resops_ross_diablo",
    "camels_br_12",
)

HISTORICAL_REPLAY_SCHEMA_VERSION = 1
HISTORICAL_REPLAY_DIRECTORY = "historical_replay"
_HISTORICAL_ARRAY_FIELDS = (
    "inflow",
    "hydrology_state",
    "hydrology_regime",
    "calendar",
    "demand",
    "source_path_id",
)


@dataclass(frozen=True)
class PreparedCase:
    """Minimal common contract consumed by every optimization method."""

    case_id: str
    system: ReservoirSystem
    fit_inflow: np.ndarray
    fit_calendar: np.ndarray
    annual_component: bool
    model: PeriodicARModel
    provenance: Mapping[str, Any]
    historical_replay: TrajectoryBatch | None = None

    def __post_init__(self) -> None:
        inflow = np.array(self.fit_inflow, dtype=np.float64, copy=True)
        calendar = np.array(self.fit_calendar, dtype=np.int16, copy=True)
        if inflow.ndim != 2 or inflow.shape[1] != self.system.n_reservoirs:
            raise ValueError("fit_inflow must have shape [time, reservoir].")
        if calendar.shape != (len(inflow),):
            raise ValueError("fit_calendar must align with fit_inflow.")
        if np.any(~np.isfinite(inflow)) or np.any(inflow < 0.0):
            raise ValueError("Prepared fit inflow must be finite and nonnegative.")
        if self.model.n_series != self.system.n_reservoirs:
            raise ValueError("Stochastic model width differs from the system.")
        historical = self.historical_replay
        if historical is not None:
            historical = _readonly_trajectory_batch(historical)
            _validate_historical_replay(historical, self.system)
        inflow.setflags(write=False)
        calendar.setflags(write=False)
        object.__setattr__(self, "fit_inflow", inflow)
        object.__setattr__(self, "fit_calendar", calendar)
        object.__setattr__(self, "historical_replay", historical)


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trajectory_arrays(batch: TrajectoryBatch) -> dict[str, np.ndarray]:
    arrays = {
        "inflow": np.asarray(batch.inflow),
        "hydrology_state": np.asarray(batch.hydrology_state),
        "hydrology_regime": np.asarray(batch.hydrology_regime),
        "calendar": np.asarray(batch.calendar),
    }
    if batch.demand is not None:
        arrays["demand"] = np.asarray(batch.demand)
    if batch.source_path_id is not None:
        arrays["source_path_id"] = np.asarray(batch.source_path_id)
    for name, values in arrays.items():
        if values.dtype.hasobject:
            raise ValueError(
                f"Historical replay array {name} cannot have object dtype."
            )
    return arrays


def _readonly_trajectory_batch(batch: TrajectoryBatch) -> TrajectoryBatch:
    arrays = {
        name: np.array(values, copy=True)
        for name, values in _trajectory_arrays(batch).items()
    }
    for values in arrays.values():
        values.setflags(write=False)
    return TrajectoryBatch(
        name=str(batch.name),
        seed=int(batch.seed),
        inflow=arrays["inflow"],
        hydrology_state=arrays["hydrology_state"],
        hydrology_regime=arrays["hydrology_regime"],
        calendar=arrays["calendar"],
        demand=arrays.get("demand"),
        source_path_id=arrays.get("source_path_id"),
    )


def _validate_historical_replay(
    batch: TrajectoryBatch, system: ReservoirSystem
) -> None:
    if batch.horizon != system.horizon:
        raise ValueError(
            "Historical replay horizon differs from the reservoir system."
        )
    if batch.n_reservoirs != system.n_reservoirs:
        raise ValueError(
            "Historical replay width differs from the reservoir system."
        )
    expected_calendar = np.arange(system.horizon, dtype=np.int64) % system.periods
    if not np.array_equal(np.asarray(batch.calendar), expected_calendar):
        raise ValueError(
            "Historical replay calendar must align stage 0 with system period 0."
        )


def historical_replay_fingerprint(batch: TrajectoryBatch) -> str:
    """Fingerprint batch identity, optional fields, dtypes, shapes, and bytes."""

    arrays = _trajectory_arrays(batch)
    digest = sha256()
    identity = json.dumps(
        {
            "schema_version": HISTORICAL_REPLAY_SCHEMA_VERSION,
            "name": str(batch.name),
            "seed": int(batch.seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(identity)
    for name in _HISTORICAL_ARRAY_FIELDS:
        digest.update(b"\0field\0" + name.encode("ascii") + b"\0")
        if name not in arrays:
            digest.update(b"absent")
            continue
        values = np.ascontiguousarray(arrays[name])
        descriptor = json.dumps(
            {"dtype": values.dtype.str, "shape": list(values.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(descriptor)
        digest.update(b"\0data\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _historical_metadata(
    batch: TrajectoryBatch,
    *,
    fingerprint: str,
    arrays_file: str,
    arrays_sha256: str,
) -> dict[str, Any]:
    arrays = _trajectory_arrays(batch)
    return {
        "schema_version": HISTORICAL_REPLAY_SCHEMA_VERSION,
        "artifact_type": "real_historical_trajectory_batch",
        "name": str(batch.name),
        "seed": int(batch.seed),
        "n_paths": batch.n,
        "horizon": batch.horizon,
        "n_reservoirs": batch.n_reservoirs,
        "hydrology_state_dimension": int(batch.hydrology_state.shape[2]),
        "has_demand": batch.demand is not None,
        "has_source_path_id": batch.source_path_id is not None,
        "array_shapes": {
            name: list(values.shape) for name, values in arrays.items()
        },
        "array_dtypes": {
            name: values.dtype.str for name, values in arrays.items()
        },
        "arrays_file": arrays_file,
        "arrays_sha256": arrays_sha256,
        "fingerprint_sha256": fingerprint,
    }


def _prepared_relative_path(destination: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Historical replay metadata contains an invalid path.")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Historical replay artifact path must remain inside the case.")
    base = destination.resolve()
    resolved = (destination / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            "Historical replay artifact path escapes the prepared case."
        ) from exc
    return resolved


def _persist_historical_replay(
    destination: Path,
    batch: TrajectoryBatch,
    system: ReservoirSystem,
) -> tuple[TrajectoryBatch, dict[str, Any]]:
    replay = _readonly_trajectory_batch(batch)
    _validate_historical_replay(replay, system)
    fingerprint = historical_replay_fingerprint(replay)
    relative_arrays = (
        Path(HISTORICAL_REPLAY_DIRECTORY) / f"{fingerprint}.npz"
    )
    arrays_path = destination / relative_arrays
    _atomic_npz(arrays_path, **_trajectory_arrays(replay))
    arrays_sha = sha256_file(arrays_path)

    relative_metadata = (
        Path(HISTORICAL_REPLAY_DIRECTORY) / f"{fingerprint}.json"
    )
    metadata_path = destination / relative_metadata
    metadata = _historical_metadata(
        replay,
        fingerprint=fingerprint,
        arrays_file=relative_arrays.as_posix(),
        arrays_sha256=arrays_sha,
    )
    _atomic_json(metadata_path, metadata)
    link = {
        "schema_version": HISTORICAL_REPLAY_SCHEMA_VERSION,
        "fingerprint_sha256": fingerprint,
        "arrays_file": relative_arrays.as_posix(),
        "arrays_sha256": arrays_sha,
        "metadata_file": relative_metadata.as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "name": replay.name,
        "seed": replay.seed,
        "n_paths": replay.n,
        "horizon": replay.horizon,
        "has_demand": replay.demand is not None,
        "has_source_path_id": replay.source_path_id is not None,
    }
    return replay, link


def _load_historical_replay(
    destination: Path,
    link: object,
    system: ReservoirSystem,
) -> TrajectoryBatch:
    if not isinstance(link, Mapping):
        raise ValueError("Historical replay provenance link must be a mapping.")
    if link.get("schema_version") != HISTORICAL_REPLAY_SCHEMA_VERSION:
        raise ValueError("Historical replay provenance schema is unsupported.")
    metadata_path = _prepared_relative_path(destination, link.get("metadata_file"))
    if sha256_file(metadata_path) != link.get("metadata_sha256"):
        raise ValueError("Historical replay metadata SHA-256 is invalid.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != HISTORICAL_REPLAY_SCHEMA_VERSION:
        raise ValueError("Historical replay metadata schema is unsupported.")
    for key in ("fingerprint_sha256", "arrays_file", "arrays_sha256"):
        if metadata.get(key) != link.get(key):
            raise ValueError(f"Historical replay {key} metadata is inconsistent.")

    arrays_path = _prepared_relative_path(destination, metadata.get("arrays_file"))
    if sha256_file(arrays_path) != metadata.get("arrays_sha256"):
        raise ValueError("Historical replay array SHA-256 is invalid.")
    has_demand = bool(metadata.get("has_demand"))
    has_source_path_id = bool(metadata.get("has_source_path_id"))
    expected_fields = {
        "inflow",
        "hydrology_state",
        "hydrology_regime",
        "calendar",
    }
    if has_demand:
        expected_fields.add("demand")
    if has_source_path_id:
        expected_fields.add("source_path_id")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise ValueError("Historical replay NPZ fields are invalid.")
        arrays = {name: archive[name].copy() for name in archive.files}
    replay = TrajectoryBatch(
        name=str(metadata.get("name")),
        seed=int(metadata.get("seed")),
        inflow=arrays["inflow"],
        hydrology_state=arrays["hydrology_state"],
        hydrology_regime=arrays["hydrology_regime"],
        calendar=arrays["calendar"],
        demand=arrays.get("demand"),
        source_path_id=arrays.get("source_path_id"),
    )
    _validate_historical_replay(replay, system)
    actual_arrays = _trajectory_arrays(replay)
    expected_shapes = metadata.get("array_shapes")
    expected_dtypes = metadata.get("array_dtypes")
    if not isinstance(expected_shapes, Mapping) or not isinstance(
        expected_dtypes, Mapping
    ):
        raise ValueError("Historical replay array metadata is missing.")
    actual_shapes = {
        name: list(values.shape) for name, values in actual_arrays.items()
    }
    actual_dtypes = {
        name: values.dtype.str for name, values in actual_arrays.items()
    }
    if dict(expected_shapes) != actual_shapes or dict(expected_dtypes) != actual_dtypes:
        raise ValueError("Historical replay array metadata does not match its NPZ.")
    fingerprint = historical_replay_fingerprint(replay)
    if fingerprint != metadata.get("fingerprint_sha256"):
        raise ValueError("Historical replay content fingerprint is invalid.")
    if fingerprint != link.get("fingerprint_sha256"):
        raise ValueError("Historical replay provenance fingerprint is invalid.")
    return _readonly_trajectory_batch(replay)


def system_from_jsonable(payload: Mapping[str, Any]) -> ReservoirSystem:
    values = dict(payload)
    objective_values = dict(values["objective"])
    for key, item in tuple(objective_values.items()):
        if isinstance(item, list):
            objective_values[key] = tuple(
                tuple(row) if isinstance(row, list) else row for row in item
            )
    values["objective"] = ObjectiveConfig(**objective_values)
    for key in (
        "reservoir_ids",
        "storage_capacity",
        "release_capacity",
        "initial_storage",
        "routing",
        "metadata",
    ):
        if key in values:
            values[key] = tuple(
                tuple(row) if isinstance(row, list) else row for row in values[key]
            )
    return ReservoirSystem(**values)


def raw_manifest(paths: Iterable[str | Path], root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    records = []
    for item in sorted({Path(path).resolve() for path in paths}):
        if not item.is_file():
            raise FileNotFoundError(item)
        try:
            relative = item.relative_to(root_path)
        except ValueError:
            relative = item
        records.append(
            {
                "path": str(relative),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return records


def save_prepared_case(
    case_id: str,
    system: ReservoirSystem,
    fit_inflow: np.ndarray,
    fit_calendar: np.ndarray,
    *,
    annual_component: bool,
    raw_files: Iterable[str | Path],
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    max_order: int = 3,
    regime_count: int = 6,
    extra_provenance: Mapping[str, Any] | None = None,
    historical_replay: TrajectoryBatch | None = None,
) -> PreparedCase:
    """Fit the frozen PAR model and atomically write a reusable case bundle."""

    if case_id not in CASE_IDS:
        raise KeyError(f"Unknown case_id {case_id!r}; choose from {CASE_IDS}.")
    fit_inflow = np.asarray(fit_inflow, dtype=np.float64)
    fit_calendar = np.asarray(fit_calendar, dtype=np.int16)
    model = fit_periodic_ar(
        fit_inflow,
        fit_calendar,
        system.periods,
        transform="log1p",
        max_order=max_order,
        annual_component=annual_component,
        regime_count=regime_count,
    )
    destination = Path(processed_root) / case_id
    destination.mkdir(parents=True, exist_ok=True)
    manifest = raw_manifest(raw_files, raw_root)
    provenance: dict[str, Any] = {
        "case_id": case_id,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files": manifest,
        "fit_shape": list(fit_inflow.shape),
        "fit_calendar_periods": system.periods,
        "fit_fingerprint_sha256": model.training_fingerprint,
        "par_order": model.order,
        "par_max_order": max_order,
        "par_aic_by_order": list(model.aic_by_order),
        "par_annual_component": annual_component,
        "hydrology_regimes": regime_count,
        "historical_replay": None,
    }
    if extra_provenance:
        provenance.update(dict(extra_provenance))
    _atomic_json(destination / "system.json", system.as_jsonable())
    _atomic_npz(
        destination / "fit_data.npz",
        fit_inflow=fit_inflow,
        fit_calendar=fit_calendar,
    )
    model.save(destination / "stochastic_model")
    replay = None
    if historical_replay is not None:
        replay, replay_link = _persist_historical_replay(
            destination, historical_replay, system
        )
        provenance["historical_replay"] = replay_link
    _atomic_json(destination / "provenance.json", provenance)
    return PreparedCase(
        case_id=case_id,
        system=system,
        fit_inflow=fit_inflow,
        fit_calendar=fit_calendar,
        annual_component=annual_component,
        model=model,
        provenance=provenance,
        historical_replay=replay,
    )


def attach_historical_replay(
    case_id: str,
    historical_replay: TrajectoryBatch,
    *,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> PreparedCase:
    """Atomically attach a real historical batch to an existing prepared case.

    Content-addressed NPZ and metadata files are written first.  The provenance
    manifest is atomically replaced only after both artifacts are durable, so a
    failed attachment cannot expose a partial replay through the loader.
    """

    destination = Path(processed_root) / case_id
    current = load_prepared_case(case_id, processed_root=processed_root)
    _, replay_link = _persist_historical_replay(
        destination, historical_replay, current.system
    )
    replay_link["attached_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance = dict(current.provenance)
    provenance["historical_replay"] = replay_link
    _atomic_json(destination / "provenance.json", provenance)
    return load_prepared_case(case_id, processed_root=processed_root)


def load_prepared_case(
    case_id: str, processed_root: str | Path = DEFAULT_PROCESSED_ROOT
) -> PreparedCase:
    destination = Path(processed_root) / case_id
    system = system_from_jsonable(
        json.loads((destination / "system.json").read_text(encoding="utf-8"))
    )
    with np.load(destination / "fit_data.npz") as arrays:
        fit_inflow = arrays["fit_inflow"].copy()
        fit_calendar = arrays["fit_calendar"].copy()
    model = PeriodicARModel.load(destination / "stochastic_model")
    provenance = json.loads(
        (destination / "provenance.json").read_text(encoding="utf-8")
    )
    fingerprint_input = np.ascontiguousarray(fit_inflow).tobytes()
    fingerprint_input += np.ascontiguousarray(fit_calendar).tobytes()
    fingerprint = sha256(fingerprint_input).hexdigest()
    if fingerprint != model.training_fingerprint:
        raise ValueError(f"Prepared case {case_id} fit fingerprint is invalid.")
    if provenance.get("fit_fingerprint_sha256") != fingerprint:
        raise ValueError(f"Prepared case {case_id} provenance fingerprint is invalid.")
    historical_link = provenance.get("historical_replay")
    historical_replay = None
    if historical_link is not None:
        historical_replay = _load_historical_replay(
            destination, historical_link, system
        )
    return PreparedCase(
        case_id=case_id,
        system=system,
        fit_inflow=fit_inflow,
        fit_calendar=fit_calendar,
        annual_component=bool(provenance["par_annual_component"]),
        model=model,
        provenance=provenance,
        historical_replay=historical_replay,
    )


__all__ = [
    "CASE_IDS",
    "DEFAULT_PROCESSED_ROOT",
    "DEFAULT_RAW_ROOT",
    "HISTORICAL_REPLAY_SCHEMA_VERSION",
    "PreparedCase",
    "attach_historical_replay",
    "historical_replay_fingerprint",
    "load_prepared_case",
    "raw_manifest",
    "save_prepared_case",
    "sha256_file",
    "system_from_jsonable",
]
