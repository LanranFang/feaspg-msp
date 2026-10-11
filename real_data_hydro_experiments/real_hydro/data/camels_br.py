"""Pinned CAMELS-BR v1.2 acquisition and leakage-safe preprocessing.

This module deliberately stops at a real-hydrology, semi-synthetic reservoir
interface.  CAMELS-BR contains catchment discharge and aggregate intervention
attributes; it does not contain the plant-level physics required to identify a
real reservoir dispatch problem.

The production defaults are pinned to Zenodo record 15025488.  Tests can pass a
small local manifest, which keeps the parser and checksum paths fully offline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import md5
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import os
import re
import shutil
import sqlite3
import struct

import numpy as np
import pandas as pd


CAMELS_BR_RECORD_ID = 15025488
CAMELS_BR_RECORD_URL = f"https://zenodo.org/records/{CAMELS_BR_RECORD_ID}"
CAMELS_BR_API_URL = f"https://zenodo.org/api/records/{CAMELS_BR_RECORD_ID}"


@dataclass(frozen=True)
class RemoteFile:
    """A content-addressed file in the pinned Zenodo record."""

    filename: str
    md5: str
    size_bytes: int

    @property
    def url(self) -> str:
        return f"{CAMELS_BR_API_URL}/files/{self.filename}/content"


CAMELS_BR_FILES: Mapping[str, RemoteFile] = MappingProxyType(
    {
        "readme": RemoteFile(
            "CAMELS_BR_readme.txt",
            "fbb422eddabd323d1c7eb64452adb974",
            23_461,
        ),
        "attributes": RemoteFile(
            "01_CAMELS_BR_attributes.zip",
            "7de033200bdb25e4da9cf40addcd77aa",
            273_282,
        ),
        "selected_streamflow": RemoteFile(
            "03_CAMELS_BR_streamflow_selected_catchments.zip",
            "ea61398d3c30400ea8edcb3fcb0ac016",
            62_466_395,
        ),
        "catchment_boundaries": RemoteFile(
            "12_CAMELS_BR_catchment_boundaries.zip",
            "f4352f3eee96062b5deea7e0cdddd6ad",
            2_136_249,
        ),
        "gauge_locations": RemoteFile(
            "13_CAMELS_BR_gauge_location.zip",
            "f80fc4aeba4340c11264bf17b570da00",
            798_606,
        ),
    }
)

CORE_FILE_KEYS = ("attributes", "selected_streamflow")
SPATIAL_FILE_KEYS = ("catchment_boundaries", "gauge_locations")
DEFAULT_ACCEPTED_QC_FLAGS = (1, 2, 6)
SECONDS_PER_DAY = 86_400.0
CUBIC_METRES_PER_HM3 = 1_000_000.0
GPKG_APPLICATION_ID = 0x47504B47


class CamelsBRError(RuntimeError):
    """Base class for CAMELS-BR acquisition and validation errors."""


class ChecksumMismatch(CamelsBRError):
    """Raised when a cached or downloaded archive is not the pinned object."""


class DataValidationError(CamelsBRError):
    """Raised when an archive or tabular input violates the data contract."""


@dataclass(frozen=True)
class GeographicCRS:
    """Audited geographic coordinate reference system for a GeoPackage layer."""

    srs_id: int
    organization: str
    organization_coordsys_id: int
    definition: str
    axis_interpretation: str = "x=longitude, y=latitude"
    unit: str = "degree"
    datum: str = "WGS 84"


@dataclass(frozen=True)
class SpatialNestingResult:
    """Strict and boundary-ambiguous catchment containment relations.

    A directed pair is ``(upstream_gauge, downstream_gauge)``: the upstream
    gauge point lies in the downstream gauge's contributing-area polygon.
    ``nesting_pairs`` is the relation passed to :func:`select_non_nested_basins`.
    Under the default conservative policy it includes boundary cases as
    conflicts, while preserving them separately for audit.
    """

    requested_gauge_ids: tuple[str, ...]
    strict_pairs: tuple[tuple[str, str], ...]
    boundary_pairs: tuple[tuple[str, str], ...]
    nesting_pairs: tuple[tuple[str, str], ...]
    boundary_policy: str
    boundary_tolerance_degrees: float
    catchment_crs: GeographicCRS
    gauge_crs: GeographicCRS
    topology_validation: str
    repaired_gauge_ids: tuple[str, ...]
    max_repair_area_relative_change: float

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Allow direct use as the conflict iterable in basin selection."""

        return iter(self.nesting_pairs)

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "requested_gauge_ids": self.requested_gauge_ids,
                "strict_pair_count": len(self.strict_pairs),
                "boundary_pair_count": len(self.boundary_pairs),
                "nesting_pair_count": len(self.nesting_pairs),
                "boundary_policy": self.boundary_policy,
                "boundary_tolerance_degrees": self.boundary_tolerance_degrees,
                "catchment_srs_id": self.catchment_crs.srs_id,
                "gauge_srs_id": self.gauge_crs.srs_id,
                "crs_equivalence": "WGS84 geographic longitude/latitude degrees",
                "topology_validation": self.topology_validation,
                "repaired_gauge_ids": self.repaired_gauge_ids,
                "max_repair_area_relative_change": (
                    self.max_repair_area_relative_change
                ),
            }
        )


@dataclass(frozen=True)
class HistoricalAvailabilityAudit:
    """Outcome-blind monthly usability masks for post-fit history.

    ``usable`` has shape ``[basin, water_year, water_month]``.  It records only
    whether the daily observations passed the frozen QC/coverage/gap rules; no
    discharge magnitude is retained in this audit interface.
    """

    basin_ids: tuple[str, ...]
    water_years: tuple[int, ...]
    usable: np.ndarray
    table: pd.DataFrame
    criteria: Mapping[str, object]

    def __post_init__(self) -> None:
        usable = np.array(self.usable, dtype=bool, copy=True)
        expected = (len(self.basin_ids), len(self.water_years), 12)
        if usable.shape != expected:
            raise ValueError(f"usable must have shape {expected}, got {usable.shape}.")
        if tuple(_normalise_gauge_id(value) for value in self.basin_ids) != (
            self.basin_ids
        ):
            raise ValueError("Availability basin IDs must be normalized strings.")
        if len(self.basin_ids) != len(set(self.basin_ids)):
            raise ValueError("Availability audit repeats a basin ID.")
        if self.water_years and self.water_years != tuple(
            range(self.water_years[0], self.water_years[-1] + 1)
        ):
            raise ValueError("Availability water years must be contiguous and increasing.")
        if len(self.table) != len(self.basin_ids):
            raise ValueError("Availability table must have one row per basin.")
        if "gauge_id" not in self.table or tuple(
            self.table["gauge_id"].map(_normalise_gauge_id)
        ) != self.basin_ids:
            raise ValueError("Availability table gauge IDs must match basin_ids order.")
        usable.setflags(write=False)
        object.__setattr__(self, "usable", usable)
        object.__setattr__(self, "table", self.table.copy(deep=True))
        object.__setattr__(self, "criteria", MappingProxyType(dict(self.criteria)))

    def locked_mask(self, water_years: tuple[int, int]) -> np.ndarray:
        start, end = water_years
        indices = [
            index
            for index, water_year in enumerate(self.water_years)
            if start <= water_year <= end
        ]
        expected = end - start + 1
        if len(indices) != expected:
            raise ValueError(
                f"Availability audit does not cover every water year {start}-{end}."
            )
        result = np.array(self.usable[:, indices, :], copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class AvailabilitySelectionResult:
    """Non-nested basin choice and its outcome-blind availability objective."""

    selected: pd.DataFrame
    mode: str
    quality_only_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    joint_locked_complete_water_years: tuple[int, ...]
    joint_locked_48_month_window_starts: tuple[int, ...]
    quality_ranking_changed: bool
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if tuple(self.selected["gauge_id"].astype(str)) != self.selected_ids:
            raise ValueError("Selected table order does not match selected_ids.")
        object.__setattr__(self, "selected", self.selected.copy(deep=True))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


Ring = tuple[tuple[float, float], ...]
Polygon = tuple[Ring, ...]
MultiPolygon = tuple[Polygon, ...]


class _WKBReader:
    """Small strict reader for the POINT/POLYGON/MULTIPOLYGON subset in v1.2."""

    def __init__(self, data: bytes) -> None:
        self.data = memoryview(data)
        self.offset = 0

    def _take(self, size: int) -> memoryview:
        end = self.offset + size
        if end > len(self.data):
            raise DataValidationError("Truncated WKB geometry in CAMELS-BR GeoPackage.")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def byte_order(self) -> str:
        value = int.from_bytes(self._take(1), "little")
        if value == 0:
            return ">"
        if value == 1:
            return "<"
        raise DataValidationError(f"Invalid WKB byte-order marker: {value}.")

    def uint32(self, order: str) -> int:
        return int(struct.unpack(f"{order}I", self._take(4))[0])

    def coordinate(self, order: str) -> tuple[float, float]:
        x, y = struct.unpack(f"{order}dd", self._take(16))
        if not np.isfinite(x) or not np.isfinite(y):
            raise DataValidationError("GeoPackage geometry contains non-finite coordinates.")
        return float(x), float(y)


def _validate_ring(ring: Ring) -> None:
    if len(ring) < 4:
        raise DataValidationError("A catchment polygon ring has fewer than four vertices.")
    if not np.allclose(ring[0], ring[-1], rtol=0.0, atol=1.0e-12):
        raise DataValidationError("A catchment polygon ring is not closed.")
    signed_twice_area = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(ring[:-1], ring[1:], strict=True)
    )
    if not np.isfinite(signed_twice_area) or abs(signed_twice_area) <= 1.0e-14:
        raise DataValidationError("A catchment polygon ring has zero geographic area.")


def _read_wkb_geometry(reader: _WKBReader) -> tuple[str, object]:
    order = reader.byte_order()
    geometry_type = reader.uint32(order)
    if geometry_type == 1:
        return "POINT", reader.coordinate(order)
    if geometry_type == 3:
        rings = []
        for _ in range(reader.uint32(order)):
            ring = tuple(
                reader.coordinate(order) for _ in range(reader.uint32(order))
            )
            _validate_ring(ring)
            rings.append(ring)
        if not rings:
            raise DataValidationError("A catchment POLYGON contains no rings.")
        return "POLYGON", tuple(rings)
    if geometry_type == 6:
        polygons = []
        for _ in range(reader.uint32(order)):
            child_type, child = _read_wkb_geometry(reader)
            if child_type != "POLYGON":
                raise DataValidationError("A MULTIPOLYGON contains a non-POLYGON child.")
            polygons.append(child)
        if not polygons:
            raise DataValidationError("A catchment MULTIPOLYGON contains no polygons.")
        return "MULTIPOLYGON", tuple(polygons)
    raise DataValidationError(
        f"Unsupported WKB geometry type {geometry_type}; CAMELS-BR v1.2 should "
        "contain 2D POINT or MULTIPOLYGON features."
    )


def _parse_geopackage_geometry(
    value: object,
    *,
    expected_srs_id: int,
    expected_geometry_type: str,
) -> tuple[object, bytes, tuple[float, float, float, float] | None]:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise DataValidationError("GeoPackage feature has no binary geometry value.")
    payload = bytes(value)
    if len(payload) < 13 or payload[:2] != b"GP":
        raise DataValidationError("Invalid GeoPackage geometry magic/header.")
    if payload[2] != 0:
        raise DataValidationError(
            f"Unsupported GeoPackage geometry version {payload[2]}."
        )
    flags = payload[3]
    if flags & 0b1100_0000:
        raise DataValidationError("GeoPackage geometry uses nonzero reserved flags.")
    if flags & 0b0010_0000:
        raise DataValidationError("Extended GeoPackage geometries are not supported.")
    if flags & 0b0001_0000:
        raise DataValidationError("Empty GeoPackage geometry is not usable.")
    order = "<" if flags & 1 else ">"
    srs_id = int(struct.unpack(f"{order}i", payload[4:8])[0])
    if srs_id != expected_srs_id:
        raise DataValidationError(
            f"Geometry header SRS {srs_id} disagrees with layer SRS "
            f"{expected_srs_id}."
        )
    envelope_indicator = (flags >> 1) & 0b111
    envelope_dimensions = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}
    if envelope_indicator not in envelope_dimensions:
        raise DataValidationError(
            f"Unsupported GeoPackage envelope code {envelope_indicator}."
        )
    envelope_count = envelope_dimensions[envelope_indicator]
    wkb_offset = 8 + envelope_count * 8
    if len(payload) <= wkb_offset:
        raise DataValidationError("GeoPackage geometry has a truncated envelope.")
    envelope: tuple[float, float, float, float] | None = None
    if envelope_count:
        values = struct.unpack(
            f"{order}{envelope_count}d", payload[8:wkb_offset]
        )
        if np.any(~np.isfinite(values)):
            raise DataValidationError("GeoPackage geometry envelope is non-finite.")
        envelope = tuple(float(item) for item in values[:4])

    wkb = payload[wkb_offset:]
    reader = _WKBReader(wkb)
    geometry_type, geometry = _read_wkb_geometry(reader)
    if reader.offset != len(reader.data):
        raise DataValidationError("GeoPackage WKB geometry has trailing bytes.")
    if geometry_type != expected_geometry_type:
        raise DataValidationError(
            f"Expected {expected_geometry_type}, found {geometry_type} geometry."
        )
    return geometry, wkb, envelope


def _multipolygon_bounds(polygons: MultiPolygon) -> tuple[float, float, float, float]:
    coordinates = [point for polygon in polygons for ring in polygon for point in ring]
    x = np.asarray([point[0] for point in coordinates], dtype=np.float64)
    y = np.asarray([point[1] for point in coordinates], dtype=np.float64)
    bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    if bounds[0] < -180.0 or bounds[1] > 180.0:
        raise DataValidationError("Catchment longitude lies outside [-180, 180].")
    if bounds[2] < -90.0 or bounds[3] > 90.0:
        raise DataValidationError("Catchment latitude lies outside [-90, 90].")
    return bounds


def _validate_envelope(
    envelope: tuple[float, float, float, float] | None,
    bounds: tuple[float, float, float, float],
) -> None:
    if envelope is not None and not np.allclose(
        envelope, bounds, rtol=0.0, atol=1.0e-10
    ):
        raise DataValidationError("GeoPackage envelope disagrees with geometry bounds.")


def _shapely_multipolygon_coordinates(geometry: object) -> MultiPolygon:
    geometry_type = str(getattr(geometry, "geom_type", ""))
    if geometry_type == "Polygon":
        polygon_geometries = (geometry,)
    elif geometry_type == "MultiPolygon":
        polygon_geometries = tuple(geometry.geoms)
    else:
        raise DataValidationError(
            f"Shapely repair produced unsupported {geometry_type} geometry."
        )
    polygons: list[Polygon] = []
    for polygon_geometry in polygon_geometries:
        rings = [
            tuple((float(x), float(y)) for x, y in polygon_geometry.exterior.coords)
        ]
        rings.extend(
            tuple((float(x), float(y)) for x, y in interior.coords)
            for interior in polygon_geometry.interiors
        )
        for ring in rings:
            _validate_ring(ring)
        polygons.append(tuple(rings))
    return tuple(polygons)


def _optional_shapely_validation(
    wkb: bytes, parsed: MultiPolygon
) -> tuple[MultiPolygon, bool, bool, float]:
    """Run a full topology check when Shapely is available.

    The official archives are GeoPackages rather than shapefiles, so pyshp is
    not applicable.  Geometry decoding remains dependency-free; Shapely adds a
    stricter self-intersection/topology check in production environments.
    """

    try:
        try:
            from shapely import from_wkb

            geometry = from_wkb(wkb)
        except ImportError:
            from shapely.wkb import loads

            geometry = loads(wkb)
    except ImportError:
        return parsed, False, False, 0.0
    if geometry.is_empty:
        raise DataValidationError("Catchment geometry is empty under Shapely.")
    if geometry.is_valid:
        return parsed, True, False, 0.0

    try:
        try:
            from shapely import make_valid
        except ImportError:
            from shapely.validation import make_valid
        repaired = make_valid(geometry)
    except ImportError as exc:
        raise DataValidationError(
            "Invalid catchment geometry requires Shapely make_valid support."
        ) from exc
    if repaired.is_empty or not repaired.is_valid:
        raise DataValidationError("Shapely could not repair an invalid catchment.")
    original_area = float(geometry.area)
    repaired_area = float(repaired.area)
    relative_change = abs(repaired_area - original_area) / max(original_area, 1.0e-30)
    if not np.isfinite(relative_change) or relative_change > 1.0e-8:
        raise DataValidationError(
            "Catchment topology repair changes geographic area by more than 1e-8."
        )
    return (
        _shapely_multipolygon_coordinates(repaired),
        True,
        True,
        relative_change,
    )


def _point_on_segment(
    point: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
    tolerance: float,
) -> bool:
    px, py = point
    ax, ay = left
    bx, by = right
    dx = bx - ax
    dy = by - ay
    length = float(np.hypot(dx, dy))
    if length <= tolerance:
        return float(np.hypot(px - ax, py - ay)) <= tolerance
    perpendicular_distance = abs(dx * (ay - py) - (ax - px) * dy) / length
    return (
        perpendicular_distance <= tolerance
        and min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance
        and min(ay, by) - tolerance <= py <= max(ay, by) + tolerance
    )


def _ring_location(
    point: tuple[float, float], ring: Ring, tolerance: float
) -> int:
    """Return -1 outside, 0 on boundary, and 1 inside a closed ring."""

    x, y = point
    inside = False
    for left, right in zip(ring[:-1], ring[1:], strict=True):
        if _point_on_segment(point, left, right, tolerance):
            return 0
        ax, ay = left
        bx, by = right
        crosses = (ay > y) != (by > y)
        if crosses:
            intersection_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if intersection_x > x:
                inside = not inside
    return 1 if inside else -1


def _multipolygon_location(
    point: tuple[float, float], polygons: MultiPolygon, tolerance: float
) -> int:
    location = -1
    for polygon in polygons:
        exterior = _ring_location(point, polygon[0], tolerance)
        if exterior == 0:
            return 0
        if exterior < 0:
            continue
        in_polygon = True
        for hole in polygon[1:]:
            hole_location = _ring_location(point, hole, tolerance)
            if hole_location == 0:
                return 0
            if hole_location > 0:
                in_polygon = False
                break
        if in_polygon:
            location = 1
    return location


def _read_geographic_crs(
    connection: sqlite3.Connection,
    *,
    srs_id: int,
    expected_organization: str,
    expected_organization_id: int,
) -> GeographicCRS:
    rows = connection.execute(
        "SELECT organization, organization_coordsys_id, definition "
        "FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (srs_id,),
    ).fetchall()
    if len(rows) != 1:
        raise DataValidationError(f"GeoPackage has {len(rows)} definitions for SRS {srs_id}.")
    organization, organization_id, definition = rows[0]
    if (
        str(organization).upper() != expected_organization.upper()
        or int(organization_id) != expected_organization_id
    ):
        raise DataValidationError(
            f"Unexpected authority for SRS {srs_id}: "
            f"{organization}:{organization_id}."
        )
    normalized = re.sub(r"\s+", "", str(definition)).upper()
    required_tokens = (
        "GEOGCS[",
        "WGS_1984",
        'SPHEROID["WGS84",6378137,298.257223563',
        'UNIT["DEGREE",0.0174532925199433',
    )
    if any(token not in normalized for token in required_tokens):
        raise DataValidationError(
            f"SRS {srs_id} is not the expected WGS84 geographic-degree CRS."
        )
    return GeographicCRS(
        srs_id=srs_id,
        organization=str(organization),
        organization_coordsys_id=int(organization_id),
        definition=str(definition),
    )


def _validate_feature_layer(
    connection: sqlite3.Connection,
    *,
    table: str,
    expected_geometry_type: str,
    expected_srs_id: int,
    expected_organization: str,
    expected_organization_id: int,
) -> tuple[str, GeographicCRS]:
    if int(connection.execute("PRAGMA application_id").fetchone()[0]) != GPKG_APPLICATION_ID:
        raise DataValidationError("Spatial archive member is not a GeoPackage database.")
    integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    if integrity.lower() != "ok":
        raise DataValidationError(f"GeoPackage integrity check failed: {integrity}")
    contents = connection.execute(
        "SELECT data_type, srs_id, min_x, min_y, max_x, max_y "
        "FROM gpkg_contents WHERE table_name = ?",
        (table,),
    ).fetchall()
    if len(contents) != 1:
        raise DataValidationError(f"GeoPackage does not define exactly one {table} layer.")
    data_type, content_srs, min_x, min_y, max_x, max_y = contents[0]
    if data_type != "features" or int(content_srs) != expected_srs_id:
        raise DataValidationError(
            f"Layer {table} has unexpected type/SRS: {data_type}/{content_srs}."
        )
    extent = np.asarray((min_x, min_y, max_x, max_y), dtype=np.float64)
    if (
        np.any(~np.isfinite(extent))
        or not -180.0 <= extent[0] <= extent[2] <= 180.0
        or not -90.0 <= extent[1] <= extent[3] <= 90.0
    ):
        raise DataValidationError(f"Layer {table} has an invalid lon/lat extent.")
    geometry_rows = connection.execute(
        "SELECT column_name, geometry_type_name, srs_id, z, m "
        "FROM gpkg_geometry_columns WHERE table_name = ?",
        (table,),
    ).fetchall()
    if len(geometry_rows) != 1:
        raise DataValidationError(
            f"GeoPackage does not define exactly one geometry column for {table}."
        )
    geometry_column, geometry_type, geometry_srs, z, m = geometry_rows[0]
    if (
        str(geometry_type).upper() != expected_geometry_type
        or int(geometry_srs) != expected_srs_id
        or int(z) != 0
        or int(m) != 0
    ):
        raise DataValidationError(
            f"Layer {table} must contain 2D {expected_geometry_type} in SRS "
            f"{expected_srs_id}."
        )
    crs = _read_geographic_crs(
        connection,
        srs_id=expected_srs_id,
        expected_organization=expected_organization,
        expected_organization_id=expected_organization_id,
    )
    return str(geometry_column), crs


def _normalise_gauge_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise DataValidationError(f"Invalid CAMELS-BR gauge_id: {value!r}")
    return text.zfill(8)


def _file_md5(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = md5()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class CamelsBRSource:
    """Access verified CAMELS-BR archives without extracting them.

    Parameters
    ----------
    cache_dir:
        Directory holding the immutable downloaded archives.
    manifest:
        Defaults to the official v1.2 manifest.  Dependency injection is
        supported so unit tests can create tiny, checksummed local archives.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: Mapping[str, RemoteFile] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = dict(CAMELS_BR_FILES if manifest is None else manifest)
        missing = set(CORE_FILE_KEYS).difference(self.manifest)
        if missing:
            raise ValueError(f"Manifest is missing core files: {sorted(missing)}")

    def cached_path(self, key: str) -> Path:
        try:
            spec = self.manifest[key]
        except KeyError as exc:
            raise KeyError(f"Unknown CAMELS-BR file key: {key}") from exc
        return self.cache_dir / spec.filename

    def verify_file(self, key: str, path: str | Path | None = None) -> Path:
        spec = self.manifest[key]
        candidate = self.cached_path(key) if path is None else Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        actual_size = candidate.stat().st_size
        actual_md5 = _file_md5(candidate)
        if actual_size != spec.size_bytes or actual_md5 != spec.md5:
            raise ChecksumMismatch(
                f"{candidate} is not the pinned CAMELS-BR object: "
                f"expected size={spec.size_bytes}, md5={spec.md5}; "
                f"got size={actual_size}, md5={actual_md5}."
            )
        return candidate

    def ensure_file(
        self,
        key: str,
        *,
        allow_download: bool = True,
        timeout: float = 120.0,
    ) -> Path:
        """Return a verified cache path, atomically downloading when allowed."""

        target = self.cached_path(key)
        if target.exists():
            return self.verify_file(key, target)
        if not allow_download:
            raise FileNotFoundError(
                f"Missing verified CAMELS-BR cache file {target}; network access "
                "was disabled for this call."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        spec = self.manifest[key]
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                prefix=f".{spec.filename}.", suffix=".part", dir=self.cache_dir,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                request = Request(spec.url, headers={"User-Agent": "real-hydro/1.0"})
                with urlopen(request, timeout=timeout) as response:
                    shutil.copyfileobj(response, output)
            self.verify_file(key, temporary)
            os.replace(temporary, target)
            temporary = None
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def ensure_core_files(
        self, *, allow_download: bool = True, timeout: float = 120.0
    ) -> dict[str, Path]:
        return {
            key: self.ensure_file(
                key, allow_download=allow_download, timeout=timeout
            )
            for key in CORE_FILE_KEYS
        }

    def ensure_spatial_files(
        self, *, allow_download: bool = True, timeout: float = 120.0
    ) -> dict[str, Path]:
        """Return the two verified GeoPackage ZIPs used for nesting checks."""

        missing = set(SPATIAL_FILE_KEYS).difference(self.manifest)
        if missing:
            raise DataValidationError(
                f"Manifest is missing spatial files: {sorted(missing)}"
            )
        return {
            key: self.ensure_file(
                key, allow_download=allow_download, timeout=timeout
            )
            for key in SPATIAL_FILE_KEYS
        }

    @contextmanager
    def _open_zipped_geopackage(
        self,
        key: str,
        member_basename: str,
        *,
        allow_download: bool,
    ) -> Iterator[sqlite3.Connection]:
        archive = self.ensure_file(key, allow_download=allow_download)
        try:
            with ZipFile(archive) as bundle:
                matches = [
                    member
                    for member in bundle.namelist()
                    if Path(member).name == member_basename
                ]
                if len(matches) != 1:
                    raise DataValidationError(
                        f"Expected one {member_basename} in {archive}; "
                        f"found {len(matches)}."
                    )
                with NamedTemporaryFile(suffix=".gpkg") as temporary:
                    with bundle.open(matches[0]) as source:
                        shutil.copyfileobj(source, temporary)
                    temporary.flush()
                    connection = sqlite3.connect(
                        f"file:{temporary.name}?mode=ro", uri=True
                    )
                    try:
                        yield connection
                    finally:
                        connection.close()
        except BadZipFile as exc:
            raise DataValidationError(f"Invalid spatial ZIP: {archive}") from exc

    @staticmethod
    def _requested_gauge_ids(
        gauge_ids: Iterable[str | int] | None,
    ) -> tuple[str, ...] | None:
        if gauge_ids is None:
            return None
        identifiers = tuple(_normalise_gauge_id(value) for value in gauge_ids)
        if not identifiers:
            raise ValueError("gauge_ids cannot be empty.")
        if len(identifiers) != len(set(identifiers)):
            raise DataValidationError("gauge_ids contains duplicates.")
        return identifiers

    def _load_catchment_boundaries(
        self,
        gauge_ids: Iterable[str | int] | None,
        *,
        allow_download: bool,
    ) -> tuple[pd.DataFrame, GeographicCRS, str]:
        requested = self._requested_gauge_ids(gauge_ids)
        requested_set = None if requested is None else set(requested)
        rows: list[dict[str, object]] = []
        with self._open_zipped_geopackage(
            "catchment_boundaries",
            "camels_br_catchments.gpkg",
            allow_download=allow_download,
        ) as connection:
            geometry_column, crs = _validate_feature_layer(
                connection,
                table="camels_br_catchments",
                expected_geometry_type="MULTIPOLYGON",
                expected_srs_id=4326,
                expected_organization="EPSG",
                expected_organization_id=4326,
            )
            if geometry_column != "geom":
                raise DataValidationError(
                    "CAMELS-BR catchment layer geometry column must be geom."
                )
            query = (
                "SELECT gauge_id, gauge_lon, gauge_lat, geom "
                "FROM camels_br_catchments"
            )
            for raw_id, raw_lon, raw_lat, geometry_blob in connection.execute(query):
                identifier = _normalise_gauge_id(raw_id)
                if requested_set is not None and identifier not in requested_set:
                    continue
                longitude = float(raw_lon)
                latitude = float(raw_lat)
                if (
                    not np.isfinite(longitude)
                    or not np.isfinite(latitude)
                    or not -180.0 <= longitude <= 180.0
                    or not -90.0 <= latitude <= 90.0
                ):
                    raise DataValidationError(
                        f"Catchment {identifier} has invalid gauge coordinates."
                    )
                parsed, wkb, envelope = _parse_geopackage_geometry(
                    geometry_blob,
                    expected_srs_id=crs.srs_id,
                    expected_geometry_type="MULTIPOLYGON",
                )
                polygons = parsed
                bounds = _multipolygon_bounds(polygons)
                _validate_envelope(envelope, bounds)
                (
                    polygons,
                    shapely_validated,
                    shapely_repaired,
                    repair_area_relative_change,
                ) = _optional_shapely_validation(wkb, polygons)
                bounds = _multipolygon_bounds(polygons)
                rows.append(
                    {
                        "gauge_id": identifier,
                        "gauge_lon": longitude,
                        "gauge_lat": latitude,
                        "geometry": polygons,
                        "bounds": bounds,
                        "shapely_validated": shapely_validated,
                        "shapely_repaired": shapely_repaired,
                        "repair_area_relative_change": repair_area_relative_change,
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise DataValidationError("No requested catchment boundaries were found.")
        if frame["gauge_id"].duplicated().any():
            raise DataValidationError("Catchment layer repeats a gauge_id.")
        found = set(frame["gauge_id"])
        if requested_set is not None and found != requested_set:
            raise DataValidationError(
                f"Catchment layer is missing gauge IDs: {sorted(requested_set - found)}"
            )
        if frame["shapely_validated"].all():
            repairs = int(frame["shapely_repaired"].sum())
            validation = (
                "structural WKB plus Shapely topology; "
                f"GEOS make_valid repairs={repairs}"
            )
        else:
            validation = "structural WKB (Shapely unavailable)"
        frame = frame.sort_values("gauge_id", kind="stable").reset_index(drop=True)
        frame.attrs.update(
            {
                "crs": crs,
                "axis_interpretation": crs.axis_interpretation,
                "topology_validation": validation,
            }
        )
        return frame, crs, validation

    def load_catchment_boundaries(
        self,
        gauge_ids: Iterable[str | int] | None = None,
        *,
        allow_download: bool = False,
    ) -> pd.DataFrame:
        """Parse selected WGS84 catchment multipolygons from the pinned ZIP."""

        frame, _, _ = self._load_catchment_boundaries(
            gauge_ids, allow_download=allow_download
        )
        return frame

    def _load_gauge_locations(
        self,
        gauge_ids: Iterable[str | int] | None,
        *,
        allow_download: bool,
        coordinate_tolerance_degrees: float,
    ) -> tuple[pd.DataFrame, GeographicCRS]:
        requested = self._requested_gauge_ids(gauge_ids)
        requested_set = None if requested is None else set(requested)
        rows: list[dict[str, object]] = []
        with self._open_zipped_geopackage(
            "gauge_locations",
            "location_gauges_streamflow.gpkg",
            allow_download=allow_download,
        ) as connection:
            geometry_column, crs = _validate_feature_layer(
                connection,
                table="location_gauges_streamflow",
                expected_geometry_type="POINT",
                expected_srs_id=100000,
                expected_organization="NONE",
                expected_organization_id=100000,
            )
            if geometry_column != "geom":
                raise DataValidationError(
                    "CAMELS-BR gauge-location geometry column must be geom."
                )
            query = (
                "SELECT gauge_id, gauge_lon, gauge_lat, geom "
                "FROM location_gauges_streamflow"
            )
            for raw_id, raw_lon, raw_lat, geometry_blob in connection.execute(query):
                identifier = _normalise_gauge_id(raw_id)
                if requested_set is not None and identifier not in requested_set:
                    continue
                longitude = float(raw_lon)
                latitude = float(raw_lat)
                parsed, _, envelope = _parse_geopackage_geometry(
                    geometry_blob,
                    expected_srs_id=crs.srs_id,
                    expected_geometry_type="POINT",
                )
                point = tuple(float(value) for value in parsed)
                if envelope is not None:
                    expected_envelope = (point[0], point[0], point[1], point[1])
                    _validate_envelope(envelope, expected_envelope)
                if (
                    not -180.0 <= point[0] <= 180.0
                    or not -90.0 <= point[1] <= 90.0
                ):
                    raise DataValidationError(
                        f"Gauge {identifier} point lies outside geographic bounds."
                    )
                if not np.allclose(
                    point,
                    (longitude, latitude),
                    rtol=0.0,
                    atol=coordinate_tolerance_degrees,
                ):
                    raise DataValidationError(
                        f"Gauge {identifier} POINT disagrees with gauge_lon/gauge_lat; "
                        "the CRS axis order cannot be verified."
                    )
                rows.append(
                    {
                        "gauge_id": identifier,
                        "gauge_lon": longitude,
                        "gauge_lat": latitude,
                        "point": point,
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise DataValidationError("No requested streamflow gauge locations were found.")
        if frame["gauge_id"].duplicated().any():
            raise DataValidationError("Gauge-location layer repeats a gauge_id.")
        found = set(frame["gauge_id"])
        if requested_set is not None and found != requested_set:
            raise DataValidationError(
                f"Gauge-location layer is missing IDs: {sorted(requested_set - found)}"
            )
        frame = frame.sort_values("gauge_id", kind="stable").reset_index(drop=True)
        frame.attrs.update(
            {"crs": crs, "axis_interpretation": crs.axis_interpretation}
        )
        return frame, crs

    def load_gauge_locations(
        self,
        gauge_ids: Iterable[str | int] | None = None,
        *,
        allow_download: bool = False,
        coordinate_tolerance_degrees: float = 1.0e-8,
    ) -> pd.DataFrame:
        """Parse streamflow gauge points and verify x/y against lon/lat fields."""

        if coordinate_tolerance_degrees < 0.0:
            raise ValueError("coordinate_tolerance_degrees must be nonnegative.")
        frame, _ = self._load_gauge_locations(
            gauge_ids,
            allow_download=allow_download,
            coordinate_tolerance_degrees=coordinate_tolerance_degrees,
        )
        return frame

    def derive_nesting_pairs(
        self,
        gauge_ids: Iterable[str | int],
        *,
        allow_download: bool = False,
        boundary_tolerance_degrees: float = 1.0e-5,
        coordinate_tolerance_degrees: float = 1.0e-8,
        boundary_policy: str = "conflict",
        require_shapely_validation: bool = True,
    ) -> SpatialNestingResult:
        """Derive directed nesting relations from polygons and gauge points.

        For catchment ``D``, any other requested gauge point ``U`` strictly
        inside ``D``'s polygon creates ``(U, D)``.  A point within the explicit
        boundary tolerance is separately audited.  The default treats such a
        pair as a conflict, which is conservative for non-nested selection.
        Formal selection requires Shapely so the minor self-intersections in
        the official v1.2 polygons are repaired by GEOS ``make_valid`` and
        audited before containment is evaluated.
        """

        if boundary_tolerance_degrees < 0.0:
            raise ValueError("boundary_tolerance_degrees must be nonnegative.")
        if coordinate_tolerance_degrees < 0.0:
            raise ValueError("coordinate_tolerance_degrees must be nonnegative.")
        if boundary_policy not in {"conflict", "error", "exclude"}:
            raise ValueError("boundary_policy must be conflict, error, or exclude.")
        identifiers = self._requested_gauge_ids(gauge_ids)
        assert identifiers is not None
        catchments, catchment_crs, topology_validation = (
            self._load_catchment_boundaries(
                identifiers, allow_download=allow_download
            )
        )
        gauges, gauge_crs = self._load_gauge_locations(
            identifiers,
            allow_download=allow_download,
            coordinate_tolerance_degrees=coordinate_tolerance_degrees,
        )
        if require_shapely_validation and not catchments["shapely_validated"].all():
            raise DataValidationError(
                "Shapely is required for full catchment topology validation but "
                "is not installed. Install shapely before the formal selection."
            )
        repaired_ids = tuple(
            catchments.loc[
                catchments["shapely_repaired"].astype(bool), "gauge_id"
            ].astype(str)
        )
        maximum_repair_change = float(
            catchments["repair_area_relative_change"].max()
        )

        catchment_coordinates = catchments.set_index("gauge_id")[[
            "gauge_lon",
            "gauge_lat",
        ]]
        gauge_coordinates = gauges.set_index("gauge_id")[[
            "gauge_lon",
            "gauge_lat",
        ]]
        if not np.allclose(
            catchment_coordinates.loc[list(identifiers)].to_numpy(dtype=float),
            gauge_coordinates.loc[list(identifiers)].to_numpy(dtype=float),
            rtol=0.0,
            atol=coordinate_tolerance_degrees,
        ):
            raise DataValidationError(
                "Gauge coordinates disagree between the boundary and location "
                "archives; CRS alignment cannot be established."
            )

        point_lookup = gauges.set_index("gauge_id")["point"].to_dict()
        strict: set[tuple[str, str]] = set()
        boundary: set[tuple[str, str]] = set()
        for row in catchments.itertuples(index=False):
            downstream = str(row.gauge_id)
            min_x, max_x, min_y, max_y = row.bounds
            for upstream in identifiers:
                if upstream == downstream:
                    continue
                point = point_lookup[upstream]
                if not (
                    min_x - boundary_tolerance_degrees
                    <= point[0]
                    <= max_x + boundary_tolerance_degrees
                    and min_y - boundary_tolerance_degrees
                    <= point[1]
                    <= max_y + boundary_tolerance_degrees
                ):
                    continue
                location = _multipolygon_location(
                    point, row.geometry, boundary_tolerance_degrees
                )
                pair = (upstream, downstream)
                if location > 0:
                    strict.add(pair)
                elif location == 0:
                    boundary.add(pair)

        strict_pairs = tuple(sorted(strict))
        boundary_pairs = tuple(sorted(boundary - strict))
        if boundary_pairs and boundary_policy == "error":
            raise DataValidationError(
                "Ambiguous gauge-on-catchment-boundary cases found: "
                f"{boundary_pairs[:10]}"
            )
        selected_pairs = set(strict_pairs)
        if boundary_policy == "conflict":
            selected_pairs.update(boundary_pairs)
        return SpatialNestingResult(
            requested_gauge_ids=identifiers,
            strict_pairs=strict_pairs,
            boundary_pairs=boundary_pairs,
            nesting_pairs=tuple(sorted(selected_pairs)),
            boundary_policy=boundary_policy,
            boundary_tolerance_degrees=boundary_tolerance_degrees,
            catchment_crs=catchment_crs,
            gauge_crs=gauge_crs,
            topology_validation=topology_validation,
            repaired_gauge_ids=repaired_ids,
            max_repair_area_relative_change=maximum_repair_change,
        )

    @staticmethod
    def _read_whitespace_table(handle: BinaryIO) -> pd.DataFrame:
        frame = pd.read_csv(handle, sep=r"\s+", na_values=("nan", "NaN"))
        if "gauge_id" in frame:
            frame["gauge_id"] = frame["gauge_id"].map(_normalise_gauge_id)
        return frame

    def load_attribute_tables(
        self, *, allow_download: bool = False
    ) -> dict[str, pd.DataFrame]:
        archive = self.ensure_file("attributes", allow_download=allow_download)
        tables: dict[str, pd.DataFrame] = {}
        try:
            with ZipFile(archive) as bundle:
                members = sorted(
                    member
                    for member in bundle.namelist()
                    if re.search(r"/camels_br_[^/]+\.txt$", member)
                )
                if not members:
                    raise DataValidationError(
                        f"No attribute text tables found in {archive}."
                    )
                for member in members:
                    stem = Path(member).stem
                    category = stem.removeprefix("camels_br_")
                    with bundle.open(member) as handle:
                        frame = self._read_whitespace_table(handle)
                    if "gauge_id" not in frame:
                        raise DataValidationError(
                            f"Attribute table {member} has no gauge_id column."
                        )
                    if frame["gauge_id"].duplicated().any():
                        raise DataValidationError(
                            f"Attribute table {member} has duplicate gauge IDs."
                        )
                    tables[category] = frame
        except BadZipFile as exc:
            raise DataValidationError(f"Invalid attribute ZIP: {archive}") from exc
        return tables

    def load_attributes(self, *, allow_download: bool = False) -> pd.DataFrame:
        """Inner-join all attribute groups on the selected catchment IDs.

        The location table contains more gauges than the selected 897 in v1.2;
        an inner join across attribute groups intentionally retains only IDs with
        the complete selected-catchment attribute record.
        """

        tables = self.load_attribute_tables(allow_download=allow_download)
        ordered = sorted(tables.items(), key=lambda item: (len(item[1]), item[0]))
        merged = ordered[0][1].copy()
        for category, table in ordered[1:]:
            overlapping = (set(merged.columns) & set(table.columns)) - {"gauge_id"}
            if overlapping:
                raise DataValidationError(
                    f"Attribute category {category} repeats columns: "
                    f"{sorted(overlapping)}"
                )
            merged = merged.merge(table, on="gauge_id", how="inner", validate="one_to_one")
        return merged.sort_values("gauge_id", kind="stable").reset_index(drop=True)

    def available_gauge_ids(self, *, allow_download: bool = False) -> tuple[str, ...]:
        archive = self.ensure_file(
            "selected_streamflow", allow_download=allow_download
        )
        pattern = re.compile(r"/(\d{8})_streamflow\.txt$")
        try:
            with ZipFile(archive) as bundle:
                identifiers = [
                    match.group(1)
                    for member in bundle.namelist()
                    if (match := pattern.search(member))
                ]
        except BadZipFile as exc:
            raise DataValidationError(f"Invalid streamflow ZIP: {archive}") from exc
        if len(identifiers) != len(set(identifiers)):
            raise DataValidationError("The streamflow ZIP repeats a gauge ID.")
        return tuple(sorted(identifiers))

    def load_streamflow(
        self, gauge_id: str | int, *, allow_download: bool = False
    ) -> pd.DataFrame:
        identifier = _normalise_gauge_id(gauge_id)
        archive = self.ensure_file(
            "selected_streamflow", allow_download=allow_download
        )
        suffix = f"/{identifier}_streamflow.txt"
        try:
            with ZipFile(archive) as bundle:
                matches = [name for name in bundle.namelist() if name.endswith(suffix)]
                if len(matches) != 1:
                    raise KeyError(
                        f"Expected one selected-streamflow member for {identifier}; "
                        f"found {len(matches)}."
                    )
                with bundle.open(matches[0]) as handle:
                    frame = pd.read_csv(
                        handle, sep=r"\s+", na_values=("nan", "NaN")
                    )
        except BadZipFile as exc:
            raise DataValidationError(f"Invalid streamflow ZIP: {archive}") from exc

        required = {
            "year",
            "month",
            "day",
            "streamflow_m3s",
            "streamflow_mm",
            "qual_control_by_ana",
            "qual_flag",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise DataValidationError(
                f"Streamflow member for {identifier} lacks columns: {sorted(missing)}"
            )
        date = pd.to_datetime(
            frame[["year", "month", "day"]].rename(
                columns={"year": "year", "month": "month", "day": "day"}
            ),
            errors="coerce",
        )
        if date.isna().any():
            raise DataValidationError(f"Gauge {identifier} contains invalid dates.")
        if date.duplicated().any():
            raise DataValidationError(f"Gauge {identifier} contains duplicate dates.")
        for column in required - {"year", "month", "day"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.insert(0, "gauge_id", identifier)
        frame.insert(1, "date", date)
        return frame.sort_values("date", kind="stable").reset_index(drop=True)

    def iter_streamflows(
        self,
        gauge_ids: Iterable[str | int] | None = None,
        *,
        allow_download: bool = False,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        identifiers = (
            self.available_gauge_ids(allow_download=allow_download)
            if gauge_ids is None
            else tuple(_normalise_gauge_id(value) for value in gauge_ids)
        )
        for identifier in identifiers:
            yield identifier, self.load_streamflow(
                identifier, allow_download=allow_download
            )


def derive_nesting_pairs(
    source: CamelsBRSource,
    gauge_ids: Iterable[str | int],
    *,
    allow_download: bool = False,
    boundary_tolerance_degrees: float = 1.0e-5,
    coordinate_tolerance_degrees: float = 1.0e-8,
    boundary_policy: str = "conflict",
    require_shapely_validation: bool = True,
) -> SpatialNestingResult:
    """Public functional interface for the pinned spatial nesting audit."""

    return source.derive_nesting_pairs(
        gauge_ids,
        allow_download=allow_download,
        boundary_tolerance_degrees=boundary_tolerance_degrees,
        coordinate_tolerance_degrees=coordinate_tolerance_degrees,
        boundary_policy=boundary_policy,
        require_shapely_validation=require_shapely_validation,
    )


def add_water_year_columns(
    frame: pd.DataFrame, *, date_column: str = "date"
) -> pd.DataFrame:
    """Add September--August water-year labels without mutating ``frame``."""

    if date_column not in frame:
        raise DataValidationError(f"Missing date column: {date_column}")
    result = frame.copy()
    dates = pd.to_datetime(result[date_column], errors="coerce")
    if dates.isna().any():
        raise DataValidationError("Cannot assign water years to invalid dates.")
    result["water_year"] = np.where(dates.dt.month >= 9, dates.dt.year + 1, dates.dt.year)
    result["water_month"] = ((dates.dt.month - 9) % 12) + 1
    return result


@dataclass(frozen=True)
class HistoricalWaterYearSplit:
    """Chronological, non-overlapping September--August history partitions."""

    fit: tuple[int, int] = (1981, 2005)
    generator_validation: tuple[int, int] = (2006, 2014)
    locked_real_test: tuple[int, int] = (2015, 2024)

    def __post_init__(self) -> None:
        windows = (self.fit, self.generator_validation, self.locked_real_test)
        if any(start > end for start, end in windows):
            raise ValueError("Every water-year window must have start <= end.")
        if not (
            self.fit[1] < self.generator_validation[0]
            and self.generator_validation[1] < self.locked_real_test[0]
        ):
            raise ValueError("Historical water-year windows must be ordered and disjoint.")


def water_year_split(
    frame: pd.DataFrame,
    split: HistoricalWaterYearSplit | None = None,
    *,
    date_column: str = "date",
) -> dict[str, pd.DataFrame]:
    """Return leakage-safe chronological history partitions."""

    definition = HistoricalWaterYearSplit() if split is None else split
    labelled = (
        frame.copy()
        if "water_year" in frame
        else add_water_year_columns(frame, date_column=date_column)
    )
    output: dict[str, pd.DataFrame] = {}
    for name, (start, end) in (
        ("fit", definition.fit),
        ("generator_validation", definition.generator_validation),
        ("locked_real_test", definition.locked_real_test),
    ):
        selected = labelled["water_year"].between(start, end, inclusive="both")
        output[name] = labelled.loc[selected].copy().reset_index(drop=True)
    return output


def accepted_streamflow_mask(
    streamflow: pd.DataFrame,
    *,
    accepted_qc_flags: Sequence[int] = DEFAULT_ACCEPTED_QC_FLAGS,
    require_ana_quality_control: bool = True,
) -> pd.Series:
    """Identify valid, nonnegative daily observations under the strict policy."""

    required = {"streamflow_m3s", "qual_flag", "qual_control_by_ana"}
    missing = required.difference(streamflow.columns)
    if missing:
        raise DataValidationError(f"Streamflow data lacks columns: {sorted(missing)}")
    discharge = pd.to_numeric(streamflow["streamflow_m3s"], errors="coerce")
    flags = pd.to_numeric(streamflow["qual_flag"], errors="coerce")
    mask = discharge.notna() & discharge.ge(0.0) & flags.isin(tuple(accepted_qc_flags))
    if require_ana_quality_control:
        controlled = pd.to_numeric(
            streamflow["qual_control_by_ana"], errors="coerce"
        )
        mask &= controlled.eq(1.0)
    return mask


def _consecutive_missing_run_lengths(values: pd.Series) -> pd.Series:
    missing = values.isna()
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    lengths = missing.groupby(groups).transform("sum")
    return lengths.where(missing, 0).astype(int)


def build_monthly_inflow(
    streamflow: pd.DataFrame,
    *,
    accepted_qc_flags: Sequence[int] = DEFAULT_ACCEPTED_QC_FLAGS,
    require_ana_quality_control: bool = True,
    min_valid_daily_fraction: float = 0.90,
    max_consecutive_missing_days: int = 3,
) -> pd.DataFrame:
    """Aggregate verified daily discharge into monthly inflow volume.

    A month is usable only when both its coverage threshold and its maximum
    consecutive-gap threshold pass.  For a usable month, the mean of its valid
    daily discharges is scaled by the number of calendar days.  No temporal
    interpolation is performed, and a long gap that crosses a month boundary
    invalidates every month touched by that gap.
    """

    if not 0.0 < min_valid_daily_fraction <= 1.0:
        raise ValueError("min_valid_daily_fraction must lie in (0, 1].")
    if max_consecutive_missing_days < 0:
        raise ValueError("max_consecutive_missing_days must be nonnegative.")
    if "date" not in streamflow:
        raise DataValidationError("Streamflow data needs a date column.")

    frame = streamflow.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise DataValidationError("Streamflow data contains invalid dates.")
    if frame["date"].duplicated().any():
        raise DataValidationError("Streamflow data contains duplicate dates.")
    if frame.empty:
        raise DataValidationError("Cannot aggregate an empty streamflow table.")

    identifiers = (
        frame["gauge_id"].dropna().map(_normalise_gauge_id).unique()
        if "gauge_id" in frame
        else np.asarray([], dtype=object)
    )
    if len(identifiers) > 1:
        raise DataValidationError("build_monthly_inflow accepts one gauge at a time.")
    identifier = str(identifiers[0]) if len(identifiers) else ""

    valid = accepted_streamflow_mask(
        frame,
        accepted_qc_flags=accepted_qc_flags,
        require_ana_quality_control=require_ana_quality_control,
    )
    verified = pd.to_numeric(frame["streamflow_m3s"], errors="coerce").where(valid)
    verified.index = pd.DatetimeIndex(frame["date"])
    verified = verified.sort_index()

    first_month = verified.index.min().to_period("M")
    last_month = verified.index.max().to_period("M")
    calendar = pd.date_range(
        first_month.start_time, last_month.end_time.normalize(), freq="D"
    )
    verified = verified.reindex(calendar)
    gap_lengths = _consecutive_missing_run_lengths(verified)

    rows: list[dict[str, object]] = []
    for period, values in verified.groupby(verified.index.to_period("M")):
        expected_days = int(period.days_in_month)
        valid_days = int(values.notna().sum())
        valid_fraction = valid_days / expected_days
        maximum_gap = int(gap_lengths.loc[values.index].max())
        usable = (
            valid_days > 0
            and valid_fraction >= min_valid_daily_fraction
            and maximum_gap <= max_consecutive_missing_days
        )
        inflow = (
            float(values.mean())
            * expected_days
            * SECONDS_PER_DAY
            / CUBIC_METRES_PER_HM3
            if usable
            else np.nan
        )
        month = period.start_time
        rows.append(
            {
                "gauge_id": identifier,
                "month": month,
                "inflow_hm3": inflow,
                "valid_days": valid_days,
                "expected_days": expected_days,
                "valid_fraction": valid_fraction,
                "max_consecutive_missing_days": maximum_gap,
                "water_year": month.year + 1 if month.month >= 9 else month.year,
                "water_month": ((month.month - 9) % 12) + 1,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BasinSelectionCriteria:
    """Pre-registered static and training-history screening thresholds."""

    area_quality: str = "high"
    max_area_relative_difference: float = 0.10
    min_metadata_quality_control_percent: float = 95.0
    min_metadata_stream_stage_percent: float = 90.0
    min_complete_water_years: int = 25
    min_train_month_coverage: float = 0.95
    max_regulation_degree_percent: float = 5.0
    max_consumptive_use_percent: float = 5.0
    accepted_qc_flags: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_ACCEPTED_QC_FLAGS
    )
    require_ana_quality_control: bool = True
    min_valid_daily_fraction_per_month: float = 0.90
    max_consecutive_missing_days: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_area_relative_difference <= 1.0:
            raise ValueError("max_area_relative_difference must lie in [0, 1].")
        if not 0.0 < self.min_train_month_coverage <= 1.0:
            raise ValueError("min_train_month_coverage must lie in (0, 1].")
        if self.min_complete_water_years < 1:
            raise ValueError("min_complete_water_years must be positive.")
        if not self.accepted_qc_flags:
            raise ValueError("accepted_qc_flags cannot be empty.")


def screen_static_attributes(
    attributes: pd.DataFrame,
    criteria: BasinSelectionCriteria | None = None,
) -> pd.DataFrame:
    """Audit area, metadata quality, and low-intervention requirements."""

    rule = BasinSelectionCriteria() if criteria is None else criteria
    required = {
        "gauge_id",
        "area_ana",
        "area_gsim",
        "area_gsim_quality",
        "q_quality_control_perc",
        "q_stream_stage_perc",
        "regulation_degree",
        "consumptive_use_perc",
    }
    missing = required.difference(attributes.columns)
    if missing:
        raise DataValidationError(f"Attributes lack columns: {sorted(missing)}")

    audit = attributes.copy()
    audit["gauge_id"] = audit["gauge_id"].map(_normalise_gauge_id)
    numeric_columns = (
        "area_ana",
        "area_gsim",
        "q_quality_control_perc",
        "q_stream_stage_perc",
        "regulation_degree",
        "consumptive_use_perc",
    )
    for column in numeric_columns:
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    area_ana = audit["area_ana"]
    area_gsim = audit["area_gsim"]
    audit["area_relative_difference"] = (area_ana - area_gsim).abs() / area_gsim

    reasons: list[str] = []
    eligibility: list[bool] = []
    for _, row in audit.iterrows():
        rejected: list[str] = []
        area_difference = row["area_relative_difference"]
        if (
            not np.isfinite(area_difference)
            or float(row["area_ana"]) <= 0.0
            or float(row["area_gsim"]) <= 0.0
            or area_difference > rule.max_area_relative_difference
        ):
            rejected.append("area")
        if str(row["area_gsim_quality"]).strip().lower() != rule.area_quality.lower():
            rejected.append("area_quality")
        qc_percent = row["q_quality_control_perc"]
        if not np.isfinite(qc_percent) or qc_percent < rule.min_metadata_quality_control_percent:
            rejected.append("metadata_qc")
        stage_percent = row["q_stream_stage_perc"]
        if (
            not np.isfinite(stage_percent)
            or stage_percent < rule.min_metadata_stream_stage_percent
        ):
            rejected.append("metadata_stage")
        regulation = row["regulation_degree"]
        if not np.isfinite(regulation) or regulation >= rule.max_regulation_degree_percent:
            rejected.append("regulation")
        consumption = row["consumptive_use_perc"]
        if not np.isfinite(consumption) or consumption >= rule.max_consumptive_use_percent:
            rejected.append("consumptive_use")
        eligibility.append(not rejected)
        reasons.append(";".join(rejected))
    audit["static_eligible"] = eligibility
    audit["static_rejection_reasons"] = reasons
    return audit


@dataclass(frozen=True)
class BasinSelectionResult:
    """Selected basins plus a complete, reproducible rejection audit."""

    eligible: pd.DataFrame
    audit: pd.DataFrame

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        return tuple(self.eligible["gauge_id"].astype(str))


def select_eligible_basins(
    source: CamelsBRSource,
    *,
    criteria: BasinSelectionCriteria | None = None,
    fit_water_years: tuple[int, int] = (1981, 2005),
    allow_download: bool = False,
) -> BasinSelectionResult:
    """Apply static rules and training-only complete-year screening."""

    rule = BasinSelectionCriteria() if criteria is None else criteria
    fit_start, fit_end = fit_water_years
    if fit_start > fit_end:
        raise ValueError("fit_water_years must be ordered.")
    expected_months = (fit_end - fit_start + 1) * 12

    static = screen_static_attributes(
        source.load_attributes(allow_download=allow_download), rule
    )
    available = set(source.available_gauge_ids(allow_download=allow_download))
    audit_rows: list[dict[str, object]] = []
    for _, attribute_row in static.iterrows():
        row = attribute_row.to_dict()
        identifier = str(row["gauge_id"])
        dynamic_rejections: list[str] = []
        complete_years = 0
        train_month_coverage = 0.0
        usable_train_months = 0

        if bool(row["static_eligible"]):
            if identifier not in available:
                dynamic_rejections.append("missing_streamflow")
            else:
                daily = source.load_streamflow(
                    identifier, allow_download=allow_download
                )
                monthly = build_monthly_inflow(
                    daily,
                    accepted_qc_flags=rule.accepted_qc_flags,
                    require_ana_quality_control=rule.require_ana_quality_control,
                    min_valid_daily_fraction=rule.min_valid_daily_fraction_per_month,
                    max_consecutive_missing_days=rule.max_consecutive_missing_days,
                )
                train = monthly[
                    monthly["water_year"].between(
                        fit_start, fit_end, inclusive="both"
                    )
                ]
                usable_train_months = int(train["inflow_hm3"].notna().sum())
                train_month_coverage = usable_train_months / expected_months
                counts = (
                    train.assign(usable=train["inflow_hm3"].notna())
                    .groupby("water_year", sort=False)["usable"]
                    .sum()
                )
                complete_years = int(counts.eq(12).sum())
                if complete_years < rule.min_complete_water_years:
                    dynamic_rejections.append("complete_water_years")
                if train_month_coverage < rule.min_train_month_coverage:
                    dynamic_rejections.append("train_month_coverage")

        row["fit_water_year_start"] = fit_start
        row["fit_water_year_end"] = fit_end
        row["usable_train_months"] = usable_train_months
        row["complete_water_years"] = complete_years
        row["train_month_coverage"] = train_month_coverage
        row["dynamic_rejection_reasons"] = ";".join(dynamic_rejections)
        row["eligible"] = bool(row["static_eligible"]) and not dynamic_rejections
        combined = [
            value
            for value in (
                str(row["static_rejection_reasons"]),
                str(row["dynamic_rejection_reasons"]),
            )
            if value
        ]
        row["rejection_reasons"] = ";".join(combined)
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows).sort_values("gauge_id", kind="stable")
    audit = audit.reset_index(drop=True)
    eligible = audit.loc[audit["eligible"]].copy().reset_index(drop=True)
    return BasinSelectionResult(eligible=eligible, audit=audit)


def audit_historical_availability(
    source: CamelsBRSource,
    candidates: pd.DataFrame | Iterable[str | int],
    *,
    criteria: BasinSelectionCriteria | None = None,
    water_years: tuple[int, int] = (2006, 2024),
    locked_real_test_water_years: tuple[int, int] = (2015, 2024),
    allow_download: bool = False,
) -> HistoricalAvailabilityAudit:
    """Build post-fit usability masks without retaining discharge magnitudes.

    Using held-out *availability* is permitted only as a frozen data-quality
    eligibility rule.  The same daily flags, ANA-control requirement, monthly
    coverage threshold, and maximum-gap rule used for fit screening are applied.
    No mean, quantile, extreme, or other flow value enters the returned audit.
    """

    rule = BasinSelectionCriteria() if criteria is None else criteria
    start, end = water_years
    locked_start, locked_end = locked_real_test_water_years
    if start > end:
        raise ValueError("water_years must be ordered.")
    if not start <= locked_start <= locked_end <= end:
        raise ValueError("Locked real-test years must lie inside water_years.")
    if isinstance(candidates, pd.DataFrame):
        if "gauge_id" not in candidates:
            raise DataValidationError("Candidates need a gauge_id column.")
        identifiers = tuple(
            _normalise_gauge_id(value) for value in candidates["gauge_id"]
        )
    else:
        identifiers = tuple(_normalise_gauge_id(value) for value in candidates)
    if not identifiers:
        raise ValueError("At least one availability candidate is required.")
    if len(identifiers) != len(set(identifiers)):
        raise DataValidationError("Availability candidates repeat a gauge_id.")

    years = tuple(range(start, end + 1))
    usable = np.zeros((len(identifiers), len(years), 12), dtype=bool)
    rows: list[dict[str, object]] = []
    locked_offset = locked_start - start
    locked_count = locked_end - locked_start + 1
    for basin_index, identifier in enumerate(identifiers):
        daily = source.load_streamflow(identifier, allow_download=allow_download)
        monthly = build_monthly_inflow(
            daily,
            accepted_qc_flags=rule.accepted_qc_flags,
            require_ana_quality_control=rule.require_ana_quality_control,
            min_valid_daily_fraction=rule.min_valid_daily_fraction_per_month,
            max_consecutive_missing_days=rule.max_consecutive_missing_days,
        )
        observed = monthly.set_index(["water_year", "water_month"])[
            "inflow_hm3"
        ].notna()
        for year_index, water_year in enumerate(years):
            for water_month in range(1, 13):
                usable[basin_index, year_index, water_month - 1] = bool(
                    observed.get((water_year, water_month), False)
                )
        complete_years = usable[basin_index].all(axis=1)
        locked = usable[
            basin_index,
            locked_offset : locked_offset + locked_count,
        ]
        complete_locked_years = locked.all(axis=1)
        complete_windows = tuple(
            locked_start + offset
            for offset in range(max(locked_count - 3, 0))
            if locked[offset : offset + 4].all()
        )
        rows.append(
            {
                "gauge_id": identifier,
                "availability_water_year_start": start,
                "availability_water_year_end": end,
                "usable_months": int(usable[basin_index].sum()),
                "availability_month_coverage": float(
                    usable[basin_index].mean()
                ),
                "availability_complete_water_years": int(complete_years.sum()),
                "full_period_available": bool(usable[basin_index].all()),
                "locked_usable_months": int(locked.sum()),
                "locked_complete_water_years": int(complete_locked_years.sum()),
                "locked_48_month_windows": len(complete_windows),
                "locked_48_month_window_starts": complete_windows,
            }
        )
    table = pd.DataFrame(rows)
    criteria_audit: dict[str, object] = {
        "outcome_blind": True,
        "retains_flow_magnitudes": False,
        "accepted_qc_flags": rule.accepted_qc_flags,
        "require_ana_quality_control": rule.require_ana_quality_control,
        "min_valid_daily_fraction_per_month": (
            rule.min_valid_daily_fraction_per_month
        ),
        "max_consecutive_missing_days": rule.max_consecutive_missing_days,
        "water_years": water_years,
        "locked_real_test_water_years": locked_real_test_water_years,
    }
    return HistoricalAvailabilityAudit(
        basin_ids=identifiers,
        water_years=years,
        usable=usable,
        table=table,
        criteria=criteria_audit,
    )


def select_non_nested_basins(
    candidates: pd.DataFrame,
    nesting_pairs: Iterable[tuple[str | int, str | int]],
    *,
    n_basins: int | None = None,
    score_columns: Sequence[str] = (
        "complete_water_years",
        "train_month_coverage",
        "q_quality_control_perc",
        "q_stream_stage_perc",
    ),
) -> pd.DataFrame:
    """Greedily choose high-quality basins with no containment conflicts.

    ``nesting_pairs`` is an iterable of ``(upstream, downstream)`` gauge IDs,
    normally produced by :func:`derive_nesting_pairs`.  The relation is treated
    as an undirected conflict: total discharge from nested gauges must not be
    used twice as independent local reservoir inflow.
    """

    if "gauge_id" not in candidates:
        raise DataValidationError("Candidates need a gauge_id column.")
    if n_basins is not None and n_basins < 1:
        raise ValueError("n_basins must be positive when supplied.")
    pool = candidates.copy()
    pool["gauge_id"] = pool["gauge_id"].map(_normalise_gauge_id)
    if "eligible" in pool:
        pool = pool.loc[pool["eligible"].astype(bool)].copy()
    present_scores = [column for column in score_columns if column in pool]
    pool = pool.sort_values(
        [*present_scores, "gauge_id"],
        ascending=[False] * len(present_scores) + [True],
        kind="stable",
    )

    conflicts: dict[str, set[str]] = {}
    for left_raw, right_raw in nesting_pairs:
        left = _normalise_gauge_id(left_raw)
        right = _normalise_gauge_id(right_raw)
        if left == right:
            raise DataValidationError(f"Self-nesting pair supplied for {left}.")
        conflicts.setdefault(left, set()).add(right)
        conflicts.setdefault(right, set()).add(left)

    selected: list[str] = []
    selected_set: set[str] = set()
    for identifier in pool["gauge_id"]:
        if conflicts.get(identifier, set()).isdisjoint(selected_set):
            selected.append(identifier)
            selected_set.add(identifier)
            if n_basins is not None and len(selected) == n_basins:
                break
    if n_basins is not None and len(selected) < n_basins:
        raise DataValidationError(
            f"Only {len(selected)} non-nested basins are available; "
            f"{n_basins} were requested."
        )
    order = {identifier: index for index, identifier in enumerate(selected)}
    chosen = pool.loc[pool["gauge_id"].isin(selected)].copy()
    chosen["selection_order"] = chosen["gauge_id"].map(order)
    return chosen.sort_values("selection_order", kind="stable").reset_index(drop=True)


def _quality_ordered_candidates(
    candidates: pd.DataFrame,
    score_columns: Sequence[str],
) -> pd.DataFrame:
    if "gauge_id" not in candidates:
        raise DataValidationError("Candidates need a gauge_id column.")
    pool = candidates.copy()
    pool["gauge_id"] = pool["gauge_id"].map(_normalise_gauge_id)
    if pool["gauge_id"].duplicated().any():
        raise DataValidationError("Candidates repeat a gauge_id.")
    if "eligible" in pool:
        pool = pool.loc[pool["eligible"].astype(bool)].copy()
    present_scores = [column for column in score_columns if column in pool]
    return pool.sort_values(
        [*present_scores, "gauge_id"],
        ascending=[False] * len(present_scores) + [True],
        kind="stable",
    ).reset_index(drop=True)


def _conflict_map(
    gauge_ids: Iterable[str],
    nesting_pairs: Iterable[tuple[str | int, str | int]],
) -> dict[str, set[str]]:
    identifiers = set(gauge_ids)
    conflicts = {identifier: set() for identifier in identifiers}
    for left_raw, right_raw in nesting_pairs:
        left = _normalise_gauge_id(left_raw)
        right = _normalise_gauge_id(right_raw)
        if left == right:
            raise DataValidationError(f"Self-nesting pair supplied for {left}.")
        if left in identifiers and right in identifiers:
            conflicts[left].add(right)
            conflicts[right].add(left)
    return conflicts


def _first_independent_group(
    ordered_ids: Sequence[str],
    conflicts: Mapping[str, set[str]],
    n_basins: int,
) -> tuple[str, ...] | None:
    """Exact include-first search; returns the best quality-rank tuple."""

    identifiers = tuple(ordered_ids)

    def visit(start: int, chosen: tuple[str, ...]) -> tuple[str, ...] | None:
        if len(chosen) == n_basins:
            return chosen
        needed = n_basins - len(chosen)
        if len(identifiers) - start < needed:
            return None
        chosen_set = set(chosen)
        last_start = len(identifiers) - needed
        for index in range(start, last_start + 1):
            candidate = identifiers[index]
            if conflicts[candidate].isdisjoint(chosen_set):
                result = visit(index + 1, (*chosen, candidate))
                if result is not None:
                    return result
        return None

    return visit(0, ())


def _masks_with_count(width: int, count: int) -> Iterator[int]:
    for positions in combinations(range(width), count):
        mask = 0
        for position in positions:
            mask |= 1 << position
        yield mask


def _boolean_row_codes(values: np.ndarray) -> np.ndarray:
    codes = np.zeros(values.shape[0], dtype=np.int64)
    for position in range(values.shape[1]):
        codes |= values[:, position].astype(np.int64) << position
    return codes


def select_non_nested_basins_with_availability(
    candidates: pd.DataFrame,
    nesting_pairs: Iterable[tuple[str | int, str | int]],
    availability: HistoricalAvailabilityAudit,
    *,
    n_basins: int = 12,
    locked_real_test_water_years: tuple[int, int] = (2015, 2024),
    score_columns: Sequence[str] = (
        "complete_water_years",
        "train_month_coverage",
        "q_quality_control_perc",
        "q_stream_stage_perc",
    ),
) -> AvailabilitySelectionResult:
    """Select a non-nested portfolio under a frozen availability-only rule.

    The preferred case requires every selected basin to have every month usable
    throughout the availability audit.  If no non-nested portfolio of the
    requested size meets that condition, an exact deterministic search uses the
    following lexicographic objective on the locked historical period:

    1. maximize joint water-year-aligned 48-month windows;
    2. maximize joint complete water years;
    3. retain the best original fit-period quality-rank tuple.

    Only Boolean QC/missingness masks enter this search.  Flow magnitudes remain
    unavailable to the selector and all physical/objective calibration remains
    fit-only.
    """

    if n_basins < 1:
        raise ValueError("n_basins must be positive.")
    pool = _quality_ordered_candidates(candidates, score_columns)
    if len(pool) < n_basins:
        raise DataValidationError(
            f"Only {len(pool)} eligible basins remain; {n_basins} are required."
        )
    ordered_ids = tuple(pool["gauge_id"])
    availability_ids = set(availability.basin_ids)
    if availability_ids != set(ordered_ids):
        raise DataValidationError(
            "Availability audit must cover exactly the eligible candidate IDs."
        )
    availability_index = {
        identifier: index for index, identifier in enumerate(availability.basin_ids)
    }
    usable = np.stack(
        [
            availability.usable[availability_index[identifier]]
            for identifier in ordered_ids
        ]
    )
    pair_tuple = tuple(nesting_pairs)
    conflicts = _conflict_map(ordered_ids, pair_tuple)
    quality_only_ids = _first_independent_group(
        ordered_ids, conflicts, n_basins
    )
    if quality_only_ids is None:
        raise DataValidationError(
            "No non-nested candidate portfolio of the requested size exists."
        )

    full_indices = np.flatnonzero(usable.all(axis=(1, 2)))
    full_ids = tuple(ordered_ids[index] for index in full_indices)
    selected_ids = _first_independent_group(full_ids, conflicts, n_basins)
    mode = "complete_availability_period"

    locked_start, locked_end = locked_real_test_water_years
    locked = availability.locked_mask(locked_real_test_water_years)
    locked = np.stack(
        [locked[availability_index[identifier]] for identifier in ordered_ids]
    )
    n_locked_years = locked_end - locked_start + 1
    if n_locked_years < 4:
        raise ValueError("Locked real-test period must contain at least four years.")

    if selected_ids is None:
        mode = "maximize_joint_locked_availability"
        complete_year = locked.all(axis=2)
        complete_window = np.stack(
            [
                locked[:, offset : offset + 4, :].all(axis=(1, 2))
                for offset in range(n_locked_years - 3)
            ],
            axis=1,
        )
        year_codes = _boolean_row_codes(complete_year)
        window_codes = _boolean_row_codes(complete_window)
        feasible_window_targets: list[int] = []
        for window_count in range(complete_window.shape[1], -1, -1):
            for target in _masks_with_count(complete_window.shape[1], window_count):
                supported = [
                    ordered_ids[index]
                    for index, code in enumerate(window_codes)
                    if int(code) & target == target
                ]
                if _first_independent_group(supported, conflicts, n_basins) is not None:
                    feasible_window_targets.append(target)
            if feasible_window_targets:
                break

        rank = {identifier: index for index, identifier in enumerate(ordered_ids)}
        solutions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        for year_count in range(n_locked_years, -1, -1):
            for year_target in _masks_with_count(n_locked_years, year_count):
                for window_target in feasible_window_targets:
                    supported = [
                        ordered_ids[index]
                        for index, (year_code, window_code) in enumerate(
                            zip(year_codes, window_codes, strict=True)
                        )
                        if int(year_code) & year_target == year_target
                        and int(window_code) & window_target == window_target
                    ]
                    group = _first_independent_group(
                        supported, conflicts, n_basins
                    )
                    if group is not None:
                        solutions.append(
                            (tuple(rank[identifier] for identifier in group), group)
                        )
            if solutions:
                break
        if not solutions:
            raise DataValidationError(
                "No non-nested availability portfolio of the requested size exists."
            )
        selected_ids = min(solutions, key=lambda item: item[0])[1]

    selected_indices = [ordered_ids.index(identifier) for identifier in selected_ids]
    joint_locked = locked[selected_indices].all(axis=0)
    joint_complete_year = joint_locked.all(axis=1)
    joint_complete_years = tuple(
        locked_start + index
        for index, complete in enumerate(joint_complete_year)
        if complete
    )
    joint_windows = tuple(
        locked_start + offset
        for offset in range(n_locked_years - 3)
        if joint_locked[offset : offset + 4].all()
    )
    order = {identifier: index for index, identifier in enumerate(selected_ids)}
    selected = pool.loc[pool["gauge_id"].isin(selected_ids)].copy()
    selected["selection_order"] = selected["gauge_id"].map(order)
    selected = selected.sort_values("selection_order", kind="stable")
    selected = selected.merge(
        availability.table,
        on="gauge_id",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    audit: dict[str, object] = {
        "outcome_blind_availability_only": True,
        "flow_magnitudes_used_for_selection": False,
        "candidate_count": len(pool),
        "full_availability_candidate_count": len(full_ids),
        "selection_mode": mode,
        "objective_priority": (
            "joint locked 48-month windows",
            "joint locked complete water years",
            "fit-period quality rank",
            "gauge_id",
        ),
        "locked_real_test_water_years": locked_real_test_water_years,
        "joint_locked_complete_water_years": joint_complete_years,
        "joint_locked_48_month_window_starts": joint_windows,
        "quality_only_ids": quality_only_ids,
        "selected_ids": selected_ids,
        "quality_ranking_changed": selected_ids != quality_only_ids,
    }
    return AvailabilitySelectionResult(
        selected=selected,
        mode=mode,
        quality_only_ids=quality_only_ids,
        selected_ids=selected_ids,
        joint_locked_complete_water_years=joint_complete_years,
        joint_locked_48_month_window_starts=joint_windows,
        quality_ranking_changed=selected_ids != quality_only_ids,
        audit=audit,
    )


def calibrate_semisynthetic_reservoirs(
    monthly_inflow: pd.DataFrame,
    *,
    fit_water_years: tuple[int, int] = (1981, 2005),
    storage_months: float = 6.0,
    release_mean_multiplier: float = 1.5,
    initial_storage_fraction: float = 0.64,
    terminal_target_fraction: float = 0.62,
    min_fit_months: int = 12,
) -> pd.DataFrame:
    """Calibrate stylized reservoir sizes from fit-period inflow only."""

    required = {"gauge_id", "inflow_hm3"}
    missing = required.difference(monthly_inflow.columns)
    if missing:
        raise DataValidationError(f"Monthly inflow lacks columns: {sorted(missing)}")
    if storage_months <= 0.0 or release_mean_multiplier <= 0.0:
        raise ValueError("Capacity multipliers must be positive.")
    if not 0.0 <= initial_storage_fraction <= 1.0:
        raise ValueError("initial_storage_fraction must lie in [0, 1].")
    if not 0.0 <= terminal_target_fraction <= 1.0:
        raise ValueError("terminal_target_fraction must lie in [0, 1].")
    if min_fit_months < 1:
        raise ValueError("min_fit_months must be positive.")
    fit_start, fit_end = fit_water_years
    if fit_start > fit_end:
        raise ValueError("fit_water_years must be ordered.")

    data = monthly_inflow.copy()
    data["gauge_id"] = data["gauge_id"].map(_normalise_gauge_id)
    if "water_year" not in data:
        if "month" not in data:
            raise DataValidationError(
                "Monthly inflow needs water_year or month for leakage-safe calibration."
            )
        dates = pd.to_datetime(data["month"], errors="coerce")
        if dates.isna().any():
            raise DataValidationError("Monthly inflow contains invalid month values.")
        data["water_year"] = np.where(
            dates.dt.month >= 9, dates.dt.year + 1, dates.dt.year
        )
    fit = data.loc[
        data["water_year"].between(fit_start, fit_end, inclusive="both")
    ].copy()
    fit["inflow_hm3"] = pd.to_numeric(fit["inflow_hm3"], errors="coerce")
    fit = fit.loc[np.isfinite(fit["inflow_hm3"]) & fit["inflow_hm3"].ge(0.0)]

    rows: list[dict[str, object]] = []
    all_identifiers = tuple(sorted(data["gauge_id"].unique()))
    for identifier in all_identifiers:
        values = fit.loc[fit["gauge_id"].eq(identifier), "inflow_hm3"]
        if len(values) < min_fit_months:
            raise DataValidationError(
                f"Gauge {identifier} has {len(values)} fit months; "
                f"at least {min_fit_months} are required."
            )
        mean_inflow = float(values.mean())
        if not np.isfinite(mean_inflow) or mean_inflow <= 0.0:
            raise DataValidationError(
                f"Gauge {identifier} has no positive finite mean fit inflow."
            )
        storage_capacity = storage_months * mean_inflow
        release_capacity = release_mean_multiplier * mean_inflow
        rows.append(
            {
                "gauge_id": identifier,
                "fit_water_year_start": fit_start,
                "fit_water_year_end": fit_end,
                "fit_months": int(len(values)),
                "mean_monthly_inflow_hm3": mean_inflow,
                "storage_capacity_hm3": storage_capacity,
                "release_capacity_hm3_per_month": release_capacity,
                "initial_storage_hm3": initial_storage_fraction * storage_capacity,
                "terminal_target_hm3": terminal_target_fraction * storage_capacity,
                "storage_months": storage_months,
                "release_mean_multiplier": release_mean_multiplier,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BasinSelectionCriteria",
    "BasinSelectionResult",
    "AvailabilitySelectionResult",
    "CAMELS_BR_API_URL",
    "CAMELS_BR_FILES",
    "CAMELS_BR_RECORD_ID",
    "CAMELS_BR_RECORD_URL",
    "CORE_FILE_KEYS",
    "SPATIAL_FILE_KEYS",
    "CamelsBRError",
    "CamelsBRSource",
    "ChecksumMismatch",
    "DataValidationError",
    "GeographicCRS",
    "HistoricalAvailabilityAudit",
    "HistoricalWaterYearSplit",
    "RemoteFile",
    "SpatialNestingResult",
    "accepted_streamflow_mask",
    "add_water_year_columns",
    "audit_historical_availability",
    "build_monthly_inflow",
    "calibrate_semisynthetic_reservoirs",
    "derive_nesting_pairs",
    "screen_static_attributes",
    "select_eligible_basins",
    "select_non_nested_basins",
    "select_non_nested_basins_with_availability",
    "water_year_split",
]
