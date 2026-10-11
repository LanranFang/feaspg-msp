"""Reproducible access and normalization for public Brazilian ONS data.

The module deliberately separates downloading from parsing.  Importing it never
performs network I/O.  Dynamic data are restricted to the last fully completed
calendar year used by this benchmark (2000--2025); the partial 2026 files are
therefore never selected accidentally.

ONS publishes the resources below under CC BY 4.0.  ONS may revise files in
place, so :class:`ONSCache` records the URL, response metadata, byte count, and
SHA256 digest for every cached object.

Hydrology terminology matters for a cascade model:

* ``incremental`` is local natural inflow between upstream reservoirs and the
  current reservoir.  Use it when upstream releases are routed explicitly.
* ``natural`` is total natural flow at the reservoir.  It already contains the
  upstream catchment contribution and must not be combined with explicit
  upstream routing.

The loader enforces this distinction to prevent double-counting water.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Literal, Mapping, Sequence, TextIO
from urllib.request import Request, urlopen

import pandas as pd


ONS_PROVIDER = "Operador Nacional do Sistema Eletrico (ONS)"
ONS_LICENSE = "CC-BY-4.0"
ONS_LICENSE_URL = "https://registry.opendata.aws/ons-opendata-portal/"
ONS_CKAN_API = "https://dados.ons.org.br/api/3/action/package_show"
ONS_S3_BASE = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"

COMPLETE_START_YEAR = 2000
COMPLETE_END_YEAR = 2025
COMPLETE_YEARS = tuple(range(COMPLETE_START_YEAR, COMPLETE_END_YEAR + 1))
THERMAL_CROSSWALK_YEARS = (2013, 2014)


@dataclass(frozen=True)
class TemporalSplit:
    """Inclusive calendar-year boundary for a chronological data split."""

    name: str
    start_year: int
    end_year: int


TEMPORAL_SPLITS = (
    TemporalSplit("train", 2000, 2014),
    TemporalSplit("validation", 2015, 2018),
    TemporalSplit("calibration", 2019, 2021),
    TemporalSplit("historical_test", 2022, 2025),
)


@dataclass(frozen=True)
class ONSResource:
    """A fixed official ONS resource location."""

    key: str
    url_template: str
    filename_template: str
    annual: bool
    first_year: int | None = None
    last_year: int | None = None

    def url(self, year: int | None = None) -> str:
        if self.annual:
            checked_year = _validate_resource_year(self, year)
            return self.url_template.format(year=checked_year)
        if year is not None:
            raise ValueError(f"{self.key!r} is not annual; do not pass year")
        return self.url_template

    def filename(self, year: int | None = None) -> str:
        if self.annual:
            checked_year = _validate_resource_year(self, year)
            return self.filename_template.format(year=checked_year)
        if year is not None:
            raise ValueError(f"{self.key!r} is not annual; do not pass year")
        return self.filename_template


ONS_RESOURCES: Mapping[str, ONSResource] = {
    "hydrology_daily": ONSResource(
        key="hydrology_daily",
        url_template=(
            f"{ONS_S3_BASE}/dados_hidrologicos_di/"
            "DADOS_HIDROLOGICOS_RES_{year}.csv"
        ),
        filename_template="DADOS_HIDROLOGICOS_RES_{year}.csv",
        annual=True,
        first_year=2000,
        last_year=2025,
    ),
    "reservoir_static": ONSResource(
        key="reservoir_static",
        url_template=f"{ONS_S3_BASE}/reservatorio/RESERVATORIOS.csv",
        filename_template="RESERVATORIOS.csv",
        annual=False,
    ),
    "reservoir_ena_daily": ONSResource(
        key="reservoir_ena_daily",
        url_template=(
            f"{ONS_S3_BASE}/ena_reservatorio_di/"
            "ENA_DIARIO_RESERVATORIOS_{year}.csv"
        ),
        filename_template="ENA_DIARIO_RESERVATORIOS_{year}.csv",
        annual=True,
        first_year=2000,
        last_year=2025,
    ),
    "subsystem_ena_daily": ONSResource(
        key="subsystem_ena_daily",
        url_template=(
            f"{ONS_S3_BASE}/ena_subsistema_di/"
            "ENA_DIARIO_SUBSISTEMA_{year}.csv"
        ),
        filename_template="ENA_DIARIO_SUBSISTEMA_{year}.csv",
        annual=True,
        first_year=2000,
        last_year=2025,
    ),
    "load_monthly": ONSResource(
        key="load_monthly",
        url_template=f"{ONS_S3_BASE}/carga_energia_me/CARGA_MENSAL.csv",
        filename_template="CARGA_MENSAL.csv",
        annual=False,
    ),
    "cvu_weekly": ONSResource(
        key="cvu_weekly",
        url_template=(
            f"{ONS_S3_BASE}/cvu_usitermica_se/"
            "CVU_USINA_TERMICA_{year}.csv"
        ),
        filename_template="CVU_USINA_TERMICA_{year}.csv",
        annual=True,
        first_year=2005,
        last_year=2025,
    ),
    "generation_capacity": ONSResource(
        key="generation_capacity",
        url_template=(
            f"{ONS_S3_BASE}/capacidade-geracao/CAPACIDADE_GERACAO.csv"
        ),
        filename_template="CAPACIDADE_GERACAO.csv",
        annual=False,
    ),
    "thermal_dispatch_crosswalk": ONSResource(
        key="thermal_dispatch_crosswalk",
        url_template=(
            f"{ONS_S3_BASE}/geracao_termica_despacho_2_ho/"
            "GERACAO_TERMICA_DESPACHO_{year}.parquet"
        ),
        filename_template="GERACAO_TERMICA_DESPACHO_{year}.parquet",
        annual=True,
        first_year=THERMAL_CROSSWALK_YEARS[0],
        last_year=THERMAL_CROSSWALK_YEARS[-1],
    ),
}


ONS_DICTIONARY_URLS: Mapping[str, str] = {
    "hydrology_daily": (
        f"{ONS_S3_BASE}/dados_hidrologicos_di/"
        "DicionarioDados_DadosHidrologicosDiarios.json"
    ),
    "reservoir_static": (
        f"{ONS_S3_BASE}/reservatorio/DicionarioDados_Reservatorio.json"
    ),
    "reservoir_ena_daily": (
        f"{ONS_S3_BASE}/ena_reservatorio_di/"
        "DicionarioDados_EnaPorReservatorio.json"
    ),
    "subsystem_ena_daily": (
        f"{ONS_S3_BASE}/ena_subsistema_di/"
        "DicionarioDados_EnaPorSubsistema.json"
    ),
    "load_monthly": (
        f"{ONS_S3_BASE}/carga_energia_me/"
        "DicionarioDados_Carga_Energia_Mensal.json"
    ),
    "cvu_weekly": (
        f"{ONS_S3_BASE}/cvu_usitermica_se/"
        "DicionarioDados_CVU_UsinaTermica.json"
    ),
    "generation_capacity": (
        f"{ONS_S3_BASE}/capacidade-geracao/"
        "DicionarioDados_Capacidade_Instalada_Geracao.json"
    ),
}


CSVSource = str | os.PathLike[str] | TextIO | BinaryIO
FrameSource = CSVSource | pd.DataFrame | Sequence[CSVSource | pd.DataFrame]
InflowKind = Literal["incremental", "natural"]


def _validate_resource_year(resource: ONSResource, year: int | None) -> int:
    if year is None:
        raise ValueError(f"{resource.key!r} requires a calendar year")
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError("year must be an integer")
    first = resource.first_year or COMPLETE_START_YEAR
    last = resource.last_year or COMPLETE_END_YEAR
    if not first <= year <= last:
        raise ValueError(
            f"{resource.key!r} supports complete years {first}--{last}; got {year}"
        )
    return year


def official_url(resource: str, year: int | None = None) -> str:
    """Return a fixed official HTTPS URL without performing a download."""

    try:
        spec = ONS_RESOURCES[resource]
    except KeyError as exc:
        raise KeyError(
            f"unknown ONS resource {resource!r}; choose from {sorted(ONS_RESOURCES)}"
        ) from exc
    return spec.url(year)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """Compute a file digest without loading the full object into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ONSCache:
    """Download-once cache with an auditable SHA256 manifest.

    Existing files are reused only when they agree with the manifest (and an
    optional caller-supplied expected digest).  Set ``offline=True`` to prohibit
    network access and fail if a verified cached copy is unavailable.
    """

    MANIFEST_VERSION = 1

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        manifest_name: str = "ons_manifest.json",
        timeout: float = 120.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / manifest_name
        self.timeout = timeout

    def read_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {
                "manifest_version": self.MANIFEST_VERSION,
                "provider": ONS_PROVIDER,
                "license": ONS_LICENSE,
                "license_url": ONS_LICENSE_URL,
                "files": {},
            }
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ONS manifest: {self.manifest_path}") from exc
        if manifest.get("manifest_version") != self.MANIFEST_VERSION:
            raise ValueError("unsupported ONS manifest version")
        if not isinstance(manifest.get("files"), dict):
            raise ValueError("ONS manifest 'files' must be an object")
        return manifest

    def _write_manifest(self, manifest: Mapping[str, object]) -> None:
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=".ons-manifest-", suffix=".json", dir=self.cache_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def fetch(
        self,
        resource: str,
        year: int | None = None,
        *,
        expected_sha256: str | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Path:
        """Fetch one item from :data:`ONS_RESOURCES`."""

        try:
            spec = ONS_RESOURCES[resource]
        except KeyError as exc:
            raise KeyError(
                f"unknown ONS resource {resource!r}; choose from "
                f"{sorted(ONS_RESOURCES)}"
            ) from exc
        return self.fetch_url(
            spec.url(year),
            filename=spec.filename(year),
            expected_sha256=expected_sha256,
            refresh=refresh,
            offline=offline,
            resource=resource,
            year=year,
        )

    def fetch_url(
        self,
        url: str,
        *,
        filename: str | None = None,
        expected_sha256: str | None = None,
        refresh: bool = False,
        offline: bool = False,
        resource: str | None = None,
        year: int | None = None,
    ) -> Path:
        """Fetch an HTTPS object and atomically update the manifest.

        ``fetch_url`` is public mainly so data dictionaries can use the same
        cache.  Benchmark data should normally use :meth:`fetch`, which selects
        only fixed official URLs.
        """

        if filename is None:
            filename = url.rstrip("/").rsplit("/", 1)[-1]
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a non-empty basename")
        if expected_sha256 is not None:
            expected_sha256 = expected_sha256.lower()
            if len(expected_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in expected_sha256
            ):
                raise ValueError("expected_sha256 must contain 64 hexadecimal digits")

        target = self.cache_dir / filename
        manifest = self.read_manifest()
        files = manifest["files"]
        assert isinstance(files, dict)  # checked by read_manifest
        recorded = files.get(filename)

        if target.exists() and not refresh:
            actual = sha256_file(target)
            recorded_digest = (
                recorded.get("sha256") if isinstance(recorded, dict) else None
            )
            recorded_url = recorded.get("url") if isinstance(recorded, dict) else None
            expected_matches = expected_sha256 is None or actual == expected_sha256
            manifest_matches = recorded_digest is None or actual == recorded_digest
            url_matches = recorded_url is None or recorded_url == url
            if expected_matches and manifest_matches and url_matches:
                if recorded is None:
                    files[filename] = self._record(
                        target, url=url, resource=resource, year=year
                    )
                    self._write_manifest(manifest)
                return target

        if offline:
            raise FileNotFoundError(
                f"no verified cached copy of {filename!r} is available in offline mode"
            )

        request = Request(url, headers={"User-Agent": "real-hydro-ons/1.0"})
        temporary: Path | None = None
        response_headers: dict[str, str | None] = {}
        try:
            with urlopen(request, timeout=self.timeout) as response:
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{filename}.", suffix=".part", dir=self.cache_dir
                )
                temporary = Path(temporary_name)
                with os.fdopen(fd, "wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                response_headers = {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }

            actual = sha256_file(temporary)
            if expected_sha256 is not None and actual != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch for {url}: expected {expected_sha256}, "
                    f"got {actual}"
                )
            os.replace(temporary, target)
            temporary = None
            files[filename] = self._record(
                target,
                url=url,
                resource=resource,
                year=year,
                etag=response_headers["etag"],
                last_modified=response_headers["last_modified"],
            )
            self._write_manifest(manifest)
            return target
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @staticmethod
    def _record(
        path: Path,
        *,
        url: str,
        resource: str | None,
        year: int | None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> dict[str, object]:
        return {
            "resource": resource,
            "year": year,
            "url": url,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "etag": etag,
            "last_modified": last_modified,
        }

    def verify_manifest(self) -> dict[str, bool]:
        """Return per-file digest status without modifying cache contents."""

        manifest = self.read_manifest()
        files = manifest["files"]
        assert isinstance(files, dict)
        status: dict[str, bool] = {}
        for filename, record in files.items():
            path = self.cache_dir / filename
            expected = record.get("sha256") if isinstance(record, dict) else None
            status[filename] = bool(
                path.is_file() and expected and sha256_file(path) == expected
            )
        return status


def _decode_csv(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", payload, 0, 1, "unsupported ONS CSV encoding")


def _source_text(source: CSVSource) -> str:
    if hasattr(source, "read"):
        payload = source.read()  # type: ignore[union-attr]
        return _decode_csv(payload) if isinstance(payload, bytes) else str(payload)
    return _decode_csv(Path(source).read_bytes())


def _detect_delimiter(text: str) -> str:
    header = next((line for line in text.splitlines() if line.strip()), "")
    if not header:
        raise ValueError("ONS CSV is empty")
    # ONS uses semicolons with decimal commas.  Inspecting the header avoids
    # mistaking those decimal commas for field delimiters.
    semicolons = header.count(";")
    commas = header.count(",")
    if semicolons != commas:
        return ";" if semicolons > commas else ","
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=";,").delimiter
    except csv.Error as exc:
        raise ValueError("could not detect ONS CSV delimiter") from exc


_MISSING_NUMBERS = {"", "-", "--", "na", "n/a", "nan", "null", "none"}


def parse_ons_number(value: object) -> float:
    """Parse Brazilian or international decimal/thousands conventions."""

    if value is None:
        return math.nan
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return math.nan
        except (TypeError, ValueError):
            pass
        return float(value)

    token = value.strip().replace("\u00a0", "").replace(" ", "")
    if token.lower() in _MISSING_NUMBERS:
        return math.nan
    negative_parentheses = token.startswith("(") and token.endswith(")")
    if negative_parentheses:
        token = token[1:-1]
    if token.endswith("%"):
        token = token[:-1]

    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = token.replace(",", ".")

    number = float(token)
    return -number if negative_parentheses else number


def read_ons_csv(
    source: CSVSource,
    *,
    numeric_columns: Iterable[str] = (),
    date_columns: Iterable[str] = (),
    numeric_errors: Literal["raise", "coerce"] = "raise",
) -> pd.DataFrame:
    """Read ONS CSV robustly while preserving identifiers as strings."""

    if numeric_errors not in {"raise", "coerce"}:
        raise ValueError("numeric_errors must be 'raise' or 'coerce'")
    text = _source_text(source)
    frame = pd.read_csv(
        io.StringIO(text),
        sep=_detect_delimiter(text),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    frame.columns = [str(column).strip().lstrip("\ufeff").lower() for column in frame]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"duplicate ONS CSV columns after normalization: {duplicates}")
    for column in frame.columns:
        frame[column] = frame[column].str.strip()

    for column in numeric_columns:
        if column not in frame:
            raise KeyError(f"missing numeric column {column!r}")
        if numeric_errors == "raise":
            frame[column] = frame[column].map(parse_ons_number)
        else:
            def coerce(value: object) -> float:
                try:
                    return parse_ons_number(value)
                except (TypeError, ValueError):
                    return math.nan

            frame[column] = frame[column].map(coerce)

    for column in date_columns:
        if column not in frame:
            raise KeyError(f"missing date column {column!r}")
        values = frame[column].replace("", pd.NA)
        try:
            parsed = pd.to_datetime(
                values, errors="coerce", dayfirst=True, format="mixed"
            )
        except TypeError:  # pandas < 2.0
            parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
        invalid = values.notna() & parsed.isna()
        if invalid.any():
            examples = values.loc[invalid].head(3).tolist()
            raise ValueError(f"invalid dates in {column!r}: {examples}")
        frame[column] = parsed
    return frame


def _read_frames(source: FrameSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        frame.columns = [str(column).strip().lower() for column in frame]
        return frame
    if isinstance(source, Sequence) and not isinstance(
        source, (str, bytes, os.PathLike)
    ):
        frames = [_read_frames(item) for item in source]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return read_ons_csv(source)  # type: ignore[arg-type]


def _read_parquet_frames(source: FrameSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        frame.columns = [str(column).strip().lower() for column in frame]
        return frame
    if isinstance(source, Sequence) and not isinstance(
        source, (str, bytes, os.PathLike)
    ):
        frames = [_read_parquet_frames(item) for item in source]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    try:
        frame = pd.read_parquet(source)  # type: ignore[arg-type]
    except ImportError as exc:
        raise ImportError(
            "reading the ONS thermal crosswalk parquet requires pyarrow"
        ) from exc
    frame.columns = [str(column).strip().lower() for column in frame]
    return frame


def _parse_dates(values: pd.Series, name: str) -> pd.Series:
    # ONS represents an unknown date both as a true null and as an empty
    # (occasionally whitespace-only) field.  Treat those representations alike
    # while continuing to reject every non-empty value that cannot be parsed.
    cleaned = values.copy()
    if not pd.api.types.is_datetime64_any_dtype(cleaned):
        cleaned = cleaned.map(
            lambda value: (
                pd.NA
                if isinstance(value, str) and not value.strip()
                else value
            )
        )
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
    else:
        try:
            parsed = pd.to_datetime(
                cleaned,
                errors="coerce",
                dayfirst=True,
                format="mixed",
            )
        except TypeError:
            parsed = pd.to_datetime(
                cleaned, errors="coerce", dayfirst=True
            )
    invalid = cleaned.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].head(3).tolist()
        raise ValueError(f"invalid dates in {name!r}: {examples}")
    return parsed


def _numeric(values: pd.Series, name: str) -> pd.Series:
    try:
        return values.map(parse_ons_number).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value in {name!r}") from exc


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"missing required ONS columns: {missing}")


def filter_complete_year_range(
    frame: pd.DataFrame, *, time_column: str = "time"
) -> pd.DataFrame:
    """Keep observations in 2000--2025, excluding partial current-year data."""

    _require_columns(frame, [time_column])
    result = frame.loc[
        frame[time_column].dt.year.between(COMPLETE_START_YEAR, COMPLETE_END_YEAR)
    ].copy()
    return result.reset_index(drop=True)


def assign_temporal_split(times: pd.Series) -> pd.Series:
    """Assign train/validation/calibration/historical_test by calendar year."""

    if not pd.api.types.is_datetime64_any_dtype(times):
        times = _parse_dates(times, "time")
    labels = pd.Series(pd.NA, index=times.index, dtype="string")
    years = times.dt.year
    for boundary in TEMPORAL_SPLITS:
        in_boundary = years.between(boundary.start_year, boundary.end_year)
        labels.loc[in_boundary] = boundary.name
    return labels


def validate_inflow_routing(
    inflow_kind: InflowKind, *, route_upstream_releases: bool
) -> None:
    """Reject the combination that double-counts upstream catchment water."""

    if inflow_kind not in {"incremental", "natural"}:
        raise ValueError("inflow_kind must be 'incremental' or 'natural'")
    if inflow_kind == "natural" and route_upstream_releases:
        raise ValueError(
            "total natural inflow already includes upstream catchment flow; "
            "use incremental inflow when routing upstream releases explicitly"
        )


def load_hydrology_daily(
    source: FrameSource,
    *,
    inflow_kind: InflowKind = "incremental",
    route_upstream_releases: bool = True,
) -> pd.DataFrame:
    """Normalize daily reservoir hydrology to a leakage-safe long table."""

    validate_inflow_routing(
        inflow_kind, route_upstream_releases=route_upstream_releases
    )
    frame = _read_frames(source)
    source_column = {
        "incremental": "val_vazaoincremental",
        "natural": "val_vazaonatural",
    }[inflow_kind]
    _require_columns(frame, ["din_instante", "id_reservatorio", source_column])

    result = pd.DataFrame(
        {
            "dataset": "ons_hydrology",
            "time": _parse_dates(frame["din_instante"], "din_instante"),
            "unit_id": frame["id_reservatorio"].astype("string").str.strip(),
            "inflow_m3s": _numeric(frame[source_column], source_column),
            "inflow_kind": inflow_kind,
            "route_upstream_releases": bool(route_upstream_releases),
            "source_column": source_column,
        }
    )
    result = filter_complete_year_range(result)
    if result["unit_id"].isna().any() or result["unit_id"].eq("").any():
        raise ValueError("hydrology contains an empty reservoir identifier")
    duplicated = result.duplicated(["unit_id", "time"], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, ["unit_id", "time"]].head(3)
        raise ValueError(f"duplicate reservoir/day observations:\n{examples}")
    result["split"] = assign_temporal_split(result["time"])
    return result.reset_index(drop=True)


def aggregate_daily_inflow_to_monthly(
    daily: pd.DataFrame, *, require_complete_months: bool = False
) -> pd.DataFrame:
    """Integrate daily mean m3/s to monthly hm3.

    Each daily value is multiplied by ``86_400 / 1e6``.  The output reports
    observed and expected days so callers can either retain incomplete months
    for diagnostics or remove them with ``require_complete_months=True``.
    Negative incremental-flow reconstructions are retained rather than silently
    clipped.
    """

    required = [
        "dataset",
        "time",
        "unit_id",
        "inflow_m3s",
        "inflow_kind",
        "route_upstream_releases",
    ]
    _require_columns(daily, required)
    frame = daily.copy()
    frame["time"] = _parse_dates(frame["time"], "time").dt.normalize()
    duplicated = frame.duplicated(["dataset", "unit_id", "time"], keep=False)
    if duplicated.any():
        raise ValueError("cannot integrate duplicate reservoir/day observations")
    frame["month"] = frame["time"].dt.to_period("M").dt.to_timestamp()
    frame["daily_volume_hm3"] = frame["inflow_m3s"] * 86_400.0 / 1_000_000.0

    keys = [
        "dataset",
        "month",
        "unit_id",
        "inflow_kind",
        "route_upstream_releases",
    ]
    grouped = frame.groupby(keys, dropna=False, sort=True)
    monthly = grouped.agg(
        inflow_hm3=(
            "daily_volume_hm3",
            lambda values: values.sum(min_count=1),
        ),
        observed_days=("inflow_m3s", "count"),
    ).reset_index()
    monthly = monthly.rename(columns={"month": "time"})
    monthly["expected_days"] = monthly["time"].dt.days_in_month.astype(int)
    monthly["is_complete_month"] = (
        monthly["observed_days"] == monthly["expected_days"]
    )
    if require_complete_months:
        monthly = monthly.loc[monthly["is_complete_month"]].copy()
    monthly["split"] = assign_temporal_split(monthly["time"])
    columns = [
        "dataset",
        "time",
        "unit_id",
        "inflow_hm3",
        "inflow_kind",
        "route_upstream_releases",
        "observed_days",
        "expected_days",
        "is_complete_month",
        "split",
    ]
    return monthly.loc[:, columns].reset_index(drop=True)


def load_hydrology_monthly(
    source: FrameSource,
    *,
    inflow_kind: InflowKind = "incremental",
    route_upstream_releases: bool = True,
    require_complete_months: bool = False,
) -> pd.DataFrame:
    """Load daily ONS hydrology and return monthly inflow volumes."""

    daily = load_hydrology_daily(
        source,
        inflow_kind=inflow_kind,
        route_upstream_releases=route_upstream_releases,
    )
    return aggregate_daily_inflow_to_monthly(
        daily, require_complete_months=require_complete_months
    )


def load_reservoir_static(
    source: FrameSource,
    *,
    commissioned_by: str | pd.Timestamp | None = "2025-12-31",
) -> pd.DataFrame:
    """Normalize the current ONS reservoir master-data snapshot.

    The source is not a historical table.  By default, plants commissioned
    after the benchmark's last complete year are removed.  An earlier
    ``commissioned_by`` cutoff can prevent plant-set look-ahead in a historical
    fit.  Callers must still pin and hash the downloaded snapshot because the
    physical attributes themselves can be revised.
    """

    frame = _read_frames(source)
    identifier_column = (
        "id_reservatorio"
        if "id_reservatorio" in frame
        else "res_id" if "res_id" in frame else None
    )
    if identifier_column is None:
        raise KeyError(
            "missing required ONS reservoir identifier: expected "
            "'id_reservatorio' or current-schema 'res_id'"
        )
    numeric_map = {
        "val_volmin": "volume_min_hm3",
        "val_volmax": "volume_max_hm3",
        "val_volutiltot": "storage_capacity_hm3",
        "val_cotaminima": "level_min_m",
        "val_cotamaxima": "level_max_m",
        "val_produtibilidadeespecifica": (
            "specific_productivity_mw_per_m3s_per_m"
        ),
        "val_produtividade65volutil": "productivity_65pct_mw_per_m3s",
        "val_latitude": "latitude",
        "val_longitude": "longitude",
    }
    text_map = {
        "nom_reservatorio": "unit_name",
        "cod_resplanejamento": "planning_id",
        "cod_posto": "flow_station_id",
        "nom_usina": "plant_name",
        "ceg": "ceg",
        "id_subsistema": "subsystem_id",
        "nom_subsistema": "subsystem_name",
        "nom_bacia": "basin_name",
        "nom_rio": "river_name",
        "nom_ree": "ree_name",
        "tip_reservatorio": "reservoir_type",
    }
    result = pd.DataFrame(
        {
            "dataset": "ons_reservoir_static",
            "unit_id": frame[identifier_column].astype("string").str.strip(),
        }
    )
    for source_name, target_name in text_map.items():
        result[target_name] = (
            frame[source_name].astype("string").str.strip()
            if source_name in frame
            else pd.Series(pd.NA, index=frame.index, dtype="string")
        )
    for source_name, target_name in numeric_map.items():
        result[target_name] = (
            _numeric(frame[source_name], source_name)
            if source_name in frame
            else math.nan
        )
    if "dat_entrada" in frame:
        result["commissioned_on"] = _parse_dates(frame["dat_entrada"], "dat_entrada")
    else:
        result["commissioned_on"] = pd.NaT

    if commissioned_by is not None:
        cutoff = pd.Timestamp(commissioned_by)
        result = result.loc[
            result["commissioned_on"].isna()
            | (result["commissioned_on"] <= cutoff)
        ].copy()
    if result["unit_id"].isna().any() or result["unit_id"].eq("").any():
        raise ValueError("reservoir master data contains an empty identifier")
    if result["unit_id"].duplicated().any():
        # The current ONS snapshot has one row per plant for a few shared
        # physical reservoirs (for example Itaipu 50/60 Hz and Paulo Afonso
        # I/II/III).  Collapse only when every control-relevant reservoir field
        # agrees; this avoids counting useful volume once per powerhouse.
        invariant = [
            "planning_id",
            "flow_station_id",
            "subsystem_id",
            "ree_name",
            "storage_capacity_hm3",
            "productivity_65pct_mw_per_m3s",
        ]
        collapsed = []
        for unit_id, group in result.groupby("unit_id", sort=False, dropna=False):
            conflicts = [
                column
                for column in invariant
                if column in group and group[column].dropna().nunique() > 1
            ]
            if conflicts:
                raise ValueError(
                    f"reservoir {unit_id!r} repeats with conflicting fields: "
                    f"{conflicts}"
                )
            row = group.iloc[0].copy()
            row["commissioned_on"] = group["commissioned_on"].min()
            for column in ("plant_name", "ceg"):
                values = tuple(
                    dict.fromkeys(
                        str(value) for value in group[column].dropna() if str(value)
                    )
                )
                row[column] = " | ".join(values) if values else pd.NA
            row["plant_records_at_reservoir"] = int(len(group))
            collapsed.append(row)
        result = pd.DataFrame(collapsed)
    else:
        result["plant_records_at_reservoir"] = 1
    return result.reset_index(drop=True)


def load_reservoir_ena_daily(source: FrameSource) -> pd.DataFrame:
    """Normalize daily reservoir ENA and published MLT diagnostics.

    ENA is an energy quantity in average MW, not a water volume, and must not be
    inserted into the hm3 storage balance.  ``published_mlt_ena_mw_mean`` is
    retained for descriptive checks only: ONS computes it from a long record
    that can extend beyond a historical benchmark split.  Any predictive
    normalization must instead be estimated from training rows alone.

    ``unit_id`` is the ONS planning-reservoir code because this table does not
    publish ``id_reservatorio``.  Join it to the static table through
    ``planning_id``.
    """

    frame = _read_frames(source)
    required = [
        "cod_resplanejamento",
        "ena_data",
        "ena_bruta_res_mwmed",
        "ena_armazenavel_res_mwmed",
        "mlt_ena",
    ]
    _require_columns(frame, required)
    result = pd.DataFrame(
        {
            "dataset": "ons_reservoir_ena",
            "time": _parse_dates(frame["ena_data"], "ena_data"),
            "unit_id": (
                frame["cod_resplanejamento"].astype("string").str.strip()
            ),
            "ena_gross_mw_mean": _numeric(
                frame["ena_bruta_res_mwmed"], "ena_bruta_res_mwmed"
            ),
            "ena_storable_mw_mean": _numeric(
                frame["ena_armazenavel_res_mwmed"],
                "ena_armazenavel_res_mwmed",
            ),
            "published_mlt_ena_mw_mean": _numeric(frame["mlt_ena"], "mlt_ena"),
        }
    )
    optional_numeric = {
        "ena_bruta_res_percentualmlt": "ena_gross_pct_published_mlt",
        "ena_armazenavel_res_percentualmlt": "ena_storable_pct_published_mlt",
        "ena_queda_bruta": "ena_gross_head_mw_mean",
    }
    optional_text = {
        "nom_reservatorio": "unit_name",
        "tip_reservatorio": "reservoir_type",
        "nom_bacia": "basin_name",
        "nom_ree": "ree_name",
        "id_subsistema": "subsystem_id",
        "nom_subsistema": "subsystem_name",
    }
    for source_name, target_name in optional_numeric.items():
        if source_name in frame:
            result[target_name] = _numeric(frame[source_name], source_name)
    for source_name, target_name in optional_text.items():
        if source_name in frame:
            result[target_name] = frame[source_name].astype("string").str.strip()
    result = filter_complete_year_range(result)
    missing_identifier = result["unit_id"].isna() | result["unit_id"].eq("")
    if missing_identifier.any():
        # A small number of post-2021 source rows (notably Canastra/Suica) are
        # published without a planning code.  They cannot be joined to a
        # reservoir and are excluded with an explicit audit count.  Missing IDs
        # in the fit interval remain fatal because they could change the model.
        if result.loc[missing_identifier, "time"].dt.year.le(2014).any():
            raise ValueError(
                "training-period reservoir ENA contains an empty planning identifier"
            )
        dropped_missing_identifier = int(missing_identifier.sum())
        result = result.loc[~missing_identifier].copy()
    else:
        dropped_missing_identifier = 0
    duplicated = result.duplicated(["unit_id", "time"], keep=False)
    dropped_duplicate_rows = 0
    if duplicated.any():
        duplicate_rows = result.loc[duplicated]
        if duplicate_rows["time"].dt.year.le(2014).any():
            raise ValueError(
                "training-period reservoir ENA contains duplicate reservoir/day observations"
            )
        numeric = [
            column
            for column in (
                "ena_gross_mw_mean",
                "ena_storable_mw_mean",
                "published_mlt_ena_mw_mean",
                "ena_gross_head_mw_mean",
            )
            if column in result
        ]
        keep_indices: list[int] = []
        for (unit_id, _), group in duplicate_rows.groupby(
            ["unit_id", "time"], sort=False
        ):
            conflicts = [
                column for column in numeric if group[column].dropna().nunique() > 1
            ]
            if conflicts:
                raise ValueError(
                    f"post-training ENA duplicate for {unit_id!r} has "
                    f"conflicting numeric fields: {conflicts}"
                )
            candidates = group
            if "ree_name" in result:
                unit_names = result.loc[
                    result["unit_id"].eq(unit_id), "ree_name"
                ].replace("", pd.NA).dropna()
                if not unit_names.empty:
                    modal_name = unit_names.mode().iloc[0]
                    matched = group.loc[group["ree_name"].eq(modal_name)]
                    if not matched.empty:
                        candidates = matched
            keep_indices.append(int(candidates.index[-1]))
        remove = duplicated & ~result.index.isin(keep_indices)
        dropped_duplicate_rows = int(remove.sum())
        result = result.loc[~remove].copy()
    result["split"] = assign_temporal_split(result["time"])
    result = result.reset_index(drop=True)
    result.attrs["dropped_post_train_rows_missing_unit_id"] = (
        dropped_missing_identifier
    )
    result.attrs["dropped_post_train_duplicate_rows"] = dropped_duplicate_rows
    return result


def load_subsystem_ena_daily(source: FrameSource) -> pd.DataFrame:
    """Normalize the official daily ENA aggregate for N/NE/S/SE.

    This is the formal subsystem benchmark's inflow source.  Unlike a sum of
    reservoir rows, it does not implicitly replace unpublished reservoir
    components with zero.  Every published date must contain exactly one
    finite gross-ENA observation for each of the four electrical subsystems;
    missing calendar dates are detected again when the benchmark forms months.
    """

    frame = _read_frames(source)
    required = [
        "id_subsistema",
        "ena_data",
        "ena_bruta_regiao_mwmed",
    ]
    _require_columns(frame, required)
    result = pd.DataFrame(
        {
            "dataset": "ons_subsystem_ena",
            "time": _parse_dates(frame["ena_data"], "ena_data").dt.normalize(),
            "unit_id": frame["id_subsistema"].astype("string").str.strip(),
            "ena_gross_mw_mean": _numeric(
                frame["ena_bruta_regiao_mwmed"], "ena_bruta_regiao_mwmed"
            ),
        }
    )
    optional_numeric = {
        "ena_bruta_regiao_percentualmlt": "ena_gross_pct_published_mlt",
        "ena_armazenavel_regiao_mwmed": "ena_storable_mw_mean",
        "ena_armazenavel_regiao_percentualmlt": (
            "ena_storable_pct_published_mlt"
        ),
    }
    for source_name, target_name in optional_numeric.items():
        if source_name in frame:
            result[target_name] = _numeric(frame[source_name], source_name)
    if "nom_subsistema" in frame:
        result["unit_name"] = frame["nom_subsistema"].astype("string").str.strip()
    result = filter_complete_year_range(result)
    expected_subsystems = {"N", "NE", "S", "SE"}
    if result.empty:
        raise ValueError("subsystem ENA has no observations in 2000--2025")
    invalid_ids = set(result["unit_id"].dropna()).difference(expected_subsystems)
    if result["unit_id"].isna().any() or result["unit_id"].eq("").any() or invalid_ids:
        raise ValueError(
            "subsystem ENA must use exactly N/NE/S/SE identifiers; "
            f"unexpected={sorted(invalid_ids)}"
        )
    if result["ena_gross_mw_mean"].isna().any() or (
        result["ena_gross_mw_mean"] < 0.0
    ).any():
        raise ValueError("subsystem gross ENA must be finite and nonnegative")
    duplicated = result.duplicated(["unit_id", "time"], keep=False)
    if duplicated.any():
        raise ValueError("subsystem ENA contains duplicate subsystem/day rows")
    daily_ids = result.groupby("time", sort=True)["unit_id"].agg(
        lambda values: frozenset(values)
    )
    incomplete = daily_ids.ne(frozenset(expected_subsystems))
    if incomplete.any():
        examples = [str(value.date()) for value in daily_ids.index[incomplete][:3]]
        raise ValueError(
            "subsystem ENA date does not contain exactly N/NE/S/SE: "
            f"{examples}"
        )
    result["split"] = assign_temporal_split(result["time"])
    return result.sort_values(["time", "unit_id"]).reset_index(drop=True)


def load_monthly_load(source: FrameSource) -> pd.DataFrame:
    """Normalize ONS subsystem demand to a 2000--2025 monthly long table."""

    frame = _read_frames(source)
    required = ["id_subsistema", "din_instante", "val_cargaenergiamwmed"]
    _require_columns(frame, required)
    time = _parse_dates(frame["din_instante"], "din_instante")
    result = pd.DataFrame(
        {
            "dataset": "ons_load",
            "time": time.dt.to_period("M").dt.to_timestamp(),
            "unit_id": frame["id_subsistema"].astype("string").str.strip(),
            "load_mw_mean": _numeric(
                frame["val_cargaenergiamwmed"], "val_cargaenergiamwmed"
            ),
        }
    )
    if "nom_subsistema" in frame:
        result["unit_name"] = frame["nom_subsistema"].astype("string").str.strip()
    result = filter_complete_year_range(result)
    duplicated = result.duplicated(["unit_id", "time"], keep=False)
    if duplicated.any():
        raise ValueError("monthly load contains duplicate subsystem/month observations")
    result["split"] = assign_temporal_split(result["time"])
    return result.reset_index(drop=True)


def load_weekly_cvu(
    source: FrameSource, *, drop_cross_split_weeks: bool = True
) -> pd.DataFrame:
    """Normalize plant-level thermal CVU observations to a weekly long table."""

    frame = _read_frames(source)
    required = [
        "dat_iniciosemana",
        "cod_usinaplanejamento",
        "id_subsistema",
        "val_cvu",
    ]
    _require_columns(frame, required)
    result = pd.DataFrame(
        {
            "dataset": "ons_cvu",
            "time": _parse_dates(frame["dat_iniciosemana"], "dat_iniciosemana"),
            "unit_id": frame["cod_usinaplanejamento"].astype("string").str.strip(),
            "subsystem_id": frame["id_subsistema"].astype("string").str.strip(),
            "cvu_brl_per_mwh": _numeric(frame["val_cvu"], "val_cvu"),
        }
    )
    if "dat_fimsemana" in frame:
        result["time_end"] = _parse_dates(frame["dat_fimsemana"], "dat_fimsemana")
    if "nom_usina" in frame:
        result["unit_name"] = frame["nom_usina"].astype("string").str.strip()
    if "nom_subsistema" in frame:
        result["subsystem_name"] = (
            frame["nom_subsistema"].astype("string").str.strip()
        )
    result = filter_complete_year_range(result)
    result["split"] = assign_temporal_split(result["time"])
    if "time_end" in result:
        end_is_complete = result["time_end"].dt.year.between(
            COMPLETE_START_YEAR, COMPLETE_END_YEAR
        )
        result = result.loc[end_is_complete].copy()
        end_split = assign_temporal_split(result["time_end"])
        crosses_boundary = result["split"].ne(end_split)
        if drop_cross_split_weeks:
            result = result.loc[~crosses_boundary].copy()
        else:
            result.loc[crosses_boundary, "split"] = pd.NA
    return result.reset_index(drop=True)


def load_thermal_dispatch_crosswalk(source: FrameSource) -> pd.DataFrame:
    """Build a training-only ONS planning-code-to-CEG identity crosswalk.

    The official hourly thermal-dispatch files contain both
    ``cod_usinaplanejamento`` (used by the CVU files) and ``ceg`` (used by the
    installed-capacity snapshot).  Only the 2013--2014 files are admitted so a
    plant identity first published after the fit period cannot leak into the
    benchmark.  Hourly duplicates are collapsed to one audited row per planning
    plant; conflicting CEG or subsystem identities are rejected.
    """

    frame = _read_parquet_frames(source)
    required = [
        "din_instante",
        "cod_usinaplanejamento",
        "id_subsistema",
        "nom_usina",
        "ceg",
    ]
    _require_columns(frame, required)
    times = _parse_dates(frame["din_instante"], "din_instante")
    planning_numeric = _numeric(
        frame["cod_usinaplanejamento"], "cod_usinaplanejamento"
    )
    invalid_planning = planning_numeric.notna() & planning_numeric.mod(1.0).ne(0.0)
    if invalid_planning.any():
        examples = frame.loc[
            invalid_planning, "cod_usinaplanejamento"
        ].head(3).tolist()
        raise ValueError(f"invalid thermal planning codes: {examples}")
    result = pd.DataFrame(
        {
            "time": times.dt.normalize(),
            "planning_id": planning_numeric.astype("Int64").astype("string"),
            "subsystem_id": frame["id_subsistema"].astype("string").str.strip(),
            "plant_name": frame["nom_usina"].astype("string").str.strip(),
            "ceg": frame["ceg"].astype("string").str.strip().str.upper(),
        }
    )
    input_rows = len(result)
    result = result.loc[
        result["time"].dt.year.between(
            THERMAL_CROSSWALK_YEARS[0], THERMAL_CROSSWALK_YEARS[-1]
        )
    ].copy()
    if result.empty:
        raise ValueError("thermal crosswalk has no observations in 2013--2014")
    # The hourly dispatch table also contains system-level rows without a plant
    # identity.  They are not crosswalk evidence and are removed explicitly.
    identity_present = pd.Series(True, index=result.index)
    for column in ("planning_id", "subsystem_id", "plant_name", "ceg"):
        identity_present &= result[column].notna() & result[column].ne("")
    dropped_outside_training_years = input_rows - len(result)
    dropped_missing_identity_rows = int((~identity_present).sum())
    result = result.loc[identity_present].copy()
    if result.empty:
        raise ValueError("thermal crosswalk contains no complete plant identity")

    identity = (
        result.groupby("planning_id", sort=True)
        .agg(
            ceg=("ceg", "first"),
            ceg_count=("ceg", "nunique"),
            subsystem_id=("subsystem_id", "first"),
            subsystem_count=("subsystem_id", "nunique"),
            plant_name=("plant_name", "first"),
            plant_name_count=("plant_name", "nunique"),
            first_observed_on=("time", "min"),
            last_observed_on=("time", "max"),
            source_years=(
                "time",
                lambda values: ",".join(
                    str(year) for year in sorted(values.dt.year.unique())
                ),
            ),
            dispatch_observations=("time", "size"),
        )
        .reset_index()
    )
    ambiguous = identity["ceg_count"].ne(1) | identity["subsystem_count"].ne(1)
    if ambiguous.any():
        examples = identity.loc[
            ambiguous, ["planning_id", "ceg_count", "subsystem_count"]
        ].head(3).to_dict("records")
        raise ValueError(f"ambiguous training thermal identity crosswalk: {examples}")
    identity.insert(0, "dataset", "ons_thermal_dispatch_crosswalk")
    identity.attrs.update(
        {
            "input_rows": input_rows,
            "dropped_outside_training_years": dropped_outside_training_years,
            "dropped_missing_identity_rows": dropped_missing_identity_rows,
            "retained_identity_observations": int(
                identity["dispatch_observations"].sum()
            ),
        }
    )
    return identity.reset_index(drop=True)


def load_generation_capacity(
    source: FrameSource,
    *,
    active_at: str | pd.Timestamp | None = "2014-12-31",
) -> pd.DataFrame:
    """Normalize ONS's current unit-capacity snapshot with an as-of audit.

    ONS explicitly identifies this source as non-historical.  Commissioning and
    deactivation dates permit a best-effort reconstruction of units active at
    the end of the fit interval, but cannot eliminate snapshot survivor bias.
    The requested cutoff is retained in every output row for disclosure.
    """

    frame = _read_frames(source)
    required = [
        "id_subsistema",
        "nom_tipousina",
        "nom_usina",
        "cod_equipamento",
        "val_potenciaefetiva",
    ]
    _require_columns(frame, required)
    result = pd.DataFrame(
        {
            "dataset": "ons_generation_capacity_snapshot",
            "subsystem_id": frame["id_subsistema"].astype("string").str.strip(),
            "plant_type": frame["nom_tipousina"].astype("string").str.strip(),
            "plant_name": frame["nom_usina"].astype("string").str.strip(),
            "unit_id": frame["cod_equipamento"].astype("string").str.strip(),
            "effective_capacity_mw": _numeric(
                frame["val_potenciaefetiva"], "val_potenciaefetiva"
            ),
        }
    )
    for source_name, target_name in {
        "nom_subsistema": "subsystem_name",
        "ceg": "ceg",
        "nom_unidadegeradora": "unit_name",
        "nom_combustivel": "fuel",
    }.items():
        if source_name in frame:
            result[target_name] = frame[source_name].astype("string").str.strip()
    for source_name, target_name in {
        "dat_entradateste": "commissioning_on",
        "dat_entradaoperacao": "operation_on",
        "dat_desativacao": "deactivated_on",
    }.items():
        result[target_name] = (
            _parse_dates(frame[source_name], source_name)
            if source_name in frame
            else pd.NaT
        )
    if result["unit_id"].isna().any() or result["unit_id"].eq("").any():
        raise ValueError("generation-capacity snapshot contains an empty unit ID")
    if result["unit_id"].duplicated().any():
        raise ValueError("generation-capacity snapshot contains duplicate unit IDs")
    if (result["effective_capacity_mw"] < 0.0).any():
        raise ValueError("generation capacity cannot be negative")
    if active_at is not None:
        cutoff = pd.Timestamp(active_at)
        entered = result["operation_on"].fillna(result["commissioning_on"])
        active = (entered.isna() | entered.le(cutoff)) & (
            result["deactivated_on"].isna() | result["deactivated_on"].gt(cutoff)
        )
        result = result.loc[active].copy()
        result["capacity_reconstruction_as_of"] = cutoff
    else:
        result["capacity_reconstruction_as_of"] = pd.NaT
    return result.reset_index(drop=True)


__all__ = [
    "COMPLETE_END_YEAR",
    "COMPLETE_START_YEAR",
    "COMPLETE_YEARS",
    "ONSCache",
    "ONS_DICTIONARY_URLS",
    "ONS_LICENSE",
    "ONS_LICENSE_URL",
    "ONS_RESOURCES",
    "TEMPORAL_SPLITS",
    "THERMAL_CROSSWALK_YEARS",
    "aggregate_daily_inflow_to_monthly",
    "assign_temporal_split",
    "filter_complete_year_range",
    "load_hydrology_daily",
    "load_hydrology_monthly",
    "load_generation_capacity",
    "load_monthly_load",
    "load_reservoir_static",
    "load_reservoir_ena_daily",
    "load_subsystem_ena_daily",
    "load_thermal_dispatch_crosswalk",
    "load_weekly_cvu",
    "official_url",
    "parse_ons_number",
    "read_ons_csv",
    "sha256_file",
    "validate_inflow_routing",
]
