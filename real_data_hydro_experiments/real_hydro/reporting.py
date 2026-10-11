"""Deterministic, plan-driven reporting for the real-hydrology suites.

The reporter never discovers the intended experiment from whatever result
files happen to exist.  Each input suite's ``suite_plan.json`` is the sole
source of planned case/method/budget/seed keys.  Result directories outside
that plan are rejected, while planned failures, running keys, and missing keys
remain explicit rows in every relevant output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence


SUITE_PLAN_SCHEMA_VERSION = 2
RUNNER_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 1
SDDP_METHOD = "hydrology_state_sddp"
FORMAL_FIT_BUDGETS = (100, 250, 500, 1000, 2500)
REQUIRED_SPLIT_FINGERPRINTS = frozenset(
    {"fit", "validation", "test", "calibration", "evaluation", "historical_replay"}
)

CASE_ORDER = (
    "ons_subsystems",
    "resops_lucky_peak",
    "resops_ross_diablo",
    "camels_br_12",
)
CASE_LABELS = {
    "ons_subsystems": "ONS subsystems",
    "resops_lucky_peak": "ResOps Lucky Peak",
    "resops_ross_diablo": "ResOps Ross–Diablo",
    "camels_br_12": "CAMELS-BR 12",
}
METHOD_ORDER = (
    "seasonal_reference",
    "algorithm1_mlp",
    "algorithm1_graph",
    "algorithm1_deepsets",
    "projected_shac",
    "rl_ppo",
    "rl_sac",
    SDDP_METHOD,
)
METHOD_LABELS = {
    "seasonal_reference": "Seasonal reference (fixed)",
    "algorithm1_mlp": "Algorithm 1 MLP",
    "algorithm1_graph": "Algorithm 1 Graph",
    "algorithm1_deepsets": "Algorithm 1 DeepSets",
    "projected_shac": "Projected SHAC",
    "rl_ppo": "PPO",
    "rl_sac": "SAC",
    SDDP_METHOD: "Hydrology-state SDDP",
}
METHOD_STYLES = {
    "seasonal_reference": ("#666666", "o"),
    "algorithm1_mlp": ("#1f77b4", "s"),
    "algorithm1_graph": ("#2ca02c", "D"),
    "algorithm1_deepsets": ("#17becf", "P"),
    "projected_shac": ("#ff7f0e", "^"),
    "rl_ppo": ("#9467bd", "v"),
    "rl_sac": ("#d62728", "X"),
}

CORE_LONG_FILENAME = "core_data_budget_long.csv"
ANCHOR_LONG_FILENAME = "n_fit_2500_all_methods_long.csv"
RAW_LONG_FILENAME = "raw_metrics_long.csv"
HISTORICAL_LONG_FILENAME = "historical_replay_long.csv"
MARKDOWN_FILENAME = "paper_table.md"
REPORT_MANIFEST_FILENAME = "report_manifest.json"
REPORT_COMPLETED_FILENAME = "COMPLETED.json"


class ReportingError(RuntimeError):
    """Raised when a suite is incomplete or violates its frozen plan."""


@dataclass(frozen=True)
class SuiteAudit:
    role: str
    root: Path
    plan: Mapping[str, object]
    plan_fingerprint: str
    case_manifests: Mapping[str, Mapping[str, object]]
    records: tuple[Mapping[str, object], ...]
    marker_complete: bool
    suite_complete: bool
    counts: Mapping[str, int]


@dataclass(frozen=True)
class ReportingSummary:
    output_dir: Path
    partial: bool
    files: tuple[Path, ...]
    core_counts: Mapping[str, int]
    sddp_counts: Mapping[str, int]
    plot_methods_by_case: Mapping[str, tuple[str, ...]]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportingError(f"Required JSON file is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportingError(f"JSON file must contain an object: {path}")
    return value


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise ReportingError(f"Cannot construct a path component from {value!r}")
    return result


def _plan_identity(plan: Mapping[str, object]) -> dict[str, object]:
    required = (
        "schema_version",
        "experiment",
        "mode",
        "protocol",
        "device",
        "all_case_ids",
        "cases",
    )
    missing = [field for field in required if field not in plan]
    if missing:
        raise ReportingError(f"suite_plan.json is missing fields: {missing}")
    return {field: plan[field] for field in required}


def _validate_plan_fingerprint(plan: Mapping[str, object], path: Path) -> str:
    claimed = plan.get("suite_plan_fingerprint")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ReportingError(f"Invalid suite plan fingerprint in {path}")
    actual = sha256(_canonical_json(_plan_identity(plan))).hexdigest()
    if actual != claimed:
        raise ReportingError(f"Suite plan fingerprint mismatch in {path}")
    return actual


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReportingError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _int_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ReportingError(f"{field} must be a JSON list")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ReportingError(f"{field} must contain integers") from error
    if len(result) != len(set(result)):
        raise ReportingError(f"{field} contains duplicates")
    return result


def _method_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ReportingError(f"{field} must be a JSON list of method names")
    result = tuple(value)
    if not result or len(result) != len(set(result)):
        raise ReportingError(f"{field} must be nonempty and contain no duplicates")
    unknown = sorted(set(result) - set(METHOD_ORDER))
    if unknown:
        raise ReportingError(f"{field} contains unsupported methods: {unknown}")
    return result


def _validate_suite_role(
    role: str,
    protocol: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> None:
    protocol_budgets = _int_tuple(protocol.get("fit_budgets"), "protocol.fit_budgets")
    protocol_seeds = _int_tuple(protocol.get("train_seeds"), "protocol.train_seeds")
    if protocol_budgets != FORMAL_FIT_BUDGETS:
        raise ReportingError(
            f"{role} protocol must retain the full formal fit budgets "
            f"{FORMAL_FIT_BUDGETS}, got {protocol_budgets}"
        )
    if protocol_seeds != (0, 1, 2, 3, 4):
        raise ReportingError(f"{role} protocol must use formal train seeds 0..4")
    max_checks = int(protocol.get("max_validation_checks", -1))
    if role == "core" and max_checks != 50:
        raise ReportingError("Core suite must retain max_validation_checks=50")
    if role == "sddp_anchor" and max_checks != 4:
        raise ReportingError("SDDP anchor must freeze max_validation_checks=4")

    for record in records:
        case_id = str(record["case_id"])
        methods = _method_tuple(record.get("methods"), f"{case_id}.methods")
        budgets = _int_tuple(
            record.get("requested_fit_budgets"),
            f"{case_id}.requested_fit_budgets",
        )
        seeds = _int_tuple(record.get("train_seeds"), f"{case_id}.train_seeds")
        if seeds != protocol_seeds:
            raise ReportingError(f"{case_id} train_seeds do not match the protocol")
        if role == "core":
            if SDDP_METHOD in methods:
                raise ReportingError("Core suite plan must not contain SDDP")
            if budgets != protocol_budgets:
                raise ReportingError(
                    f"Core case {case_id} must schedule every formal fit budget"
                )
        else:
            if methods != (SDDP_METHOD,):
                raise ReportingError(
                    f"SDDP anchor case {case_id} must schedule only {SDDP_METHOD}"
                )
            if budgets != (2500,):
                raise ReportingError(
                    f"SDDP anchor case {case_id} must request only n_fit=2500"
                )
        expected = len(methods) * len(budgets) * len(seeds)
        if int(record.get("expected_runs", -1)) != expected:
            raise ReportingError(f"{case_id} expected_runs does not match its plan")


def _planned_case_records(plan: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    raw = plan.get("cases")
    if not isinstance(raw, list):
        raise ReportingError("suite_plan.cases must be a list")
    records: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    seen_locations: set[tuple[str, str]] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise ReportingError("suite_plan.cases contains a non-object record")
        for field in (
            "case_id",
            "benchmark",
            "system_case_id",
            "methods",
            "requested_fit_budgets",
            "train_seeds",
            "expected_runs",
            "output_dir",
        ):
            if field not in value:
                raise ReportingError(f"Suite case record is missing {field}")
        case_id = str(value["case_id"])
        if case_id not in CASE_ORDER:
            raise ReportingError(f"Unknown planned case_id {case_id!r}")
        if case_id in seen_cases:
            raise ReportingError(f"Duplicate case record in suite plan: {case_id}")
        location = (str(value["benchmark"]), str(value["system_case_id"]))
        if location in seen_locations:
            raise ReportingError(f"Duplicate case output location in suite plan: {location}")
        declared = Path(str(value["output_dir"]))
        if declared.name != location[1] or declared.parent.name != location[0]:
            raise ReportingError(f"Declared output_dir does not match case {case_id}")
        seen_cases.add(case_id)
        seen_locations.add(location)
        records.append(dict(value))
    case_rank = {case_id: index for index, case_id in enumerate(CASE_ORDER)}
    return tuple(sorted(records, key=lambda item: case_rank[str(item["case_id"])]))


def _validate_case_manifest(
    root: Path,
    plan: Mapping[str, object],
    record: Mapping[str, object],
) -> dict[str, object]:
    case_id = str(record["case_id"])
    case_dir = root / _slug(str(record["benchmark"])) / _slug(
        str(record["system_case_id"])
    )
    path = case_dir / "manifest.json"
    manifest = _read_json(path)
    if manifest.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise ReportingError(f"Unsupported runner manifest schema for {case_id}: {path}")
    _require_sha256(
        manifest.get("experiment_fingerprint"),
        f"{case_id}.manifest.experiment_fingerprint",
    )
    system = manifest.get("system")
    if not isinstance(system, Mapping):
        raise ReportingError(f"Case manifest has no system object: {path}")
    if system.get("benchmark") != record["benchmark"]:
        raise ReportingError(f"Case manifest benchmark does not match its plan: {path}")
    if system.get("case_id") != record["system_case_id"]:
        raise ReportingError(f"Case manifest case_id does not match its plan: {path}")
    if manifest.get("protocol") != plan.get("protocol"):
        raise ReportingError(f"Case manifest protocol does not match its suite plan: {path}")
    if manifest.get("device") != plan.get("device"):
        raise ReportingError(f"Case manifest device does not match its suite plan: {path}")
    _require_sha256(
        manifest.get("model_training_fingerprint"),
        f"{case_id}.manifest.model_training_fingerprint",
    )
    raw_splits = manifest.get("split_fingerprints")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != REQUIRED_SPLIT_FINGERPRINTS:
        raise ReportingError(
            f"Case manifest split_fingerprints must contain exactly "
            f"{sorted(REQUIRED_SPLIT_FINGERPRINTS)}: {path}"
        )
    for name, fingerprint in raw_splits.items():
        _require_sha256(fingerprint, f"{case_id}.manifest.split_fingerprints.{name}")
    historical = manifest.get("historical_replay")
    if not isinstance(historical, Mapping):
        raise ReportingError(f"Case manifest has no locked historical replay: {path}")
    if historical.get("locked") is not True or historical.get(
        "evaluated_after_selection"
    ) is not True:
        raise ReportingError(f"Case manifest historical replay is not locked: {path}")
    if historical.get("fingerprint") != raw_splits["historical_replay"]:
        raise ReportingError(
            f"Case manifest historical replay fingerprint is inconsistent: {path}"
        )
    if not isinstance(historical.get("n_paths"), int) or int(
        historical["n_paths"]
    ) < 1:
        raise ReportingError(f"Case manifest historical replay n_paths is invalid: {path}")
    if not isinstance(historical.get("seed"), int):
        raise ReportingError(f"Case manifest historical replay seed is invalid: {path}")
    return manifest


def _run_key_payload(method: str, n_fit: int, train_seed: int) -> dict[str, object]:
    return {"method": method, "n_fit": int(n_fit), "train_seed": int(train_seed)}


def _parse_run_dir(run_dir: Path) -> tuple[str, int, int]:
    try:
        method, budget, seed = run_dir.parts[-3:]
        if not budget.startswith("n_fit_") or not seed.startswith("train_seed_"):
            raise ValueError
        return method, int(budget.removeprefix("n_fit_")), int(
            seed.removeprefix("train_seed_")
        )
    except (TypeError, ValueError) as error:
        raise ReportingError(f"Malformed run directory: {run_dir}") from error


def _safe_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportingError("Result metrics must be finite")
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_complete_result(
    payload: Mapping[str, object],
    expected_key: Mapping[str, object],
    record: Mapping[str, object],
    case_manifest: Mapping[str, object],
    path: Path,
) -> None:
    if payload.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise ReportingError(f"Complete result has the wrong runner schema: {path}")
    if payload.get("status") != "complete" or payload.get("run_key") != dict(expected_key):
        raise ReportingError(f"Complete result has the wrong status/run_key: {path}")
    expected_key_string = (
        f"{expected_key['method']}/n_fit={expected_key['n_fit']}/"
        f"train_seed={expected_key['train_seed']}"
    )
    if payload.get("run_key_string") != expected_key_string:
        raise ReportingError(f"Complete result has the wrong run_key_string: {path}")
    experiment_fingerprint = _require_sha256(
        payload.get("experiment_fingerprint"),
        f"{path}.experiment_fingerprint",
    )
    if experiment_fingerprint != case_manifest.get("experiment_fingerprint"):
        raise ReportingError(
            f"Result experiment fingerprint does not match its case manifest: {path}"
        )
    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        raise ReportingError(f"Complete result has no settings object: {path}")
    if settings.get("method") != expected_key["method"] or settings.get(
        "n_fit"
    ) != expected_key["n_fit"]:
        raise ReportingError(f"Result settings do not match its planned run key: {path}")
    if settings.get("device") != case_manifest.get("device"):
        raise ReportingError(f"Result settings device does not match its manifest: {path}")
    claimed_run_fingerprint = _require_sha256(
        payload.get("run_fingerprint"),
        f"{path}.run_fingerprint",
    )
    expected_run_fingerprint = sha256(
        _canonical_json(
            {
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "experiment_fingerprint": experiment_fingerprint,
                "run_key": dict(expected_key),
                "settings": settings,
            }
        )
    ).hexdigest()
    if claimed_run_fingerprint != expected_run_fingerprint:
        raise ReportingError(f"Result run fingerprint is invalid: {path}")
    system = payload.get("system")
    manifest_system = case_manifest["system"]
    if not isinstance(system, Mapping) or not isinstance(manifest_system, Mapping):
        raise ReportingError(f"Complete result has no system object: {path}")
    expected_system = {
        "benchmark": record["benchmark"],
        "case_id": record["system_case_id"],
        "horizon": manifest_system.get("horizon"),
        "periods": manifest_system.get("periods"),
        "n_reservoirs": len(manifest_system.get("reservoir_ids", [])),
    }
    for field, expected in expected_system.items():
        if system.get(field) != expected:
            raise ReportingError(
                f"Result system.{field} does not match its case manifest/plan: {path}"
            )
    data = payload.get("data")
    historical = case_manifest["historical_replay"]
    if not isinstance(data, Mapping) or not isinstance(historical, Mapping):
        raise ReportingError(f"Complete result has no data audit: {path}")
    if data.get("fit_paths") != expected_key["n_fit"]:
        raise ReportingError(f"Result fit_paths does not match n_fit: {path}")
    if data.get("historical_replay_fingerprint") != historical.get("fingerprint"):
        raise ReportingError(
            f"Result historical replay fingerprint does not match its manifest: {path}"
        )
    if data.get("historical_replay_paths") != historical.get("n_paths") or data.get(
        "historical_replay_seed"
    ) != historical.get("seed"):
        raise ReportingError(
            f"Result historical replay metadata does not match its manifest: {path}"
        )
    metrics = payload.get("metrics")
    required_metrics = {"validation", "test", "evaluation", "historical_replay"}
    if not isinstance(metrics, Mapping) or set(metrics) != required_metrics:
        raise ReportingError(f"Complete result has the wrong metric interfaces: {path}")
    for name in required_metrics:
        if not isinstance(metrics[name], Mapping):
            raise ReportingError(f"Complete result has invalid {name} metrics: {path}")
    cost = metrics["test"].get("cost_mean")
    if not isinstance(cost, (int, float)) or not math.isfinite(float(cost)):
        raise ReportingError(f"Complete result has invalid test cost_mean: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReportingError(f"Complete result has no artifacts object: {path}")
    for artifact_name in ("checkpoint", "path_costs"):
        artifact = artifacts.get(artifact_name)
        if not isinstance(artifact, Mapping):
            raise ReportingError(f"Complete result has no {artifact_name} artifact: {path}")
        filename = artifact.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ReportingError(f"Result artifact has an unsafe filename: {path}")
        artifact_path = path.parent / filename
        if not artifact_path.is_file():
            raise ReportingError(f"Result artifact is missing: {artifact_path}")
        if not isinstance(artifact.get("size_bytes"), int) or artifact_path.stat().st_size != int(
            artifact["size_bytes"]
        ):
            raise ReportingError(f"Result artifact size mismatch: {artifact_path}")
        expected_hash = _require_sha256(
            artifact.get("sha256"), f"{path}.artifacts.{artifact_name}.sha256"
        )
        if _sha256_file(artifact_path) != expected_hash:
            raise ReportingError(f"Result artifact hash mismatch: {artifact_path}")


def _case_data_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "system_fingerprint": sha256(_canonical_json(manifest["system"])).hexdigest(),
        "model_training_fingerprint": manifest["model_training_fingerprint"],
        "split_fingerprints": dict(manifest["split_fingerprints"]),
        "historical_replay": dict(manifest["historical_replay"]),
    }


def _validate_cross_suite_case_identity(core: SuiteAudit, anchor: SuiteAudit) -> None:
    core_cases = set(core.case_manifests)
    anchor_cases = set(anchor.case_manifests)
    if core_cases != anchor_cases:
        raise ReportingError(
            "Core and SDDP anchor suite plans must contain the same cases for reporting"
        )
    for case_id in sorted(core_cases):
        left = _case_data_identity(core.case_manifests[case_id])
        right = _case_data_identity(anchor.case_manifests[case_id])
        if left["system_fingerprint"] != right["system_fingerprint"]:
            raise ReportingError(
                f"Core/anchor system fingerprints differ for case {case_id}"
            )
        if left["model_training_fingerprint"] != right["model_training_fingerprint"]:
            raise ReportingError(
                f"Core/anchor model training fingerprints differ for case {case_id}"
            )
        if left["split_fingerprints"] != right["split_fingerprints"]:
            raise ReportingError(
                f"Core/anchor split fingerprints differ for case {case_id}"
            )
        if left["historical_replay"] != right["historical_replay"]:
            raise ReportingError(
                f"Core/anchor historical replay identities differ for case {case_id}"
            )


def _completion_marker_valid(
    root: Path,
    plan_fingerprint: str,
    planned_cases: Sequence[Mapping[str, object]],
) -> bool:
    marker_path = root / "COMPLETED.json"
    if not marker_path.is_file():
        return False
    marker = _read_json(marker_path)
    if marker.get("suite_plan_fingerprint") != plan_fingerprint:
        raise ReportingError(f"COMPLETED.json has the wrong suite-plan fingerprint: {root}")
    planned_ids = {str(record["case_id"]) for record in planned_cases}
    marker_ids = marker.get("case_ids")
    if not isinstance(marker_ids, list) or set(map(str, marker_ids)) != planned_ids:
        raise ReportingError(f"COMPLETED.json case_ids do not match suite_plan.json: {root}")
    return True


def _load_suite(root: str | Path, role: str) -> SuiteAudit:
    root = Path(root).resolve()
    plan_path = root / "suite_plan.json"
    plan = _read_json(plan_path)
    if plan.get("schema_version") != SUITE_PLAN_SCHEMA_VERSION:
        raise ReportingError(f"Unsupported suite plan schema in {plan_path}")
    if plan.get("experiment") != "real_hydrology_suite" or plan.get("mode") != "formal":
        raise ReportingError(f"Reporter accepts only formal real-hydrology suites: {root}")
    plan_fingerprint = _validate_plan_fingerprint(plan, plan_path)
    all_case_ids = plan.get("all_case_ids")
    if not isinstance(all_case_ids, list) or tuple(all_case_ids) != CASE_ORDER:
        raise ReportingError("suite_plan.all_case_ids does not match the frozen four cases")
    case_records = _planned_case_records(plan)
    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ReportingError("suite_plan.protocol must be an object")
    _validate_suite_role(role, protocol, case_records)

    case_manifests = {
        str(record["case_id"]): _validate_case_manifest(root, plan, record)
        for record in case_records
    }

    locator_to_record: dict[tuple[str, str], Mapping[str, object]] = {
        (str(record["benchmark"]), str(record["system_case_id"])): record
        for record in case_records
    }
    expected: dict[tuple[str, str, int, int], tuple[Mapping[str, object], Path]] = {}
    for record in case_records:
        case_id = str(record["case_id"])
        case_dir = (
            root
            / _slug(str(record["benchmark"]))
            / _slug(str(record["system_case_id"]))
        )
        for method in _method_tuple(record["methods"], f"{case_id}.methods"):
            for n_fit in _int_tuple(
                record["requested_fit_budgets"], f"{case_id}.requested_fit_budgets"
            ):
                for train_seed in _int_tuple(
                    record["train_seeds"], f"{case_id}.train_seeds"
                ):
                    key = (case_id, method, n_fit, train_seed)
                    if key in expected:
                        raise ReportingError(f"Duplicate planned run key: {key}")
                    expected[key] = (
                        record,
                        case_dir
                        / _slug(method)
                        / f"n_fit_{n_fit}"
                        / f"train_seed_{train_seed}",
                    )

    seen_actual: set[tuple[str, str, int, int]] = set()
    pattern = "*/*/*/n_fit_*/train_seed_*"
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        relative = run_dir.relative_to(root)
        benchmark, system_case_id = relative.parts[:2]
        record = locator_to_record.get((benchmark, system_case_id))
        method, n_fit, train_seed = _parse_run_dir(run_dir)
        if record is None:
            raise ReportingError(f"Unplanned run directory: {run_dir}")
        key = (str(record["case_id"]), method, n_fit, train_seed)
        if key not in expected:
            raise ReportingError(f"Unplanned run key: {key}")
        if key in seen_actual:
            raise ReportingError(f"Duplicate run key on disk: {key}")
        seen_actual.add(key)

    recognized_dirs = {path.resolve() for _, path in expected.values()}
    for filename in ("result.json", "failure.json", "status.json"):
        for artifact in root.rglob(filename):
            if artifact.parent.resolve() not in recognized_dirs:
                raise ReportingError(f"Run artifact is outside the suite plan: {artifact}")

    rows: list[dict[str, object]] = []
    counts = {"planned": len(expected), "complete": 0, "failed": 0, "running": 0, "missing": 0}
    case_rank = {case_id: index for index, case_id in enumerate(CASE_ORDER)}
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    for key in sorted(
        expected,
        key=lambda item: (case_rank[item[0]], method_rank[item[1]], item[2], item[3]),
    ):
        case_id, method, n_fit, train_seed = key
        record, run_dir = expected[key]
        expected_payload = _run_key_payload(method, n_fit, train_seed)
        result_path = run_dir / "result.json"
        failure_path = run_dir / "failure.json"
        status_path = run_dir / "status.json"
        result: Mapping[str, object] | None = None
        failure: Mapping[str, object] | None = None
        if result_path.is_file():
            if failure_path.is_file():
                raise ReportingError(f"Run has both result.json and failure.json: {run_dir}")
            result = _read_json(result_path)
            _validate_complete_result(
                result,
                expected_payload,
                record,
                case_manifests[case_id],
                result_path,
            )
            run_status = "complete"
        elif failure_path.is_file():
            failure = _read_json(failure_path)
            if failure.get("status") != "failed" or failure.get("run_key") != expected_payload:
                raise ReportingError(f"Failure record has the wrong status/run_key: {failure_path}")
            run_status = "failed"
        elif status_path.is_file():
            status = _read_json(status_path)
            if status.get("run_key") != expected_payload:
                raise ReportingError(f"Status record has the wrong run_key: {status_path}")
            run_status = "running" if status.get("status") == "running" else "missing"
        else:
            run_status = "missing"
        counts[run_status] += 1
        rows.append(
            {
                "suite_role": role,
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id],
                "benchmark": str(record["benchmark"]),
                "system_case_id": str(record["system_case_id"]),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "fixed_policy": method == "seasonal_reference",
                "n_fit": n_fit,
                "train_seed": train_seed,
                "run_status": run_status,
                "result_path": str(result_path),
                "result": result,
                "failure": failure,
            }
        )

    marker_complete = _completion_marker_valid(root, plan_fingerprint, case_records)
    all_cases_planned = {str(record["case_id"]) for record in case_records} == set(CASE_ORDER)
    suite_complete = (
        marker_complete
        and all_cases_planned
        and counts["complete"] == counts["planned"]
        and counts["failed"] == counts["running"] == counts["missing"] == 0
    )
    return SuiteAudit(
        role=role,
        root=root,
        plan=plan,
        plan_fingerprint=plan_fingerprint,
        case_manifests=case_manifests,
        records=tuple(rows),
        marker_complete=marker_complete,
        suite_complete=suite_complete,
        counts=counts,
    )


def _flatten_mapping(prefix: str, payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    return {f"{prefix}{key}": _safe_scalar(value) for key, value in payload.items()}


def _run_row(record: Mapping[str, object], report_status: str) -> dict[str, object]:
    row = {
        key: record[key]
        for key in (
            "suite_role",
            "case_id",
            "case_label",
            "benchmark",
            "system_case_id",
            "method",
            "method_label",
            "fixed_policy",
            "n_fit",
            "train_seed",
            "run_status",
            "result_path",
        )
    }
    row["report_status"] = report_status
    result = record.get("result")
    failure = record.get("failure")
    if isinstance(result, Mapping):
        row["experiment_fingerprint"] = result.get("experiment_fingerprint")
        row["run_fingerprint"] = result.get("run_fingerprint")
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            for interface in ("validation", "test", "evaluation", "historical_replay"):
                row.update(_flatten_mapping(f"{interface}_", metrics.get(interface)))
        row.update(_flatten_mapping("training_", result.get("training")))
        selection = result.get("selection")
        if isinstance(selection, Mapping):
            for field in (
                "selected_candidate_index",
                "selected_validation_cost",
                "selected_validation_cost_raw",
                "selected_validation_cost_normalized",
            ):
                if field in selection:
                    row[f"selection_{field}"] = _safe_scalar(selection[field])
    if isinstance(failure, Mapping):
        row["failure_type"] = failure.get("exception_type")
        row["failure_message"] = failure.get("message")
    return row


def _raw_metric_rows(
    records: Iterable[Mapping[str, object]], report_status: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        base = {
            key: record[key]
            for key in (
                "suite_role",
                "case_id",
                "method",
                "n_fit",
                "train_seed",
                "run_status",
            )
        }
        base["report_status"] = report_status
        result = record.get("result")
        metrics = result.get("metrics") if isinstance(result, Mapping) else None
        emitted = False
        if isinstance(metrics, Mapping):
            for interface in sorted(metrics):
                values = metrics[interface]
                if not isinstance(values, Mapping):
                    continue
                for metric in sorted(values):
                    rows.append(
                        {
                            **base,
                            "interface": interface,
                            "metric": metric,
                            "value": _safe_scalar(values[metric]),
                        }
                    )
                    emitted = True
        if not emitted:
            failure = record.get("failure")
            rows.append(
                {
                    **base,
                    "interface": "",
                    "metric": "",
                    "value": "",
                    "failure_type": (
                        failure.get("exception_type")
                        if isinstance(failure, Mapping)
                        else ""
                    ),
                    "failure_message": (
                        failure.get("message") if isinstance(failure, Mapping) else ""
                    ),
                }
            )
    return rows


def _historical_rows(
    records: Iterable[Mapping[str, object]], report_status: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = {
            key: record[key]
            for key in (
                "suite_role",
                "case_id",
                "method",
                "n_fit",
                "train_seed",
                "run_status",
            )
        }
        row["report_status"] = report_status
        result = record.get("result")
        metrics = result.get("metrics") if isinstance(result, Mapping) else None
        historical = metrics.get("historical_replay") if isinstance(metrics, Mapping) else None
        row.update(_flatten_mapping("historical_replay_", historical))
        failure = record.get("failure")
        if isinstance(failure, Mapping):
            row["failure_type"] = failure.get("exception_type")
            row["failure_message"] = failure.get("message")
        rows.append(row)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], preferred: Sequence[str]) -> None:
    fields = list(preferred)
    extras = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extras)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    _atomic_text(path, stream.getvalue())


def _test_cost(record: Mapping[str, object]) -> float | None:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("test"), Mapping):
        return None
    value = metrics["test"].get("cost_mean")
    return float(value) if isinstance(value, (int, float)) else None


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("At least one value is required")
    return float(statistics.fmean(values)), (
        float(statistics.stdev(values)) if len(values) > 1 else 0.0
    )


def _format_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1.0e6 or (0.0 < magnitude < 1.0e-3):
        return f"{value:.3e}"
    return f"{value:,.3f}"


def _paper_markdown(
    records_2500: Sequence[Mapping[str, object]],
    core: SuiteAudit,
    anchor: SuiteAudit,
    partial: bool,
) -> str:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in records_2500:
        grouped.setdefault((str(record["case_id"]), str(record["method"])), []).append(
            record
        )
    lines = ["# Real-hydrology formal comparison", ""]
    if partial:
        lines.extend(
            [
                "> **PARTIAL PREVIEW — failures and missing runs are not silently dropped.**",
                "",
            ]
        )
    lines.extend(
        [
            "Lower test cost is better. Cells report mean ± train-seed SD and successful runs/planned seeds at `n_fit=2500`.",
            "",
            f"Core audit: {core.counts['complete']}/{core.counts['planned']} complete, {core.counts['failed']} failed, {core.counts['running']} running, {core.counts['missing']} missing.",
            "",
            f"SDDP anchor audit: {anchor.counts['complete']}/{anchor.counts['planned']} complete, {anchor.counts['failed']} failed, {anchor.counts['running']} running, {anchor.counts['missing']} missing.",
            "",
            "| Case | " + " | ".join(METHOD_LABELS[method] for method in METHOD_ORDER) + " |",
            "| --- | " + " | ".join("---:" for _ in METHOD_ORDER) + " |",
        ]
    )
    for case_id in CASE_ORDER:
        cells = []
        for method in METHOD_ORDER:
            planned = grouped.get((case_id, method), [])
            if not planned:
                cells.append("—")
                continue
            values = [value for value in (_test_cost(item) for item in planned) if value is not None]
            successes = len(values)
            expected = len(planned)
            if not values:
                cells.append(f"— ({successes}/{expected})")
                continue
            mean, sd = _mean_sd(values)
            cells.append(
                f"{_format_number(mean)} ± {_format_number(sd)} ({successes}/{expected})"
            )
        lines.append(f"| {CASE_LABELS[case_id]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The seasonal reference is fixed; repeated seed rows verify the common evaluation interface rather than represent fitted-policy randomness.",
            "",
            f"Per-run historical replay metrics are retained in `{HISTORICAL_LONG_FILENAME}` and all unaggregated interface metrics in `{RAW_LONG_FILENAME}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _budget_series(
    core_records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, list[dict[str, object]]]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    for record in core_records:
        method = str(record["method"])
        if method == SDDP_METHOD:
            continue
        key = (str(record["case_id"]), method, int(record["n_fit"]))
        grouped.setdefault(key, []).append(record)
    result: dict[str, dict[str, list[dict[str, object]]]] = {
        case_id: {} for case_id in CASE_ORDER
    }
    for (case_id, method, n_fit), records in grouped.items():
        values = [value for value in (_test_cost(item) for item in records) if value is not None]
        if not values:
            continue
        mean, sd = _mean_sd(values)
        result[case_id].setdefault(method, []).append(
            {
                "n_fit": n_fit,
                "mean": mean,
                "sd": sd,
                "success": len(values),
                "expected": len(records),
            }
        )
    for methods in result.values():
        for points in methods.values():
            points.sort(key=lambda item: int(item["n_fit"]))
    return result


def _write_plots(
    output_dir: Path,
    series: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    partial: bool,
) -> tuple[list[Path], dict[str, tuple[str, ...]]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ReportingError(
            "matplotlib is required to write data-budget PNGs; install it or run "
            "the reporter in the frozen scientific environment"
        ) from error

    paths: list[Path] = []
    plotted: dict[str, tuple[str, ...]] = {}
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    for case_id in CASE_ORDER:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        methods = tuple(
            sorted(series[case_id], key=lambda method: method_rank[method])
        )
        if SDDP_METHOD in methods:
            raise AssertionError("SDDP must never enter a data-budget curve")
        plotted[case_id] = methods
        for method in methods:
            points = series[case_id][method]
            x = [int(point["n_fit"]) for point in points]
            y = [float(point["mean"]) for point in points]
            yerr = [float(point["sd"]) for point in points]
            color, marker = METHOD_STYLES[method]
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                marker=marker,
                color=color,
                linewidth=1.6,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        if not methods:
            axis.text(0.5, 0.5, "No completed planned runs", ha="center", va="center")
        axis.set_xscale("log")
        axis.set_xlabel("Unique fit trajectories (n_fit)")
        axis.set_ylabel("Test cost (mean ± train-seed SD; lower is better)")
        prefix = "PARTIAL PREVIEW — " if partial else ""
        axis.set_title(f"{prefix}{CASE_LABELS[case_id]} data budget")
        axis.grid(True, alpha=0.25)
        if methods:
            axis.legend(fontsize=8)
        figure.tight_layout()
        destination = output_dir / f"data_budget_{case_id}.png"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".png",
                dir=output_dir,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            figure.savefig(
                temporary,
                dpi=180,
                metadata={"Software": "real_hydro.reporting"},
            )
            os.replace(temporary, destination)
            temporary = None
        finally:
            plt.close(figure)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        paths.append(destination)
    return paths, plotted


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    core_suite: str | Path,
    sddp_suite: str | Path,
    output_dir: str | Path,
    *,
    allow_partial: bool = False,
) -> ReportingSummary:
    """Audit two formal suites and write deterministic paper/report artifacts."""

    core = _load_suite(core_suite, "core")
    anchor = _load_suite(sddp_suite, "sddp_anchor")
    _validate_cross_suite_case_identity(core, anchor)
    if not allow_partial and (not core.suite_complete or not anchor.suite_complete):
        raise ReportingError(
            "Both suites must be complete; pass --allow-partial only for an explicitly "
            f"marked preview (core={dict(core.counts)}, sddp={dict(anchor.counts)})"
        )
    partial = not (core.suite_complete and anchor.suite_complete)
    report_status = "partial" if partial else "complete"

    core_records = list(core.records)
    anchor_records = list(anchor.records)
    records_2500 = [
        record for record in (*core_records, *anchor_records) if int(record["n_fit"]) == 2500
    ]
    combined_keys: set[tuple[str, str, int, int]] = set()
    for record in records_2500:
        key = (
            str(record["case_id"]),
            str(record["method"]),
            int(record["n_fit"]),
            int(record["train_seed"]),
        )
        if key in combined_keys:
            raise ReportingError(f"Duplicate core/anchor run key: {key}")
        combined_keys.add(key)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    core_rows = [_run_row(record, report_status) for record in core_records]
    anchor_rows = [_run_row(record, report_status) for record in records_2500]
    all_records = [*core_records, *anchor_records]
    raw_rows = _raw_metric_rows(all_records, report_status)
    historical_rows = _historical_rows(all_records, report_status)

    common_fields = (
        "report_status",
        "suite_role",
        "case_id",
        "case_label",
        "benchmark",
        "system_case_id",
        "method",
        "method_label",
        "fixed_policy",
        "n_fit",
        "train_seed",
        "run_status",
        "test_cost_mean",
        "test_cost_cvar95",
        "historical_replay_cost_mean",
        "failure_type",
        "failure_message",
        "experiment_fingerprint",
        "run_fingerprint",
        "result_path",
    )
    core_path = output_dir / CORE_LONG_FILENAME
    anchor_path = output_dir / ANCHOR_LONG_FILENAME
    raw_path = output_dir / RAW_LONG_FILENAME
    historical_path = output_dir / HISTORICAL_LONG_FILENAME
    markdown_path = output_dir / MARKDOWN_FILENAME
    _write_csv(core_path, core_rows, common_fields)
    _write_csv(anchor_path, anchor_rows, common_fields)
    _write_csv(
        raw_path,
        raw_rows,
        (
            "report_status",
            "suite_role",
            "case_id",
            "method",
            "n_fit",
            "train_seed",
            "run_status",
            "interface",
            "metric",
            "value",
            "failure_type",
            "failure_message",
        ),
    )
    _write_csv(
        historical_path,
        historical_rows,
        (
            "report_status",
            "suite_role",
            "case_id",
            "method",
            "n_fit",
            "train_seed",
            "run_status",
            "historical_replay_cost_mean",
            "historical_replay_cost_cvar95",
            "failure_type",
            "failure_message",
        ),
    )
    _atomic_text(
        markdown_path,
        _paper_markdown(records_2500, core, anchor, partial),
    )
    series = _budget_series(core_records)
    plot_paths, plotted = _write_plots(output_dir, series, partial)

    artifact_paths = [
        core_path,
        anchor_path,
        raw_path,
        historical_path,
        markdown_path,
        *plot_paths,
    ]
    manifest_path = output_dir / REPORT_MANIFEST_FILENAME
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_status": report_status,
        "lower_is_better": True,
        "core_suite": {
            "root": str(core.root),
            "suite_plan_fingerprint": core.plan_fingerprint,
            "counts": dict(core.counts),
            "marker_complete": core.marker_complete,
        },
        "sddp_anchor_suite": {
            "root": str(anchor.root),
            "suite_plan_fingerprint": anchor.plan_fingerprint,
            "counts": dict(anchor.counts),
            "marker_complete": anchor.marker_complete,
        },
        "method_order": list(METHOD_ORDER),
        "method_labels": dict(METHOD_LABELS),
        "case_order": list(CASE_ORDER),
        "data_budget_plot_methods": {
            case_id: list(methods) for case_id, methods in plotted.items()
        },
        "artifacts": {
            path.name: {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    completed_path = output_dir / REPORT_COMPLETED_FILENAME
    if partial:
        completed_path.unlink(missing_ok=True)
        final_files = (*artifact_paths, manifest_path)
    else:
        _atomic_text(
            completed_path,
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "experiment": "real_hydrology_reporting",
                    "status": "complete",
                    "core_suite_plan_fingerprint": core.plan_fingerprint,
                    "sddp_suite_plan_fingerprint": anchor.plan_fingerprint,
                    "report_manifest": REPORT_MANIFEST_FILENAME,
                    "report_manifest_sha256": _sha256_file(manifest_path),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
        )
        final_files = (*artifact_paths, manifest_path, completed_path)
    return ReportingSummary(
        output_dir=output_dir,
        partial=partial,
        files=tuple(final_files),
        core_counts=core.counts,
        sddp_counts=anchor.counts,
        plot_methods_by_case=plotted,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m real_hydro.reporting",
        description="Audit and summarize the frozen core and SDDP-anchor suites",
    )
    parser.add_argument("--core-suite", type=Path, required=True)
    parser.add_argument("--sddp-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write an unmistakably marked preview when planned runs are incomplete",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_report(
            args.core_suite,
            args.sddp_suite,
            args.output_dir,
            allow_partial=args.allow_partial,
        )
    except ReportingError as error:
        raise SystemExit(f"error: {error}") from error
    state = "PARTIAL PREVIEW" if summary.partial else "complete"
    print(f"report {state}: {summary.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_LABELS",
    "CASE_ORDER",
    "METHOD_LABELS",
    "METHOD_ORDER",
    "ReportingError",
    "ReportingSummary",
    "SuiteAudit",
    "build_parser",
    "build_report",
    "main",
]
