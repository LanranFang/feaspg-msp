"""Command line entry points for the frozen real-hydrology suite.

The CLI deliberately separates immutable data validation from optimization.
``audit`` never downloads or trains, while ``run`` consumes only prepared,
content-addressed case bundles.  Every run is delegated to the per-key atomic
runner, so reissuing the same command after a disconnect is safe.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import FormalProtocol
from .preparation import (
    CASE_IDS,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    PreparedCase,
    load_prepared_case,
    sha256_file,
)
from .runner import (
    DEFAULT_METHODS,
    RunnerExecutionError,
    resolve_requested_fit_budgets,
    run_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUITE_SCHEMA_VERSION = 2
SUITE_PLAN_FILENAME = "suite_plan.json"

CASE_METHODS: Mapping[str, tuple[str, ...]] = {
    "ons_subsystems": DEFAULT_METHODS,
    "resops_lucky_peak": DEFAULT_METHODS,
    "resops_ross_diablo": (*DEFAULT_METHODS, "algorithm1_graph"),
    "camels_br_12": (*DEFAULT_METHODS, "algorithm1_deepsets"),
}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _selected_cases(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "all", False):
        return CASE_IDS
    values = tuple(dict.fromkeys(getattr(args, "case", ()) or ()))
    if not values:
        raise ValueError("choose --all or at least one --case")
    return values


def _resolve_raw_path(record: Mapping[str, object], raw_root: Path) -> Path:
    value = Path(str(record["path"]))
    return value if value.is_absolute() else raw_root / value


def audit_case(
    case_id: str,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Validate one prepared bundle and every raw artifact in its manifest."""

    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    case = load_prepared_case(case_id, processed_root)
    records = case.provenance.get("raw_files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{case_id}: provenance has no raw_files manifest")
    total_bytes = 0
    checked = 0
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError(f"{case_id}: malformed raw manifest record")
        path = _resolve_raw_path(item, raw_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_bytes = int(item["bytes"])
        actual_bytes = int(path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{case_id}: byte-count mismatch for {path}: "
                f"{actual_bytes} != {expected_bytes}"
            )
        if verify_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != str(item["sha256"]):
                raise ValueError(f"{case_id}: SHA-256 mismatch for {path}")
        total_bytes += actual_bytes
        checked += 1

    if case.model.training_fingerprint != case.provenance.get(
        "fit_fingerprint_sha256"
    ):
        raise ValueError(f"{case_id}: stochastic-model fingerprint mismatch")
    if case.model.order > int(case.provenance.get("par_max_order", -1)):
        raise ValueError(f"{case_id}: selected PAR order exceeds frozen maximum")
    if case.system.horizon < 1 or case.system.periods < 1:
        raise ValueError(f"{case_id}: invalid horizon/calendar")
    if not np.array_equal(
        np.unique(case.fit_calendar), np.arange(case.system.periods)
    ):
        raise ValueError(f"{case_id}: fit history does not cover every calendar period")

    return {
        "case_id": case_id,
        "status": "valid",
        "benchmark": case.system.benchmark,
        "system_case_id": case.system.case_id,
        "reservoirs": case.system.n_reservoirs,
        "horizon": case.system.horizon,
        "fit_rows": int(len(case.fit_inflow)),
        "fit_series": int(case.fit_inflow.shape[1]),
        "par_order": int(case.model.order),
        "annual_component": bool(case.annual_component),
        "historical_replay_paths": (
            0 if case.historical_replay is None else int(case.historical_replay.n)
        ),
        "raw_files_checked": checked,
        "raw_bytes_checked": total_bytes,
        "sha256_checked": bool(verify_hashes),
        "fit_fingerprint_sha256": case.model.training_fingerprint,
    }


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is unavailable")
    return value


def _output_base(output_root: Path, smoke: bool) -> Path:
    return output_root / ("smoke_v1" if smoke else "formal_v1")


def _case_output_dir(base: Path, case: PreparedCase) -> Path:
    return base / case.system.benchmark / case.system.case_id


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _case_plan_record(
    case: PreparedCase,
    methods: Sequence[str],
    requested_fit_budgets: Sequence[int],
    protocol: FormalProtocol,
    output_base: Path,
) -> dict[str, object]:
    methods = tuple(str(method) for method in methods)
    budgets = tuple(int(value) for value in requested_fit_budgets)
    return {
        "case_id": case.case_id,
        "benchmark": case.system.benchmark,
        "system_case_id": case.system.case_id,
        "methods": list(methods),
        "requested_fit_budgets": list(budgets),
        "train_seeds": list(protocol.train_seeds),
        "expected_runs": len(methods) * len(budgets) * len(protocol.train_seeds),
        "output_dir": str(_case_output_dir(output_base, case).resolve()),
    }


def _suite_plan_identity(
    *,
    mode: str,
    protocol: FormalProtocol,
    device: str,
    cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    case_order = {case_id: index for index, case_id in enumerate(CASE_IDS)}
    ordered = sorted(cases, key=lambda item: case_order[str(item["case_id"])])
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "experiment": "real_hydrology_suite",
        "mode": mode,
        "protocol": asdict(protocol),
        "device": device,
        "all_case_ids": list(CASE_IDS),
        "cases": [dict(record) for record in ordered],
    }


def _plan_fingerprint(identity: Mapping[str, object]) -> str:
    return sha256(_canonical_json(identity)).hexdigest()


def _run_directory_key(run_dir: Path) -> tuple[str, int, int] | None:
    try:
        method, budget_part, seed_part = run_dir.parts[-3:]
        return (
            method,
            int(budget_part.removeprefix("n_fit_")),
            int(seed_part.removeprefix("train_seed_")),
        )
    except (TypeError, ValueError):
        return None


def _unexpected_run_directories(
    output_base: Path,
    case_records: Sequence[Mapping[str, object]],
) -> list[str]:
    """Return existing run directories outside the declared suite schedule."""

    declared: list[tuple[Path, set[tuple[str, int, int]]]] = []
    for record in case_records:
        case_dir = Path(str(record["output_dir"])).resolve()
        keys = {
            (str(method), int(n_fit), int(train_seed))
            for method in record["methods"]
            for n_fit in record["requested_fit_budgets"]
            for train_seed in record["train_seeds"]
        }
        declared.append((case_dir, keys))

    unexpected: list[str] = []
    pattern = "*/*/*/n_fit_*/train_seed_*"
    for run_dir in output_base.resolve().glob(pattern):
        if not run_dir.is_dir():
            continue
        match = next(
            (
                (case_dir, keys)
                for case_dir, keys in declared
                if run_dir.is_relative_to(case_dir)
            ),
            None,
        )
        key = _run_directory_key(run_dir)
        if match is None or key is None or key not in match[1]:
            unexpected.append(str(run_dir))
    return sorted(unexpected)


def _merge_suite_plan(
    output_base: Path,
    *,
    mode: str,
    protocol: FormalProtocol,
    device: str,
    proposed_cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Create or monotonically extend the immutable per-case suite schedule."""

    plan_path = output_base / SUITE_PLAN_FILENAME
    current = _read_json(plan_path) if plan_path.is_file() else None
    current_cases: dict[str, dict[str, object]] = {}
    created_at = datetime.now(timezone.utc).isoformat()
    if current is not None:
        if current.get("schema_version") != SUITE_SCHEMA_VERSION:
            raise ValueError("Existing suite plan has an unsupported schema version")
        for field, expected in (
            ("experiment", "real_hydrology_suite"),
            ("mode", mode),
            ("protocol", asdict(protocol)),
            ("device", device),
            ("all_case_ids", list(CASE_IDS)),
        ):
            if _canonical_json(current.get(field)) != _canonical_json(expected):
                raise ValueError(
                    f"Existing suite plan has a different {field}; use a new output root"
                )
        raw_cases = current.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("Existing suite plan has no valid cases list")
        for record in raw_cases:
            if not isinstance(record, dict) or "case_id" not in record:
                raise ValueError("Existing suite plan contains a malformed case record")
            current_cases[str(record["case_id"])] = dict(record)
        created_at = str(current.get("created_at", created_at))

    for proposed in proposed_cases:
        case_id = str(proposed["case_id"])
        previous = current_cases.get(case_id)
        if previous is not None and previous != dict(proposed):
            raise ValueError(
                f"Suite plan already freezes a different schedule for {case_id}; "
                "use a new output root"
            )
        current_cases[case_id] = dict(proposed)

    identity = _suite_plan_identity(
        mode=mode,
        protocol=protocol,
        device=device,
        cases=tuple(current_cases.values()),
    )
    unexpected = _unexpected_run_directories(output_base, identity["cases"])
    if unexpected:
        preview = ", ".join(unexpected[:3])
        suffix = "" if len(unexpected) <= 3 else ", ..."
        raise ValueError(
            "Existing output contains run keys outside the requested suite plan; "
            f"use a new output root ({preview}{suffix})"
        )
    now = datetime.now(timezone.utc).isoformat()
    plan = {
        **identity,
        "suite_plan_fingerprint": _plan_fingerprint(identity),
        "created_at": created_at,
        "updated_at": now,
    }
    if current != plan:
        _atomic_json(plan_path, plan)
    return plan


def _matches_run_key(payload: Mapping[str, object] | None, key: Mapping[str, object]) -> bool:
    return payload is not None and payload.get("run_key") == dict(key)


def _case_progress(record: Mapping[str, object]) -> dict[str, object]:
    case_dir = Path(str(record["output_dir"]))
    complete = 0
    failed = 0
    running = 0
    for method in record["methods"]:
        for n_fit in record["requested_fit_budgets"]:
            for train_seed in record["train_seeds"]:
                key = {
                    "method": str(method),
                    "n_fit": int(n_fit),
                    "train_seed": int(train_seed),
                }
                run_dir = (
                    case_dir
                    / str(method)
                    / f"n_fit_{int(n_fit)}"
                    / f"train_seed_{int(train_seed)}"
                )
                result = _read_json(run_dir / "result.json")
                if _matches_run_key(result, key) and result.get("status") == "complete":
                    complete += 1
                    continue
                failure = _read_json(run_dir / "failure.json")
                if _matches_run_key(failure, key) and failure.get("status") == "failed":
                    failed += 1
                    continue
                status = _read_json(run_dir / "status.json")
                if _matches_run_key(status, key) and status.get("status") == "running":
                    running += 1
    expected = int(record["expected_runs"])
    return {
        "case_id": str(record["case_id"]),
        "planned": True,
        "complete": complete,
        "expected": expected,
        "failed_files": failed,
        "running_files": running,
        "missing": max(0, expected - complete),
        "methods": list(record["methods"]),
        "requested_fit_budgets": list(record["requested_fit_budgets"]),
        "output_dir": str(case_dir),
    }


def _write_suite_state(output_base: Path, plan: Mapping[str, object]) -> dict[str, object]:
    rows = [_case_progress(record) for record in plan["cases"]]
    unexpected = _unexpected_run_directories(output_base, plan["cases"])
    all_cases_planned = {row["case_id"] for row in rows} == set(CASE_IDS)
    complete = (
        all_cases_planned
        and not unexpected
        and all(
            row["complete"] == row["expected"]
            and row["failed_files"] == 0
            and row["running_files"] == 0
            for row in rows
        )
    )
    status = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "experiment": "real_hydrology_suite",
        "mode": plan["mode"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_plan": SUITE_PLAN_FILENAME,
        "suite_plan_fingerprint": plan["suite_plan_fingerprint"],
        "protocol": plan["protocol"],
        "device": plan["device"],
        "all_cases_planned": all_cases_planned,
        "suite_complete": complete,
        "unplanned_run_directories": unexpected,
        "cases": rows,
    }
    _atomic_json(output_base / "suite_status.json", status)
    completed_path = output_base / "COMPLETED.json"
    if complete:
        _atomic_json(
            completed_path,
            {
                "schema_version": SUITE_SCHEMA_VERSION,
                "experiment": "real_hydrology_suite",
                "mode": plan["mode"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "suite_plan": SUITE_PLAN_FILENAME,
                "suite_plan_fingerprint": plan["suite_plan_fingerprint"],
                "case_ids": list(CASE_IDS),
                "cases": rows,
            },
        )
    else:
        completed_path.unlink(missing_ok=True)
    return status


def _run_cases(args: argparse.Namespace) -> int:
    cases = _selected_cases(args)
    protocol = FormalProtocol.smoke() if args.smoke else FormalProtocol()
    max_validation_checks = getattr(args, "max_validation_checks", None)
    if max_validation_checks is not None:
        if args.smoke:
            raise ValueError("--max-validation-checks is a formal-only override")
        protocol = replace(
            protocol, max_validation_checks=int(max_validation_checks)
        )
    output_base = _output_base(Path(args.output_root), args.smoke)
    device = _device(args.device)
    requested_methods = tuple(args.methods) if args.methods else None
    requested_fit_budgets = resolve_requested_fit_budgets(
        protocol, getattr(args, "requested_fit_budgets", None)
    )
    prepared_cases = {
        case_id: load_prepared_case(case_id, args.processed_root) for case_id in cases
    }
    methods_by_case = {
        case_id: requested_methods or CASE_METHODS[case_id] for case_id in cases
    }
    proposed_cases = [
        _case_plan_record(
            prepared_cases[case_id],
            methods_by_case[case_id],
            requested_fit_budgets,
            protocol,
            output_base,
        )
        for case_id in cases
    ]
    plan = _merge_suite_plan(
        output_base,
        mode="smoke" if args.smoke else "formal",
        protocol=protocol,
        device=device,
        proposed_cases=proposed_cases,
    )
    any_failures = False

    for case_id in cases:
        case = prepared_cases[case_id]
        methods = methods_by_case[case_id]
        print(
            f"[{case_id}] start mode={'smoke' if args.smoke else 'formal'} "
            f"device={device} methods={','.join(methods)} "
            f"fit_budgets={','.join(str(value) for value in requested_fit_budgets)}",
            flush=True,
        )
        try:
            summary = run_benchmark(
                case.system,
                case.model,
                protocol,
                output_base,
                methods=methods,
                requested_fit_budgets=requested_fit_budgets,
                device=device,
                resume=not args.no_resume,
                raise_on_failure=False,
                historical_replay=case.historical_replay,
            )
            failed = summary.n_failed
            any_failures = any_failures or failed > 0
        except Exception as error:
            any_failures = True
            print(f"[{case_id}] failed: {type(error).__name__}: {error}", flush=True)
        _write_suite_state(output_base, plan)

    status = _write_suite_state(output_base, plan)
    any_failures = any_failures or any(
        int(record["failed_files"]) > 0 for record in status["cases"]
    )
    return 1 if any_failures else 0


def _audit_cases(args: argparse.Namespace) -> int:
    records: list[dict[str, object]] = []
    failed = False
    for case_id in _selected_cases(args):
        try:
            record = audit_case(
                case_id,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                verify_hashes=not args.fast,
            )
            print(
                f"[{case_id}] valid: fit={record['fit_rows']}x{record['fit_series']} "
                f"PAR({record['par_order']}), "
                f"historical={record['historical_replay_paths']}, "
                f"raw_files={record['raw_files_checked']}",
                flush=True,
            )
        except Exception as error:
            failed = True
            record = {
                "case_id": case_id,
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            print(f"[{case_id}] invalid: {type(error).__name__}: {error}", flush=True)
        records.append(record)
    destination = Path(args.processed_root) / "audit.json"
    _atomic_json(
        destination,
        {
            "schema_version": SUITE_SCHEMA_VERSION,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "hashes_verified": not args.fast,
            "cases": records,
        },
    )
    print(f"audit report: {destination}", flush=True)
    return 1 if failed else 0


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _status(args: argparse.Namespace) -> int:
    smoke = bool(args.smoke)
    protocol = FormalProtocol.smoke() if smoke else FormalProtocol()
    base = _output_base(Path(args.output_root), smoke)
    plan = _read_json(base / SUITE_PLAN_FILENAME)
    planned_cases: dict[str, Mapping[str, object]] = {}
    if plan is not None:
        if plan.get("schema_version") != SUITE_SCHEMA_VERSION:
            raise ValueError("suite plan has an unsupported schema version")
        records = plan.get("cases")
        if not isinstance(records, list):
            raise ValueError("suite plan has no valid cases list")
        planned_cases = {
            str(record["case_id"]): record
            for record in records
            if isinstance(record, Mapping) and "case_id" in record
        }
    rows: list[dict[str, object]] = []
    for case_id in _selected_cases(args):
        if plan is not None and case_id not in planned_cases:
            row = {
                "case_id": case_id,
                "planned": False,
                "complete": 0,
                "expected": 0,
                "failed_files": 0,
                "running_files": 0,
                "missing": 0,
                "methods": [],
                "requested_fit_budgets": [],
                "output_dir": None,
            }
        elif plan is not None:
            row = _case_progress(planned_cases[case_id])
        else:
            # Backward-compatible view for output roots created before suite
            # plans existed.
            case = load_prepared_case(case_id, args.processed_root)
            record = _case_plan_record(
                case,
                CASE_METHODS[case_id],
                protocol.fit_budgets,
                protocol,
                base,
            )
            row = _case_progress(record)
        rows.append(row)
        if not row["planned"]:
            print(f"{case_id}: not present in suite plan", flush=True)
        else:
            print(
                f"{case_id}: {row['complete']}/{row['expected']} complete, "
                f"{row['failed_files']} failure record(s), "
                f"{row['running_files']} running marker(s), "
                f"fit_budgets={','.join(str(value) for value in row['requested_fit_budgets'])}",
                flush=True,
            )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    return 0


def _add_case_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="select all frozen cases")
    group.add_argument(
        "--case",
        action="append",
        choices=CASE_IDS,
        help="select one case; repeat to select multiple",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m real_hydro.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit", help="validate prepared bundles and their frozen raw manifests"
    )
    _add_case_selector(audit)
    audit.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    audit.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    audit.add_argument(
        "--fast", action="store_true", help="check sizes but skip raw SHA-256 hashing"
    )
    audit.set_defaults(handler=_audit_cases)

    prepare = subparsers.add_parser(
        "prepare",
        help="verify the already frozen raw and processed source bundles",
    )
    _add_case_selector(prepare)
    prepare.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    prepare.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    prepare.add_argument("--fast", action="store_true")
    prepare.add_argument(
        "--offline",
        action="store_true",
        help="document that no network access is permitted (the command never downloads)",
    )
    prepare.set_defaults(handler=_audit_cases)

    run = subparsers.add_parser("run", help="run or resume benchmark optimization")
    _add_case_selector(run)
    run.add_argument("--smoke", action="store_true", help="use the tiny protocol")
    run.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    run.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument(
        "--methods",
        nargs="+",
        help="override the preregistered case-specific method set",
    )
    run.add_argument(
        "--fit-budgets",
        "--requested-fit-budgets",
        dest="requested_fit_budgets",
        nargs="+",
        type=int,
        help=(
            "schedule only these registered fit budgets without changing the full "
            "FormalProtocol or its split pools"
        ),
    )
    run.add_argument(
        "--max-validation-checks",
        type=int,
        help=(
            "formal-only protocol override; for the SDDP anchor use 4 so "
            "80 iterations have validation_every=20"
        ),
    )
    run.set_defaults(handler=_run_cases)

    status = subparsers.add_parser("status", help="count resumable run artifacts")
    _add_case_selector(status)
    status.add_argument("--smoke", action="store_true")
    status.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RunnerExecutionError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
