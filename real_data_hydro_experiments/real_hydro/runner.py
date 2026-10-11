"""Unified, resumable execution of one real-hydrology benchmark.

The runner enforces a narrow information boundary: training helpers receive
only the nested fit prefix and the generator-validation batch.  Test and fresh
evaluation batches are passed to :func:`evaluate_policy` only after a candidate
has been selected.  Each ``method/n_fit/train_seed`` run is independently
recoverable from a content-addressed JSON result and atomic checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import traceback
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from .config import FormalProtocol, ReservoirSystem
from .dataset import TrajectoryBatch, TrajectorySplits
from .evaluation import evaluate_policy
from .policies import (
    ProjectedDeepSetsPolicy,
    ProjectedGraphPolicy,
    ProjectedMLPPolicy,
    SeasonalReferencePolicy,
    count_parameters,
    matched_mlp_width,
)
from .stochastic import PeriodicARModel, generate_splits
from .train import DirectTrainingConfig, tune_direct


RUNNER_SCHEMA_VERSION = 2

DEFAULT_METHODS = (
    "seasonal_reference",
    "algorithm1_mlp",
    "projected_shac",
    "rl_ppo",
    "rl_sac",
    "hydrology_state_sddp",
)

OPTIONAL_METHODS = (
    "algorithm1_graph",
    "algorithm1_deepsets",
)

SUPPORTED_METHODS = frozenset((*DEFAULT_METHODS, *OPTIONAL_METHODS))

_DIRECT_LEARNING_RATES = {
    "algorithm1_mlp": (1.0e-4, 3.0e-4, 6.0e-4),
    "algorithm1_graph": (3.0e-4, 6.0e-4, 1.0e-3),
    "algorithm1_deepsets": (3.0e-4, 6.0e-4, 1.0e-3),
}
_SHAC_LEARNING_RATES = (1.0e-4, 3.0e-4, 6.0e-4)
_RL_LEARNING_RATES = (1.0e-4, 3.0e-4, 1.0e-3)

_OBJECTIVE_COST_VECTOR_FIELDS = (
    "shortage_cost",
    "excess_release_cost",
    "low_storage_cost",
    "flood_cost",
    "spill_cost",
    "ramp_cost",
    "terminal_shortfall_cost",
    "terminal_excess_cost",
    "thermal_block_cost",
)
_REFERENCE_COST_TARGET = 100.0
_MINIMUM_COST_DIVISOR = 1.0e-6
_SDDP_IMPLEMENTATION_VERSION = "hydrology_state_sddp_complete_recourse_v2"
_SDDP_NUMERICAL_RECOVERY_RULE = (
    "standard_solve_then_robust_retry_then_penalized_complete_recourse_"
    "after_infeasible"
)
_SDDP_COMPLETE_RECOURSE_PENALTY_RULE = (
    "1e3_times_system_influence_bound_times_max_normalized_marginal_cost_v1"
)


def _print_progress(message: str) -> None:
    print(message, flush=True)


class RunnerInputError(ValueError):
    """Raised before execution when the system/protocol/data contract is invalid."""


class RunnerExecutionError(RuntimeError):
    """Raised after all requested runs were attempted and at least one failed."""

    def __init__(self, summary: "RunnerSummary") -> None:
        self.summary = summary
        keys = ", ".join(item["run_key_string"] for item in summary.failures[:5])
        suffix = "" if len(summary.failures) <= 5 else ", ..."
        super().__init__(f"{len(summary.failures)} benchmark run(s) failed: {keys}{suffix}")


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise RunnerInputError(f"Cannot construct an output key from {value!r}.")
    return result


@dataclass(frozen=True, order=True)
class RunKey:
    """The unique unit of restart and result validation."""

    method: str
    n_fit: int
    train_seed: int

    @property
    def key_string(self) -> str:
        return f"{self.method}/n_fit={self.n_fit}/train_seed={self.train_seed}"

    @property
    def relative_directory(self) -> Path:
        return (
            Path(_slug(self.method))
            / f"n_fit_{int(self.n_fit)}"
            / f"train_seed_{int(self.train_seed)}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "n_fit": int(self.n_fit),
            "train_seed": int(self.train_seed),
        }


@dataclass(frozen=True)
class RunnerSummary:
    """In-memory index of newly completed, skipped, and failed run keys."""

    output_dir: Path
    experiment_fingerprint: str
    completed: tuple[Mapping[str, object], ...]
    skipped: tuple[Mapping[str, object], ...]
    failures: tuple[Mapping[str, object], ...]
    generated_splits: bool
    requested_fit_budgets: tuple[int, ...] = ()

    @property
    def n_complete(self) -> int:
        return len(self.completed)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    @property
    def n_failed(self) -> int:
        return len(self.failures)


def resolve_requested_fit_budgets(
    protocol: FormalProtocol,
    requested_fit_budgets: Iterable[int] | None,
) -> tuple[int, ...]:
    """Return a canonical execution-only subset of the frozen fit budgets.

    The returned values follow ``protocol.fit_budgets`` order.  This helper is
    deliberately a scheduling layer: callers must continue to pass the full
    :class:`FormalProtocol`, so selecting one budget does not change the split
    pools, experiment fingerprint, or any per-run fingerprint.
    """

    if requested_fit_budgets is None:
        return tuple(protocol.fit_budgets)
    try:
        requested = tuple(int(value) for value in requested_fit_budgets)
    except (TypeError, ValueError) as error:
        raise RunnerInputError("requested_fit_budgets must contain integers.") from error
    if not requested:
        raise RunnerInputError("requested_fit_budgets cannot be empty.")
    if len(requested) != len(set(requested)):
        raise RunnerInputError("requested_fit_budgets contains duplicates.")
    unknown = sorted(set(requested) - set(protocol.fit_budgets))
    if unknown:
        raise RunnerInputError(
            "requested_fit_budgets must be a subset of protocol.fit_budgets; "
            f"unknown values: {unknown}."
        )
    selected = set(requested)
    return tuple(value for value in protocol.fit_budgets if value in selected)


@dataclass
class _MethodOutcome:
    policy: Any
    source_checkpoint: Path | None
    selection: dict[str, object]
    training: dict[str, object]
    cleanup: Callable[[], None]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(digest: Any, name: str, values: object | None) -> None:
    digest.update(name.encode("utf-8") + b"\0")
    if values is None:
        digest.update(b"NONE\0")
        return
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def _batch_fingerprint(batch: TrajectoryBatch) -> str:
    digest = sha256()
    digest.update(str(batch.name).encode("utf-8") + b"\0")
    digest.update(str(int(batch.seed)).encode("ascii") + b"\0")
    for name in (
        "inflow",
        "hydrology_state",
        "hydrology_regime",
        "calendar",
        "demand",
        "source_path_id",
    ):
        _hash_array(digest, name, getattr(batch, name))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
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
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: object) -> None:
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
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output, source.open("rb") as input_stream:
            temporary = Path(output.name)
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_costs(path: Path, costs: Mapping[str, np.ndarray]) -> None:
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
            np.savez_compressed(
                stream,
                **{
                    name: np.asarray(values, dtype=np.float64)
                    for name, values in costs.items()
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _protocol_view(splits: TrajectorySplits, protocol: FormalProtocol) -> TrajectorySplits:
    expected = {
        "fit": protocol.fit_budgets[-1],
        "validation": protocol.n_validation,
        "test": protocol.n_test,
        "calibration": protocol.n_calibration,
        "evaluation": protocol.n_evaluation,
    }
    batches: dict[str, TrajectoryBatch] = {}
    for name, count in expected.items():
        source = getattr(splits, name)
        if source.n < count:
            raise RunnerInputError(
                f"Split {name!r} has {source.n} paths; protocol requires {count}."
            )
        batches[name] = source.take(slice(0, count), name=source.name)
    result = TrajectorySplits(**batches)
    result.validate_independence()
    return result


def _validate_splits(
    splits: TrajectorySplits,
    system: ReservoirSystem,
    protocol: FormalProtocol,
) -> int:
    expected_seeds = asdict(protocol.data_seeds)
    calendars: list[np.ndarray] = []
    n_regimes = 1
    for name, batch in splits.as_dict().items():
        if int(batch.seed) != int(expected_seeds[name]):
            raise RunnerInputError(
                f"Split {name!r} seed {batch.seed} does not match protocol seed "
                f"{expected_seeds[name]}."
            )
        if batch.horizon != system.horizon:
            raise RunnerInputError(
                f"Split {name!r} horizon {batch.horizon} != system horizon "
                f"{system.horizon}."
            )
        if batch.n_reservoirs != system.n_reservoirs:
            raise RunnerInputError(
                f"Split {name!r} reservoir width does not match the system."
            )
        expected_calendar = np.asarray(
            [system.period(stage) for stage in range(system.horizon)], dtype=np.int16
        )
        if not np.array_equal(batch.calendar, expected_calendar):
            raise RunnerInputError(
                f"Split {name!r} calendar is not aligned with system.period(stage)."
            )
        calendars.append(np.asarray(batch.calendar))
        n_regimes = max(n_regimes, batch.n_regimes)
    if any(not np.array_equal(calendars[0], value) for value in calendars[1:]):
        raise RunnerInputError("All trajectory interfaces must share one calendar.")
    return n_regimes


def _validate_historical_replay(
    historical_replay: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
) -> None:
    """Validate a locked replay without letting it define model dimensions."""

    if historical_replay.n < 1:
        raise RunnerInputError("historical_replay must contain at least one path.")
    if historical_replay.horizon != system.horizon:
        raise RunnerInputError(
            "historical_replay horizon does not match ReservoirSystem.horizon."
        )
    if historical_replay.n_reservoirs != system.n_reservoirs:
        raise RunnerInputError(
            "historical_replay reservoir width does not match the system."
        )
    expected_calendar = np.asarray(
        [system.period(stage) for stage in range(system.horizon)], dtype=np.int16
    )
    if not np.array_equal(historical_replay.calendar, expected_calendar):
        raise RunnerInputError(
            "historical_replay calendar is not aligned with system.period(stage)."
        )
    if historical_replay.n_regimes > n_regimes:
        raise RunnerInputError(
            "historical_replay regime labels exceed the model dimension fixed "
            "without historical data."
        )


def _resolve_splits(
    system: ReservoirSystem,
    hydrology: PeriodicARModel | TrajectorySplits,
    protocol: FormalProtocol,
    supplied: TrajectorySplits | None,
) -> tuple[TrajectorySplits, bool, str | None, int | None]:
    if isinstance(hydrology, PeriodicARModel):
        if hydrology.periods != system.periods:
            raise RunnerInputError("PAR periods do not match ReservoirSystem.periods.")
        if hydrology.n_series != system.n_reservoirs:
            raise RunnerInputError("PAR series width does not match ReservoirSystem.")
        splits = (
            generate_splits(hydrology, protocol, system.horizon)
            if supplied is None
            else supplied
        )
        generated = supplied is None
        training_fingerprint = hydrology.training_fingerprint
        model_n_regimes: int | None = hydrology.n_regimes
    elif isinstance(hydrology, TrajectorySplits):
        if supplied is not None:
            raise RunnerInputError(
                "Pass either TrajectorySplits as hydrology or supplied splits, not both."
            )
        splits = hydrology
        generated = False
        training_fingerprint = None
        model_n_regimes = None
    else:
        raise TypeError("hydrology must be PeriodicARModel or TrajectorySplits.")
    return (
        _protocol_view(splits, protocol),
        generated,
        training_fingerprint,
        model_n_regimes,
    )


def _nearest_mlp_width(
    system: ReservoirSystem,
    feature_dim: int,
    target_parameters: int,
    *,
    depth: int = 2,
    maximum: int = 2048,
) -> tuple[int, int]:
    """Return the MLP width/count nearest an arbitrary structured policy."""

    input_dim = 2 * system.n_reservoirs + feature_dim

    def parameter_count(width: int) -> int:
        return int(
            (input_dim + 1) * width
            + max(depth - 1, 0) * (width + 1) * width
            + (width + 1) * system.n_reservoirs
        )

    width = min(
        range(1, maximum + 1),
        key=lambda value: abs(parameter_count(value) - target_parameters),
    )
    # Verify the closed-form count against the actual module.  This makes a
    # future architecture edit fail loudly instead of silently breaking the
    # parameter-matching claim.
    actual = count_parameters(
        ProjectedMLPPolicy(system, feature_dim, width=width, depth=depth)
    )
    if actual != parameter_count(width):
        raise RuntimeError("ProjectedMLPPolicy parameter-count formula is stale.")
    return width, actual


def _parameter_record(
    *,
    status: str,
    target_method: str | None,
    target_parameters: int | None,
    actual_parameters: int,
) -> dict[str, object]:
    return {
        "status": status,
        "target_method": target_method,
        "target_parameters": target_parameters,
        "actual_parameters": int(actual_parameters),
        "absolute_difference": (
            None
            if target_parameters is None
            else abs(int(actual_parameters) - int(target_parameters))
        ),
    }


def _architecture_plan(
    system: ReservoirSystem,
    n_regimes: int,
    methods: tuple[str, ...],
    *,
    smoke: bool,
) -> dict[str, dict[str, object]]:
    """Freeze widths and auditable parameter counts before any training run."""

    graph_width = 16 if smoke else 64
    deepsets_width = 16 if smoke else 64
    structured = tuple(
        method
        for method in ("algorithm1_graph", "algorithm1_deepsets")
        if method in methods
    )
    if "algorithm1_mlp" in methods and len(structured) > 1:
        raise RunnerInputError(
            "One MLP cannot be parameter-matched to graph and DeepSets at once; "
            "run those structured comparisons as separate suites."
        )

    plan: dict[str, dict[str, object]] = {}
    graph_parameters: int | None = None
    if "algorithm1_graph" in methods:
        graph_parameters = count_parameters(
            ProjectedGraphPolicy(system, n_regimes, width=graph_width)
        )
        plan["algorithm1_graph"] = {
            "width": graph_width,
            "depth": 2,
            "parameter_count": graph_parameters,
            "parameter_matching": _parameter_record(
                status="structured_target",
                target_method="algorithm1_graph",
                target_parameters=graph_parameters,
                actual_parameters=graph_parameters,
            ),
        }

    deepsets_parameters: int | None = None
    if "algorithm1_deepsets" in methods:
        deepsets_parameters = count_parameters(
            ProjectedDeepSetsPolicy(system, n_regimes, width=deepsets_width)
        )
        plan["algorithm1_deepsets"] = {
            "width": deepsets_width,
            "depth": 2,
            "parameter_count": deepsets_parameters,
            "parameter_matching": _parameter_record(
                status="structured_target",
                target_method="algorithm1_deepsets",
                target_parameters=deepsets_parameters,
                actual_parameters=deepsets_parameters,
            ),
        }

    if "algorithm1_mlp" in methods:
        depth = 2
        if structured == ("algorithm1_graph",):
            width, actual, target = matched_mlp_width(
                system,
                n_regimes,
                graph_width=graph_width,
                depth=depth,
            )
            if graph_parameters is not None and target != graph_parameters:
                raise RuntimeError("Graph parameter counts disagree.")
            matching = _parameter_record(
                status="matched_to_requested_structured_policy",
                target_method="algorithm1_graph",
                target_parameters=target,
                actual_parameters=actual,
            )
        elif structured == ("algorithm1_deepsets",):
            if deepsets_parameters is None:
                raise AssertionError("Missing DeepSets parameter count.")
            width, actual = _nearest_mlp_width(
                system,
                n_regimes,
                deepsets_parameters,
                depth=depth,
            )
            matching = _parameter_record(
                status="matched_to_requested_structured_policy",
                target_method="algorithm1_deepsets",
                target_parameters=deepsets_parameters,
                actual_parameters=actual,
            )
        else:
            width = 16 if smoke else 128
            actual = count_parameters(
                ProjectedMLPPolicy(system, n_regimes, width=width, depth=depth)
            )
            matching = _parameter_record(
                status="no_structured_policy_requested",
                target_method=None,
                target_parameters=None,
                actual_parameters=actual,
            )
        plan["algorithm1_mlp"] = {
            "width": int(width),
            "depth": depth,
            "parameter_count": int(actual),
            "parameter_matching": matching,
        }
    return plan


def _fit_objective_normalization(
    fit: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    device: str,
) -> dict[str, object]:
    """Derive one positive cost divisor using only the nested fit prefix."""

    reference = SeasonalReferencePolicy(system, n_regimes).to(device)
    _, costs = evaluate_policy(reference, fit, system, n_regimes, device=device)
    costs = np.asarray(costs, dtype=np.float64)
    raw_mean = float(costs.mean())
    raw_median = float(np.median(costs))
    if not np.isfinite(raw_mean) or not np.isfinite(raw_median):
        raise RunnerInputError("Fit-prefix SeasonalReference costs must be finite.")
    divisor = max(abs(raw_mean) / _REFERENCE_COST_TARGET, _MINIMUM_COST_DIVISOR)
    normalized_mean = raw_mean / divisor
    normalized_median = raw_median / divisor
    # SHAC's critic and model-free rewards each need a positive typical-return
    # scale after the common objective rescaling.  A one-unit floor handles the
    # degenerate all-zero-cost fixture without changing any objective ranking.
    normalized_return_scale = max(abs(normalized_mean), 1.0)
    return {
        "source": "fit_prefix_seasonal_reference",
        "source_policy": "seasonal_reference",
        "fit_paths": int(fit.n),
        "fit_seed": int(fit.seed),
        "reference_cost_mean_raw": raw_mean,
        "reference_cost_median_raw": raw_median,
        "target_reference_cost": _REFERENCE_COST_TARGET,
        "minimum_raw_cost_divisor": _MINIMUM_COST_DIVISOR,
        "raw_cost_divisor": float(divisor),
        "reference_cost_mean_normalized": float(normalized_mean),
        "reference_cost_median_normalized": float(normalized_median),
        "normalized_return_scale": float(normalized_return_scale),
        "training_objective_rule": "divide_every_cost_coefficient_by_raw_cost_divisor",
        "held_out_interfaces_used": [],
    }


def _normalized_training_system(
    system: ReservoirSystem,
    normalization: Mapping[str, object],
) -> ReservoirSystem:
    """Uniformly rescale every cost coefficient, leaving physics unchanged."""

    divisor = float(normalization["raw_cost_divisor"])
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise RunnerInputError("Objective normalization divisor must be finite and positive.")
    objective = system.objective
    scaled_vectors = {
        name: tuple(float(value) / divisor for value in getattr(objective, name))
        for name in _OBJECTIVE_COST_VECTOR_FIELDS
    }
    scaled_objective = replace(
        objective,
        **scaled_vectors,
        deficit_cost=float(objective.deficit_cost) / divisor,
    )
    return replace(system, objective=scaled_objective)


def _sddp_complete_recourse_penalty(system: ReservoirSystem) -> float:
    """Freeze the scale-aware SDDP fallback penalty into the run identity."""

    objective = system.objective
    coefficients = [abs(float(objective.deficit_cost))]
    for name in _OBJECTIVE_COST_VECTOR_FIELDS:
        coefficients.extend(abs(float(value)) for value in getattr(objective, name))
    maximum_cost = max(coefficients, default=0.0)
    if maximum_cost == 0.0:
        return 1.0
    generation_scale = sum(
        abs(float(value)) for value in objective.generation_coefficient
    )
    influence_bound = system.horizon + system.n_reservoirs + generation_scale
    return 1.0e3 * max(influence_bound, 1.0) * maximum_cost


def _method_settings(
    method: str,
    n_fit: int,
    system: ReservoirSystem,
    protocol: FormalProtocol,
    device: str,
    architecture: Mapping[str, Mapping[str, object]],
    normalization: Mapping[str, object],
) -> dict[str, object]:
    iterations = math.ceil(n_fit / protocol.batch_size)
    validation_every = max(1, iterations // protocol.max_validation_checks)
    smoke = bool(protocol.smoke_mode)
    common = {
        "method": method,
        "n_fit": int(n_fit),
        "device": device,
        "smoke_mode": smoke,
        "objective_normalization": dict(normalization),
    }
    if method == "seasonal_reference":
        return {**common, "family": "fixed_reference"}
    if method in _DIRECT_LEARNING_RATES:
        policy_architecture = architecture[method]
        rates = (
            (_DIRECT_LEARNING_RATES[method][1],)
            if smoke
            else _DIRECT_LEARNING_RATES[method]
        )
        return {
            **common,
            "family": "algorithm1_pathwise",
            "iterations": iterations,
            "batch_size": protocol.batch_size,
            "validation_every": validation_every,
            "learning_rates": rates,
            "width": int(policy_architecture["width"]),
            "depth": int(policy_architecture["depth"]),
            "parameter_count": int(policy_architecture["parameter_count"]),
            "parameter_matching": dict(policy_architecture["parameter_matching"]),
            "weight_decay": 1.0e-6,
            "gradient_clip": 10.0,
        }
    if method == "projected_shac":
        return {
            **common,
            "family": "first_order_actor_critic",
            "iterations": iterations,
            "batch_size": protocol.batch_size,
            "validation_every": validation_every,
            "learning_rates": (3.0e-4,) if smoke else _SHAC_LEARNING_RATES,
            "actor_width": 16 if smoke else 128,
            "actor_depth": 2,
            "window": min(3 if smoke else 8, system.horizon),
            "critic_learning_rate": 1.0e-3,
            "critic_epochs": 1 if smoke else 2,
            "critic_tau": 0.05,
            "critic_width": 16 if smoke else 256,
            "critic_depth": 1 if smoke else 2,
            "cost_scale": float(normalization["normalized_return_scale"]),
            "cost_scale_units": "normalized_total_cost",
            "weight_decay": 1.0e-6,
            "gradient_clip": 10.0,
        }
    if method in {"rl_ppo", "rl_sac"}:
        return {
            **common,
            "family": "model_free_rl",
            "algorithm": method.removeprefix("rl_"),
            "learning_rates": (3.0e-4,) if smoke else _RL_LEARNING_RATES,
            "reward_scale": float(normalization["normalized_return_scale"]),
            "reward_scale_units": "normalized_total_cost",
            "gamma": 1.0,
            "validation_every_episodes": (
                max(1, min(2, n_fit))
                if smoke
                else min(50, max(1, n_fit // protocol.max_validation_checks))
            ),
            "network_width": 16 if smoke else 256,
            "network_depth": 1 if smoke else 2,
            "ppo_rollout_episodes": min(2 if smoke else 10, n_fit),
            "ppo_epochs": 1 if smoke else 10,
            "ppo_batch_size": 16 if smoke else 256,
            "replay_buffer_size": 64 if smoke else 100_000,
            "learning_starts": 0 if smoke else 1_000,
            "off_policy_batch_size": 8 if smoke else 256,
            "off_policy_gradient_steps": 1,
        }
    if method == "hydrology_state_sddp":
        return {
            **common,
            "family": "hydrology_state_sddp",
            "implementation_version": _SDDP_IMPLEMENTATION_VERSION,
            "iterations": protocol.sddp_iterations,
            "validation_every": (
                1
                if smoke
                else max(1, protocol.sddp_iterations // protocol.max_validation_checks)
            ),
            "max_validation_checks": protocol.max_validation_checks,
            "noise_scenarios": protocol.sddp_noise_scenarios,
            "complete_recourse_penalty": _sddp_complete_recourse_penalty(system),
            "complete_recourse_penalty_rule": (
                _SDDP_COMPLETE_RECOURSE_PENALTY_RULE
            ),
            "numerical_recovery_rule": _SDDP_NUMERICAL_RECOVERY_RULE,
        }
    raise RunnerInputError(f"Unsupported method {method!r}.")


def _validate_method_applicability(method: str, system: ReservoirSystem) -> None:
    routing = np.asarray(system.routing, dtype=np.float64)
    if method == "algorithm1_graph" and (
        system.n_reservoirs < 2 or not np.any(routing > 0.0)
    ):
        raise RunnerInputError(
            "algorithm1_graph requires an explicitly routed multi-reservoir system."
        )
    if method == "algorithm1_deepsets" and (
        "camels" not in system.benchmark.casefold() or np.any(routing != 0.0)
    ):
        raise RunnerInputError(
            "algorithm1_deepsets is reserved for the parallel CAMELS portfolio."
        )


def _selected_index(candidates: Iterable[object], selected: object) -> int:
    for index, candidate in enumerate(candidates):
        if candidate is selected:
            return index
    raise RuntimeError("Training API returned a selected candidate outside its candidate list.")


def _direct_outcome(
    method: str,
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    settings: Mapping[str, object],
    train_seed: int,
    candidate_dir: Path,
    provenance: dict[str, object],
) -> _MethodOutcome:
    feature_dim = n_regimes
    width = int(settings["width"])
    depth = int(settings["depth"])

    def builder() -> torch.nn.Module:
        if method == "algorithm1_mlp":
            return ProjectedMLPPolicy(system, feature_dim, width=width, depth=depth)
        if method == "algorithm1_graph":
            return ProjectedGraphPolicy(system, feature_dim, width=width)
        if method == "algorithm1_deepsets":
            return ProjectedDeepSetsPolicy(system, feature_dim, width=width)
        raise AssertionError(method)

    config = DirectTrainingConfig(
        iterations=int(settings["iterations"]),
        batch_size=int(settings["batch_size"]),
        validation_every=int(settings["validation_every"]),
        learning_rate=float(tuple(settings["learning_rates"])[0]),
        weight_decay=float(settings["weight_decay"]),
        gradient_clip=float(settings["gradient_clip"]),
        device=str(settings["device"]),
    )
    selection = tune_direct(
        builder,
        fit,
        validation,
        system,
        n_regimes,
        config,
        train_seed,
        tuple(float(value) for value in settings["learning_rates"]),
        candidate_dir,
        provenance,
    )
    selected = selection.selected
    selected_index = _selected_index(selection.candidates, selected)
    candidates = [
        {
            "candidate_index": index,
            "learning_rate": float(candidate.learning_rate),
            "best_validation_cost": float(candidate.best_validation_cost),
            "selected_iteration": int(candidate.selected_iteration),
            "selected": index == selected_index,
        }
        for index, candidate in enumerate(selection.candidates)
    ]
    selected_parameters = count_parameters(selected.policy)
    return _MethodOutcome(
        policy=selected.policy,
        source_checkpoint=Path(selected.checkpoint_path),
        selection={
            "dataset": "validation",
            "selected_candidate_index": selected_index,
            "selected_validation_cost": float(selected.best_validation_cost),
            "candidates": candidates,
        },
        training={
            "selected_fit_time_seconds": float(selected.fit_time_seconds),
            "total_tuning_time_seconds": float(selection.total_tuning_time_seconds),
            "selected_environment_transitions": int(selected.environment_transitions),
            "total_environment_transitions": int(
                sum(item.environment_transitions for item in selection.candidates)
            ),
            "selected_actor_updates": int(selected.gradient_updates),
            "total_actor_updates": int(
                sum(item.gradient_updates for item in selection.candidates)
            ),
            "selected_critic_updates": 0,
            "total_critic_updates": 0,
            "selected_optimizer_updates": int(selected.gradient_updates),
            "total_optimizer_updates": int(
                sum(item.gradient_updates for item in selection.candidates)
            ),
            "deployment_parameters": int(selected_parameters),
            "training_parameters": int(selected_parameters),
        },
        cleanup=lambda: None,
    )


def _shac_outcome(
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    settings: Mapping[str, object],
    train_seed: int,
    candidate_dir: Path,
    provenance: dict[str, object],
) -> _MethodOutcome:
    from .shac import SHACConfig, tune_shac

    width = int(settings["actor_width"])
    depth = int(settings["actor_depth"])
    config = SHACConfig(
        iterations=int(settings["iterations"]),
        batch_size=int(settings["batch_size"]),
        validation_every=int(settings["validation_every"]),
        window=int(settings["window"]),
        actor_learning_rate=float(tuple(settings["learning_rates"])[0]),
        critic_learning_rate=float(settings["critic_learning_rate"]),
        critic_epochs=int(settings["critic_epochs"]),
        critic_tau=float(settings["critic_tau"]),
        cost_scale=float(settings["cost_scale"]),
        weight_decay=float(settings["weight_decay"]),
        gradient_clip=float(settings["gradient_clip"]),
        critic_width=int(settings["critic_width"]),
        critic_depth=int(settings["critic_depth"]),
        device=str(settings["device"]),
    )
    selection = tune_shac(
        lambda: ProjectedMLPPolicy(system, n_regimes, width=width, depth=depth),
        fit,
        validation,
        system,
        n_regimes,
        config,
        train_seed,
        tuple(float(value) for value in settings["learning_rates"]),
        candidate_dir,
        provenance,
    )
    selected = selection.selected
    selected_index = _selected_index(selection.candidates, selected)
    candidates = [
        {
            "candidate_index": index,
            "actor_learning_rate": float(candidate.actor_learning_rate),
            "best_validation_cost": float(candidate.best_validation_cost),
            "selected_iteration": int(candidate.selected_iteration),
            "selected": index == selected_index,
        }
        for index, candidate in enumerate(selection.candidates)
    ]
    actor_parameters = count_parameters(selected.policy)
    critic_parameters = count_parameters(selected.critic)
    return _MethodOutcome(
        policy=selected.policy,
        source_checkpoint=Path(selected.checkpoint_path),
        selection={
            "dataset": "validation",
            "selected_candidate_index": selected_index,
            "selected_validation_cost": float(selected.best_validation_cost),
            "candidates": candidates,
        },
        training={
            "selected_fit_time_seconds": float(selected.fit_time_seconds),
            "total_tuning_time_seconds": float(selection.total_tuning_time_seconds),
            "selected_environment_transitions": int(selected.environment_transitions),
            "total_environment_transitions": int(
                sum(item.environment_transitions for item in selection.candidates)
            ),
            "selected_actor_updates": int(selected.actor_gradient_updates),
            "total_actor_updates": int(
                sum(item.actor_gradient_updates for item in selection.candidates)
            ),
            "selected_critic_updates": int(selected.critic_gradient_updates),
            "total_critic_updates": int(
                sum(item.critic_gradient_updates for item in selection.candidates)
            ),
            "selected_optimizer_updates": int(
                selected.actor_gradient_updates + selected.critic_gradient_updates
            ),
            "total_optimizer_updates": int(
                sum(
                    item.actor_gradient_updates + item.critic_gradient_updates
                    for item in selection.candidates
                )
            ),
            "deployment_parameters": int(actor_parameters),
            "training_parameters": int(actor_parameters + critic_parameters),
        },
        cleanup=lambda: None,
    )


def _rl_outcome(
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    settings: Mapping[str, object],
    train_seed: int,
    candidate_dir: Path,
    provenance: dict[str, object],
) -> _MethodOutcome:
    from .rl import RLConfig, tune_rl

    algorithm = str(settings["algorithm"])
    config = RLConfig(
        algorithm=algorithm,
        learning_rate=float(tuple(settings["learning_rates"])[0]),
        reward_scale=float(settings["reward_scale"]),
        gamma=float(settings["gamma"]),
        validation_every_episodes=int(settings["validation_every_episodes"]),
        network_width=int(settings["network_width"]),
        network_depth=int(settings["network_depth"]),
        ppo_rollout_episodes=int(settings["ppo_rollout_episodes"]),
        ppo_epochs=int(settings["ppo_epochs"]),
        ppo_batch_size=int(settings["ppo_batch_size"]),
        replay_buffer_size=int(settings["replay_buffer_size"]),
        learning_starts=int(settings["learning_starts"]),
        off_policy_batch_size=int(settings["off_policy_batch_size"]),
        off_policy_gradient_steps=int(settings["off_policy_gradient_steps"]),
        device=str(settings["device"]),
    )
    selection = tune_rl(
        algorithm,
        fit,
        validation,
        system,
        n_regimes,
        config,
        train_seed,
        tuple(float(value) for value in settings["learning_rates"]),
        candidate_dir,
        provenance,
    )
    selected = selection.selected
    selected_index = _selected_index(selection.candidates, selected)

    def updates(candidate: object) -> int | None:
        value = getattr(candidate.policy.model, "_n_updates", None)
        return None if value is None else int(value)

    candidate_updates = [updates(candidate) for candidate in selection.candidates]
    candidates = [
        {
            "candidate_index": index,
            "learning_rate": float(candidate.learning_rate),
            "best_validation_cost": float(candidate.best_validation_cost),
            "selected_timesteps": int(candidate.selected_timesteps),
            "sb3_updates": candidate_updates[index],
            "selected": index == selected_index,
        }
        for index, candidate in enumerate(selection.candidates)
    ]
    deployment_parameters = int(selected.policy.deployment_parameters)

    def cleanup() -> None:
        seen: set[int] = set()
        for candidate in selection.candidates:
            if id(candidate.policy) not in seen:
                candidate.close()
                seen.add(id(candidate.policy))

    selected_updates = candidate_updates[selected_index]
    known_updates = [value for value in candidate_updates if value is not None]
    return _MethodOutcome(
        policy=selected.policy,
        source_checkpoint=Path(selected.checkpoint_path),
        selection={
            "dataset": "validation",
            "selected_candidate_index": selected_index,
            "selected_validation_cost": float(selected.best_validation_cost),
            "candidates": candidates,
        },
        training={
            "selected_fit_time_seconds": float(selected.fit_time_seconds),
            "total_tuning_time_seconds": float(selection.total_tuning_time_seconds),
            "selected_environment_transitions": int(selected.environment_transitions),
            "total_environment_transitions": int(
                sum(item.environment_transitions for item in selection.candidates)
            ),
            "selected_actor_updates": None,
            "total_actor_updates": None,
            "selected_critic_updates": None,
            "total_critic_updates": None,
            "selected_optimizer_updates": selected_updates,
            "total_optimizer_updates": (
                int(sum(known_updates))
                if len(known_updates) == len(candidate_updates)
                else None
            ),
            "update_counter_source": "stable_baselines3._n_updates",
            "deployment_parameters": deployment_parameters,
            "training_parameters": deployment_parameters,
        },
        cleanup=cleanup,
    )


def _sddp_outcome(
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    settings: Mapping[str, object],
    train_seed: int,
    candidate_dir: Path,
    provenance: dict[str, object],
) -> _MethodOutcome:
    from .sddp import SDDPConfig, fit_sddp

    config = SDDPConfig(
        iterations=int(settings["iterations"]),
        validation_every=int(settings["validation_every"]),
        noise_scenarios=int(settings["noise_scenarios"]),
        device=str(settings["device"]),
        max_validation_checks=int(settings["max_validation_checks"]),
        complete_recourse_penalty=float(settings["complete_recourse_penalty"]),
    )
    result = fit_sddp(
        fit,
        validation,
        system,
        n_regimes,
        config,
        train_seed,
        candidate_dir / f"sddp_seed{train_seed}.pt",
        provenance,
    )
    stage_lp_solves = int(
        sum(int(row.get("stage_lp_solves", 0)) for row in result.history)
    )
    return _MethodOutcome(
        policy=result.policy,
        source_checkpoint=Path(result.checkpoint_path),
        selection={
            "dataset": "validation",
            "selected_candidate_index": 0,
            "selected_validation_cost": float(result.best_validation_cost),
            "candidates": [
                {
                    "candidate_index": 0,
                    "best_validation_cost": float(result.best_validation_cost),
                    "selected_iteration": int(result.selected_iteration),
                    "selected": True,
                }
            ],
        },
        training={
            "selected_fit_time_seconds": float(result.fit_time_seconds),
            "total_tuning_time_seconds": float(result.fit_time_seconds),
            "selected_environment_transitions": int(
                int(settings["iterations"]) * system.horizon
            ),
            "total_environment_transitions": int(
                int(settings["iterations"]) * system.horizon
            ),
            "selected_actor_updates": 0,
            "total_actor_updates": 0,
            "selected_critic_updates": 0,
            "total_critic_updates": 0,
            "selected_optimizer_updates": 0,
            "total_optimizer_updates": 0,
            "sddp_iterations": int(settings["iterations"]),
            "stage_lp_solves": stage_lp_solves,
            "number_of_cuts": int(result.number_of_cuts),
            "effective_complete_recourse_penalty": float(
                result.feasibility_audit.get(
                    "complete_recourse_penalty",
                    settings["complete_recourse_penalty"],
                )
            ),
            "fit_feasibility_audit": dict(result.feasibility_audit),
            "deployment_parameters": 0,
            "training_parameters": 0,
        },
        cleanup=result.policy.close,
    )


def _fit_method(
    key: RunKey,
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    settings: Mapping[str, object],
    run_dir: Path,
    provenance: dict[str, object],
) -> _MethodOutcome:
    candidate_dir = run_dir / "candidates"
    if key.method == "seasonal_reference":
        policy = SeasonalReferencePolicy(system, n_regimes).to(str(settings["device"]))
        return _MethodOutcome(
            policy=policy,
            source_checkpoint=None,
            selection={
                "dataset": "validation",
                "selected_candidate_index": 0,
                "selected_validation_cost": None,
                "candidates": [
                    {
                        "candidate_index": 0,
                        "fixed_policy": True,
                        "selected": True,
                    }
                ],
            },
            training={
                "selected_fit_time_seconds": 0.0,
                "total_tuning_time_seconds": 0.0,
                "selected_environment_transitions": 0,
                "total_environment_transitions": 0,
                "selected_actor_updates": 0,
                "total_actor_updates": 0,
                "selected_critic_updates": 0,
                "total_critic_updates": 0,
                "selected_optimizer_updates": 0,
                "total_optimizer_updates": 0,
                "deployment_parameters": count_parameters(policy),
                "training_parameters": 0,
            },
            cleanup=lambda: None,
        )
    if key.method in _DIRECT_LEARNING_RATES:
        return _direct_outcome(
            key.method,
            fit,
            validation,
            system,
            n_regimes,
            settings,
            key.train_seed,
            candidate_dir,
            provenance,
        )
    if key.method == "projected_shac":
        return _shac_outcome(
            fit,
            validation,
            system,
            n_regimes,
            settings,
            key.train_seed,
            candidate_dir,
            provenance,
        )
    if key.method in {"rl_ppo", "rl_sac"}:
        return _rl_outcome(
            fit,
            validation,
            system,
            n_regimes,
            settings,
            key.train_seed,
            candidate_dir,
            provenance,
        )
    if key.method == "hydrology_state_sddp":
        return _sddp_outcome(
            fit,
            validation,
            system,
            n_regimes,
            settings,
            key.train_seed,
            candidate_dir,
            provenance,
        )
    raise AssertionError(key.method)


def _evaluate_selected(
    policy: Any,
    splits: TrajectorySplits,
    system: ReservoirSystem,
    n_regimes: int,
    device: str,
    historical_replay: TrajectoryBatch | None,
) -> tuple[dict[str, dict[str, object]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, object]] = {}
    costs: dict[str, np.ndarray] = {}
    evaluation_interfaces: list[tuple[str, TrajectoryBatch]] = [
        ("validation", splits.validation),
        ("test", splits.test),
        ("evaluation", splits.evaluation),
    ]
    if historical_replay is not None:
        evaluation_interfaces.append(("historical_replay", historical_replay))
    for name, dataset in evaluation_interfaces:
        start = time.perf_counter()
        split_metrics, split_costs = evaluate_policy(
            policy,
            dataset,
            system,
            n_regimes,
            device=device,
        )
        elapsed = time.perf_counter() - start
        metrics[name] = {
            **{key: float(value) for key, value in split_metrics.items()},
            "n_paths": int(dataset.n),
            "dataset_seed": int(dataset.seed),
            "evaluation_time_seconds": float(elapsed),
        }
        costs[name] = np.asarray(split_costs, dtype=np.float64)
    return metrics, costs


def _run_fingerprint(
    experiment_fingerprint: str,
    key: RunKey,
    settings: Mapping[str, object],
) -> str:
    payload = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "experiment_fingerprint": experiment_fingerprint,
        "run_key": key.as_dict(),
        "settings": settings,
    }
    return sha256(_canonical_json(payload)).hexdigest()


def _valid_result(
    run_dir: Path,
    expected_fingerprint: str,
    key: RunKey,
    *,
    expect_historical_replay: bool,
) -> dict[str, object] | None:
    path = run_dir / "result.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != RUNNER_SCHEMA_VERSION:
            return None
        if payload.get("status") != "complete":
            return None
        if payload.get("run_fingerprint") != expected_fingerprint:
            return None
        if payload.get("run_key") != key.as_dict():
            return None
        metrics = payload.get("metrics")
        expected_metrics = {
            "validation",
            "test",
            "evaluation",
        }
        if expect_historical_replay:
            expected_metrics.add("historical_replay")
        if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
            return None
        for split_metrics in metrics.values():
            if not isinstance(split_metrics, dict):
                return None
            cost_mean = split_metrics.get("cost_mean")
            n_paths = split_metrics.get("n_paths")
            if not isinstance(cost_mean, (int, float)) or not np.isfinite(cost_mean):
                return None
            if not isinstance(n_paths, int) or n_paths < 1:
                return None
        selection = payload.get("selection")
        if not isinstance(selection, dict) or selection.get("dataset") != "validation":
            return None
        if not isinstance(payload.get("training"), dict):
            return None
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        for artifact_name in ("checkpoint", "path_costs"):
            artifact = artifacts.get(artifact_name)
            if not isinstance(artifact, dict):
                return None
            filename = artifact.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                return None
            artifact_path = run_dir / filename
            if not artifact_path.is_file():
                return None
            if artifact_path.stat().st_size != int(artifact.get("size_bytes", -1)):
                return None
            if _sha256_file(artifact_path) != artifact.get("sha256"):
                return None
        costs_path = run_dir / str(artifacts["path_costs"]["filename"])
        with np.load(costs_path, allow_pickle=False) as stored_costs:
            if set(stored_costs.files) != expected_metrics:
                return None
            for name in expected_metrics:
                values = np.asarray(stored_costs[name])
                if values.shape != (int(metrics[name]["n_paths"]),):
                    return None
                if np.any(~np.isfinite(values)):
                    return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_manifest(
    base_dir: Path,
    experiment_fingerprint: str,
    system: ReservoirSystem,
    protocol: FormalProtocol,
    split_fingerprints: Mapping[str, str],
    model_training_fingerprint: str | None,
    device: str,
    historical_replay: TrajectoryBatch | None,
    historical_replay_fingerprint: str | None,
    *,
    resume: bool,
) -> None:
    manifest_path = base_dir / "manifest.json"
    if manifest_path.is_file() and resume:
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RunnerInputError(f"Cannot read existing manifest: {error}") from error
        if not isinstance(current, dict):
            raise RunnerInputError("Existing manifest must be a JSON object.")
        if current.get("experiment_fingerprint") != experiment_fingerprint:
            raise RunnerInputError(
                "Existing output manifest belongs to different system/protocol/data."
            )
        return
    _atomic_json(
        manifest_path,
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "experiment_fingerprint": experiment_fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "system": system.as_jsonable(),
            "protocol": asdict(protocol),
            "split_fingerprints": dict(split_fingerprints),
            "model_training_fingerprint": model_training_fingerprint,
            "device": device,
            "selection_rule": "hyperparameters/checkpoints selected on validation only",
            **(
                {}
                if historical_replay is None
                else {
                    "historical_replay": {
                        "locked": True,
                        "evaluated_after_selection": True,
                        "n_paths": int(historical_replay.n),
                        "seed": int(historical_replay.seed),
                        "fingerprint": historical_replay_fingerprint,
                    }
                }
            ),
        },
    )


def _execute_one(
    key: RunKey,
    run_dir: Path,
    run_fingerprint: str,
    experiment_fingerprint: str,
    settings: Mapping[str, object],
    splits: TrajectorySplits,
    system: ReservoirSystem,
    training_system: ReservoirSystem,
    protocol: FormalProtocol,
    n_regimes: int,
    device: str,
    historical_replay: TrajectoryBatch | None,
    historical_replay_fingerprint: str | None,
) -> dict[str, object]:
    fit = splits.fit.take(protocol.fit_prefix(key.n_fit), name=f"fit_n{key.n_fit}")
    fit_fingerprint = _batch_fingerprint(fit)
    validation_fingerprint = _batch_fingerprint(splits.validation)
    training_information_fingerprint = sha256(
        _canonical_json(
            {
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "training_system": training_system.as_jsonable(),
                "run_key": key.as_dict(),
                "settings": settings,
                "fit_fingerprint": fit_fingerprint,
                "validation_fingerprint": validation_fingerprint,
            }
        )
    ).hexdigest()
    # This dictionary is the *only* runner provenance passed into a fitting
    # API.  It intentionally contains no full-experiment/run fingerprint,
    # because those hashes include locked test, evaluation, and optional
    # historical-replay interfaces.
    provenance = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "run_key": key.as_dict(),
        "selection_dataset": "validation",
        "fit_fingerprint": fit_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "training_information_fingerprint": training_information_fingerprint,
        "excluded_interfaces": (
            "test",
            "calibration",
            "evaluation",
            "historical_replay",
        ),
    }
    wall_start = time.perf_counter()
    outcome: _MethodOutcome | None = None
    try:
        fit_start = time.perf_counter()
        outcome = _fit_method(
            key,
            fit,
            splits.validation,
            training_system,
            n_regimes,
            settings,
            run_dir,
            provenance,
        )
        outcome.training["runner_fit_wall_time_seconds"] = float(
            time.perf_counter() - fit_start
        )
        metrics, costs = _evaluate_selected(
            outcome.policy,
            splits,
            system,
            n_regimes,
            device,
            historical_replay,
        )
        combined_feasibility_audit = getattr(
            outcome.policy, "feasibility_audit", None
        )
        if isinstance(combined_feasibility_audit, Mapping):
            deployment_feasibility_audit = getattr(
                outcome.policy, "deployment_feasibility_audit", None
            )
            if isinstance(deployment_feasibility_audit, Mapping):
                outcome.training["deployment_feasibility_audit"] = dict(
                    deployment_feasibility_audit
                )
            else:
                outcome.training["deployment_feasibility_audit"] = dict(
                    combined_feasibility_audit
                )
            outcome.training["combined_feasibility_audit"] = dict(
                combined_feasibility_audit
            )
        normalization = dict(settings["objective_normalization"])
        divisor = float(normalization["raw_cost_divisor"])
        raw_validation_cost = float(metrics["validation"]["cost_mean"])
        normalized_validation_cost = outcome.selection["selected_validation_cost"]
        if normalized_validation_cost is None:
            normalized_validation_cost = raw_validation_cost / divisor
        outcome.selection["selected_validation_cost_normalized"] = float(
            normalized_validation_cost
        )
        outcome.selection["selected_validation_cost_raw"] = raw_validation_cost
        # Preserve the public field in report/evaluation units.  Candidate
        # ranking fields remain in the explicitly documented training units.
        outcome.selection["selected_validation_cost"] = raw_validation_cost
        outcome.selection["candidate_cost_units"] = (
            "raw_objective_divided_by_objective_normalization.raw_cost_divisor"
        )

        cost_path = run_dir / "path_costs.npz"
        _atomic_costs(cost_path, costs)
        if outcome.source_checkpoint is None:
            checkpoint_path = run_dir / "checkpoint.pt"
            _atomic_torch_save(
                checkpoint_path,
                {
                    "runner_schema_version": RUNNER_SCHEMA_VERSION,
                    "method": key.method,
                    "system": training_system.as_jsonable(),
                    "evaluation_system": system.as_jsonable(),
                    "n_regimes": n_regimes,
                    "policy_class": outcome.policy.__class__.__name__,
                    "state_dict": outcome.policy.state_dict(),
                    "provenance": provenance,
                },
            )
        else:
            extension = outcome.source_checkpoint.suffix or ".bin"
            checkpoint_path = run_dir / f"checkpoint{extension}"
            _atomic_copy(outcome.source_checkpoint, checkpoint_path)

        result = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_key": key.as_dict(),
            "run_key_string": key.key_string,
            "run_fingerprint": run_fingerprint,
            "experiment_fingerprint": experiment_fingerprint,
            "system": {
                "benchmark": system.benchmark,
                "case_id": system.case_id,
                "horizon": system.horizon,
                "periods": system.periods,
                "n_reservoirs": system.n_reservoirs,
            },
            "data": {
                "fit_paths": fit.n,
                "fit_seed": fit.seed,
                "validation_paths": splits.validation.n,
                "validation_seed": splits.validation.seed,
                "test_paths": splits.test.n,
                "test_seed": splits.test.seed,
                "calibration_paths": splits.calibration.n,
                "calibration_seed": splits.calibration.seed,
                "evaluation_paths": splits.evaluation.n,
                "evaluation_seed": splits.evaluation.seed,
                "n_regimes": n_regimes,
                "nested_fit_prefix": True,
                **(
                    {}
                    if historical_replay is None
                    else {
                        "historical_replay_paths": historical_replay.n,
                        "historical_replay_seed": historical_replay.seed,
                        "historical_replay_fingerprint": historical_replay_fingerprint,
                    }
                ),
            },
            "settings": dict(settings),
            "selection": outcome.selection,
            "training": {
                **outcome.training,
                "total_run_wall_time_seconds": float(time.perf_counter() - wall_start),
            },
            "metrics": metrics,
            "artifacts": {
                "checkpoint": _artifact(checkpoint_path),
                "path_costs": _artifact(cost_path),
            },
            "information_audit": {
                "selection_dataset": "validation",
                "training_received_test": False,
                "training_received_calibration": False,
                "training_received_evaluation": False,
                "training_received_historical_replay": False,
                "objective_normalization_source": normalization["source"],
                "objective_normalization_fit_paths": normalization["fit_paths"],
                "objective_normalization_received_validation": False,
                "objective_normalization_received_test": False,
                "objective_normalization_received_calibration": False,
                "objective_normalization_received_evaluation": False,
                "objective_normalization_received_historical_replay": False,
                "training_objective_uniformly_scaled": True,
                "reported_metrics_use_raw_objective_units": True,
                "test_evaluated_after_selection": True,
                "evaluation_evaluated_after_selection": True,
                "historical_replay_evaluated_after_selection": (
                    historical_replay is not None
                ),
            },
        }
        _atomic_json(run_dir / "result.json", result)
        (run_dir / "failure.json").unlink(missing_ok=True)
        return result
    finally:
        if outcome is not None:
            outcome.cleanup()


def run_benchmark(
    system: ReservoirSystem,
    hydrology: PeriodicARModel | TrajectorySplits,
    protocol: FormalProtocol,
    output_dir: str | Path,
    *,
    splits: TrajectorySplits | None = None,
    historical_replay: TrajectoryBatch | None = None,
    methods: Iterable[str] = DEFAULT_METHODS,
    requested_fit_budgets: Iterable[int] | None = None,
    device: str = "cpu",
    resume: bool = True,
    raise_on_failure: bool = True,
    progress: Callable[[str], None] | None = _print_progress,
) -> RunnerSummary:
    """Run every requested method/budget/seed for one reservoir system.

    ``hydrology`` may be a fitted :class:`PeriodicARModel`, in which case the
    frozen split pools are generated, or an already generated
    :class:`TrajectorySplits`.  Supplying ``splits=`` alongside a model reuses
    those exact pools while retaining the model training fingerprint.

    ``historical_replay`` is an optional locked interface.  It contributes to
    experiment identity but is evaluated only after validation has selected a
    candidate; it is never passed to normalization or fitting code.

    ``requested_fit_budgets`` selects which registered run keys are scheduled.
    It is not part of experiment identity: the full ``protocol`` still fixes
    split generation and fingerprints.  Omitting it preserves the historical
    behavior of running every registered fit budget.

    A valid existing ``result.json`` is skipped only when its run fingerprint
    and both artifact hashes match.  Interrupted or corrupt runs are therefore
    retried without invalidating already completed keys.  All requested keys
    are attempted before :class:`RunnerExecutionError` is raised.
    """

    selected_methods = tuple(str(method) for method in methods)
    if not selected_methods:
        raise RunnerInputError("At least one method is required.")
    if len(selected_methods) != len(set(selected_methods)):
        raise RunnerInputError("methods contains duplicates.")
    unsupported = sorted(set(selected_methods) - SUPPORTED_METHODS)
    if unsupported:
        raise RunnerInputError(f"Unsupported methods: {unsupported}.")
    if not isinstance(protocol, FormalProtocol):
        raise TypeError("protocol must be a FormalProtocol.")
    scheduled_fit_budgets = resolve_requested_fit_budgets(
        protocol, requested_fit_budgets
    )
    if historical_replay is not None and not isinstance(
        historical_replay, TrajectoryBatch
    ):
        raise TypeError("historical_replay must be a TrajectoryBatch or None.")

    (
        trajectory_splits,
        generated,
        model_training_fingerprint,
        model_n_regimes,
    ) = _resolve_splits(system, hydrology, protocol, splits)
    observed_n_regimes = _validate_splits(trajectory_splits, system, protocol)
    n_regimes = observed_n_regimes if model_n_regimes is None else model_n_regimes
    if n_regimes < observed_n_regimes:
        raise RunnerInputError(
            "Trajectory regime labels exceed the fitted PAR model's regime count."
        )
    if historical_replay is not None:
        _validate_historical_replay(historical_replay, system, n_regimes)
    architecture = _architecture_plan(
        system,
        n_regimes,
        selected_methods,
        smoke=bool(protocol.smoke_mode),
    )
    split_fingerprints = {
        name: _batch_fingerprint(batch)
        for name, batch in trajectory_splits.as_dict().items()
    }
    historical_replay_fingerprint = (
        None
        if historical_replay is None
        else _batch_fingerprint(historical_replay)
    )
    if historical_replay_fingerprint is not None:
        split_fingerprints["historical_replay"] = historical_replay_fingerprint
    experiment_payload = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "system": system.as_jsonable(),
        "protocol": asdict(protocol),
        "split_fingerprints": split_fingerprints,
        # Split content fully determines every optimization/evaluation call.
        # Keeping the optional PAR provenance outside this hash makes a result
        # resumable whether the caller supplies the model plus splits or the
        # exact same TrajectorySplits directly.
        "n_regimes": n_regimes,
        "device": device,
    }
    experiment_fingerprint = sha256(_canonical_json(experiment_payload)).hexdigest()
    base_dir = (
        Path(output_dir)
        / _slug(system.benchmark)
        / _slug(system.case_id)
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        base_dir,
        experiment_fingerprint,
        system,
        protocol,
        split_fingerprints,
        model_training_fingerprint,
        device,
        historical_replay,
        historical_replay_fingerprint,
        resume=resume,
    )

    completed: list[Mapping[str, object]] = []
    skipped: list[Mapping[str, object]] = []
    failures: list[Mapping[str, object]] = []
    for n_fit in scheduled_fit_budgets:
        fit_prefix = trajectory_splits.fit.take(
            protocol.fit_prefix(n_fit), name=f"fit_n{n_fit}"
        )
        normalization = _fit_objective_normalization(
            fit_prefix,
            system,
            n_regimes,
            device,
        )
        training_system = _normalized_training_system(system, normalization)
        for method in selected_methods:
            for train_seed in protocol.train_seeds:
                key = RunKey(method, n_fit, train_seed)
                settings = _method_settings(
                    method,
                    n_fit,
                    training_system,
                    protocol,
                    device,
                    architecture,
                    normalization,
                )
                run_fingerprint = _run_fingerprint(
                    experiment_fingerprint, key, settings
                )
                run_dir = base_dir / key.relative_directory
                if resume:
                    existing = _valid_result(
                        run_dir,
                        run_fingerprint,
                        key,
                        expect_historical_replay=historical_replay is not None,
                    )
                    if existing is not None:
                        # ``result.json`` is committed before these supervisory
                        # files.  Heal the narrow crash window on every valid
                        # resume so external validators see one canonical state.
                        _atomic_json(
                            run_dir / "status.json",
                            {
                                "schema_version": RUNNER_SCHEMA_VERSION,
                                "status": "complete",
                                "completed_at": existing["completed_at"],
                                "run_key": existing["run_key"],
                                "run_key_string": key.key_string,
                                "result": "result.json",
                            },
                        )
                        (run_dir / "failure.json").unlink(missing_ok=True)
                        if progress is not None:
                            progress(f"[skip] {key.key_string}")
                        skipped.append(existing)
                        continue
                try:
                    if progress is not None:
                        progress(f"[run] {key.key_string}")
                    _validate_method_applicability(method, system)
                    _atomic_json(
                        run_dir / "status.json",
                        {
                            "schema_version": RUNNER_SCHEMA_VERSION,
                            "status": "running",
                            "started_at": datetime.now(timezone.utc).isoformat(),
                            "pid": os.getpid(),
                            "run_key": key.as_dict(),
                            "run_key_string": key.key_string,
                            "run_fingerprint": run_fingerprint,
                        },
                    )
                    result = _execute_one(
                        key,
                        run_dir,
                        run_fingerprint,
                        experiment_fingerprint,
                        settings,
                        trajectory_splits,
                        system,
                        training_system,
                        protocol,
                        n_regimes,
                        device,
                        historical_replay,
                        historical_replay_fingerprint,
                    )
                    completed.append(result)
                    _atomic_json(
                        run_dir / "status.json",
                        {
                            "schema_version": RUNNER_SCHEMA_VERSION,
                            "status": "complete",
                            "completed_at": result["completed_at"],
                            "run_key": key.as_dict(),
                            "run_key_string": key.key_string,
                            "result": "result.json",
                        },
                    )
                    if progress is not None:
                        progress(f"[done] {key.key_string}")
                except Exception as error:  # Keep later independent keys runnable.
                    failure = {
                        "schema_version": RUNNER_SCHEMA_VERSION,
                        "status": "failed",
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "run_key": key.as_dict(),
                        "run_key_string": key.key_string,
                        "run_fingerprint": run_fingerprint,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    _atomic_json(run_dir / "failure.json", failure)
                    _atomic_json(
                        run_dir / "status.json",
                        {
                            "schema_version": RUNNER_SCHEMA_VERSION,
                            "status": "failed",
                            "failed_at": failure["failed_at"],
                            "run_key": key.as_dict(),
                            "run_key_string": key.key_string,
                            "failure": "failure.json",
                        },
                    )
                    failures.append(failure)
                    if progress is not None:
                        progress(
                            f"[failed] {key.key_string}: "
                            f"{type(error).__name__}: {error}"
                        )

    summary = RunnerSummary(
        output_dir=base_dir,
        experiment_fingerprint=experiment_fingerprint,
        completed=tuple(completed),
        skipped=tuple(skipped),
        failures=tuple(failures),
        generated_splits=generated,
        requested_fit_budgets=scheduled_fit_budgets,
    )
    if failures and raise_on_failure:
        raise RunnerExecutionError(summary)
    return summary


run_single_benchmark = run_benchmark


__all__ = [
    "DEFAULT_METHODS",
    "OPTIONAL_METHODS",
    "RUNNER_SCHEMA_VERSION",
    "RunKey",
    "RunnerExecutionError",
    "RunnerInputError",
    "RunnerSummary",
    "SUPPORTED_METHODS",
    "resolve_requested_fit_budgets",
    "run_benchmark",
    "run_single_benchmark",
]
