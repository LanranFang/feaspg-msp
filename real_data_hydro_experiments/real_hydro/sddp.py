"""Finite-hydrology-state SDDP for the common classic convex simulator.

The fitted PAR lag vector is quantized into calendar-specific regimes before
optimization.  Every policy receives the same regime indicator.  Unlike the
older stagewise-independent baseline, continuation cuts here are indexed by the
next hydrology regime and include both storage and previous release in the
state.  The latter is required because the common objective contains ramping.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import warnings

import gurobipy as gp
import numpy as np
import torch
from torch import nn

from .config import ObjectiveConfig, ReservoirSystem
from .dataset import TrajectoryBatch
from .evaluation import evaluate_policy


Cut = tuple[float, np.ndarray]


@dataclass(frozen=True)
class SDDPConfig:
    iterations: int = 80
    validation_every: int = 10
    noise_scenarios: int = 5
    device: str = "cpu"
    max_validation_checks: int = 50
    complete_recourse_penalty: float | None = None

    def __post_init__(self) -> None:
        if min(
            self.iterations,
            self.validation_every,
            self.max_validation_checks,
            self.noise_scenarios,
        ) < 1:
            raise ValueError("SDDP counts must be positive.")
        if self.complete_recourse_penalty is not None and (
            not np.isfinite(self.complete_recourse_penalty)
            or self.complete_recourse_penalty <= 0
        ):
            raise ValueError("SDDP complete-recourse penalty must be finite and positive.")


def validation_schedule(config: SDDPConfig) -> tuple[int, ...]:
    """Return deterministic validation checkpoints, always including the final iterate."""

    cadence = max(
        config.validation_every,
        math.ceil(config.iterations / config.max_validation_checks),
    )
    checkpoints = list(range(cadence, config.iterations + 1, cadence))
    if not checkpoints or checkpoints[-1] != config.iterations:
        checkpoints.append(config.iterations)
    if len(checkpoints) > config.max_validation_checks:
        raise AssertionError("Internal error: validation schedule exceeds its configured cap.")
    return tuple(checkpoints)


@dataclass(frozen=True)
class ScenarioSet:
    inflow: np.ndarray
    demand: np.ndarray
    next_regime: np.ndarray
    probability: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.probability)
        if self.inflow.ndim != 2 or self.inflow.shape[0] != n:
            raise ValueError("Scenario inflows must align with probabilities.")
        if self.demand.shape != (n,) or self.next_regime.shape != (n,):
            raise ValueError("Scenario demand/regime arrays must align.")
        if n < 1 or np.any(self.probability < 0.0):
            raise ValueError("Scenario probabilities must be nonnegative and nonempty.")
        if not np.isclose(self.probability.sum(), 1.0):
            raise ValueError("Scenario probabilities must sum to one.")


@dataclass(frozen=True)
class MarkovDiscretization:
    n_regimes: int
    scenarios: tuple[tuple[ScenarioSet, ...], ...]
    fallback_bins: tuple[tuple[bool, ...], ...]

    @property
    def horizon(self) -> int:
        return len(self.scenarios)


@dataclass(frozen=True)
class StageSolution:
    objective: float
    action: np.ndarray
    state_dual: np.ndarray
    optimization_attempts: int = 1
    feasibility_event: dict[str, object] | None = None


class SDDPSolveError(RuntimeError):
    """An unrecovered stage solve, including the attempted recovery audit."""

    def __init__(
        self,
        stage: int,
        statuses: tuple[int, ...],
        diagnostic: str,
    ) -> None:
        self.stage = int(stage)
        self.statuses = tuple(int(status) for status in statuses)
        self.status_names = tuple(_gurobi_status_name(status) for status in statuses)
        self.diagnostic = str(diagnostic)
        sequence = " -> ".join(
            f"{name}({status})"
            for name, status in zip(self.status_names, self.statuses, strict=True)
        )
        super().__init__(
            f"SDDP stage {self.stage} failed after auditable recovery attempts: "
            f"{sequence}. Diagnostic: {self.diagnostic}"
        )


@dataclass
class SDDPFitResult:
    policy: "HydrologyStateSDDPPolicy"
    train_seed: int
    best_validation_cost: float
    selected_iteration: int
    number_of_cuts: int
    fit_time_seconds: float
    history: list[dict[str, float | int | str]]
    feasibility_audit: dict[str, object]
    checkpoint_path: str


_GUROBI_STATUS_NAMES = {
    int(getattr(gp.GRB, name)): name
    for name in (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    )
    if hasattr(gp.GRB, name)
}


def _gurobi_status_name(status: int) -> str:
    return _GUROBI_STATUS_NAMES.get(int(status), "UNKNOWN")


def _summarize_feasibility_events(
    events: list[dict[str, object]],
    *,
    example_limit: int = 25,
) -> dict[str, object]:
    by_resolution: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    artificial_total = 0.0
    artificial_weighted_total = 0.0
    artificial_max = 0.0
    for event in events:
        resolution = str(event["resolution"])
        stage = str(int(event["stage"]))
        by_resolution[resolution] = by_resolution.get(resolution, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1
        artificial_total += float(event["artificial_water_total"])
        artificial_weighted_total += float(event["artificial_water_weighted"])
        artificial_max = max(artificial_max, float(event["artificial_water_max"]))
    return {
        "recovered_solve_count": len(events),
        "numerical_retry_count": by_resolution.get("numerical_retry", 0),
        "complete_recourse_count": by_resolution.get("complete_recourse", 0),
        "artificial_water_total": artificial_total,
        "artificial_water_probability_weighted_total": artificial_weighted_total,
        "artificial_water_max": artificial_max,
        "by_resolution": by_resolution,
        "by_stage": by_stage,
        "event_examples": copy.deepcopy(events[:example_limit]),
        "event_examples_truncated": max(0, len(events) - example_limit),
    }


def _merge_feasibility_audits(
    first: dict[str, object],
    second: dict[str, object],
    *,
    example_limit: int = 25,
) -> dict[str, object]:
    def merged_counts(name: str) -> dict[str, int]:
        left = dict(first.get(name, {}))
        right = dict(second.get(name, {}))
        keys = set(left) | set(right)
        return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}

    examples = list(first.get("event_examples", [])) + list(
        second.get("event_examples", [])
    )
    known_truncated = int(first.get("event_examples_truncated", 0)) + int(
        second.get("event_examples_truncated", 0)
    )
    merged = {
        "recovered_solve_count": int(first.get("recovered_solve_count", 0))
        + int(second.get("recovered_solve_count", 0)),
        "numerical_retry_count": int(first.get("numerical_retry_count", 0))
        + int(second.get("numerical_retry_count", 0)),
        "complete_recourse_count": int(first.get("complete_recourse_count", 0))
        + int(second.get("complete_recourse_count", 0)),
        "artificial_water_total": float(first.get("artificial_water_total", 0.0))
        + float(second.get("artificial_water_total", 0.0)),
        "artificial_water_probability_weighted_total": float(
            first.get("artificial_water_probability_weighted_total", 0.0)
        )
        + float(second.get("artificial_water_probability_weighted_total", 0.0)),
        "artificial_water_max": max(
            float(first.get("artificial_water_max", 0.0)),
            float(second.get("artificial_water_max", 0.0)),
        ),
        "by_resolution": merged_counts("by_resolution"),
        "by_stage": merged_counts("by_stage"),
        "event_examples": copy.deepcopy(examples[:example_limit]),
        "event_examples_truncated": known_truncated
        + max(0, len(examples) - example_limit),
    }
    penalties = {
        float(audit["complete_recourse_penalty"])
        for audit in (first, second)
        if audit.get("complete_recourse_penalty") is not None
    }
    if penalties:
        if len(penalties) != 1:
            raise ValueError("Cannot merge feasibility audits with different penalties.")
        merged["complete_recourse_penalty"] = penalties.pop()
    return merged


def _default_complete_recourse_penalty(system: ReservoirSystem) -> float:
    """Choose a scale-aware penalty well above any physical marginal cost."""

    objective = system.objective
    coefficient_names = (
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
    coefficients = [abs(float(objective.deficit_cost))]
    for name in coefficient_names:
        coefficients.extend(abs(float(value)) for value in getattr(objective, name))
    maximum_cost = max(coefficients, default=0.0)
    if maximum_cost == 0.0:
        return 1.0
    # A unit of water can affect storage penalties over the horizon and can be
    # routed through several generators.  The 1e3 separation is deliberate,
    # while scaling by the actual normalized objective avoids the 1e12
    # coefficient range produced by a fixed 1e6 penalty in the ONS case.
    generation_scale = sum(
        abs(float(value)) for value in objective.generation_coefficient
    )
    influence_bound = system.horizon + system.n_reservoirs + generation_scale
    return 1.0e3 * max(influence_bound, 1.0) * maximum_cost


def _resolve_complete_recourse_penalty(
    system: ReservoirSystem,
    configured: float | None,
) -> float:
    penalty = (
        _default_complete_recourse_penalty(system)
        if configured is None
        else float(configured)
    )
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("Complete-recourse penalty must be finite and positive.")
    return penalty


def _kmeans(values: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or n_clusters < 1:
        raise ValueError("k-means requires a nonempty matrix and positive k.")
    if not np.all(np.isfinite(values)):
        raise ValueError("k-means observations must all be finite.")
    k = min(n_clusters, len(values))
    scale = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.ones(values.shape[1])
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-10), scale, 1.0)
    normalized = values / scale
    centers_normalized = [
        normalized[np.square(normalized - normalized.mean(axis=0)).sum(axis=1).argmin()]
    ]
    for _ in range(1, k):
        distances = np.min(
            np.stack(
                [np.square(normalized - center).sum(axis=1) for center in centers_normalized]
            ),
            axis=0,
        )
        centers_normalized.append(normalized[int(distances.argmax())])
    centers_normalized = np.stack(centers_normalized)
    labels = np.full(len(values), -1, dtype=np.int64)
    for _ in range(100):
        new_labels = np.square(
            normalized[:, None] - centers_normalized[None]
        ).sum(axis=2).argmin(axis=1)
        updated = centers_normalized.copy()
        for cluster in range(k):
            members = normalized[new_labels == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(updated, centers_normalized):
            break
        labels, centers_normalized = new_labels, updated
    # Duplicate observations can make several initialized centers identical.
    # Deterministic argmin then assigns those observations to the first copy,
    # leaving later clusters empty.  Treat the requested cluster count as an
    # upper bound and retain only empirical support with positive mass; taking
    # the mean of an empty cluster would otherwise inject NaNs into every SDDP
    # stage LP.
    counts = np.bincount(labels, minlength=k)
    active = np.flatnonzero(counts > 0)
    if active.size < 1:
        raise RuntimeError("k-means produced no nonempty empirical support.")
    centers = np.stack(
        [values[labels == cluster].mean(axis=0) for cluster in active]
    )
    probabilities = counts[active].astype(np.float64)
    probabilities /= probabilities.sum()
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("k-means produced nonfinite empirical support.")
    return centers, probabilities


def _allocate_clusters(counts: np.ndarray, budget: int) -> np.ndarray:
    allocation = np.ones(len(counts), dtype=np.int64)
    total = max(int(budget), len(counts))
    while allocation.sum() < total:
        score = counts / allocation
        allocation[int(score.argmax())] += 1
    return np.minimum(allocation, counts)


def _conditional_support(
    continuous: np.ndarray,
    next_regime: np.ndarray,
    budget: int,
    n_reservoirs: int,
) -> ScenarioSet:
    labels, counts = np.unique(next_regime.astype(np.int64), return_counts=True)
    allocation = _allocate_clusters(counts, budget)
    centers = []
    probabilities = []
    next_labels = []
    total = len(continuous)
    for label, clusters in zip(labels, allocation, strict=True):
        members = continuous[next_regime == label]
        group_centers, group_weights = _kmeans(members, int(clusters))
        centers.append(group_centers)
        probabilities.append(group_weights * len(members) / total)
        next_labels.extend([int(label)] * len(group_centers))
    combined = np.concatenate(centers, axis=0)
    probability = np.concatenate(probabilities)
    probability /= probability.sum()
    return ScenarioSet(
        inflow=np.maximum(combined[:, :n_reservoirs], 0.0),
        demand=np.maximum(combined[:, n_reservoirs], 0.0),
        next_regime=np.asarray(next_labels, dtype=np.int16),
        probability=probability,
    )


def build_markov_discretization(
    fit: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    noise_scenarios: int,
) -> MarkovDiscretization:
    """Estimate joint inflow/demand/regime transitions from fit paths only."""

    if fit.horizon != system.horizon or fit.n_reservoirs != system.n_reservoirs:
        raise ValueError("Fit trajectories and reservoir system do not align.")
    if n_regimes <= int(fit.hydrology_regime.max(initial=0)):
        raise ValueError("n_regimes does not cover every fit regime.")
    stage_sets = []
    fallback_flags = []
    for stage in range(system.horizon):
        demand = (
            fit.demand[:, stage].astype(np.float64)
            if fit.demand is not None
            else np.full(
                fit.n,
                float(system.objective.seasonal_demand[system.period(stage)])
                if system.objective.kind == "hydrothermal"
                else 0.0,
            )
        )
        continuous_all = np.concatenate(
            (fit.inflow[:, stage].astype(np.float64), demand[:, None]), axis=1
        )
        next_all = (
            fit.hydrology_regime[:, stage + 1]
            if stage + 1 < system.horizon
            else fit.hydrology_regime[:, stage]
        )
        regime_sets = []
        regime_fallback = []
        for regime in range(n_regimes):
            members = fit.hydrology_regime[:, stage] == regime
            fallback = not np.any(members)
            if fallback:
                members = np.ones(fit.n, dtype=bool)
            regime_sets.append(
                _conditional_support(
                    continuous_all[members],
                    next_all[members],
                    noise_scenarios,
                    system.n_reservoirs,
                )
            )
            regime_fallback.append(fallback)
        stage_sets.append(tuple(regime_sets))
        fallback_flags.append(tuple(regime_fallback))
    return MarkovDiscretization(
        n_regimes=n_regimes,
        scenarios=tuple(stage_sets),
        fallback_bins=tuple(fallback_flags),
    )


class _DecisionHazardStage:
    def __init__(
        self,
        env: gp.Env,
        system: ReservoirSystem,
        stage: int,
        scenarios: ScenarioSet,
        future_cuts: list[list[Cut]],
        complete_recourse_penalty: float | None = None,
    ) -> None:
        self.system = system
        self.stage = stage
        self.scenarios = scenarios
        self.complete_recourse_penalty = _resolve_complete_recourse_penalty(
            system, complete_recourse_penalty
        )
        objective = system.objective
        R = system.n_reservoirs
        capacity = np.asarray(system.storage_capacity, dtype=np.float64)
        release_capacity = np.asarray(system.release_capacity, dtype=np.float64)
        routing = np.asarray(system.routing, dtype=np.float64)
        period = system.period(stage)

        self.model = gp.Model(f"real_hydro_sddp_{stage}", env=env)
        self.model.Params.OutputFlag = 0
        self._standard_solver_parameters = {
            name: getattr(self.model.Params, name)
            for name in (
                "DualReductions",
                "FeasibilityTol",
                "Method",
                "NumericFocus",
                "OptimalityTol",
                "Presolve",
                "ScaleFlag",
            )
        }
        self.incoming_storage = [
            self.model.addVar(lb=-gp.GRB.INFINITY, name=f"incoming_storage[{r}]")
            for r in range(R)
        ]
        self.incoming_previous = [
            self.model.addVar(lb=-gp.GRB.INFINITY, name=f"incoming_previous[{r}]")
            for r in range(R)
        ]
        self.fix_storage = [
            self.model.addConstr(variable == 0.0, name=f"fix_storage[{r}]")
            for r, variable in enumerate(self.incoming_storage)
        ]
        self.fix_previous = [
            self.model.addConstr(variable == 0.0, name=f"fix_previous[{r}]")
            for r, variable in enumerate(self.incoming_previous)
        ]
        self.action = [
            self.model.addVar(lb=0.0, ub=float(release_capacity[r]), name=f"release[{r}]")
            for r in range(R)
        ]
        for r in range(R):
            self.model.addConstr(
                self.action[r] <= self.incoming_storage[r], name=f"available_water[{r}]"
            )
        if system.shared_release_capacity is not None:
            self.model.addConstr(
                gp.quicksum(self.action) <= float(system.shared_release_capacity),
                name="shared_release",
            )

        common_cost = gp.LinExpr()
        for r in range(R):
            shortage = self.model.addVar(lb=0.0, name=f"shortage[{r}]")
            excess = self.model.addVar(lb=0.0, name=f"excess[{r}]")
            ramp = self.model.addVar(lb=0.0, name=f"ramp[{r}]")
            target = float(objective.release_target[period][r])
            self.model.addConstr(shortage >= target - self.action[r])
            self.model.addConstr(excess >= self.action[r] - target)
            self.model.addConstr(ramp >= self.action[r] - self.incoming_previous[r])
            self.model.addConstr(ramp >= self.incoming_previous[r] - self.action[r])
            common_cost += float(objective.shortage_cost[r]) * shortage
            common_cost += float(objective.excess_release_cost[r]) * excess
            common_cost += float(objective.ramp_cost[r]) * ramp

        self.outgoing_storage: list[list[gp.Var]] = []
        self.continuation: list[gp.Var | None] = []
        self.artificial_water: list[list[gp.Var]] = []
        model_objective = common_cost
        for scenario_index, probability in enumerate(scenarios.probability):
            outgoing = [
                self.model.addVar(
                    lb=0.0,
                    ub=float(capacity[r]),
                    name=f"outgoing[{scenario_index},{r}]",
                )
                for r in range(R)
            ]
            self.outgoing_storage.append(outgoing)
            artificial_water = [
                self.model.addVar(
                    lb=0.0,
                    ub=0.0,
                    name=f"complete_recourse_water[{scenario_index},{r}]",
                )
                for r in range(R)
            ]
            self.artificial_water.append(artificial_water)
            scenario_cost = gp.LinExpr()
            for r in range(R):
                spill = self.model.addVar(lb=0.0, name=f"spill[{scenario_index},{r}]")
                low = self.model.addVar(lb=0.0, name=f"low[{scenario_index},{r}]")
                flood = self.model.addVar(lb=0.0, name=f"flood[{scenario_index},{r}]")
                routed = gp.quicksum(
                    float(routing[source, r]) * self.action[source]
                    for source in range(R)
                )
                self.model.addConstr(
                    outgoing[r] + spill
                    == self.incoming_storage[r]
                    - self.action[r]
                    + float(scenarios.inflow[scenario_index, r])
                    + routed
                    + artificial_water[r],
                    name=f"balance[{scenario_index},{r}]",
                )
                self.model.addConstr(
                    low >= float(objective.guide_lower[period][r]) - outgoing[r]
                )
                self.model.addConstr(
                    flood >= outgoing[r] - float(objective.guide_upper[period][r])
                )
                scenario_cost += float(objective.spill_cost[r]) * spill
                scenario_cost += float(objective.low_storage_cost[r]) * low
                scenario_cost += float(objective.flood_cost[r]) * flood

            if objective.kind == "hydrothermal":
                blocks = [
                    self.model.addVar(
                        lb=0.0,
                        ub=float(block_capacity),
                        name=f"thermal[{scenario_index},{block}]",
                    )
                    for block, block_capacity in enumerate(objective.thermal_block_capacity)
                ]
                deficit = self.model.addVar(lb=0.0, name=f"deficit[{scenario_index}]")
                self.model.addConstr(
                    gp.quicksum(blocks)
                    + deficit
                    + gp.quicksum(
                        float(objective.generation_coefficient[r]) * self.action[r]
                        for r in range(R)
                    )
                    >= float(scenarios.demand[scenario_index]),
                    name=f"load_balance[{scenario_index}]",
                )
                for block, marginal_cost in zip(
                    blocks, objective.thermal_block_cost, strict=True
                ):
                    scenario_cost += float(marginal_cost) * block
                scenario_cost += float(objective.deficit_cost) * deficit

            if stage < system.horizon - 1:
                continuation = self.model.addVar(
                    lb=0.0, name=f"continuation[{scenario_index}]"
                )
                self.continuation.append(continuation)
                scenario_cost += continuation
            else:
                self.continuation.append(None)
                for r in range(R):
                    shortfall = self.model.addVar(
                        lb=0.0, name=f"terminal_shortfall[{scenario_index},{r}]"
                    )
                    excess = self.model.addVar(
                        lb=0.0, name=f"terminal_excess[{scenario_index},{r}]"
                    )
                    target = float(objective.terminal_target[r])
                    self.model.addConstr(shortfall >= target - outgoing[r])
                    self.model.addConstr(excess >= outgoing[r] - target)
                    scenario_cost += float(objective.terminal_shortfall_cost[r]) * shortfall
                    scenario_cost += float(objective.terminal_excess_cost[r]) * excess
            model_objective += float(probability) * scenario_cost

        # A deterministic, negligible tie-breaker removes path-order dependence.
        model_objective += 1.0e-9 * gp.quicksum(
            (r + 1) * action for r, action in enumerate(self.action)
        )
        self.model.setObjective(model_objective, gp.GRB.MINIMIZE)
        for next_regime, cuts in enumerate(future_cuts):
            for cut in cuts:
                self.add_future_cut(next_regime, cut)
        self.model.update()

    def add_future_cut(self, next_regime: int, cut: Cut) -> None:
        if self.stage >= self.system.horizon - 1:
            return
        intercept, slope = cut
        R = self.system.n_reservoirs
        for scenario_index, continuation in enumerate(self.continuation):
            if int(self.scenarios.next_regime[scenario_index]) != next_regime:
                continue
            assert continuation is not None
            self.model.addConstr(
                continuation
                >= float(intercept)
                + gp.quicksum(
                    float(slope[r]) * self.outgoing_storage[scenario_index][r]
                    for r in range(R)
                )
                + gp.quicksum(
                    float(slope[R + r]) * self.action[r] for r in range(R)
                )
            )

    def _optimize(self) -> int:
        self.model.optimize()
        return int(self.model.Status)

    def _configure_robust_retry(self) -> None:
        # Large ONS energy states (order 1e7 MWh-month) occasionally cause
        # presolve to report a false status-3 infeasibility.  This retry uses a
        # different, conservative LP path and distinguishes infeasible from
        # unbounded models.
        self.model.Params.DualReductions = 0
        self.model.Params.FeasibilityTol = 1.0e-8
        self.model.Params.Method = 0
        self.model.Params.NumericFocus = 3
        self.model.Params.OptimalityTol = 1.0e-8
        self.model.Params.Presolve = 0
        self.model.Params.ScaleFlag = 2

    def _restore_solver_parameters(self) -> None:
        for name, value in self._standard_solver_parameters.items():
            setattr(self.model.Params, name, value)

    def _set_complete_recourse(self, enabled: bool) -> None:
        upper_bound = gp.GRB.INFINITY if enabled else 0.0
        for scenario_index, scenario_variables in enumerate(self.artificial_water):
            for variable in scenario_variables:
                variable.UB = upper_bound
                variable.Obj = (
                    float(self.scenarios.probability[scenario_index])
                    * self.complete_recourse_penalty
                    if enabled
                    else 0.0
                )
        self.model.update()

    def _diagnose_failure(self) -> str:
        try:
            self.model.computeIIS()
            constraints = [
                constraint.ConstrName
                for constraint in self.model.getConstrs()
                if constraint.IISConstr
            ]
            bounds = []
            for variable in self.model.getVars():
                if variable.IISLB:
                    bounds.append(f"{variable.VarName}:LB")
                if variable.IISUB:
                    bounds.append(f"{variable.VarName}:UB")
            return f"IIS constraints={constraints[:25]!r}, bounds={bounds[:25]!r}"
        except gp.GurobiError as error:
            # This exact diagnostic is useful: in the observed ONS failure,
            # computeIIS itself reported that the allegedly infeasible model
            # was feasible, confirming a numerical/presolve misclassification.
            return f"IIS unavailable ({error})"

    def _solution(
        self,
        *,
        optimization_attempts: int,
        feasibility_event: dict[str, object] | None,
    ) -> StageSolution:
        dual = np.asarray(
            [constraint.Pi for constraint in self.fix_storage]
            + [constraint.Pi for constraint in self.fix_previous],
            dtype=np.float64,
        )
        return StageSolution(
            objective=float(self.model.ObjVal),
            action=np.asarray([variable.X for variable in self.action]),
            state_dual=dual,
            optimization_attempts=optimization_attempts,
            feasibility_event=feasibility_event,
        )

    def _feasibility_event(
        self,
        *,
        resolution: str,
        statuses: list[int],
        storage: np.ndarray,
        previous_release: np.ndarray,
    ) -> dict[str, object]:
        artificial = np.asarray(
            [
                [variable.X for variable in scenario_variables]
                for scenario_variables in self.artificial_water
            ],
            dtype=np.float64,
        )
        probabilities = np.asarray(self.scenarios.probability, dtype=np.float64)
        return {
            "stage": self.stage,
            "resolution": resolution,
            "status_sequence": [int(status) for status in statuses],
            "status_name_sequence": [
                _gurobi_status_name(status) for status in statuses
            ],
            "optimization_attempts": len(statuses),
            "complete_recourse_penalty": self.complete_recourse_penalty,
            "artificial_water_total": float(artificial.sum()),
            "artificial_water_weighted": float(
                (artificial.sum(axis=1) * probabilities).sum()
            ),
            "artificial_water_max": float(artificial.max(initial=0.0)),
            "storage_min": float(storage.min()),
            "storage_max": float(storage.max()),
            "previous_release_min": float(previous_release.min()),
            "previous_release_max": float(previous_release.max()),
        }

    def solve(self, storage: np.ndarray, previous_release: np.ndarray) -> StageSolution:
        storage = np.asarray(storage, dtype=np.float64)
        previous_release = np.asarray(previous_release, dtype=np.float64)
        R = self.system.n_reservoirs
        if storage.shape != (R,) or previous_release.shape != (R,):
            raise ValueError("SDDP state vectors have the wrong shape.")
        storage = np.clip(storage, 0.0, np.asarray(self.system.storage_capacity))
        previous_release = np.clip(
            previous_release, 0.0, np.asarray(self.system.release_capacity)
        )
        for r in range(R):
            self.fix_storage[r].RHS = float(storage[r])
            self.fix_previous[r].RHS = float(previous_release[r])
        self._set_complete_recourse(False)
        self._restore_solver_parameters()
        self.model.reset(1)
        statuses = [self._optimize()]
        if statuses[-1] == gp.GRB.OPTIMAL:
            return self._solution(optimization_attempts=1, feasibility_event=None)

        recoverable = {
            gp.GRB.INFEASIBLE,
            gp.GRB.INF_OR_UNBD,
            gp.GRB.NUMERIC,
            gp.GRB.SUBOPTIMAL,
        }
        if statuses[-1] not in recoverable:
            raise SDDPSolveError(
                self.stage,
                tuple(statuses),
                f"status {_gurobi_status_name(statuses[-1])} is not a feasibility "
                "status and was not silently relaxed",
            )

        try:
            self._configure_robust_retry()
            self.model.reset(1)
            statuses.append(self._optimize())
            if statuses[-1] == gp.GRB.OPTIMAL:
                event = self._feasibility_event(
                    resolution="numerical_retry",
                    statuses=statuses,
                    storage=storage,
                    previous_release=previous_release,
                )
                return self._solution(
                    optimization_attempts=len(statuses), feasibility_event=event
                )

            if statuses[-1] != gp.GRB.INFEASIBLE:
                raise SDDPSolveError(
                    self.stage,
                    tuple(statuses),
                    f"robust retry ended with {_gurobi_status_name(statuses[-1])}; "
                    "complete recourse was not used for a non-infeasibility status",
                )

            self._set_complete_recourse(True)
            self.model.reset(1)
            statuses.append(self._optimize())
            if statuses[-1] != gp.GRB.OPTIMAL:
                self._set_complete_recourse(False)
                diagnostic = self._diagnose_failure()
                raise SDDPSolveError(self.stage, tuple(statuses), diagnostic)

            event = self._feasibility_event(
                resolution="complete_recourse",
                statuses=statuses,
                storage=storage,
                previous_release=previous_release,
            )
            return self._solution(
                optimization_attempts=len(statuses), feasibility_event=event
            )
        finally:
            self._set_complete_recourse(False)
            self._restore_solver_parameters()

    def dispose(self) -> None:
        self.model.dispose()


class _SDDPBank:
    def __init__(
        self,
        system: ReservoirSystem,
        discretization: MarkovDiscretization,
        cuts: list[list[list[Cut]]] | None = None,
        complete_recourse_penalty: float | None = None,
    ) -> None:
        self.system = system
        self.discretization = discretization
        self.complete_recourse_penalty = _resolve_complete_recourse_penalty(
            system, complete_recourse_penalty
        )
        K = discretization.n_regimes
        self.cuts = (
            [[[] for _ in range(K)] for _ in range(system.horizon)]
            if cuts is None
            else copy.deepcopy(cuts)
        )
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.solve_calls = 0
        self.optimization_attempts = 0
        self.feasibility_events: list[dict[str, object]] = []
        self._warned_recovery_resolutions: set[str] = set()
        self.models: list[list[_DecisionHazardStage]] = []
        for stage in range(system.horizon):
            stage_models = []
            future = self.cuts[stage + 1] if stage + 1 < system.horizon else [[] for _ in range(K)]
            for regime in range(K):
                stage_models.append(
                    _DecisionHazardStage(
                        self.env,
                        system,
                        stage,
                        discretization.scenarios[stage][regime],
                        future,
                        self.complete_recourse_penalty,
                    )
                )
            self.models.append(stage_models)

    def solve(self, stage: int, regime: int, storage, previous_release) -> StageSolution:
        solution = self.models[stage][regime].solve(storage, previous_release)
        self.solve_calls += 1
        self.optimization_attempts += solution.optimization_attempts
        if solution.feasibility_event is not None:
            event = copy.deepcopy(solution.feasibility_event)
            self.feasibility_events.append(event)
            resolution = str(event["resolution"])
            if resolution not in self._warned_recovery_resolutions:
                warnings.warn(
                    f"SDDP encountered {resolution!r} first at stage {stage}; "
                    "all occurrences and artificial-water amounts are retained "
                    "in policy.feasibility_audit.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_recovery_resolutions.add(resolution)
        return solution

    def add_cut(self, stage: int, regime: int, cut: Cut) -> None:
        self.cuts[stage][regime].append(cut)
        if stage > 0:
            for predecessor in self.models[stage - 1]:
                predecessor.add_future_cut(regime, cut)

    def snapshot_cuts(self) -> list[list[list[Cut]]]:
        return copy.deepcopy(self.cuts)

    @property
    def number_of_cuts(self) -> int:
        return sum(len(regime_cuts) for stage in self.cuts for regime_cuts in stage)

    @property
    def feasibility_audit(self) -> dict[str, object]:
        audit = _summarize_feasibility_events(self.feasibility_events)
        audit["complete_recourse_penalty"] = self.complete_recourse_penalty
        return audit

    def close(self) -> None:
        for stage in self.models:
            for model in stage:
                model.dispose()
        self.env.dispose()


class HydrologyStateSDDPPolicy(nn.Module):
    policy_name = "hydrology_state_sddp"

    def __init__(
        self,
        system: ReservoirSystem,
        discretization: MarkovDiscretization,
        cuts: list[list[list[Cut]]] | None = None,
        bank: _SDDPBank | None = None,
        complete_recourse_penalty: float | None = None,
        checkpoint_feasibility_audit: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.discretization = discretization
        self.bank = (
            _SDDPBank(
                system,
                discretization,
                cuts,
                complete_recourse_penalty=complete_recourse_penalty,
            )
            if bank is None
            else bank
        )
        self._owns_bank = bank is None
        self.checkpoint_feasibility_audit = copy.deepcopy(
            checkpoint_feasibility_audit
            if checkpoint_feasibility_audit is not None
            else _summarize_feasibility_events([])
        )
        self._closed_feasibility_audit: dict[str, object] | None = None

    def propose(self, stage, storage, previous_release, hydrology_features):
        regimes = hydrology_features[:, : self.discretization.n_regimes].argmax(dim=-1)
        storage_np = storage.detach().cpu().numpy()
        previous_np = previous_release.detach().cpu().numpy()
        actions = np.empty_like(storage_np)
        for path, regime in enumerate(regimes.detach().cpu().numpy()):
            actions[path] = self.bank.solve(
                stage, int(regime), storage_np[path], previous_np[path]
            ).action
        return torch.as_tensor(actions, dtype=storage.dtype, device=storage.device)

    @property
    def number_of_cuts(self) -> int:
        return self.bank.number_of_cuts

    @property
    def deployment_feasibility_audit(self) -> dict[str, object]:
        if self.bank is not None:
            return self.bank.feasibility_audit
        assert self._closed_feasibility_audit is not None
        return copy.deepcopy(self._closed_feasibility_audit)

    @property
    def feasibility_audit(self) -> dict[str, object]:
        """Combined fit/selection and post-selection deployment recovery audit."""

        return _merge_feasibility_audits(
            self.checkpoint_feasibility_audit,
            self.deployment_feasibility_audit,
        )

    def close(self) -> None:
        if self._owns_bank and self.bank is not None:
            self._closed_feasibility_audit = self.bank.feasibility_audit
            self.bank.close()
            self.bank = None


def _advance_state(system: ReservoirSystem, storage, action, inflow):
    raw = (
        np.asarray(storage, dtype=np.float64)
        - np.asarray(action, dtype=np.float64)
        + np.asarray(inflow, dtype=np.float64)
        + np.asarray(action, dtype=np.float64) @ np.asarray(system.routing)
    )
    return np.minimum(raw, np.asarray(system.storage_capacity, dtype=np.float64))


def fit_sddp(
    fit: TrajectoryBatch,
    validation: TrajectoryBatch,
    system: ReservoirSystem,
    n_regimes: int,
    config: SDDPConfig,
    train_seed: int,
    checkpoint_path: str | Path,
    provenance: dict | None = None,
) -> SDDPFitResult:
    rng = np.random.default_rng(train_seed)
    discretization = build_markov_discretization(
        fit, system, n_regimes, config.noise_scenarios
    )
    bank = _SDDPBank(
        system,
        discretization,
        complete_recourse_penalty=config.complete_recourse_penalty,
    )
    training_policy = HydrologyStateSDDPPolicy(
        system, discretization, bank=bank
    )
    best_validation = float("inf")
    best_iteration = 0
    best_cuts = bank.snapshot_cuts()
    history: list[dict[str, float | int | str]] = []
    validation_checkpoints = validation_schedule(config)
    validation_checkpoint_set = set(validation_checkpoints)
    fit_feasibility_audit = _summarize_feasibility_events([])
    start = time.perf_counter()
    try:
        for iteration in range(1, config.iterations + 1):
            solve_calls_before = bank.solve_calls
            optimization_attempts_before = bank.optimization_attempts
            feasibility_events_before = len(bank.feasibility_events)
            path = int(rng.integers(0, fit.n))
            storage = np.asarray(system.initial_storage, dtype=np.float64).copy()
            previous = np.zeros(system.n_reservoirs, dtype=np.float64)
            visited: list[tuple[int, np.ndarray, np.ndarray]] = []
            for stage in range(system.horizon):
                regime = int(fit.hydrology_regime[path, stage])
                visited.append((regime, storage.copy(), previous.copy()))
                solution = bank.solve(stage, regime, storage, previous)
                action = solution.action
                storage = _advance_state(system, storage, action, fit.inflow[path, stage])
                previous = action
            for stage in range(system.horizon - 1, -1, -1):
                regime, state_storage, state_previous = visited[stage]
                solution = bank.solve(stage, regime, state_storage, state_previous)
                state = np.concatenate((state_storage, state_previous))
                intercept = solution.objective - float(solution.state_dual @ state)
                bank.add_cut(stage, regime, (intercept, solution.state_dual.copy()))

            should_validate = iteration in validation_checkpoint_set
            validation_value = float("nan")
            is_best = 0
            if should_validate:
                metrics, _ = evaluate_policy(
                    training_policy,
                    validation,
                    system,
                    n_regimes,
                    device=config.device,
                )
                validation_value = metrics["cost_mean"]
                if validation_value < best_validation:
                    best_validation = validation_value
                    best_iteration = iteration
                    best_cuts = bank.snapshot_cuts()
                    is_best = 1
            history.append(
                {
                    "iteration": iteration,
                    "train_seed": train_seed,
                    "forward_path_index": path,
                    "number_of_cuts": bank.number_of_cuts,
                    "stage_lp_solves": (
                        bank.optimization_attempts - optimization_attempts_before
                    ),
                    "stage_solve_calls": bank.solve_calls - solve_calls_before,
                    "validation_checkpoint": int(should_validate),
                    "validation_cost": validation_value,
                    "is_best": is_best,
                    "feasibility_recoveries": (
                        len(bank.feasibility_events) - feasibility_events_before
                    ),
                    "complete_recourse_uses": sum(
                        event["resolution"] == "complete_recourse"
                        for event in bank.feasibility_events[feasibility_events_before:]
                    ),
                }
            )
    finally:
        fit_feasibility_audit = bank.feasibility_audit
        bank.close()
        training_policy.bank = None
    elapsed = time.perf_counter() - start

    selected_policy = HydrologyStateSDDPPolicy(
        system,
        discretization,
        cuts=best_cuts,
        complete_recourse_penalty=config.complete_recourse_penalty,
        checkpoint_feasibility_audit=fit_feasibility_audit,
    )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_name": selected_policy.policy_name,
        "train_seed": train_seed,
        "fit_seed": fit.seed,
        "validation_seed": validation.seed,
        "selected_iteration": best_iteration,
        "best_validation_cost": best_validation,
        "sddp_config": asdict(config),
        "system": system.as_jsonable(),
        "n_regimes": n_regimes,
        "discretization": discretization,
        "cuts": best_cuts,
        "effective_complete_recourse_penalty": (
            selected_policy.bank.complete_recourse_penalty
        ),
        "validation_schedule": validation_checkpoints,
        "feasibility_audit": fit_feasibility_audit,
        "provenance": {} if provenance is None else provenance,
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    return SDDPFitResult(
        policy=selected_policy,
        train_seed=train_seed,
        best_validation_cost=best_validation,
        selected_iteration=best_iteration,
        number_of_cuts=selected_policy.number_of_cuts,
        fit_time_seconds=elapsed,
        history=history,
        feasibility_audit=fit_feasibility_audit,
        checkpoint_path=str(checkpoint_path),
    )


def _system_from_payload(payload: dict) -> ReservoirSystem:
    values = dict(payload)
    values["objective"] = ObjectiveConfig(**values["objective"])
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


def load_sddp_checkpoint(path: str | Path) -> HydrologyStateSDDPPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    system = _system_from_payload(payload["system"])
    return HydrologyStateSDDPPolicy(
        system,
        payload["discretization"],
        cuts=payload["cuts"],
        complete_recourse_penalty=payload.get(
            "effective_complete_recourse_penalty",
            payload.get("sddp_config", {}).get("complete_recourse_penalty"),
        ),
        checkpoint_feasibility_audit=payload.get("feasibility_audit"),
    )
