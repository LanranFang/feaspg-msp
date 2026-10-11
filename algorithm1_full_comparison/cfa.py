"""Deterministic-lookahead and parametric CFA policies for nonlinear hydro.

The online optimizer uses an LP planned-state linearization of the nonlinear
hydro model. The fitted CFA changes only a scalar multiplier on the unconditional
inflow forecast; realized costs always come from the shared nonlinear simulator.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gurobipy as gp
import numpy as np
import torch
from torch import nn

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.physics import total_generation_linearization
from efficiency_switching_hydro.simulator import simulate
from efficiency_switching_hydro.train import unused_batches

from .config import CFAConfig


@dataclass(frozen=True)
class CFACandidateFit:
    learning_rate: float
    train_seed: int
    selected_iteration: int
    best_validation_cost: float
    best_theta: np.ndarray
    fit_time_seconds: float
    checkpoint_path: str
    history: list[dict[str, float | int | str]]


@dataclass(frozen=True)
class CFASelectedFit:
    policy: "HydroLookaheadPolicy"
    learning_rate: float
    selected_iteration: int
    best_validation_cost: float
    best_theta: np.ndarray
    selected_fit_time_seconds: float
    total_tuning_time_seconds: float
    checkpoint_path: str
    candidates: tuple[CFACandidateFit, ...]


class _LookaheadStage:
    """Reusable LP for the first action of one receding-horizon solve."""

    def __init__(
        self,
        env: gp.Env,
        hydro: ExperimentConfig,
        cfa: CFAConfig,
        stage: int,
    ) -> None:
        self.hydro = hydro
        self.stage = stage
        self.horizon = min(cfa.lookahead_horizon, hydro.T - stage)
        R = hydro.n_reservoirs
        D = hydro.routing_delay
        capacity = np.asarray(hydro.storage_capacity, dtype=np.float64)
        release_capacity = np.asarray(hydro.release_capacity, dtype=np.float64)
        generation_weights = np.asarray(
            hydro.generation_coefficient, dtype=np.float64
        )
        routing = np.asarray(hydro.routing, dtype=np.float64)
        queue_capacity = hydro.queue_capacity()
        planned_after = hydro.planned_storage()
        planned_action = hydro.reference_release()
        evaporation_rate = np.asarray(hydro.evaporation_rate, dtype=np.float64)

        self.mean_inflow = hydro.inflow_mean()[
            stage : stage + self.horizon
        ].copy()
        mean_demand = hydro.demand_mean()[stage : stage + self.horizon]
        price = hydro.price()[stage : stage + self.horizon]

        self.model = gp.Model(f"full_comparison_cfa_stage_{stage}", env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        self.model.Params.Method = 1
        self.model.Params.NumericFocus = 2

        self.storage = [
            [
                self.model.addVar(
                    lb=0.0,
                    ub=float(capacity[r]),
                    name=f"storage[{lag},{r}]",
                )
                for r in range(R)
            ]
            for lag in range(self.horizon + 1)
        ]
        self.release = [
            [
                self.model.addVar(
                    lb=0.0,
                    ub=float(release_capacity[r]),
                    name=f"release[{lag},{r}]",
                )
                for r in range(R)
            ]
            for lag in range(self.horizon)
        ]
        self.queue = [
            [
                [
                    self.model.addVar(
                        lb=0.0,
                        ub=float(queue_capacity[d, r]),
                        name=f"queue[{lag},{d},{r}]",
                    )
                    for r in range(R)
                ]
                for d in range(D)
            ]
            for lag in range(self.horizon + 1)
        ]
        self.initial_previous = [
            self.model.addVar(
                lb=0.0,
                ub=float(release_capacity[r]),
                name=f"initial_previous[{r}]",
            )
            for r in range(R)
        ]

        self.fix_storage = [
            self.model.addConstr(
                self.storage[0][r] == 0.0,
                name=f"fix_storage[{r}]",
            )
            for r in range(R)
        ]
        self.fix_previous = [
            self.model.addConstr(
                self.initial_previous[r] == 0.0,
                name=f"fix_previous[{r}]",
            )
            for r in range(R)
        ]
        self.fix_queue = [
            [
                self.model.addConstr(
                    self.queue[0][d][r] == 0.0,
                    name=f"fix_queue[{d},{r}]",
                )
                for r in range(R)
            ]
            for d in range(D)
        ]

        objective = gp.LinExpr()
        self.balance: list[list[gp.Constr]] = []
        self.base_balance_rhs: list[np.ndarray] = []
        ramp_limit = hydro.ramp_fraction * release_capacity
        for lag in range(self.horizon):
            absolute_stage = stage + lag
            planned_storage = (
                np.asarray(hydro.initial_storage, dtype=np.float64)
                if absolute_stage == 0
                else planned_after[absolute_stage - 1]
            )
            reference_action = planned_action[absolute_stage]
            planned_post = np.maximum(planned_storage - reference_action, 0.0)
            evaporation_slope = (
                1.0
                - 2.0 * evaporation_rate * planned_post / capacity
            )
            evaporation_constant = (
                evaporation_rate * np.square(planned_post) / capacity
            )
            generation, generation_s, generation_q = total_generation_linearization(
                planned_storage,
                reference_action,
                hydro,
            )
            generation_constant = float(
                generation
                - generation_s @ planned_storage
                - generation_q @ reference_action
            )

            previous = self.initial_previous if lag == 0 else self.release[lag - 1]
            for r in range(R):
                self.model.addConstr(
                    self.release[lag][r] <= self.storage[lag][r],
                    name=f"available_water[{lag},{r}]",
                )
                self.model.addConstr(
                    self.release[lag][r] - previous[r] <= float(ramp_limit[r]),
                    name=f"ramp_up[{lag},{r}]",
                )
                emergency = self.model.addVar(
                    lb=0.0,
                    name=f"emergency_ramp_down[{lag},{r}]",
                )
                self.model.addConstr(
                    previous[r] - self.release[lag][r]
                    <= float(ramp_limit[r]) + emergency,
                    name=f"ramp_down[{lag},{r}]",
                )
                objective += 100.0 * hydro.base_price * emergency
                if hydro.ramp_cost:
                    ramp_absolute = self.model.addVar(
                        lb=0.0,
                        name=f"ramp_absolute[{lag},{r}]",
                    )
                    self.model.addConstr(
                        ramp_absolute >= self.release[lag][r] - previous[r],
                        name=f"ramp_absolute_up[{lag},{r}]",
                    )
                    self.model.addConstr(
                        ramp_absolute >= previous[r] - self.release[lag][r],
                        name=f"ramp_absolute_down[{lag},{r}]",
                    )
                    objective += (
                        hydro.ramp_cost
                        / float(max(release_capacity[r], 1.0e-12))
                        * ramp_absolute
                    )

            self.model.addConstr(
                gp.quicksum(
                    generation_weights[r] * self.release[lag][r]
                    for r in range(R)
                )
                <= hydro.shared_release_capacity,
                name=f"shared_release[{lag}]",
            )

            routed = [
                gp.quicksum(
                    routing[source, destination]
                    * self.release[lag][source]
                    for source in range(R)
                )
                for destination in range(R)
            ]
            if D:
                arrival = self.queue[lag][0]
                for d in range(D - 1):
                    for r in range(R):
                        self.model.addConstr(
                            self.queue[lag + 1][d][r]
                            == self.queue[lag][d + 1][r],
                            name=f"queue_shift[{lag},{d},{r}]",
                        )
                for r in range(R):
                    self.model.addConstr(
                        self.queue[lag + 1][D - 1][r] == routed[r],
                        name=f"queue_route[{lag},{r}]",
                    )
            else:
                arrival = routed

            lag_balance: list[gp.Constr] = []
            for r in range(R):
                spill = self.model.addVar(
                    lb=0.0,
                    name=f"spill[{lag},{r}]",
                )
                lag_balance.append(
                    self.model.addConstr(
                        self.storage[lag + 1][r] + spill
                        - evaporation_slope[r]
                        * (self.storage[lag][r] - self.release[lag][r])
                        - arrival[r]
                        == float(evaporation_constant[r]),
                        name=f"balance[{lag},{r}]",
                    )
                )
                objective += hydro.spill_cost[r] * spill
                flood = self.model.addVar(lb=0.0, name=f"flood[{lag},{r}]")
                low = self.model.addVar(lb=0.0, name=f"low[{lag},{r}]")
                self.model.addConstr(
                    flood >= self.storage[lag + 1][r] - hydro.flood_threshold[r]
                )
                self.model.addConstr(
                    low >= hydro.low_storage_threshold[r] - self.storage[lag + 1][r]
                )
                objective += hydro.flood_cost * capacity[r] * flood
                objective += hydro.low_storage_cost * capacity[r] * low
            self.balance.append(lag_balance)
            self.base_balance_rhs.append(evaporation_constant.copy())

            thermal = self.model.addVar(lb=0.0, name=f"thermal[{lag}]")
            linear_generation = gp.LinExpr(generation_constant)
            linear_generation += gp.quicksum(
                generation_s[r] * self.storage[lag][r]
                + generation_q[r] * self.release[lag][r]
                for r in range(R)
            )
            self.model.addConstr(
                thermal + linear_generation >= float(mean_demand[lag]),
                name=f"meet_demand[{lag}]",
            )
            thermal_slope = (
                float(price[lag])
                + hydro.thermal_quadratic_cost * float(mean_demand[lag])
            )
            objective += thermal_slope * thermal
            peaker = self.model.addVar(lb=0.0, name=f"peaker[{lag}]")
            self.model.addConstr(
                peaker >= thermal - hydro.peaker_threshold,
                name=f"peaker_threshold[{lag}]",
            )
            peaker_scale = max(
                float(mean_demand[lag]) - hydro.peaker_threshold,
                0.10 * hydro.peaker_threshold,
            )
            objective += hydro.peaker_quadratic_cost * peaker_scale * peaker

        target = (
            np.asarray(hydro.terminal_target, dtype=np.float64)
            if stage + self.horizon == hydro.T
            else planned_after[stage + self.horizon - 1]
        )
        terminal_value = np.asarray(hydro.terminal_water_value, dtype=np.float64)
        for r in range(R):
            deficit = self.model.addVar(lb=0.0, name=f"terminal_deficit[{r}]")
            queued = (
                gp.quicksum(self.queue[self.horizon][d][r] for d in range(D))
                if D
                else 0.0
            )
            self.model.addConstr(
                deficit + self.storage[self.horizon][r] + queued
                >= float(target[r]),
                name=f"terminal_target[{r}]",
            )
            objective += terminal_value[r] * deficit

        objective += 1.0e-7 * gp.quicksum(
            (r + 1) * self.release[0][r] for r in range(R)
        )
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
        self.model.update()

    def solve(
        self,
        storage: np.ndarray,
        previous_release: np.ndarray,
        queue: np.ndarray,
        inflow_multiplier: float,
    ) -> np.ndarray:
        storage = np.clip(
            np.asarray(storage, dtype=np.float64),
            0.0,
            np.asarray(self.hydro.storage_capacity, dtype=np.float64),
        )
        previous_release = np.clip(
            np.asarray(previous_release, dtype=np.float64),
            0.0,
            np.asarray(self.hydro.release_capacity, dtype=np.float64),
        )
        queue = np.asarray(queue, dtype=np.float64)
        if self.hydro.routing_delay:
            queue = np.clip(queue, 0.0, self.hydro.queue_capacity())
        for r, constraint in enumerate(self.fix_storage):
            constraint.RHS = float(storage[r])
        for r, constraint in enumerate(self.fix_previous):
            constraint.RHS = float(previous_release[r])
        for d, row in enumerate(self.fix_queue):
            for r, constraint in enumerate(row):
                constraint.RHS = float(queue[d, r])
        for lag, row in enumerate(self.balance):
            for r, constraint in enumerate(row):
                constraint.RHS = float(
                    self.base_balance_rhs[lag][r]
                    + inflow_multiplier * self.mean_inflow[lag, r]
                )
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                "Hydro lookahead failed with Gurobi status "
                f"{self.model.Status} at stage {self.stage}."
            )
        return np.asarray(
            [variable.X for variable in self.release[0]],
            dtype=np.float64,
        )

    def close(self) -> None:
        self.model.dispose()


class _LookaheadBank:
    def __init__(self, hydro: ExperimentConfig, cfa: CFAConfig) -> None:
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.stages = [
            _LookaheadStage(self.env, hydro, cfa, stage)
            for stage in range(hydro.T)
        ]

    def solve(
        self,
        stage: int,
        storage: np.ndarray,
        previous_release: np.ndarray,
        queue: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        return self.stages[stage].solve(
            storage,
            previous_release,
            queue,
            float(theta[0]),
        )

    def close(self) -> None:
        for stage in self.stages:
            stage.close()
        self.env.dispose()


class HydroLookaheadPolicy(nn.Module):
    """Fixed deterministic lookahead or a frozen constant-multiplier CFA."""

    def __init__(
        self,
        hydro: ExperimentConfig,
        cfa: CFAConfig,
        mode: str,
        theta: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        normalized = mode.strip().lower().replace("-", "_")
        aliases = {
            "deterministic": "deterministic_lookahead",
            "d_la": "deterministic_lookahead",
            "constant": "constant_cfa",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"deterministic_lookahead", "constant_cfa"}:
            raise ValueError(f"Unknown lookahead mode: {mode}")
        self.hydro = hydro
        self.cfa = cfa
        self.policy_name = normalized
        self._theta = np.ones(1, dtype=np.float64)
        self.set_theta(np.ones(1) if theta is None else theta)
        self.bank = _LookaheadBank(hydro, cfa)

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def n_policy_parameters(self) -> int:
        return int(self.policy_name == "constant_cfa")

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (1,) or not np.isfinite(theta).all():
            raise ValueError("CFA theta must be one finite scalar.")
        if self.policy_name == "deterministic_lookahead":
            if not np.allclose(theta, 1.0):
                raise ValueError("Deterministic lookahead fixes theta at one.")
        elif np.any(theta < self.cfa.theta_lower) or np.any(
            theta > self.cfa.theta_upper
        ):
            raise ValueError("CFA theta is outside its registered bounds.")
        self._theta = theta.copy()

    def propose(self, stage, storage, previous_release, queue):
        storage_np = storage.detach().cpu().numpy().astype(np.float64)
        previous_np = previous_release.detach().cpu().numpy().astype(np.float64)
        queue_np = queue.detach().cpu().numpy().astype(np.float64)
        proposals = np.empty_like(storage_np)
        for path in range(storage_np.shape[0]):
            proposals[path] = self.bank.solve(
                stage,
                storage_np[path],
                previous_np[path],
                queue_np[path],
                self._theta,
            )
        return torch.as_tensor(
            proposals,
            dtype=storage.dtype,
            device=storage.device,
        )

    def close(self) -> None:
        if self.bank is not None:
            self.bank.close()
            self.bank = None


def make_lookahead_policy(
    name: str,
    hydro: ExperimentConfig,
    cfa: CFAConfig,
    theta: np.ndarray | None = None,
) -> HydroLookaheadPolicy:
    return HydroLookaheadPolicy(hydro, cfa, name, theta)


@torch.no_grad()
def _mean_cost(
    policy: HydroLookaheadPolicy,
    data: Dataset,
    hydro: ExperimentConfig,
    device: str,
) -> float:
    return float(simulate(policy, data, hydro, device=device).mean_cost.cpu())


def _checkpoint_payload(
    hydro: ExperimentConfig,
    cfa: CFAConfig,
    train_seed: int,
    learning_rate: float,
    selected_iteration: int,
    best_validation_cost: float,
    theta: np.ndarray,
) -> dict[str, object]:
    return {
        "policy_name": "constant_cfa",
        "train_seed": train_seed,
        "learning_rate": learning_rate,
        "selected_iteration": selected_iteration,
        "best_validation_cost": best_validation_cost,
        "theta": np.asarray(theta, dtype=np.float64),
        "hydro_config": asdict(hydro),
        "cfa_config": asdict(cfa),
    }


def fit_cfa_candidate(
    fit: Dataset,
    validation: Dataset,
    hydro: ExperimentConfig,
    cfa: CFAConfig,
    train_seed: int,
    learning_rate: float,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> CFACandidateFit:
    """Fit one scalar CFA candidate with a symmetric Gaussian estimator."""

    rng = np.random.default_rng(train_seed)
    batches = unused_batches(fit.n, cfa.iterations, cfa.batch_size, rng)
    policy = make_lookahead_policy("constant_cfa", hydro, cfa)
    theta = policy.theta
    rms = np.zeros_like(theta)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    best_validation = _mean_cost(policy, validation, hydro, device)
    objective_scale = max(abs(best_validation), 1.0)
    best_iteration = 0
    best_theta = theta.copy()
    history: list[dict[str, float | int | str]] = [
        {
            "policy": "constant_cfa",
            "train_seed": train_seed,
            "learning_rate": learning_rate,
            "iteration": 0,
            "fit_trajectories_seen": 0,
            "function_evaluated_paths": 0,
            "train_cost_plus": float("nan"),
            "train_cost_minus": float("nan"),
            "validation_cost": best_validation,
            "is_best": 1,
        }
    ]

    def save_best() -> None:
        torch.save(
            _checkpoint_payload(
                hydro,
                cfa,
                train_seed,
                learning_rate,
                best_iteration,
                best_validation,
                best_theta,
            ),
            checkpoint_path,
        )

    save_best()
    try:
        for iteration in range(1, cfa.iterations + 1):
            batch = fit.take(batches[iteration - 1], name="cfa_fit_batch")
            direction = rng.normal(size=theta.shape)
            plus = np.clip(
                theta + cfa.smoothing_radius * direction,
                cfa.theta_lower,
                cfa.theta_upper,
            )
            minus = np.clip(
                theta - cfa.smoothing_radius * direction,
                cfa.theta_lower,
                cfa.theta_upper,
            )
            policy.set_theta(plus)
            plus_cost = _mean_cost(policy, batch, hydro, device)
            policy.set_theta(minus)
            minus_cost = _mean_cost(policy, batch, hydro, device)
            gradient = (
                (plus_cost - minus_cost)
                / (2.0 * cfa.smoothing_radius)
                * direction
                / objective_scale
            )
            rms = cfa.rms_decay * rms + (1.0 - cfa.rms_decay) * np.square(
                gradient
            )
            theta = np.clip(
                theta
                - learning_rate * gradient / (np.sqrt(rms) + 1.0e-8),
                cfa.theta_lower,
                cfa.theta_upper,
            )
            policy.set_theta(theta)
            should_validate = (
                iteration % cfa.validation_every == 0
                or iteration == cfa.iterations
            )
            validation_cost = float("nan")
            is_best = 0
            if should_validate:
                validation_cost = _mean_cost(policy, validation, hydro, device)
                if validation_cost < best_validation:
                    best_validation = validation_cost
                    best_iteration = iteration
                    best_theta = theta.copy()
                    save_best()
                    is_best = 1
            history.append(
                {
                    "policy": "constant_cfa",
                    "train_seed": train_seed,
                    "learning_rate": learning_rate,
                    "iteration": iteration,
                    "fit_trajectories_seen": iteration * cfa.batch_size,
                    "function_evaluated_paths": 2
                    * iteration
                    * cfa.batch_size,
                    "train_cost_plus": plus_cost,
                    "train_cost_minus": minus_cost,
                    "validation_cost": validation_cost,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                    "theta": float(theta[0]),
                    "is_best": is_best,
                }
            )
    finally:
        policy.close()
    elapsed = time.perf_counter() - start
    save_best()
    return CFACandidateFit(
        learning_rate=learning_rate,
        train_seed=train_seed,
        selected_iteration=best_iteration,
        best_validation_cost=best_validation,
        best_theta=best_theta,
        fit_time_seconds=elapsed,
        checkpoint_path=str(checkpoint_path),
        history=history,
    )


def tune_and_select_cfa(
    fit: Dataset,
    validation: Dataset,
    hydro: ExperimentConfig,
    cfa: CFAConfig,
    train_seed: int,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> CFASelectedFit:
    candidates = []
    checkpoint_dir = Path(checkpoint_dir)
    for index, learning_rate in enumerate(cfa.learning_rates):
        tag = f"{learning_rate:.8g}".replace(".", "p")
        candidates.append(
            fit_cfa_candidate(
                fit,
                validation,
                hydro,
                cfa,
                train_seed,
                learning_rate,
                checkpoint_dir
                / f"constant_cfa_candidate{index}_lr{tag}_seed{train_seed}.pt",
                device,
            )
        )
    selected = min(candidates, key=lambda item: item.best_validation_cost)
    selected_path = checkpoint_dir / f"constant_cfa_selected_seed{train_seed}.pt"
    payload = torch.load(selected.checkpoint_path, weights_only=False)
    payload["candidate_learning_rates"] = list(cfa.learning_rates)
    payload["total_tuning_time_seconds"] = sum(
        candidate.fit_time_seconds for candidate in candidates
    )
    torch.save(payload, selected_path)
    policy = make_lookahead_policy(
        "constant_cfa",
        hydro,
        cfa,
        selected.best_theta,
    )
    return CFASelectedFit(
        policy=policy,
        learning_rate=selected.learning_rate,
        selected_iteration=selected.selected_iteration,
        best_validation_cost=selected.best_validation_cost,
        best_theta=copy.deepcopy(selected.best_theta),
        selected_fit_time_seconds=selected.fit_time_seconds,
        total_tuning_time_seconds=sum(
            candidate.fit_time_seconds for candidate in candidates
        ),
        checkpoint_path=str(selected_path),
        candidates=tuple(candidates),
    )
