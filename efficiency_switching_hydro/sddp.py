"""Empirical SDDP and a planned-state efficiency-switching linearization."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import gurobipy as gp
import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig, SDDPConfig
from .data import Dataset
from .physics import total_generation_linearization, transition_np
from .projection import project_action_np


Cut = tuple[float, np.ndarray]


def policy_name(config: ExperimentConfig) -> str:
    return (
        "empirical_sddp"
        if config.profile == "linear"
        else "linearized_empirical_sddp"
    )


def _kmeans(values: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    effective = min(n_clusters, len(values))
    centers = [values[np.square(values - values.mean(axis=0)).sum(axis=1).argmin()]]
    for _ in range(1, effective):
        distance = np.min(
            np.stack(
                [np.square(values - center).sum(axis=1) for center in centers],
                axis=1,
            ),
            axis=1,
        )
        centers.append(values[int(distance.argmax())])
    centers_array = np.stack(centers)
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(30):
        distances = np.square(
            values[:, None, :] - centers_array[None, :, :]
        ).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centers = centers_array.copy()
        for cluster in range(effective):
            members = values[new_labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(
            centers_array, new_centers
        ):
            break
        labels = new_labels
        centers_array = new_centers
    probabilities = np.bincount(labels, minlength=effective).astype(np.float64)
    probabilities /= probabilities.sum()
    if effective == n_clusters:
        return centers_array, probabilities
    padded = np.repeat(centers_array[-1:], n_clusters, axis=0)
    padded[:effective] = centers_array
    padded_probabilities = np.zeros(n_clusters, dtype=np.float64)
    padded_probabilities[:effective] = probabilities
    return padded, padded_probabilities


def discretize(
    fit: Dataset, config: ExperimentConfig, sddp: SDDPConfig
) -> tuple[np.ndarray, np.ndarray]:
    joint = np.concatenate((fit.inflow, fit.demand[..., None]), axis=-1)
    scenarios = np.empty(
        (config.T, sddp.noise_scenarios, config.n_reservoirs + 1),
        dtype=np.float64,
    )
    probabilities = np.empty(
        (config.T, sddp.noise_scenarios), dtype=np.float64
    )
    for stage in range(config.T):
        scenarios[stage], probabilities[stage] = _kmeans(
            joint[:, stage], sddp.noise_scenarios
        )
    return scenarios, probabilities


def pack_state(storage, previous_release, queue) -> np.ndarray:
    return np.concatenate((storage, previous_release, queue.reshape(-1)))


def unpack_state(state: np.ndarray, config: ExperimentConfig):
    R = config.n_reservoirs
    storage = state[:R]
    previous = state[R : 2 * R]
    queue = state[2 * R :].reshape(config.routing_delay, R)
    return storage, previous, queue


@dataclass(frozen=True)
class StageSolution:
    objective: float
    action: np.ndarray
    state_dual: np.ndarray


@dataclass(frozen=True)
class FitResult:
    policy: "SDDPPolicy"
    selected_iteration: int
    best_validation_cost: float
    number_of_cuts: int
    checkpoint_path: str
    history: list[dict[str, float | int | str]]


class StageModel:
    def __init__(
        self,
        env: gp.Env,
        config: ExperimentConfig,
        stage: int,
        outcomes: np.ndarray,
        probabilities: np.ndarray,
        future_cuts: list[Cut],
    ) -> None:
        self.config = config
        self.stage = stage
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        R = config.n_reservoirs
        D = config.routing_delay
        capacity = np.asarray(config.storage_capacity, dtype=np.float64)
        release = np.asarray(config.release_capacity, dtype=np.float64)
        eta = np.asarray(config.generation_coefficient, dtype=np.float64)
        routing = np.asarray(config.routing, dtype=np.float64)
        queue_capacity = config.queue_capacity()
        planned_after = config.planned_storage()
        planned_storage = (
            np.asarray(config.initial_storage, dtype=np.float64)
            if stage == 0
            else planned_after[stage - 1]
        )
        planned_action = config.reference_release()[stage]
        planned_post = np.maximum(planned_storage - planned_action, 0.0)
        evaporation_rate = np.asarray(config.evaporation_rate, dtype=np.float64)
        evaporation_slope = 1.0 - 2.0 * evaporation_rate * planned_post / capacity
        evaporation_constant = evaporation_rate * np.square(planned_post) / capacity

        planned_generation, generation_s, generation_q = (
            total_generation_linearization(
                planned_storage, planned_action, config
            )
        )
        generation_constant = (
            planned_generation
            - generation_q @ planned_action
            - generation_s @ planned_storage
        )

        self.model = gp.Model(f"efficiency_switching_sddp_{stage}", env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.NumericFocus = 2
        self.storage = [
            self.model.addVar(lb=0.0, ub=float(capacity[r]), name=f"storage[{r}]")
            for r in range(R)
        ]
        self.previous_release = [
            self.model.addVar(lb=0.0, ub=float(release[r]), name=f"previous[{r}]")
            for r in range(R)
        ]
        self.queue = [
            [
                self.model.addVar(
                    lb=0.0,
                    ub=float(queue_capacity[d, r]),
                    name=f"queue[{d},{r}]",
                )
                for r in range(R)
            ]
            for d in range(D)
        ]
        incoming = self.storage + self.previous_release + [
            variable for row in self.queue for variable in row
        ]
        self.fix_state = [
            self.model.addConstr(variable == 0.0, name=f"fix_state[{i}]")
            for i, variable in enumerate(incoming)
        ]
        self.action = [
            self.model.addVar(lb=0.0, ub=float(release[r]), name=f"release[{r}]")
            for r in range(R)
        ]
        self.emergency_ramp_down = []
        ramp = config.ramp_fraction * release
        for r in range(R):
            self.model.addConstr(self.action[r] <= self.storage[r])
            self.model.addConstr(
                self.action[r] - self.previous_release[r] <= float(ramp[r])
            )
            emergency = self.model.addVar(
                lb=0.0, name=f"emergency_ramp_down[{r}]"
            )
            self.emergency_ramp_down.append(emergency)
            self.model.addConstr(
                self.previous_release[r] - self.action[r]
                <= float(ramp[r]) + emergency
            )
        self.model.addConstr(
            gp.quicksum(eta[r] * self.action[r] for r in range(R))
            <= config.shared_release_capacity,
            name="shared_release",
        )

        routed = [
            gp.quicksum(routing[source, r] * self.action[source] for source in range(R))
            for r in range(R)
        ]
        if D == 0:
            arrival = routed
            next_queue: list[list[gp.LinExpr | gp.Var]] = []
        else:
            arrival = self.queue[0]
            next_queue = [self.queue[d + 1] for d in range(D - 1)] + [routed]

        objective = gp.QuadExpr()
        objective += 100.0 * config.base_price * gp.quicksum(
            self.emergency_ramp_down
        )
        if config.ramp_cost:
            objective += gp.quicksum(
                config.ramp_cost
                * (self.action[r] - self.previous_release[r])
                * (self.action[r] - self.previous_release[r])
                / (release[r] * release[r])
                for r in range(R)
            )
        self.outgoing_storage: list[list[gp.Var]] = []
        self.continuation: list[gp.Var] = []
        for scenario, probability in enumerate(self.probabilities):
            outgoing = [
                self.model.addVar(
                    lb=0.0,
                    ub=float(capacity[r]),
                    name=f"outgoing[{scenario},{r}]",
                )
                for r in range(R)
            ]
            self.outgoing_storage.append(outgoing)
            scenario_cost = gp.QuadExpr()
            for r in range(R):
                spill = self.model.addVar(lb=0.0, name=f"spill[{scenario},{r}]")
                self.model.addConstr(
                    outgoing[r] + spill
                    == evaporation_slope[r] * (self.storage[r] - self.action[r])
                    + evaporation_constant[r]
                    + float(max(outcomes[scenario, r], 0.0))
                    + arrival[r]
                )
                scenario_cost += config.spill_cost[r] * spill
                if config.flood_cost:
                    flood = self.model.addVar(lb=0.0, name=f"flood[{scenario},{r}]")
                    self.model.addConstr(
                        flood >= outgoing[r] - config.flood_threshold[r]
                    )
                    scenario_cost += config.flood_cost * flood * flood
                if config.low_storage_cost:
                    low = self.model.addVar(lb=0.0, name=f"low[{scenario},{r}]")
                    self.model.addConstr(
                        low >= config.low_storage_threshold[r] - outgoing[r]
                    )
                    scenario_cost += config.low_storage_cost * low * low
            thermal = self.model.addVar(lb=0.0, name=f"thermal[{scenario}]")
            linear_generation = gp.LinExpr(float(generation_constant))
            linear_generation += gp.quicksum(
                generation_q[r] * self.action[r]
                + generation_s[r] * self.storage[r]
                for r in range(R)
            )
            self.model.addConstr(
                thermal + linear_generation
                >= float(max(outcomes[scenario, R], 0.0))
            )
            scenario_cost += config.price()[stage] * thermal
            if config.thermal_quadratic_cost:
                scenario_cost += config.thermal_quadratic_cost * thermal * thermal
            if config.peaker_quadratic_cost:
                peaker = self.model.addVar(
                    lb=0.0, name=f"peaker[{scenario}]"
                )
                self.model.addConstr(
                    peaker >= thermal - config.peaker_threshold
                )
                scenario_cost += config.peaker_quadratic_cost * peaker * peaker
            if stage < config.T - 1:
                continuation = self.model.addVar(
                    lb=0.0, name=f"continuation[{scenario}]"
                )
                self.continuation.append(continuation)
                scenario_cost += continuation
            else:
                for r in range(R):
                    queued_water = gp.quicksum(
                        next_queue[d][r] for d in range(D)
                    ) if D else 0.0
                    shortfall = self.model.addVar(
                        lb=0.0, name=f"terminal_shortfall[{scenario},{r}]"
                    )
                    self.model.addConstr(
                        shortfall + outgoing[r] + queued_water
                        >= config.terminal_target[r]
                    )
                    scenario_cost += config.terminal_water_value[r] * shortfall
            objective += float(probability) * scenario_cost
        objective += 1e-7 * gp.quicksum(
            (r + 1) * self.action[r] for r in range(R)
        )
        self.next_queue = next_queue
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
        for cut in future_cuts:
            self.add_future_cut(cut)
        self.model.update()

    def add_future_cut(self, cut: Cut) -> None:
        if self.stage >= self.config.T - 1:
            return
        R = self.config.n_reservoirs
        D = self.config.routing_delay
        intercept, slope = cut
        storage_slope = slope[:R]
        previous_slope = slope[R : 2 * R]
        queue_slope = slope[2 * R :].reshape(D, R)
        for scenario, continuation in enumerate(self.continuation):
            expression = gp.LinExpr(float(intercept))
            expression += gp.quicksum(
                storage_slope[r] * self.outgoing_storage[scenario][r]
                + previous_slope[r] * self.action[r]
                for r in range(R)
            )
            if D:
                expression += gp.quicksum(
                    queue_slope[d, r] * self.next_queue[d][r]
                    for d in range(D)
                    for r in range(R)
                )
            self.model.addConstr(continuation >= expression)

    def solve(self, state: np.ndarray) -> StageSolution:
        state = np.asarray(state, dtype=np.float64)
        storage, previous, queue = unpack_state(state, self.config)
        storage = np.clip(storage, 0.0, self.config.storage_capacity)
        previous = np.clip(previous, 0.0, self.config.release_capacity)
        if self.config.routing_delay:
            queue = np.clip(queue, 0.0, self.config.queue_capacity())
        fixed = pack_state(storage, previous, queue)
        for constraint, value in zip(self.fix_state, fixed):
            constraint.RHS = float(value)
        self.model.reset()
        self.model.optimize()
        has_usable_solution = (
            self.model.Status == gp.GRB.OPTIMAL
            or (
                self.model.Status == gp.GRB.SUBOPTIMAL
                and self.model.SolCount > 0
            )
        )
        if not has_usable_solution:
            raise RuntimeError(
                f"SDDP stage {self.stage} failed with status {self.model.Status}."
            )
        if self.model.Status == gp.GRB.OPTIMAL:
            state_dual = np.asarray(
                [constraint.Pi for constraint in self.fix_state],
                dtype=np.float64,
            )
        else:
            state_dual = np.full(len(self.fix_state), np.nan, dtype=np.float64)
        return StageSolution(
            objective=float(self.model.ObjVal),
            action=np.asarray([variable.X for variable in self.action]),
            state_dual=state_dual,
        )

    def dispose(self) -> None:
        self.model.dispose()


class ModelBank:
    def __init__(
        self,
        config: ExperimentConfig,
        scenarios: np.ndarray,
        probabilities: np.ndarray,
        cuts: list[list[Cut]] | None = None,
    ) -> None:
        self.config = config
        self.scenarios = scenarios
        self.probabilities = probabilities
        self.cuts = [list() for _ in range(config.T)] if cuts is None else copy.deepcopy(cuts)
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.models = [
            StageModel(
                self.env,
                config,
                stage,
                scenarios[stage],
                probabilities[stage],
                self.cuts[stage + 1] if stage < config.T - 1 else [],
            )
            for stage in range(config.T)
        ]

    def solve(self, stage: int, state: np.ndarray) -> StageSolution:
        return self.models[stage].solve(state)

    def add_cut(self, stage: int, cut: Cut) -> None:
        self.cuts[stage].append(cut)
        if stage > 0:
            self.models[stage - 1].add_future_cut(cut)

    def snapshot(self) -> list[list[Cut]]:
        return copy.deepcopy(self.cuts)

    def number_of_cuts(self) -> int:
        return sum(len(cuts) for cuts in self.cuts)

    def close(self) -> None:
        for model in self.models:
            model.dispose()
        self.env.dispose()


class SDDPPolicy(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        scenarios: np.ndarray,
        probabilities: np.ndarray,
        cuts: list[list[Cut]] | None = None,
        bank: ModelBank | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.policy_name = policy_name(config)
        self.scenarios = scenarios
        self.probabilities = probabilities
        self.bank = bank or ModelBank(config, scenarios, probabilities, cuts)
        self._owns_bank = bank is None

    def propose(self, stage, storage, previous_release, queue):
        storage_np = storage.detach().cpu().numpy()
        previous_np = previous_release.detach().cpu().numpy()
        queue_np = queue.detach().cpu().numpy()
        actions = np.empty_like(storage_np)
        for index in range(len(storage_np)):
            state = pack_state(storage_np[index], previous_np[index], queue_np[index])
            actions[index] = self.bank.solve(stage, state).action
        return torch.as_tensor(actions, dtype=storage.dtype, device=storage.device)

    @property
    def number_of_cuts(self) -> int:
        return self.bank.number_of_cuts()

    def close(self) -> None:
        if self._owns_bank and self.bank is not None:
            self.bank.close()
            self.bank = None


@torch.no_grad()
def _mean_cost(policy, data: Dataset, config: ExperimentConfig) -> float:
    from .simulator import simulate

    return float(simulate(policy, data, config).mean_cost)


def fit_checkpoint(
    fit: Dataset,
    validation: Dataset,
    config: ExperimentConfig,
    sddp: SDDPConfig,
    train_seed: int,
    checkpoint_path: str | Path,
) -> FitResult:
    rng = np.random.default_rng(train_seed)
    scenarios, probabilities = discretize(fit, config, sddp)
    bank = ModelBank(config, scenarios, probabilities)
    current_policy = SDDPPolicy(
        config, scenarios, probabilities, bank=bank
    )
    best_validation = _mean_cost(current_policy, validation, config)
    best_iteration = 0
    best_cuts = bank.snapshot()
    history: list[dict[str, float | int | str]] = [
        {
            "policy": policy_name(config),
            "train_seed": train_seed,
            "iteration": 0,
            "train_cost": float("nan"),
            "validation_cost": best_validation,
            "lower_bound": 0.0,
            "is_best": 1,
        }
    ]
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def save_best() -> None:
        torch.save(
            {
                "policy_name": policy_name(config),
                "profile": config.profile,
                "linearization": (
                    "exact" if config.profile == "linear" else "planned_state_first_order"
                ),
                "train_seed": train_seed,
                "selected_iteration": best_iteration,
                "best_validation_cost": best_validation,
                "experiment_config": asdict(config),
                "sddp_config": asdict(sddp),
                "scenarios": scenarios,
                "probabilities": probabilities,
                "cuts": best_cuts,
            },
            checkpoint_path,
        )

    save_best()
    initial_storage = np.asarray(config.initial_storage, dtype=np.float64)
    initial_previous = np.zeros(config.n_reservoirs, dtype=np.float64)
    initial_queue = np.zeros(
        (config.routing_delay, config.n_reservoirs), dtype=np.float64
    )
    terminal_target = np.asarray(config.terminal_target, dtype=np.float64)
    terminal_value = np.asarray(config.terminal_water_value, dtype=np.float64)

    for iteration in range(1, sddp.iterations + 1):
        states = []
        storage = initial_storage.copy()
        previous = initial_previous.copy()
        queue = initial_queue.copy()
        realized_cost = 0.0
        lower_bound = 0.0
        for stage in range(config.T):
            state = pack_state(storage, previous, queue)
            states.append(state)
            solution = bank.solve(stage, state)
            if stage == 0:
                lower_bound = solution.objective
            action = project_action_np(
                solution.action[None, :],
                storage[None, :],
                previous[None, :],
                config,
            )[0]
            scenario = int(
                rng.choice(sddp.noise_scenarios, p=probabilities[stage])
            )
            outcome = scenarios[stage, scenario]
            storage, previous, queue, components = transition_np(
                storage,
                previous,
                queue,
                action,
                np.maximum(outcome[: config.n_reservoirs], 0.0),
                max(float(outcome[config.n_reservoirs]), 0.0),
                config.price()[stage],
                config,
            )
            realized_cost += sum(
                float(components[key])
                for key in ("thermal", "spill", "ramp", "storage_penalty")
            )
        terminal_water = storage + (queue.sum(axis=0) if config.routing_delay else 0.0)
        realized_cost += float(
            terminal_value @ np.maximum(terminal_target - terminal_water, 0.0)
        )

        for stage in range(config.T - 1, -1, -1):
            solution = bank.solve(stage, states[stage])
            if not np.isfinite(solution.state_dual).all():
                continue
            cut = (
                float(solution.objective - solution.state_dual @ states[stage]),
                solution.state_dual.copy(),
            )
            bank.add_cut(stage, cut)

        validate = (
            iteration % sddp.validation_every == 0
            or iteration == sddp.iterations
        )
        validation_cost = float("nan")
        is_best = 0
        if validate:
            validation_cost = _mean_cost(current_policy, validation, config)
            if validation_cost < best_validation:
                best_validation = validation_cost
                best_iteration = iteration
                best_cuts = bank.snapshot()
                save_best()
                is_best = 1
        history.append(
            {
                "policy": policy_name(config),
                "train_seed": train_seed,
                "iteration": iteration,
                "train_cost": realized_cost,
                "validation_cost": validation_cost,
                "lower_bound": lower_bound,
                "is_best": is_best,
            }
        )

    bank.close()
    save_best()
    selected = SDDPPolicy(config, scenarios, probabilities, cuts=best_cuts)
    return FitResult(
        policy=selected,
        selected_iteration=best_iteration,
        best_validation_cost=best_validation,
        number_of_cuts=selected.number_of_cuts,
        checkpoint_path=str(checkpoint_path),
        history=history,
    )
