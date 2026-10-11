"""Known hydro physics, sample-only fitting, and an isolated evaluation oracle."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
import json

import numpy as np
import torch


@dataclass(frozen=True)
class Hydro:
    prices: tuple[int, ...] = (1, 4, 2, 6, 3, 8)
    initial_storage: int = 3
    release_capacity: int = 3
    wet_inflow: int = 2
    dry_demand: int = 3
    wet_demand: int = 2
    holding_denominator: int = 8
    terminal_target: int = 3
    terminal_price: int = 4

    @property
    def T(self):
        return len(self.prices)

    def state_upper(self, stage):
        return self.initial_storage + stage * self.wet_inflow

    def exact_cost(self, stage, storage, action, wet):
        nxt = storage - action + self.wet_inflow * wet
        demand = self.wet_demand if wet else self.dry_demand
        cost = Fraction(self.prices[stage] * max(demand - action, 0))
        cost += Fraction(nxt, self.holding_denominator)
        if stage == self.T - 1:
            cost += self.terminal_price * max(self.terminal_target - nxt, 0)
        return cost

    def cost(self, stage, storage, action, wet):
        nxt = storage - action + self.wet_inflow * wet
        demand = np.where(wet, self.wet_demand, self.dry_demand)
        result = self.prices[stage] * np.maximum(demand - action, 0)
        result += nxt / self.holding_denominator
        if stage == self.T - 1:
            result += self.terminal_price * np.maximum(self.terminal_target - nxt, 0)
        return result


def support(hydro):
    return np.asarray(list(product((0, 1), repeat=hydro.T)), dtype=np.uint8)


def draw_paths(probabilities, n, seed):
    """The experiment driver owns probabilities; certificate code gets paths."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, len(probabilities))) < np.asarray(probabilities)).astype(np.uint8)


def path_indices(paths):
    return paths @ (2 ** np.arange(paths.shape[1] - 1, -1, -1))


def rollout(parameters, paths, hydro):
    storage = np.full(len(paths), float(hydro.initial_storage))
    states, actions, costs = [], [], []
    for t in range(hydro.T):
        proposed = parameters[t, 0] + parameters[t, 1] * storage
        action = np.minimum(np.maximum(proposed, 0), np.minimum(storage, hydro.release_capacity))
        states.append(storage.copy())
        actions.append(action)
        costs.append(hydro.cost(t, storage, action, paths[:, t]))
        storage = storage - action + hydro.wet_inflow * paths[:, t]
    return dict(states=np.asarray(states).T, actions=np.asarray(actions).T,
                costs=np.asarray(costs).T, terminal_storage=storage)


def torch_cost(parameters, paths, hydro):
    storage = torch.full((len(paths),), float(hydro.initial_storage), dtype=torch.float64)
    cost = torch.zeros_like(storage)
    for t, price in enumerate(hydro.prices):
        action = torch.minimum(torch.relu(parameters[t, 0] + parameters[t, 1] * storage),
                               torch.clamp(storage, max=hydro.release_capacity))
        demand = hydro.dry_demand + (hydro.wet_demand - hydro.dry_demand) * paths[:, t]
        cost = cost + price * torch.relu(demand - action)
        storage = storage - action + hydro.wet_inflow * paths[:, t]
        cost = cost + storage / hydro.holding_denominator
    return cost + hydro.terminal_price * torch.relu(hydro.terminal_target - storage)


def fit_policy(fit, validation, hydro, out, updates=800):
    """Four declared Adam candidates; selection uses fitting/validation only."""
    torch.set_num_threads(1)
    fit_tensor = torch.as_tensor(fit, dtype=torch.float64)
    validation_tensor = torch.as_tensor(validation, dtype=torch.float64)
    history, best = [], None
    for seed in (73001, 73002):
        for lr in (0.01, 0.03):
            rng = np.random.default_rng(seed)
            initial = np.tile([0.3, 0.6], (hydro.T, 1)) + rng.normal(0, 0.03, (hydro.T, 2))
            parameters = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
            optimizer = torch.optim.Adam([parameters], lr=lr)
            for step in range(updates + 1):
                if step % 40 == 0 or step == updates:
                    with torch.no_grad():
                        score = float(torch_cost(parameters, validation_tensor, hydro).mean())
                    row = dict(seed=seed, learning_rate=lr, update=step, validation_cost=score)
                    history.append(row)
                    if best is None or score < best['validation_cost']:
                        best = {**row, 'parameters': parameters.detach().numpy().tolist()}
                if step == updates:
                    break
                optimizer.zero_grad()
                batch = rng.integers(0, len(fit), size=64)
                torch_cost(parameters, fit_tensor[batch], hydro).mean().backward()
                optimizer.step()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'policy.json').write_text(json.dumps(best, indent=2) + '\n')
    (out / 'training_history.json').write_text(json.dumps(history, indent=2) + '\n')
    return np.asarray(best['parameters'])


def scenario_tree_oracle(hydro, probabilities):
    """Independent continuous LP with one release per pre-shock history node."""
    import gurobipy as gp
    with gp.Env(params={'OutputFlag': 0}) as env, gp.Model(env=env) as model:
        model.Params.Threads = 1
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        states = {(): model.addVar(lb=hydro.initial_storage, ub=hydro.initial_storage)}
        objective = gp.LinExpr()
        for t in range(hydro.T):
            for history in product((0, 1), repeat=t):
                state = states[history]
                action = model.addVar(lb=0, ub=hydro.release_capacity)
                model.addConstr(action <= state)
                for wet in (0, 1):
                    child = history + (wet,)
                    probability = np.prod([float(probabilities[j]) if bit else 1-float(probabilities[j])
                                           for j, bit in enumerate(child)])
                    nxt = model.addVar(lb=0, ub=hydro.state_upper(t + 1))
                    model.addConstr(nxt == state - action + hydro.wet_inflow * wet)
                    states[child] = nxt
                    thermal = model.addVar(lb=0)
                    model.addConstr(thermal >= (hydro.wet_demand if wet else hydro.dry_demand) - action)
                    objective += probability * (hydro.prices[t] * thermal + nxt / hydro.holding_denominator)
                    if t == hydro.T - 1:
                        shortage = model.addVar(lb=0)
                        model.addConstr(shortage >= hydro.terminal_target - nxt)
                        objective += probability * hydro.terminal_price * shortage
        model.setObjective(objective)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'Oracle LP failed: {model.Status}')
        return dict(value=model.ObjVal, bound=model.ObjBound, constraints=model.NumConstrs,
                    variables=model.NumVars, status=int(model.Status))
