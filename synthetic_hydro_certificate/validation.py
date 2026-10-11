"""Independent continuous formulations used to audit the exact inner DP."""
from fractions import Fraction
import numpy as np


def continuous_inner_mip(engine, counts, n, path):
    """Represent the same continuous PWL objective with Gurobi, including nonconvex terms."""
    import gurobipy as gp
    h, v = engine.hydro, engine.continuation
    with gp.Env(params={'OutputFlag': 0}) as env, gp.Model(env=env) as model:
        model.Params.Threads = 1
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 1e-9
        model.Params.TimeLimit = 60
        previous = None
        for t, wet in enumerate(path):
            wet = int(wet)
            p = Fraction(int(counts[t]), n)
            action = model.addVar(lb=0, ub=h.release_capacity)
            post = model.addVar(lb=0, ub=h.state_upper(t))
            incoming = h.initial_storage if t == 0 else previous + h.wet_inflow * int(path[t - 1])
            model.addConstr(post + action == incoming)
            x_points = list(range(h.release_capacity + 1))
            x_values = [h.prices[t] * ((1-p)*max(h.dry_demand-x,0) + p*max(h.wet_demand-x,0)) for x in x_points]
            y_points = list(range(h.state_upper(t) + 1))
            y_values = []
            for y in y_points:
                expected = Fraction(0)
                for donor, probability in ((0, 1-p), (1, p)):
                    nxt = y + h.wet_inflow * donor
                    response = Fraction(nxt, h.holding_denominator) + v.values[t+1][nxt]
                    if t == h.T-1:
                        response += h.terminal_price * max(h.terminal_target-nxt,0)
                    expected += probability * response
                y_values.append(expected-v.values[t+1][y+h.wet_inflow*wet])
            model.setPWLObj(action, x_points, [float(x) for x in x_values])
            model.setPWLObj(post, y_points, [float(x) for x in y_values])
            previous = post
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'Continuous inner audit failed with status {model.Status}')
        return dict(primal=model.ObjVal, lower=model.ObjBound, nodes=model.NodeCount,
                    mixed_integer=bool(model.IsMIP))


def audit_engines(engines, counts, n, path_indices=tuple(range(64))):
    rows = []
    for engine in engines:
        exact, _ = engine.inner_exact(counts, n)
        for index in path_indices:
            result = continuous_inner_mip(engine, counts, n, engine.paths[index])
            expected = float(Fraction(int(exact[index]), n * engine.denominator))
            error = max(abs(result['primal']-expected), abs(result['lower']-expected))
            if error > 1e-7:
                raise RuntimeError((engine.continuation.name,index,result,expected))
            rows.append(dict(generator=engine.continuation.name, path_index=index,
                             exact_dp=expected, absolute_error=error, **result))
    return rows
