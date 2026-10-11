"""Exact PWL inner optimization and a finite-support Algorithm 2 instantiation.

Continuous controls are retained. Integer breakpoints and the path's network
balance matrix imply that each affine cell has integral vertices, so minimizing
over integer storage/release pairs finds the continuous inner global optimum.
All DP objective comparisons use Python arbitrary-precision integer arithmetic.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import lcm, log, sqrt

import numpy as np

from .model import support


@dataclass
class Continuation:
    name: str
    values: list[list[Fraction]]

    def exact_at(self, stage, state):
        state = Fraction(state)
        if state < 0 or state > len(self.values[stage]) - 1:
            raise ValueError(f'Continuation query outside supported domain: {stage}, {state}')
        left = state.numerator // state.denominator
        if state == left:
            return self.values[stage][left]
        fraction = state - left
        return (1 - fraction) * self.values[stage][left] + fraction * self.values[stage][left + 1]

    def payload(self):
        return {'name': self.name, 'values': [[str(v) for v in row] for row in self.values]}


def bellman_values(hydro, probabilities, name='pwl'):
    """Exact empirical Bellman values, with exact linear interpolation between knots."""
    values = [[Fraction(0) for _ in range(hydro.state_upper(t) + 1)] for t in range(hydro.T + 1)]
    for t in reversed(range(hydro.T)):
        p = Fraction(probabilities[t])
        for state in range(hydro.state_upper(t) + 1):
            choices = []
            for action in range(min(state, hydro.release_capacity) + 1):
                expected = Fraction(0)
                for wet, probability in ((0, 1 - p), (1, p)):
                    nxt = state - action + hydro.wet_inflow * wet
                    expected += probability * (hydro.exact_cost(t, state, action, wet) + values[t + 1][nxt])
                choices.append(expected)
            values[t][state] = min(choices)
        slopes = np.diff(values[t])
        if any(a > b for a, b in zip(slopes, slopes[1:])):
            raise RuntimeError('Convex Bellman interpolation lost convexity.')
    return Continuation(name, values)


def generators(hydro, fit):
    probabilities = [Fraction(int(n), len(fit)) for n in fit.sum(axis=0)]
    pwl = bellman_values(hydro, probabilities)
    zero = Continuation('zero', [[Fraction(0)] * (hydro.state_upper(t) + 1) for t in range(hydro.T + 1)])
    affine = Continuation('affine', [[row[0] + Fraction(i, len(row) - 1) * (row[-1] - row[0])
                                    for i in range(len(row))] for row in pwl.values])
    return [zero, affine, pwl]


def exact_policy_trace(parameters, hydro, paths):
    """Treat saved float coefficients as exact rationals; simulate continuous actions."""
    coefficients = [[Fraction(float(x)) for x in row] for row in parameters]
    traces = []
    for path in paths:
        s = Fraction(hydro.initial_storage)
        states, actions, costs = [], [], []
        for t, wet in enumerate(path):
            x = min(max(coefficients[t][0] + coefficients[t][1] * s, 0), s, hydro.release_capacity)
            nxt = s - x + hydro.wet_inflow * int(wet)
            demand = hydro.wet_demand if wet else hydro.dry_demand
            cost = hydro.prices[t] * max(demand - x, 0) + nxt / hydro.holding_denominator
            if t == hydro.T - 1:
                cost += hydro.terminal_price * max(hydro.terminal_target - nxt, 0)
            states.append(s); actions.append(x); costs.append(cost)
            s = nxt
        traces.append(dict(states=states, actions=actions, costs=costs))
    return traces


def exact_response(hydro, continuation, t, state, action, wet):
    nxt = state - action + hydro.wet_inflow * wet
    demand = hydro.wet_demand if wet else hydro.dry_demand
    cost = hydro.prices[t] * max(demand - action, 0) + nxt / hydro.holding_denominator
    if t == hydro.T - 1:
        cost += hydro.terminal_price * max(hydro.terminal_target - nxt, 0)
    return cost + continuation.exact_at(t + 1, nxt)


def _floor_float(value):
    return float(np.nextafter(float(value), -np.inf))


def _ceil_float(value):
    return float(np.nextafter(float(value), np.inf))


class CertificateEngine:
    """Uses physical support, frozen policy/value generator, and calibration counts.

    The engine never receives the data-generating probabilities or the oracle.
    Known binary support is enumerated only to memoize the deterministic path
    solver; evaluation still uses independently sampled path frequencies.
    """

    def __init__(self, hydro, continuation, parameters):
        self.hydro, self.continuation = hydro, continuation
        self.paths = support(hydro)
        self.denominator = lcm(hydro.holding_denominator,
                               *[v.denominator for row in continuation.values for v in row])
        self.cost, self.contrast, self.valid, self.response_ranges = [], [], [], []
        for t in range(hydro.T):
            shape = (hydro.state_upper(t) + 1, hydro.release_capacity + 1)
            cost = np.zeros((*shape, 2), dtype=object)
            contrast = np.zeros(shape, dtype=object)
            valid = np.zeros(shape, dtype=bool)
            responses = []
            for s in range(shape[0]):
                for x in range(min(s, hydro.release_capacity) + 1):
                    valid[s, x] = True
                    h = []
                    for wet in (0, 1):
                        c = hydro.exact_cost(t, s, x, wet)
                        cost[s, x, wet] = int(c * self.denominator)
                        h.append(c + continuation.values[t + 1][s - x + hydro.wet_inflow * wet])
                    contrast[s, x] = int((h[1] - h[0]) * self.denominator)
                    responses.extend(h)
            self.cost.append(cost); self.contrast.append(contrast); self.valid.append(valid)
            self.response_ranges.append((min(responses), max(responses)))
        self.contrast_bounds = [Fraction(max(abs(v) for v in d[ok]), self.denominator)
                                for d, ok in zip(self.contrast, self.valid)]
        self.traces = exact_policy_trace(parameters, hydro, self.paths)
        bases, coefficients = [], []
        for path, trace in zip(self.paths, self.traces):
            base, coef = Fraction(0), []
            for t, wet in enumerate(path):
                s, x = trace['states'][t], trace['actions'][t]
                delta = exact_response(hydro, continuation, t, s, x, 1) - exact_response(hydro, continuation, t, s, x, 0)
                base += trace['costs'][t] - int(wet) * delta
                coef.append(delta)
            bases.append(base); coefficients.append(coef)
        self.policy_denominator = lcm(*[v.denominator for v in bases],
                                      *[v.denominator for row in coefficients for v in row])
        self.policy_base = np.asarray([int(v * self.policy_denominator) for v in bases], dtype=object)
        self.policy_coefficients = np.asarray([[int(v * self.policy_denominator) for v in row]
                                              for row in coefficients], dtype=object)

    def inner_exact(self, counts, n):
        """Return global optima of C + sum(Qhat-H), before the constant beta shift."""
        counts = [int(x) for x in counts]
        if n < 1 or len(counts) != self.hydro.T or any(x < 0 or x > n for x in counts):
            raise ValueError('Invalid calibration counts.')
        h = self.hydro
        next_values = np.zeros((len(self.paths), h.state_upper(h.T) + 1), dtype=object)
        decisions = [None] * h.T
        path_rows = np.arange(len(self.paths))
        for t in reversed(range(h.T)):
            wet = self.paths[:, t].astype(np.int64)
            current = np.empty((len(self.paths), h.state_upper(t) + 1), dtype=object)
            selected = np.empty(current.shape, dtype=np.int64)
            for s in range(current.shape[1]):
                candidates = []
                for x in range(min(s, h.release_capacity) + 1):
                    stage_value = n * self.cost[t][s, x, wet] + (counts[t] - n * wet).astype(object) * self.contrast[t][s, x]
                    candidates.append(stage_value + next_values[path_rows, s - x + h.wet_inflow * wet])
                candidates = np.asarray(candidates, dtype=object)
                best = np.argmin(candidates, axis=0)
                current[:, s] = candidates[best, path_rows]
                selected[:, s] = best
            next_values = current
            decisions[t] = selected
        numerators = next_values[:, h.initial_storage]
        # Independently re-evaluate the optimizing path for an exact arithmetic audit.
        states = np.full(len(self.paths), h.initial_storage, dtype=np.int64)
        recovered = np.zeros(len(self.paths), dtype=object)
        actions = np.empty(self.paths.shape, dtype=np.int64)
        for t in range(h.T):
            wet = self.paths[:, t].astype(np.int64)
            action = decisions[t][path_rows, states]
            actions[:, t] = action
            recovered += n * self.cost[t][states, action, wet] + (counts[t] - n * wet).astype(object) * self.contrast[t][states, action]
            states = states - action + h.wet_inflow * wet
        if any(a != b for a, b in zip(recovered, numerators)):
            raise RuntimeError('Exact inner objective reconstruction failed.')
        return numerators, actions

    def calibrate(self, counts, n, delta=0.025, rounding_cap=1e-8):
        h = self.hydro
        # Bernoulli Hoeffding band, unioned across stages, gives a uniform query band.
        rho = min(1.0, sqrt(log(2 * h.T / delta) / (2 * n)) + 1e-14)
        beta = np.asarray([_ceil_float(r) * rho + 1e-13 for r in self.contrast_bounds])
        numerator, actions = self.inner_exact(counts, n)
        inner = [Fraction(int(x), n * self.denominator) for x in numerator]
        policy_numerator = n * self.policy_base + self.policy_coefficients @ np.asarray(counts, dtype=object)
        policy = [Fraction(int(x), n * self.policy_denominator) for x in policy_numerator]
        exact_gaps = [a - b for a, b in zip(policy, inner)]
        if min(exact_gaps) < 0:
            raise RuntimeError('Continuous frozen-policy cost falls below exact inner optimum.')
        lower = np.asarray([_floor_float(x) for x in inner])
        upper = np.asarray([_ceil_float(x) for x in policy])
        inner_rounding = np.asarray([_ceil_float(x) for x in inner]) - lower
        policy_rounding = upper - np.asarray([_floor_float(x) for x in policy])
        if max(inner_rounding.max(), policy_rounding.max()) > rounding_cap:
            raise RuntimeError('The predeclared rational-to-float rounding cap was exceeded.')
        raw = upper - lower
        shift = float(2 * beta.sum())
        corrected = raw + shift
        # A support-wide envelope, frozen before evaluation: within-path ranges
        # of the stagewise PWL objective over the complete relaxed query domain.
        range_sum = Fraction(0)
        for t in range(h.T):
            ranges = []
            for wet in (0, 1):
                values = n * self.cost[t][..., wet] + (int(counts[t]) - n * wet) * self.contrast[t]
                valid = values[self.valid[t]]
                ranges.append(Fraction(int(max(valid) - min(valid)), n * self.denominator))
            range_sum += max(ranges)
        structural_envelope = _ceil_float(range_sum) + shift + 2 * rounding_cap
        # Enumerate the entire known 64-path support, not the evaluation sample.
        # This policy-specific supremum uses no probabilities and is frozen now.
        support_envelope = max(_ceil_float(g) for g in exact_gaps) + shift + 2 * rounding_cap
        envelope = min(structural_envelope, support_envelope)
        if corrected.min() < -rounding_cap or corrected.max() > envelope + rounding_cap:
            raise RuntimeError('The predeclared support-wide envelope failed.')
        return dict(beta=beta, rho=rho, centering=shift, envelope=envelope,
                    structural_envelope=structural_envelope,
                    raw=raw, corrected=corrected, inner_lower=lower,
                    policy_upper=upper, inner_actions=actions,
                    inner_conversion_gap=float(inner_rounding.max()),
                    policy_conversion_gap=float(policy_rounding.max()),
                    exact_inner_solver_gap=0.0)


def report_certificate(calibration, evaluation_counts, alpha=0.025):
    counts = np.asarray(evaluation_counts)
    n = int(counts.sum())
    if n < 1 or counts.shape != calibration['raw'].shape or np.any(counts < 0):
        raise ValueError('Invalid evaluation sample counts.')
    raw = float(counts @ calibration['raw'] / n)
    outer = calibration['envelope'] * sqrt(log(2 / alpha) / (2 * n))
    structural_outer = calibration['structural_envelope'] * sqrt(log(2 / alpha) / (2 * n))
    return dict(raw_gap=raw, calibration_correction=calibration['centering'],
                evaluation_correction=outer,
                certificate=raw + calibration['centering'] + outer,
                envelope=calibration['envelope'], n_evaluation=n,
                structural_envelope=calibration['structural_envelope'],
                structural_certificate=raw + calibration['centering'] + structural_outer,
                inner_conversion_gap=calibration['inner_conversion_gap'],
                exact_inner_solver_gap=0.0)
