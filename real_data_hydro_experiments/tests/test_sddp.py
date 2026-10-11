from __future__ import annotations

import warnings

import gurobipy as gp
import numpy as np
import pytest
import torch

from real_hydro.config import ObjectiveConfig, ReservoirSystem, repeated_rows
from real_hydro.dataset import TrajectoryBatch
from real_hydro.evaluation import evaluate_policy
from real_hydro.sddp import (
    SDDPConfig,
    SDDPSolveError,
    MarkovDiscretization,
    ScenarioSet,
    _DecisionHazardStage,
    _SDDPBank,
    build_markov_discretization,
    fit_sddp,
    validation_schedule,
)


def _fixture():
    periods = 3
    objective = ObjectiveConfig(
        kind="multipurpose",
        release_target=repeated_rows((1.5,), periods),
        guide_lower=repeated_rows((1.0,), periods),
        guide_upper=repeated_rows((5.0,), periods),
        terminal_target=(3.0,),
        shortage_cost=(2.0,),
        excess_release_cost=(0.2,),
        low_storage_cost=(1.0,),
        flood_cost=(1.0,),
        spill_cost=(2.0,),
        ramp_cost=(0.1,),
        terminal_shortfall_cost=(5.0,),
        terminal_excess_cost=(0.0,),
    )
    system = ReservoirSystem(
        benchmark="fixture",
        case_id="sddp",
        horizon=3,
        periods=periods,
        step_unit="mcm",
        reservoir_ids=("r",),
        storage_capacity=(6.0,),
        release_capacity=(3.0,),
        initial_storage=(3.0,),
        routing=((0.0,),),
        objective=objective,
    )
    rng = np.random.default_rng(10)
    inflow = rng.uniform(0.5, 2.0, size=(12, 3, 1)).astype(np.float32)
    regime = np.asarray([[path % 2, (path + 1) % 2, path % 2] for path in range(12)], dtype=np.int16)
    batch = TrajectoryBatch(
        name="fit",
        seed=101,
        inflow=inflow,
        hydrology_state=np.zeros((12, 3, 1), dtype=np.float32),
        hydrology_regime=regime,
        calendar=np.arange(3, dtype=np.int16),
    )
    validation = TrajectoryBatch(
        name="validation",
        seed=202,
        inflow=inflow[:4],
        hydrology_state=np.zeros((4, 3, 1), dtype=np.float32),
        hydrology_regime=regime[:4],
        calendar=np.arange(3, dtype=np.int16),
    )
    return system, batch, validation


def test_discretization_keeps_next_hydrology_state():
    system, fit, _ = _fixture()
    support = build_markov_discretization(fit, system, 2, 2)
    assert support.n_regimes == 2
    assert set(support.scenarios[0][0].next_regime) == {1}
    assert set(support.scenarios[0][1].next_regime) == {0}


def test_discretization_collapses_duplicate_empty_clusters_without_nan():
    system, fit, _ = _fixture()
    identical = TrajectoryBatch(
        name="identical_fit",
        seed=fit.seed,
        inflow=np.ones_like(fit.inflow),
        hydrology_state=np.zeros_like(fit.hydrology_state),
        hydrology_regime=np.zeros_like(fit.hydrology_regime),
        calendar=fit.calendar,
    )
    support = build_markov_discretization(
        identical,
        system,
        n_regimes=1,
        noise_scenarios=5,
    )
    for stage in support.scenarios:
        scenario = stage[0]
        assert scenario.inflow.shape == (1, 1)
        assert scenario.demand.shape == (1,)
        assert scenario.probability == pytest.approx(np.ones(1))
        assert np.isfinite(scenario.inflow).all()
        assert np.isfinite(scenario.demand).all()


def test_validation_schedule_is_capped_and_always_final():
    anchor = SDDPConfig(
        iterations=80,
        validation_every=1,
        max_validation_checks=4,
    )
    smoke = SDDPConfig(
        iterations=2,
        validation_every=1,
        max_validation_checks=3,
    )
    sparse = SDDPConfig(
        iterations=11,
        validation_every=4,
        max_validation_checks=3,
    )
    assert validation_schedule(anchor) == (20, 40, 60, 80)
    assert validation_schedule(smoke) == (1, 2)
    assert validation_schedule(sparse) == (4, 8, 11)
    assert validation_schedule(sparse)[-1] == sparse.iterations
    assert len(validation_schedule(sparse)) <= sparse.max_validation_checks


def _stage_fixture(inflow: float = 1.0):
    system, _, _ = _fixture()
    scenarios = ScenarioSet(
        inflow=np.asarray([[inflow]], dtype=np.float64),
        demand=np.zeros(1, dtype=np.float64),
        next_regime=np.zeros(1, dtype=np.int16),
        probability=np.ones(1, dtype=np.float64),
    )
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    stage = _DecisionHazardStage(env, system, 0, scenarios, [[]])
    return stage, env


def test_stage_uses_audited_numerical_retry(monkeypatch):
    stage, env = _stage_fixture()
    original_optimize = stage._optimize
    calls = 0

    def false_infeasible_then_solve():
        nonlocal calls
        calls += 1
        if calls == 1:
            return gp.GRB.INFEASIBLE
        return original_optimize()

    monkeypatch.setattr(stage, "_optimize", false_infeasible_then_solve)
    try:
        solution = stage.solve(np.asarray([3.0]), np.asarray([0.0]))
        assert solution.optimization_attempts == 2
        assert solution.feasibility_event is not None
        assert solution.feasibility_event["resolution"] == "numerical_retry"
        assert solution.feasibility_event["status_sequence"] == [
            gp.GRB.INFEASIBLE,
            gp.GRB.OPTIMAL,
        ]
        assert solution.feasibility_event["artificial_water_total"] == 0.0
        assert all(variable.UB == 0.0 for variable in stage.artificial_water[0])
    finally:
        stage.dispose()
        env.dispose()


def test_stage_complete_recourse_is_penalized_audited_and_reset():
    system, _, _ = _fixture()
    scenario = ScenarioSet(
        inflow=np.asarray([[-1.0]], dtype=np.float64),
        demand=np.zeros(1, dtype=np.float64),
        next_regime=np.zeros(1, dtype=np.int16),
        probability=np.ones(1, dtype=np.float64),
    )
    discretization = MarkovDiscretization(
        n_regimes=1,
        scenarios=tuple((scenario,) for _ in range(system.horizon)),
        fallback_bins=tuple((False,) for _ in range(system.horizon)),
    )
    bank = _SDDPBank(system, discretization)
    stage = bank.models[0][0]
    try:
        with pytest.warns(RuntimeWarning, match="policy.feasibility_audit"):
            solution = bank.solve(0, 0, np.asarray([0.0]), np.asarray([0.0]))
        assert solution.optimization_attempts == 3
        assert solution.feasibility_event is not None
        assert solution.feasibility_event["resolution"] == "complete_recourse"
        assert solution.feasibility_event["status_sequence"] == [
            gp.GRB.INFEASIBLE,
            gp.GRB.INFEASIBLE,
            gp.GRB.OPTIMAL,
        ]
        assert solution.feasibility_event["artificial_water_total"] == pytest.approx(1.0)
        assert solution.objective >= solution.feasibility_event[
            "complete_recourse_penalty"
        ]
        assert all(variable.UB == 0.0 for variable in stage.artificial_water[0])
        assert all(variable.Obj == 0.0 for variable in stage.artificial_water[0])
        assert stage.complete_recourse_penalty < 1.0e6

        with warnings.catch_warnings(record=True) as repeated_warnings:
            warnings.simplefilter("always")
            bank.solve(0, 0, np.asarray([0.0]), np.asarray([0.0]))
        assert repeated_warnings == []
        assert bank.feasibility_audit["complete_recourse_count"] == 2

        nominal = bank.solve(0, 0, np.asarray([2.0]), np.asarray([0.0]))
        assert nominal.feasibility_event is None
        assert nominal.optimization_attempts == 1
    finally:
        bank.close()


def test_stage_does_not_silently_relax_unbounded_status(monkeypatch):
    stage, env = _stage_fixture()
    statuses = iter((gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED))
    monkeypatch.setattr(stage, "_optimize", lambda: next(statuses))
    try:
        with pytest.raises(SDDPSolveError) as captured:
            stage.solve(np.asarray([3.0]), np.asarray([0.0]))
        assert captured.value.statuses == (gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED)
        assert captured.value.status_names == ("INFEASIBLE", "UNBOUNDED")
        assert "complete recourse was not used" in str(captured.value)
        assert all(variable.UB == 0.0 for variable in stage.artificial_water[0])
    finally:
        stage.dispose()
        env.dispose()


def test_sddp_smoke_has_cuts_and_feasible_actions(tmp_path):
    system, fit, validation = _fixture()
    result = fit_sddp(
        fit,
        validation,
        system,
        n_regimes=2,
        config=SDDPConfig(iterations=2, validation_every=1, noise_scenarios=2),
        train_seed=0,
        checkpoint_path=tmp_path / "sddp.pt",
    )
    try:
        metrics, costs = evaluate_policy(result.policy, validation, system, 2)
        # Validation-only model selection may legitimately retain the first
        # iteration (one cut per stage) instead of the final six-cut bank.
        assert result.number_of_cuts >= system.horizon
        assert len(costs) == validation.n
        assert metrics["max_projected_action_violation"] <= 1.0e-6
        assert (tmp_path / "sddp.pt").is_file()
        assert [
            row["iteration"]
            for row in result.history
            if row["validation_checkpoint"]
        ] == [1, 2]
        assert result.feasibility_audit["recovered_solve_count"] == 0
        payload = torch.load(tmp_path / "sddp.pt", weights_only=False)
        assert payload["validation_schedule"] == (1, 2)
        assert payload["feasibility_audit"] == result.feasibility_audit
        assert payload["effective_complete_recourse_penalty"] > 0.0
    finally:
        result.policy.close()
