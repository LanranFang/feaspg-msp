from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from real_hydro.config import ObjectiveConfig, ReservoirSystem, repeated_rows
from real_hydro.aspinall_runtime import (
    project_release_coupled,
    project_release_coupled_np,
)
from real_hydro.dataset import TrajectoryBatch
from real_hydro.policies import ProjectedMLPPolicy, SeasonalReferencePolicy
from real_hydro.projection import project_release
from real_hydro.simulator import simulate, thermal_dispatch_cost
from real_hydro.stochastic import (
    _has_contiguous_prehistory,
    fit_periodic_ar,
    observed_trajectory_batch,
)


def _system(kind: str = "multipurpose") -> ReservoirSystem:
    periods = 4
    objective = ObjectiveConfig(
        kind=kind,
        release_target=repeated_rows((2.0, 1.0), periods),
        guide_lower=repeated_rows((2.0, 1.0), periods),
        guide_upper=repeated_rows((8.0, 5.0), periods),
        terminal_target=(5.0, 3.0),
        shortage_cost=(1.0, 1.0),
        excess_release_cost=(0.1, 0.1),
        low_storage_cost=(2.0, 2.0),
        flood_cost=(1.0, 1.0),
        spill_cost=(3.0, 3.0),
        ramp_cost=(0.1, 0.1),
        terminal_shortfall_cost=(4.0, 4.0),
        terminal_excess_cost=(0.0, 0.0),
        generation_coefficient=(1.0, 1.0) if kind == "hydrothermal" else (),
        seasonal_demand=(5.0,) * periods if kind == "hydrothermal" else (),
        thermal_block_capacity=(2.0, 2.0) if kind == "hydrothermal" else (),
        thermal_block_cost=(3.0, 6.0) if kind == "hydrothermal" else (),
        deficit_cost=100.0 if kind == "hydrothermal" else 0.0,
    )
    return ReservoirSystem(
        benchmark="fixture",
        case_id=kind,
        horizon=4,
        periods=periods,
        step_unit="mcm",
        reservoir_ids=("upstream", "downstream"),
        storage_capacity=(10.0, 6.0),
        release_capacity=(4.0, 3.0),
        initial_storage=(5.0, 3.0),
        routing=((0.0, 1.0), (0.0, 0.0)),
        objective=objective,
        shared_release_capacity=5.0,
    )


def _batch(first_inflow: float = 1.0) -> TrajectoryBatch:
    inflow = np.ones((3, 4, 2), dtype=np.float32)
    inflow[:, 0, 0] = first_inflow
    return TrajectoryBatch(
        name="fixture",
        seed=7,
        inflow=inflow,
        hydrology_state=np.zeros((3, 4, 2), dtype=np.float32),
        hydrology_regime=np.zeros((3, 4), dtype=np.int16),
        calendar=np.arange(4, dtype=np.int16),
    )


def test_projection_enforces_all_action_constraints():
    system = _system()
    proposal = torch.tensor([[9.0, 9.0], [-1.0, 9.0]])
    storage = torch.tensor([[3.0, 4.0], [3.0, 1.0]])
    action = project_release(proposal, storage, system)
    assert torch.all(action >= 0.0)
    assert torch.all(action <= storage + 1e-6)
    assert torch.all(action <= torch.tensor(system.release_capacity) + 1e-6)
    assert torch.all(action.sum(dim=1) <= 5.0 + 1e-5)


def test_coupled_projection_is_euclidean_and_differentiable():
    system = replace(
        _system(),
        benchmark="rise_uc_hdb_v1",
        case_id="aspinall_3",
        release_capacity=(20.0, 20.0),
        shared_release_capacity=None,
        metadata=(("availability_semantics", "same_stage_upstream_release"),),
    )
    # The downstream reservoir is empty, but the upstream reservoir can pass
    # water through during this stage.  Sequential clipping would return zero;
    # the Euclidean projection onto u_down <= u_up is (5, 5).
    proposal = torch.tensor([[0.0, 10.0]], requires_grad=True)
    storage = torch.tensor([[10.0, 0.0]])
    action = project_release_coupled(proposal, storage, system)
    assert torch.allclose(action, torch.tensor([[5.0, 5.0]]), atol=1.0e-6)
    np.testing.assert_allclose(
        project_release_coupled_np(
            proposal.detach().numpy(), storage.numpy(), system
        ),
        action.detach().numpy(),
        atol=1.0e-10,
    )
    action[0, 1].backward()
    assert torch.allclose(proposal.grad, torch.tensor([[0.5, 0.5]]), atol=1.0e-6)


def test_current_inflow_is_not_visible_before_action():
    system = _system()
    policy = SeasonalReferencePolicy(system, feature_dim=1)
    low = simulate(policy, _batch(0.0), system, n_regimes=1)
    high = simulate(policy, _batch(9.0), system, n_regimes=1)
    assert torch.equal(low.actions[:, 0], high.actions[:, 0])
    assert torch.all(high.storage[:, 0, 0] > low.storage[:, 0, 0])


def test_read_only_historical_arrays_convert_to_torch_without_shared_storage():
    batch = _batch()
    for values in (
        batch.inflow,
        batch.hydrology_state,
        batch.hydrology_regime,
        batch.calendar,
    ):
        values.setflags(write=False)
    tensors = batch.to_torch(n_regimes=1)
    assert torch.equal(tensors["inflow"], torch.as_tensor(batch.inflow.copy()))
    assert torch.equal(
        tensors["hydrology_regime"],
        torch.as_tensor(batch.hydrology_regime.copy(), dtype=torch.long),
    )


def test_routed_release_enters_downstream_once():
    system = _system()
    policy = SeasonalReferencePolicy(system, feature_dim=1)
    trace = simulate(policy, _batch(0.0), system, n_regimes=1)
    expected = (
        trace.storage_before_action[:, 0, 1]
        - trace.actions[:, 0, 1]
        + trace.actions[:, 0, 0]
        + 1.0
    )
    assert torch.allclose(trace.storage[:, 0, 1], torch.minimum(expected, torch.tensor(6.0)))


def test_thermal_merit_order_and_deficit():
    cost, generation, deficit = thermal_dispatch_cost(
        torch.tensor([1.0, 3.0, 6.0]),
        torch.tensor([2.0, 2.0]),
        torch.tensor([3.0, 6.0]),
        100.0,
    )
    assert torch.allclose(generation, torch.tensor([1.0, 3.0, 4.0]))
    assert torch.allclose(deficit, torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(cost, torch.tensor([3.0, 12.0, 218.0]))


def test_mlp_gradient_reaches_parameters():
    system = _system()
    policy = ProjectedMLPPolicy(system, feature_dim=1, width=8)
    trace = simulate(policy, _batch(), system, n_regimes=1)
    trace.mean_cost.backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0.0)
        for parameter in policy.parameters()
    )


def test_periodic_ar_joint_sampling_and_serialization(tmp_path):
    rng = np.random.default_rng(123)
    periods = 12
    n = periods * 30
    calendar = np.arange(n) % periods
    seasonal = 5.0 + 2.0 * np.sin(2.0 * np.pi * calendar / periods)
    common = rng.normal(size=n)
    values = np.stack(
        (
            np.maximum(0.0, seasonal + common),
            np.maximum(0.0, 1.5 * seasonal + 0.7 * common + rng.normal(scale=0.2, size=n)),
        ),
        axis=1,
    )
    model = fit_periodic_ar(
        values,
        calendar,
        periods,
        transform="log1p",
        max_order=2,
        regime_count=3,
    )
    paths = model.sample("test", 7, 24, seed=88, burn_in_years=1)
    assert paths.inflow.shape == (7, 24, 2)
    assert paths.hydrology_state.shape[-1] == model.state_dimension
    assert np.all(paths.inflow >= 0.0)
    model.save(tmp_path)
    restored = type(model).load(tmp_path)
    replay = restored.sample("test", 7, 24, seed=88, burn_in_years=1)
    assert np.array_equal(paths.inflow, replay.inflow)
    assert restored.training_fingerprint == model.training_fingerprint


def test_periodic_ar_breaks_lags_across_missing_calendar_rows():
    calendar = np.asarray([0, 1, 2, 3, 1, 2, 3, 0], dtype=np.int16)
    assert _has_contiguous_prehistory(calendar, 3, 3, periods=4)
    assert not _has_contiguous_prehistory(calendar, 4, 1, periods=4)
    assert not _has_contiguous_prehistory(calendar, 6, 3, periods=4)
    assert _has_contiguous_prehistory(calendar, 7, 3, periods=4)


def test_historical_encoding_uses_only_strictly_past_flow():
    periods = 4
    calendar = np.arange(80) % periods
    values = (5.0 + np.sin(np.arange(80)[:, None] / 3.0)).astype(np.float64)
    model = fit_periodic_ar(
        values[:60], calendar[:60], periods, transform="level", max_order=1,
        regime_count=2,
    )
    original = observed_trajectory_batch(
        model, "historical", values, calendar, np.asarray([60]), 4
    )
    changed = values.copy()
    changed[60] += 100.0
    perturbed = observed_trajectory_batch(
        model, "historical", changed, calendar, np.asarray([60]), 4
    )
    assert np.array_equal(original.hydrology_state[:, 0], perturbed.hydrology_state[:, 0])
    assert not np.array_equal(original.hydrology_state[:, 1], perturbed.hydrology_state[:, 1])
