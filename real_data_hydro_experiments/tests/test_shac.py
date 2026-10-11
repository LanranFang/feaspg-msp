from __future__ import annotations

import numpy as np

from real_hydro.config import ObjectiveConfig, ReservoirSystem, repeated_rows
from real_hydro.dataset import TrajectoryBatch
from real_hydro.policies import ProjectedMLPPolicy
from real_hydro.shac import SHACConfig, fit_shac_candidate


def test_shac_smoke(tmp_path):
    objective = ObjectiveConfig(
        kind="multipurpose",
        release_target=repeated_rows((1.0,), 3),
        guide_lower=repeated_rows((1.0,), 3),
        guide_upper=repeated_rows((5.0,), 3),
        terminal_target=(3.0,),
        shortage_cost=(1.0,),
        excess_release_cost=(0.1,),
        low_storage_cost=(1.0,),
        flood_cost=(1.0,),
        spill_cost=(1.0,),
        ramp_cost=(0.1,),
        terminal_shortfall_cost=(2.0,),
        terminal_excess_cost=(0.0,),
    )
    system = ReservoirSystem(
        benchmark="fixture", case_id="shac", horizon=3, periods=3,
        step_unit="mcm", reservoir_ids=("r",), storage_capacity=(6.0,),
        release_capacity=(2.0,), initial_storage=(3.0,), routing=((0.0,),),
        objective=objective,
    )
    inflow = np.ones((8, 3, 1), dtype=np.float32)
    regime = np.zeros((8, 3), dtype=np.int16)
    fit = TrajectoryBatch("fit", 1, inflow, np.zeros_like(inflow), regime, np.arange(3))
    validation = TrajectoryBatch(
        "validation", 2, inflow[:3], np.zeros_like(inflow[:3]), regime[:3], np.arange(3)
    )
    result = fit_shac_candidate(
        ProjectedMLPPolicy(system, 1, width=8),
        fit,
        validation,
        system,
        1,
        SHACConfig(
            iterations=2,
            batch_size=2,
            validation_every=1,
            window=2,
            critic_epochs=1,
            critic_width=8,
            critic_depth=1,
            cost_scale=10.0,
        ),
        0,
        tmp_path / "shac.pt",
    )
    assert result.actor_gradient_updates == 2
    assert result.critic_gradient_updates == 2
    assert (tmp_path / "shac.pt").is_file()

