from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_hydro.benchmarks.ons import (
    EXPECTED_TRAIN_MONTHS,
    MWH_PER_HM3_PER_MW_PER_M3S,
    aggregate_monthly_ena,
    aggregate_official_subsystem_monthly_ena,
    aggregate_ree_monthly_ena,
    build_cvu_merit_order,
    build_ons_benchmark,
)


def _fixture_tables(
    *, heldout_scale: float = 1.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    rows = []
    for unit_id, ree_name, value in (
        ("P-A", "REE_A", 10.0),
        ("P-B", "REE_B", 20.0),
        ("P-C", "REE_C", 30.0),
    ):
        values = np.full(len(dates), value, dtype=float)
        values[dates >= "2015-01-01"] *= heldout_scale
        unit = pd.DataFrame(
            {
                "dataset": "ons_reservoir_ena",
                "time": dates,
                "unit_id": unit_id,
                "ree_name": ree_name,
                "subsystem_id": {
                    "P-A": "SE",
                    "P-B": "S",
                    "P-C": "NE",
                }[unit_id],
                "ena_gross_mw_mean": values,
                # Deliberately false labels: the builder must use dates.
                "split": "train",
            }
        )
        if ree_name == "REE_C":
            unit = unit.loc[unit["time"].ne(pd.Timestamp("2001-01-01"))]
        rows.append(unit)
    ena = pd.concat(rows, ignore_index=True)

    static = pd.DataFrame(
        {
            "dataset": "ons_reservoir_static",
            "unit_id": ["R-A", "R-B", "R-C", "R-D"],
            "planning_id": ["P-A", "P-B", "P-C", "P-D"],
            "unit_name": ["A", "B", "C", "D"],
            "ree_name": ["REE_A", "REE_B", "REE_C", "REE_D"],
            "subsystem_id": ["SE", "S", "NE", "N"],
            "commissioned_on": pd.to_datetime(
                ["1990-01-01", "1991-01-01", "1992-01-01", "2020-01-01"]
            ),
            "storage_capacity_hm3": [100.0, 150.0, 1_000.0, 2_000.0],
            "productivity_65pct_mw_per_m3s": [1.0, 1.0, 2.0, 2.0],
        }
    )

    months = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    load_values = np.full(len(months), 100.0)
    load_values[months >= "2015-01-01"] *= heldout_scale
    load = pd.DataFrame(
        {
            "dataset": "ons_load",
            "time": months,
            "unit_id": "SE",
            "load_mw_mean": load_values,
            "split": "train",
        }
    )

    cvu = pd.DataFrame(
        {
            "dataset": "ons_cvu",
            "time": pd.to_datetime(
                [
                    "2010-01-04",
                    "2010-01-04",
                    "2010-01-04",
                    "2022-01-03",
                ]
            ),
            "time_end": pd.to_datetime(
                [
                    "2010-01-10",
                    "2010-01-10",
                    "2010-01-10",
                    "2022-01-09",
                ]
            ),
            "unit_id": ["T1", "T2", "T3", "T1"],
            "unit_name": ["Thermal One", "Thermal Two", "Thermal Three", "Thermal One"],
            "subsystem_id": ["SE", "SE", "S", "SE"],
            "cvu_brl_per_mwh": [10.0, 20.0, 30.0, 9_999.0 * heldout_scale],
            "split": "train",
        }
    )

    generation_capacity = pd.DataFrame(
        {
            "dataset": "ons_generation_capacity",
            "plant_name": [
                "Thermal One",
                "Thermal One",
                "Thermal Two",
                "Thermal Three",
                "Future Thermal",
                "Hydro Plant",
            ],
            "subsystem_id": ["SE", "SE", "SE", "S", "SE", "SE"],
            "plant_type": [
                "TÉRMICA",
                "TÉRMICA",
                "TÉRMICA",
                "TÉRMICA",
                "TÉRMICA",
                "HIDROELÉTRICA",
            ],
            "effective_capacity_mw": [25.0, 15.0, 30.0, 20.0, 1_000.0, 9_000.0],
            "operation_on": pd.to_datetime(
                [
                    "1990-01-01",
                    "1990-01-01",
                    "1991-01-01",
                    "1992-01-01",
                    "2020-01-01",
                    "1980-01-01",
                ]
            ),
            "capacity_reconstruction_as_of": pd.Timestamp("2014-12-31"),
        }
    )
    return static, ena, load, cvu, generation_capacity


def _fixture_subsystem_ena(*, heldout_scale: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    rows = []
    for unit_id, value in (("N", 40.0), ("NE", 30.0), ("S", 20.0), ("SE", 10.0)):
        values = np.full(len(dates), value, dtype=float)
        values[dates >= "2015-01-01"] *= heldout_scale
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "ons_subsystem_ena",
                    "time": dates,
                    "unit_id": unit_id,
                    "ena_gross_mw_mean": values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _build(*, heldout_scale: float = 1.0, max_rees: int = 2):
    static, ena, load, cvu, capacity = _fixture_tables(
        heldout_scale=heldout_scale
    )
    return build_ons_benchmark(
        static,
        ena,
        load,
        cvu,
        generation_capacity=capacity,
        aggregation_level="ree",
        max_aggregates=max_rees,
        horizon=48,
    )


def _build_subsystem(subsystem_ena_override: pd.DataFrame | None = None):
    static, ena, load, cvu, capacity = _fixture_tables()
    return build_ons_benchmark(
        static,
        ena,
        load,
        cvu,
        subsystem_ena=(
            _fixture_subsystem_ena()
            if subsystem_ena_override is None
            else subsystem_ena_override
        ),
        generation_capacity=capacity,
    )


def test_daily_ena_integrates_to_monthly_mwh_and_marks_missing_day() -> None:
    _, ena, _, _, _ = _fixture_tables()
    monthly = aggregate_ree_monthly_ena(ena)
    january = monthly.loc[
        monthly["ree_name"].eq("REE_A")
        & monthly["time"].eq(pd.Timestamp("2000-01-01"))
    ].iloc[0]
    assert january["inflow_mwh"] == pytest.approx(10.0 * 24.0 * 31.0)
    missing = monthly.loc[
        monthly["ree_name"].eq("REE_C")
        & monthly["time"].eq(pd.Timestamp("2001-01-01"))
    ].iloc[0]
    assert not bool(missing["is_complete_month"])
    assert np.isnan(missing["inflow_mwh"])


def test_subsystem_month_keeps_component_missingness_auditable() -> None:
    _, ena, _, _, _ = _fixture_tables()
    extra = ena.loc[ena["unit_id"].eq("P-A")].copy()
    extra["unit_id"] = "P-A-SECOND"
    extra["ena_gross_mw_mean"] = 5.0
    extra.loc[extra["time"].eq(pd.Timestamp("2000-01-15")), "ena_gross_mw_mean"] = (
        np.nan
    )
    monthly = aggregate_monthly_ena(pd.concat([ena, extra], ignore_index=True))
    january = monthly.loc[
        monthly["aggregation_id"].eq("SE")
        & monthly["time"].eq(pd.Timestamp("2000-01-01"))
    ].iloc[0]
    assert bool(january["is_complete_month"])
    assert not bool(january["all_values_present"])
    assert january["missing_component_rows"] == 1
    assert january["component_value_coverage"] == pytest.approx(61.0 / 62.0)
    assert january["inflow_mwh"] == pytest.approx((31 * 15.0 - 5.0) * 24.0)


def test_formal_default_aggregates_by_electrical_subsystem() -> None:
    benchmark = _build_subsystem()
    assert benchmark.selected_aggregates == ("SE", "S", "NE")
    assert benchmark.system.case_id == "ons_subsystem_3"
    assert dict(benchmark.system.metadata)["aggregation_level"] == "subsystem"
    assert set(benchmark.audit.monthly_coverage["aggregation_level"]) == {
        "subsystem"
    }
    selection = benchmark.audit.aggregation_selection.set_index("aggregation_id")
    assert selection.loc["SE", "aggregation_level"] == "subsystem"
    assert bool(selection.loc["NE", "eligible"])
    capacity = benchmark.audit.reservoir_capacity
    assert set(capacity["aggregation_level"]) == {"subsystem"}
    assert set(capacity["assignment_source"]) == {"static.subsystem_id"}

    monthly = aggregate_official_subsystem_monthly_ena(_fixture_subsystem_ena())
    january = monthly.loc[
        monthly["aggregation_id"].eq("SE")
        & monthly["time"].eq(pd.Timestamp("2000-01-01"))
    ].iloc[0]
    assert january["inflow_mwh"] == pytest.approx(10.0 * 24.0 * 31.0)
    assert set(benchmark.audit.monthly_coverage["inflow_source"]) == {
        "ons_official_subsystem_ena_daily"
    }
    component_audit = benchmark.audit.reservoir_component_coverage
    assert not component_audit["used_as_benchmark_inflow"].any()


def test_formal_subsystem_path_rejects_reservoir_sum_as_inflow() -> None:
    static, ena, load, cvu, capacity = _fixture_tables()
    with pytest.raises(ValueError, match="requires official subsystem_ena"):
        build_ons_benchmark(
            static, ena, load, cvu, generation_capacity=capacity
        )


def test_post_train_subsystem_or_ree_reassignment_cannot_change_calibration() -> None:
    static, ena, load, cvu, capacity = _fixture_tables()
    subsystem_ena = _fixture_subsystem_ena()
    ordinary = build_ons_benchmark(
        static,
        ena,
        load,
        cvu,
        subsystem_ena=subsystem_ena,
        generation_capacity=capacity,
    )
    reassigned = subsystem_ena.copy()
    heldout = reassigned["time"].ge(pd.Timestamp("2015-01-01"))
    old_ids = reassigned.loc[heldout, "unit_id"].copy()
    reassigned.loc[heldout & old_ids.eq("SE"), "unit_id"] = "N"
    reassigned.loc[heldout & old_ids.eq("N"), "unit_id"] = "SE"
    changed = build_ons_benchmark(
        static,
        ena,
        load,
        cvu,
        subsystem_ena=reassigned,
        generation_capacity=capacity,
    )
    assert ordinary.system == changed.system
    assert np.array_equal(ordinary.fit_inflow, changed.fit_inflow)
    assert np.array_equal(ordinary.fit_calendar, changed.fit_calendar)
    ordinary_heldout = ordinary.historical.loc[
        ordinary.historical["split"].ne("train"), "inflow_mwh"
    ]
    changed_heldout = changed.historical.loc[
        changed.historical["split"].ne("train"), "inflow_mwh"
    ]
    assert not ordinary_heldout.equals(changed_heldout)


def test_builder_selects_largest_complete_identifiable_rees_deterministically() -> None:
    benchmark = _build()
    assert benchmark.selected_rees == ("REE_B", "REE_A")
    assert benchmark.fit_inflow.shape == (EXPECTED_TRAIN_MONTHS, 2)
    assert benchmark.calendar.tolist() == list(np.arange(180) % 12)
    assert benchmark.fit_times[0] == pd.Timestamp("2000-01-01")
    assert benchmark.fit_times[-1] == pd.Timestamp("2014-12-01")
    assert benchmark.system.step_unit == "mwh_month"
    assert benchmark.system.horizon == 48
    assert len(benchmark.historical) == 26 * 12 * 2
    assert set(benchmark.historical["split"].dropna()) == {
        "train",
        "validation",
        "calibration",
        "historical_test",
    }

    ree_audit = benchmark.audit.ree_selection.set_index("aggregation_id")
    assert bool(ree_audit.loc["REE_A", "selected"])
    assert bool(ree_audit.loc["REE_B", "selected"])
    assert not bool(ree_audit.loc["REE_C", "eligible"])
    assert "incomplete_2000_2014_ena" in ree_audit.loc[
        "REE_C", "exclusion_reason"
    ]


def test_static_capacity_formula_and_energy_units_are_auditable() -> None:
    benchmark = _build()
    expected_a = 100.0 * 1.0 * MWH_PER_HM3_PER_MW_PER_M3S
    expected_b = 150.0 * 1.0 * MWH_PER_HM3_PER_MW_PER_M3S
    by_ree = dict(zip(benchmark.selected_rees, benchmark.system.storage_capacity))
    assert by_ree["REE_A"] == pytest.approx(expected_a)
    assert by_ree["REE_B"] == pytest.approx(expected_b)
    capacity_audit = benchmark.audit.reservoir_capacity.set_index("planning_id")
    assert capacity_audit.loc["P-A", "energy_capacity_mwh"] == pytest.approx(
        expected_a
    )
    assert capacity_audit.loc["P-A", "capacity_formula_factor"] == pytest.approx(
        277.77777777777777
    )


def test_thermal_blocks_use_rated_thermal_mw_not_plant_count_proxy() -> None:
    benchmark = _build()
    blocks = benchmark.audit.thermal_blocks
    assert set(blocks["capacity_source"]) == {"ons_effective_capacity_snapshot"}
    reference_hours = blocks["reference_month_hours"].iloc[0]
    assert sum(benchmark.system.objective.thermal_block_capacity) == pytest.approx(
        (40.0 + 30.0 + 20.0) * reference_hours
    )
    assert benchmark.system.objective.thermal_block_cost == (10.0, 20.0, 30.0)
    assert not benchmark.audit.thermal_plants["capacity_match_method"].eq(
        "explicit_training_load_proxy"
    ).any()


def test_training_crosswalk_uses_terminal_catalog_and_sums_ceg_units() -> None:
    cvu = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2010-01-04", "2010-01-04", "2014-12-20"]
            ),
            "unit_id": ["OLD", "T1", "T1"],
            "unit_name": ["Retired", "Legacy Alias", "Legacy Alias"],
            "subsystem_id": ["SE", "SE", "SE"],
            "cvu_brl_per_mwh": [5.0, 10.0, 30.0],
        }
    )
    capacity = pd.DataFrame(
        {
            "subsystem_id": ["SE", "SE"],
            "plant_type": ["TÉRMICA", "TÉRMICA"],
            "plant_name": ["Official Plant", "Official Plant"],
            "ceg": ["UTE.GN.RJ.012345-6.01", "UTE.GN.RJ.012345-6.02"],
            "effective_capacity_mw": [40.0, 60.0],
        }
    )
    crosswalk = pd.DataFrame(
        {
            "planning_id": ["T1"],
            "ceg": ["UTE.GN.RJ.012345-6.01"],
            "plant_name": ["Official Plant"],
            "subsystem_id": ["SE"],
            "first_observed_on": pd.to_datetime(["2013-01-01"]),
            "last_observed_on": pd.to_datetime(["2014-12-20"]),
            "source_years": ["2013,2014"],
        }
    )
    capacities, costs, plants, blocks = build_cvu_merit_order(
        cvu,
        generation_capacity=capacity,
        thermal_crosswalk=crosswalk,
        training_peak_load_mwh=1_000.0,
        mean_training_month_hours=720.0,
    )
    assert capacities == (100.0 * 720.0,)
    assert costs == (20.0,)
    by_id = plants.set_index("unit_id")
    assert not bool(by_id.loc["OLD", "population_eligible"])
    assert by_id.loc["T1", "capacity_match_method"] == (
        "planning_id_to_ceg_registration"
    )
    assert by_id.loc["T1", "rated_capacity_mw"] == pytest.approx(100.0)
    assert blocks.loc[0, "historical_training_cvu_plants"] == 2
    assert blocks.loc[0, "terminal_catalog_cvu_plants"] == 1
    assert blocks.loc[0, "matched_terminal_cvu_plants"] == 1


def test_unique_alias_and_high_confidence_fuzzy_matches_are_audited() -> None:
    names = ["P.MEDICI B", "MAUA B5A", "ST.CRUZ 34", "TRÊS LAGOA"]
    cvu = pd.DataFrame(
        {
            "time": pd.Timestamp("2014-12-20"),
            "unit_id": ["1", "2", "3", "4"],
            "unit_name": names,
            "subsystem_id": ["S", "N", "SE", "SE"],
            "cvu_brl_per_mwh": [10.0, 20.0, 30.0, 40.0],
        }
    )
    capacity = pd.DataFrame(
        {
            "subsystem_id": ["S", "N", "SE", "SE"],
            "plant_type": ["TÉRMICA"] * 4,
            "plant_name": [
                "Presidente Médici",
                "Mauá Bloco 4",
                "Santa Cruz",
                "Três Lagoas",
            ],
            "ceg": [
                "UTE.CM.RS.000001-1.01",
                "UTE.PE.AM.000002-2.01",
                "UTE.GN.RJ.000003-3.01",
                "UTE.GN.MS.000004-4.01",
            ],
            "effective_capacity_mw": [100.0, 200.0, 300.0, 400.0],
        }
    )
    _, _, plants, _ = build_cvu_merit_order(
        cvu,
        generation_capacity=capacity,
        training_peak_load_mwh=2_000.0,
        mean_training_month_hours=720.0,
    )
    methods = plants.set_index("unit_name")["capacity_match_method"]
    assert set(methods.loc[names[:3]]) == {"unique_alias_subsystem"}
    assert methods.loc["TRÊS LAGOA"] == (
        "unique_high_confidence_fuzzy_name_subsystem"
    )
    fuzzy = plants.set_index("unit_name").loc["TRÊS LAGOA"]
    assert fuzzy["fuzzy_match_score"] >= 0.92
    assert fuzzy["fuzzy_score_margin"] >= 0.10


def test_future_thermal_crosswalk_identity_is_rejected() -> None:
    _, _, _, cvu, capacity = _fixture_tables()
    crosswalk = pd.DataFrame(
        {
            "planning_id": ["T1"],
            "ceg": ["UTE.GN.RJ.012345-6.01"],
            "first_observed_on": pd.to_datetime(["2022-01-01"]),
            "last_observed_on": pd.to_datetime(["2022-12-31"]),
        }
    )
    with pytest.raises(ValueError, match="no identity observed"):
        build_cvu_merit_order(
            cvu,
            generation_capacity=capacity,
            thermal_crosswalk=crosswalk,
            training_peak_load_mwh=1_000.0,
            mean_training_month_hours=720.0,
        )


def test_missing_generation_capacity_requires_explicit_proxy_flag() -> None:
    static, ena, load, cvu, _ = _fixture_tables()
    with pytest.raises(ValueError, match="requires generation_capacity"):
        build_ons_benchmark(
            static,
            ena,
            load,
            cvu,
            aggregation_level="ree",
            max_aggregates=1,
        )
    sensitivity = build_ons_benchmark(
        static,
        ena,
        load,
        cvu,
        aggregation_level="ree",
        max_aggregates=1,
        thermal_capacity_proxy=True,
        max_thermal_blocks=2,
    )
    assert set(sensitivity.audit.thermal_blocks["capacity_source"]) == {
        "explicit_training_peak_proxy_by_plant_count"
    }


def test_formal_path_rejects_low_cvu_capacity_match_rate() -> None:
    static, ena, load, cvu, capacity = _fixture_tables()
    capacity = capacity.loc[capacity["plant_name"].ne("Thermal Three")]
    with pytest.raises(ValueError, match="match fraction"):
        build_ons_benchmark(
            static,
            ena,
            load,
            cvu,
            generation_capacity=capacity,
            aggregation_level="ree",
            max_aggregates=1,
        )
    allowed = build_ons_benchmark(
        static,
        ena,
        load,
        cvu,
        generation_capacity=capacity,
        aggregation_level="ree",
        max_aggregates=1,
        minimum_thermal_capacity_match_fraction=0.60,
    )
    assert allowed.audit.thermal_blocks["capacity_match_fraction"].iloc[0] == (
        pytest.approx(2.0 / 3.0)
    )


def test_generation_capacity_as_of_date_cannot_cross_training_boundary() -> None:
    static, ena, load, cvu, capacity = _fixture_tables()
    capacity["capacity_reconstruction_as_of"] = pd.Timestamp("2025-12-31")
    with pytest.raises(ValueError, match="as-of date"):
        build_ons_benchmark(
            static,
            ena,
            load,
            cvu,
            generation_capacity=capacity,
            aggregation_level="ree",
            max_aggregates=1,
        )


def test_heldout_extremes_cannot_change_system_or_fit_calibration() -> None:
    ordinary = _build(heldout_scale=1.0)
    extreme = _build(heldout_scale=1_000_000.0)
    assert ordinary.system == extreme.system
    assert np.array_equal(ordinary.fit_inflow, extreme.fit_inflow)
    assert np.array_equal(ordinary.calendar, extreme.calendar)
    # Historical evidence is retained, so the locked rows do reflect the change.
    ordinary_test = ordinary.historical.loc[
        ordinary.historical["split"].eq("historical_test"), "inflow_mwh"
    ].sum()
    extreme_test = extreme.historical.loc[
        extreme.historical["split"].eq("historical_test"), "inflow_mwh"
    ].sum()
    assert extreme_test > ordinary_test


def test_training_load_must_be_complete_even_if_labels_claim_train() -> None:
    static, ena, load, cvu, capacity = _fixture_tables()
    load = load.loc[load["time"].ne(pd.Timestamp("2002-05-01"))]
    with pytest.raises(ValueError, match="every month"):
        build_ons_benchmark(
            static,
            ena,
            load,
            cvu,
            generation_capacity=capacity,
            aggregation_level="ree",
            max_aggregates=1,
        )
