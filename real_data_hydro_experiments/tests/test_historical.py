from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from real_hydro.benchmarks.camels_br import HistoricalSplitArrays
from real_hydro.benchmarks.resopsus import HistoricalWindow
from real_hydro.historical import (
    build_camels_locked_replay,
    build_ons_locked_replay,
    build_resops_locked_replay,
)
from real_hydro.stochastic import fit_periodic_ar


def _fit_model(periods: int, width: int, years: int = 10):
    time = np.arange(periods * years, dtype=np.float64)
    columns = [
        20.0
        + 0.02 * time
        + (series + 1) * np.sin(2.0 * np.pi * time / periods)
        for series in range(width)
    ]
    values = np.column_stack(columns)
    calendar = (np.arange(len(values)) % periods).astype(np.int16)
    return fit_periodic_ar(
        values,
        calendar,
        periods,
        transform="level",
        max_order=1,
        annual_component=True,
        regime_count=2,
    )


def _ons_long() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    months = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    time = np.arange(len(months), dtype=np.float64)
    reservoir_ids = ("SE", "S")
    matrix = np.column_stack(
        (
            100.0 + 0.2 * time + 5.0 * np.sin(2.0 * np.pi * time / 12.0),
            70.0 + 0.1 * time + 3.0 * np.cos(2.0 * np.pi * time / 12.0),
        )
    )
    demand = 1_000.0 + 20.0 * np.sin(2.0 * np.pi * time / 12.0)
    rows = []
    for month_index, month in enumerate(months):
        for series, reservoir_id in enumerate(reservoir_ids):
            rows.append(
                {
                    "time": month,
                    "unit_id": reservoir_id,
                    "inflow_mwh": matrix[month_index, series],
                    "load_mwh": demand[month_index],
                    "split": (
                        "historical_test" if month.year >= 2022 else "not_locked"
                    ),
                }
            )
    return pd.DataFrame(rows), matrix, demand


def test_ons_locked_replay_is_one_48_month_path_with_real_demand() -> None:
    historical, matrix, demand = _ons_long()
    calendar = (np.arange(180) % 12).astype(np.int16)
    model = fit_periodic_ar(
        matrix[:180],
        calendar,
        12,
        transform="level",
        max_order=1,
        annual_component=True,
        regime_count=2,
    )
    replay = build_ons_locked_replay(
        historical, model, reservoir_ids=("SE", "S")
    )
    assert replay.name == "ons_locked_2022_2025"
    assert replay.inflow.shape == (1, 48, 2)
    assert np.array_equal(replay.inflow[0], matrix[-48:].astype(np.float32))
    assert np.array_equal(replay.demand[0], demand[-48:].astype(np.float32))
    assert np.array_equal(replay.calendar, np.tile(np.arange(12), 4))
    assert replay.source_path_id.tolist() == [264]

    changed = historical.copy()
    current = changed["time"].eq(pd.Timestamp("2022-01-01"))
    changed.loc[current, "inflow_mwh"] += 1_000.0
    perturbed = build_ons_locked_replay(
        changed, model, reservoir_ids=("SE", "S")
    )
    assert np.array_equal(
        replay.hydrology_state[:, 0], perturbed.hydrology_state[:, 0]
    )
    assert not np.array_equal(replay.inflow[:, 0], perturbed.inflow[:, 0])


def test_ons_replay_rejects_nonlocked_or_incomplete_history() -> None:
    historical, matrix, _ = _ons_long()
    model = fit_periodic_ar(
        matrix[:180],
        np.arange(180) % 12,
        12,
        transform="level",
        max_order=1,
        annual_component=True,
        regime_count=2,
    )
    wrong_split = historical.copy()
    wrong_split.loc[
        wrong_split["time"].eq(pd.Timestamp("2022-01-01")), "split"
    ] = "validation"
    with pytest.raises(ValueError, match="historical_test"):
        build_ons_locked_replay(wrong_split, model, reservoir_ids=("SE", "S"))
    incomplete = historical.loc[
        ~historical["time"].eq(pd.Timestamp("2005-03-01"))
    ]
    with pytest.raises(ValueError, match="complete monthly grid"):
        build_ons_locked_replay(incomplete, model, reservoir_ids=("SE", "S"))


def _resops_windows() -> tuple[HistoricalWindow, ...]:
    all_dates = pd.date_range("2021-10-03", periods=3 * 52, freq="7D")
    time = np.arange(len(all_dates), dtype=np.float64)
    inflow = np.column_stack((10.0 + 0.1 * time, 20.0 + 0.2 * time))
    windows = []
    for offset, water_year in enumerate((2022, 2023, 2024)):
        selection = slice(offset * 52, (offset + 1) * 52)
        values = inflow[selection]
        windows.append(
            HistoricalWindow(
                split="test",
                water_year=water_year,
                week_end=all_dates[selection].to_numpy(),
                calendar=np.arange(52, dtype=np.int16),
                inflow=values,
                observed_storage=values + 100.0,
                observed_release=values / 2.0,
                opening_storage=(100.0, 200.0),
            )
        )
    return tuple(windows)


def test_resops_first_test_year_is_prehistory_for_later_52_week_paths() -> None:
    windows = _resops_windows()
    model = _fit_model(52, 2)
    replay = build_resops_locked_replay(windows, model)
    assert replay.inflow.shape == (2, 52, 2)
    assert np.array_equal(replay.inflow[0], windows[1].inflow.astype(np.float32))
    assert np.array_equal(replay.inflow[1], windows[2].inflow.astype(np.float32))
    assert replay.source_path_id.tolist() == [52, 104]
    assert np.array_equal(replay.calendar, np.arange(52))
    assert replay.demand is None


def test_resops_replay_accepts_one_excluded_overflow_week_only() -> None:
    windows = list(_resops_windows())
    for index in range(2, len(windows)):
        windows[index] = replace(
            windows[index], week_end=windows[index].week_end + np.timedelta64(7, "D")
        )
    model = _fit_model(52, 2)
    replay = build_resops_locked_replay(tuple(windows), model)
    assert replay.inflow.shape == (2, 52, 2)

    invalid = list(windows)
    invalid[-1] = replace(
        invalid[-1], week_end=invalid[-1].week_end + np.timedelta64(7, "D")
    )
    with pytest.raises(ValueError, match="at most one audited overflow"):
        build_resops_locked_replay(tuple(invalid), model)


def test_resops_replay_rejects_training_or_nonchronological_windows() -> None:
    windows = _resops_windows()
    model = _fit_model(52, 2)
    with pytest.raises(ValueError, match="sorted consecutive"):
        build_resops_locked_replay(windows[::-1], model)
    validation = HistoricalWindow(
        split="validation",
        water_year=windows[0].water_year,
        week_end=windows[0].week_end,
        calendar=windows[0].calendar,
        inflow=windows[0].inflow,
        observed_storage=windows[0].observed_storage,
        observed_release=windows[0].observed_release,
        opening_storage=windows[0].opening_storage,
    )
    with pytest.raises(ValueError, match="only test"):
        build_resops_locked_replay((validation, windows[1]), model)


def _camels_split(*, missing_index: int | None = None) -> HistoricalSplitArrays:
    years = 7
    months = pd.date_range("2019-09-01", periods=years * 12, freq="MS")
    time = np.arange(len(months), dtype=np.float64)
    inflow = np.column_stack((30.0 + 0.1 * time, 50.0 + 0.2 * time))
    if missing_index is not None:
        inflow[missing_index, 0] = np.nan
    return HistoricalSplitArrays(
        name="locked_real_test",
        water_year_range=(2020, 2026),
        reservoir_ids=("00000001", "00000002"),
        months=months.to_numpy(dtype="datetime64[ns]"),
        water_year=np.repeat(np.arange(2020, 2027), 12),
        calendar=np.tile(np.arange(12), years),
        inflow=inflow,
    )


def test_camels_uses_first_year_as_prehistory_and_all_complete_stride12_paths() -> None:
    split = _camels_split(missing_index=65)
    model = _fit_model(12, 2)
    replay = build_camels_locked_replay(split, model)
    assert replay.inflow.shape == (1, 48, 2)
    assert np.array_equal(
        replay.inflow[0], split.inflow[12:60].astype(np.float32)
    )
    assert replay.source_path_id.tolist() == [12]
    assert np.array_equal(replay.calendar, np.tile(np.arange(12), 4))


def test_camels_replay_rejects_nonlocked_split_or_missing_first_year() -> None:
    model = _fit_model(12, 2)
    split = _camels_split()
    wrong_name = HistoricalSplitArrays(
        name="generator_validation",
        water_year_range=split.water_year_range,
        reservoir_ids=split.reservoir_ids,
        months=split.months,
        water_year=split.water_year,
        calendar=split.calendar,
        inflow=split.inflow,
    )
    with pytest.raises(ValueError, match="locked_real_test"):
        build_camels_locked_replay(wrong_name, model)
    missing_prehistory = _camels_split(missing_index=0)
    with pytest.raises(ValueError, match="complete prehistory"):
        build_camels_locked_replay(missing_prehistory, model)
