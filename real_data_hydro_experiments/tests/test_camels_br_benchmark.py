"""Offline contract tests for the CAMELS-BR benchmark builder."""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd

from real_hydro.benchmarks.camels_br import (
    CAMELS_BR_HORIZON,
    CAMELS_BR_PERIODS,
    build_camels_br_benchmark,
)
from real_hydro.data.camels_br import (
    DataValidationError,
    calibrate_semisynthetic_reservoirs,
)


def _monthly_fixture() -> pd.DataFrame:
    months = pd.date_range("1980-09-01", "2024-08-01", freq="MS")
    calendar = ((months.month - 9) % 12).to_numpy()
    water_year = np.where(months.month >= 9, months.year + 1, months.year)
    rows: list[dict[str, object]] = []
    for identifier, scale in (("1", 10.0), ("2", 25.0)):
        # Seasonal and interannual variation make the train-only empirical guide
        # bands non-degenerate while keeping every value strictly positive.
        seasonal = 1.0 + 0.30 * np.sin(2.0 * np.pi * calendar / 12.0)
        annual = 1.0 + 0.02 * ((water_year - 1981) % 7)
        values = scale * seasonal * annual
        for month, wy, period, value in zip(
            months, water_year, calendar, values, strict=True
        ):
            rows.append(
                {
                    "gauge_id": identifier,
                    "month": month,
                    "water_year": int(wy),
                    "water_month": int(period) + 1,
                    "inflow_hm3": float(value),
                }
            )
    return pd.DataFrame(rows)


def _calibration(monthly: pd.DataFrame) -> pd.DataFrame:
    return calibrate_semisynthetic_reservoirs(
        monthly,
        fit_water_years=(1981, 2005),
        storage_months=6.0,
        release_mean_multiplier=1.5,
        initial_storage_fraction=0.64,
        terminal_target_fraction=0.62,
        min_fit_months=300,
    )


class CamelsBRBenchmarkTests(TestCase):
    def setUp(self) -> None:
        self.monthly = _monthly_fixture()
        self.calibration = _calibration(self.monthly)

    def build(self, monthly: pd.DataFrame | None = None):
        return build_camels_br_benchmark(
            self.monthly if monthly is None else monthly,
            self.calibration,
            basin_ids=("1", "2"),
            n_basins=2,
        )

    def test_builds_parallel_semisynthetic_48_month_system(self) -> None:
        benchmark = self.build()
        system = benchmark.system
        self.assertEqual(system.horizon, CAMELS_BR_HORIZON)
        self.assertEqual(system.periods, CAMELS_BR_PERIODS)
        self.assertEqual(system.reservoir_ids, ("00000001", "00000002"))
        self.assertEqual(system.objective.kind, "multipurpose")
        np.testing.assert_array_equal(system.routing, np.zeros((2, 2)))

        metadata = dict(system.metadata)
        self.assertEqual(metadata["semisynthetic_reservoirs"], "true")
        self.assertEqual(metadata["graph_topology"], "none")
        self.assertTrue(benchmark.audit["semisynthetic_reservoirs"])
        self.assertFalse(benchmark.audit["has_graph"])
        self.assertTrue(benchmark.audit["train_only_audit_passed"])

    def test_fit_calendar_and_historical_splits_are_exact(self) -> None:
        benchmark = self.build()
        self.assertEqual(benchmark.fit_matrix.shape, (25 * 12, 2))
        np.testing.assert_array_equal(
            benchmark.fit_calendar, np.tile(np.arange(12), 25)
        )
        self.assertEqual(
            benchmark.split("generator_validation").inflow.shape, (9 * 12, 2)
        )
        locked = benchmark.split("locked_real_test")
        self.assertEqual(locked.inflow.shape, (10 * 12, 2))
        self.assertEqual(locked.calendar[0], 0)
        self.assertEqual(locked.calendar[-1], 11)
        self.assertEqual(locked.contiguous_paths().shape, (7, 48, 2))
        self.assertFalse(benchmark.fit_matrix.flags.writeable)

    def test_later_observations_cannot_change_system_or_fit_contract(self) -> None:
        original = self.build()
        changed = self.monthly.copy()
        changed.loc[changed["water_year"].ge(2006), "inflow_hm3"] *= 1_000_000.0
        rebuilt = self.build(changed)

        self.assertEqual(original.system, rebuilt.system)
        np.testing.assert_array_equal(original.fit_matrix, rebuilt.fit_matrix)
        np.testing.assert_array_equal(original.fit_calendar, rebuilt.fit_calendar)
        self.assertEqual(
            original.audit["fit_fingerprint_sha256"],
            rebuilt.audit["fit_fingerprint_sha256"],
        )
        self.assertFalse(
            np.array_equal(
                original.split("generator_validation").inflow,
                rebuilt.split("generator_validation").inflow,
            )
        )

    def test_capacity_and_objective_scales_are_recomputed_from_fit_only(self) -> None:
        benchmark = self.build()
        fit_means = benchmark.fit_matrix.mean(axis=0)
        np.testing.assert_allclose(benchmark.system.storage_capacity, 6.0 * fit_means)
        np.testing.assert_allclose(benchmark.system.release_capacity, 1.5 * fit_means)
        np.testing.assert_allclose(
            benchmark.system.initial_storage, 0.64 * 6.0 * fit_means
        )
        np.testing.assert_allclose(
            benchmark.system.objective.terminal_target, 0.62 * 6.0 * fit_means
        )
        np.testing.assert_allclose(
            benchmark.system.objective.shortage_cost, 1.0 / fit_means
        )
        np.testing.assert_allclose(
            benchmark.basin_audit["objective_volume_scale_hm3"], fit_means
        )

    def test_fit_change_with_stale_calibration_is_rejected(self) -> None:
        changed = self.monthly.copy()
        fit_row = changed["water_year"].eq(1981) & changed["gauge_id"].eq("1")
        changed.loc[fit_row, "inflow_hm3"] *= 2.0
        with self.assertRaisesRegex(DataValidationError, "training-only"):
            self.build(changed)

    def test_fit_gap_is_rejected_but_post_fit_gap_is_preserved(self) -> None:
        fit_gap = self.monthly.copy()
        fit_gap.loc[
            fit_gap["gauge_id"].eq("1")
            & fit_gap["month"].eq(pd.Timestamp("1990-01-01")),
            "inflow_hm3",
        ] = np.nan
        with self.assertRaisesRegex(DataValidationError, "must contain 25 complete"):
            self.build(fit_gap)

        later_gap = self.monthly.copy()
        later_gap.loc[
            later_gap["gauge_id"].eq("1")
            & later_gap["month"].eq(pd.Timestamp("2017-01-01")),
            "inflow_hm3",
        ] = np.nan
        benchmark = self.build(later_gap)
        locked = benchmark.split("locked_real_test")
        self.assertEqual(locked.missing_cells, 1)
        self.assertTrue(np.isnan(locked.inflow).any())
        self.assertLess(locked.contiguous_paths().shape[0], 7)
        self.assertEqual(
            benchmark.basin_audit.loc[
                benchmark.basin_audit["gauge_id"].eq("00000001"),
                "locked_real_test_missing_months",
            ].item(),
            1,
        )

    def test_calendar_labels_and_formal_basin_count_are_enforced(self) -> None:
        inconsistent = self.monthly.copy()
        inconsistent.loc[0, "water_month"] = 12
        with self.assertRaisesRegex(DataValidationError, "water_month"):
            self.build(inconsistent)

        with self.assertRaisesRegex(DataValidationError, "requires 12"):
            build_camels_br_benchmark(self.monthly, self.calibration)


if __name__ == "__main__":
    import unittest

    unittest.main()
