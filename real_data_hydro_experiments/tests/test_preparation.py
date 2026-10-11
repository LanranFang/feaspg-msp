from __future__ import annotations

import json

import numpy as np
import pytest

from real_hydro.config import ObjectiveConfig, ReservoirSystem, repeated_rows
from real_hydro.dataset import TrajectoryBatch
from real_hydro.preparation import (
    attach_historical_replay,
    historical_replay_fingerprint,
    load_prepared_case,
    save_prepared_case,
    sha256_file,
)


def _system() -> ReservoirSystem:
    periods = 4
    objective = ObjectiveConfig(
        kind="multipurpose",
        release_target=repeated_rows((1.0,), periods),
        guide_lower=repeated_rows((1.0,), periods),
        guide_upper=repeated_rows((4.0,), periods),
        terminal_target=(2.0,),
        shortage_cost=(1.0,),
        excess_release_cost=(0.1,),
        low_storage_cost=(1.0,),
        flood_cost=(1.0,),
        spill_cost=(1.0,),
        ramp_cost=(0.1,),
        terminal_shortfall_cost=(1.0,),
        terminal_excess_cost=(0.0,),
    )
    return ReservoirSystem(
        benchmark="fixture",
        case_id="fixture",
        horizon=4,
        periods=periods,
        step_unit="mcm",
        reservoir_ids=("one",),
        storage_capacity=(5.0,),
        release_capacity=(2.0,),
        initial_storage=(2.0,),
        routing=((0.0,),),
        objective=objective,
    )


def _historical_replay() -> TrajectoryBatch:
    inflow = np.asarray(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[1.5], [2.5], [3.5], [4.5]],
        ],
        dtype=np.float32,
    )
    state = np.arange(16, dtype=np.float32).reshape(2, 4, 2) / 10.0
    regime = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int16)
    return TrajectoryBatch(
        name="locked_real_replay",
        seed=20_152_023,
        inflow=inflow,
        hydrology_state=state,
        hydrology_regime=regime,
        calendar=np.arange(4, dtype=np.int16),
        demand=np.asarray(
            [[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]],
            dtype=np.float32,
        ),
        source_path_id=np.asarray([2015, 2016], dtype=np.int32),
    )


def _save_fixture(tmp_path, *, historical_replay=None):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(exist_ok=True)
    source = raw_root / "fixture.csv"
    source.write_text("frozen\n", encoding="utf-8")
    calendar = np.arange(80, dtype=np.int16) % 4
    values = 2.0 + np.sin(np.arange(80, dtype=float) / 5.0)[:, None]
    return save_prepared_case(
        "resops_lucky_peak",
        _system(),
        values,
        calendar,
        annual_component=False,
        raw_files=[source],
        raw_root=raw_root,
        processed_root=tmp_path / "processed",
        max_order=1,
        regime_count=2,
        historical_replay=historical_replay,
    )


def test_prepared_case_round_trip_is_content_addressed(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    source = raw_root / "fixture.csv"
    source.write_text("frozen\n", encoding="utf-8")
    calendar = np.arange(80, dtype=np.int16) % 4
    values = 2.0 + np.sin(np.arange(80, dtype=float) / 5.0)[:, None]
    prepared = save_prepared_case(
        "resops_lucky_peak",
        _system(),
        values,
        calendar,
        annual_component=False,
        raw_files=[source],
        raw_root=raw_root,
        processed_root=tmp_path / "processed",
        max_order=1,
        regime_count=2,
    )
    restored = load_prepared_case(
        "resops_lucky_peak", processed_root=tmp_path / "processed"
    )
    assert np.array_equal(restored.fit_inflow, prepared.fit_inflow)
    assert np.array_equal(restored.fit_calendar, prepared.fit_calendar)
    assert restored.system == prepared.system
    assert restored.model.training_fingerprint == prepared.model.training_fingerprint
    assert restored.provenance["raw_files"][0]["path"] == "fixture.csv"
    assert restored.historical_replay is None


def test_historical_replay_round_trip_saves_every_batch_field(tmp_path) -> None:
    replay = _historical_replay()
    prepared = _save_fixture(tmp_path, historical_replay=replay)
    restored = load_prepared_case(
        "resops_lucky_peak", processed_root=tmp_path / "processed"
    )
    assert prepared.historical_replay is not None
    assert restored.historical_replay is not None
    actual = restored.historical_replay
    assert actual.name == replay.name
    assert actual.seed == replay.seed
    for field in (
        "inflow",
        "hydrology_state",
        "hydrology_regime",
        "calendar",
        "demand",
        "source_path_id",
    ):
        assert np.array_equal(getattr(actual, field), getattr(replay, field))
    assert not actual.inflow.flags.writeable

    link = restored.provenance["historical_replay"]
    assert link["fingerprint_sha256"] == historical_replay_fingerprint(replay)
    artifact = tmp_path / "processed" / "resops_lucky_peak" / link["arrays_file"]
    with np.load(artifact, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "inflow",
            "hydrology_state",
            "hydrology_regime",
            "calendar",
            "demand",
            "source_path_id",
        }


def test_historical_replay_can_be_atomically_attached_after_prepare(tmp_path) -> None:
    prepared = _save_fixture(tmp_path)
    assert prepared.historical_replay is None
    attached = attach_historical_replay(
        "resops_lucky_peak",
        _historical_replay(),
        processed_root=tmp_path / "processed",
    )
    assert attached.historical_replay is not None
    assert attached.provenance["historical_replay"]["attached_at_utc"]
    restored = load_prepared_case(
        "resops_lucky_peak", processed_root=tmp_path / "processed"
    )
    assert restored.historical_replay is not None
    assert np.array_equal(
        restored.historical_replay.source_path_id,
        np.asarray([2015, 2016], dtype=np.int32),
    )


def test_historical_replay_tampered_npz_is_rejected(tmp_path) -> None:
    prepared = _save_fixture(tmp_path, historical_replay=_historical_replay())
    link = prepared.provenance["historical_replay"]
    case_dir = tmp_path / "processed" / "resops_lucky_peak"
    artifact = case_dir / link["arrays_file"]
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["inflow"][0, 0, 0] += 100.0
    np.savez_compressed(artifact, **arrays)
    with pytest.raises(ValueError, match="array SHA-256"):
        load_prepared_case(
            "resops_lucky_peak", processed_root=tmp_path / "processed"
        )


def test_historical_replay_content_fingerprint_rejects_rehashed_tamper(
    tmp_path,
) -> None:
    prepared = _save_fixture(tmp_path, historical_replay=_historical_replay())
    case_dir = tmp_path / "processed" / "resops_lucky_peak"
    provenance_path = case_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    link = provenance["historical_replay"]
    artifact = case_dir / link["arrays_file"]
    metadata_path = case_dir / link["metadata_file"]

    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["inflow"][0, 0, 0] += 100.0
    np.savez_compressed(artifact, **arrays)
    replacement_sha = sha256_file(artifact)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arrays_sha256"] = replacement_sha
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    link["arrays_sha256"] = replacement_sha
    link["metadata_sha256"] = sha256_file(metadata_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="content fingerprint"):
        load_prepared_case(
            "resops_lucky_peak", processed_root=tmp_path / "processed"
        )
