"""Reproducibility metadata and incremental tabular output helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch


def canonical_seed(value: object) -> str:
    """Normalize mixed CSV seed columns while preserving fixed baselines."""

    text = str(value)
    if text == "fixed":
        return text
    return str(int(float(text)))


def write_protocol(
    output: Path,
    protocol,
    experiment: str,
    details: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "protocol": asdict(protocol),
        "primary_metric": "test cost",
        "selection_rule": "minimum validation cost",
        "test_role": "reporting only after fitting and selection are frozen",
        "uncertainty": {
            "test_path": "paired 95% CI on common test paths within train seed",
            "training": "SD of selected-policy test means across train seeds",
        },
        "counting": {
            "environment_transitions": "fit trajectories times horizon",
            "actor_gradient_updates": "optimizer steps applied to deployment policy",
            "critic_gradient_updates": "reported separately and excluded from actor updates",
        },
        "reserved_interfaces": {
            "calibration": "independent seed, not read by Algorithm 1 experiments",
            "evaluation": "independent seed, not read by Algorithm 1 experiments",
        },
        **details,
    }
    (output / "protocol.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "device": protocol.device,
        "rl_device": protocol.rl_device,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )


def dataset_fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    for values in (dataset.inflow, dataset.demand, dataset.price):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_data_manifest(
    output: Path,
    datasets: dict[str, object],
    reserved: dict[str, tuple[int, int]],
) -> None:
    payload: dict[str, object] = {}
    for name, dataset in datasets.items():
        payload[name] = {
            "seed": int(dataset.seed),
            "n_paths": int(dataset.n),
            "sha256": dataset_fingerprint(dataset),
            "role": {
                "fit_pool": "fitting only",
                "fit": "fitting only",
                "validation": "selection only",
                "test": "reporting only",
            }.get(name, name),
        }
    for name, (seed, n_paths) in reserved.items():
        payload[name] = {
            "seed": seed,
            "n_paths": n_paths,
            "role": "reserved_unread",
        }
    (output / "data_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_completion(output: Path, experiment: str, details: dict[str, object]) -> None:
    (output / "COMPLETED.json").write_text(
        json.dumps({"experiment": experiment, **details}, indent=2),
        encoding="utf-8",
    )
