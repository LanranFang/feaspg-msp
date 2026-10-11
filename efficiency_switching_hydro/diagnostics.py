"""Policy-map and turbine-band diagnostics for one completed experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DataConfig, ExperimentConfig
from .data import generate_dataset
from .physics import turbine_efficiency_np
from .policies import make_policy, normalized_flat_state
from .projection import project_action
from .simulator import simulate


BASE_POLICIES = (
    "projected_affine",
    "projected_polynomial",
)


def _load_policy(
    name: str, config: ExperimentConfig, checkpoint_dir: Path, seed: int
):
    policy = make_policy(name, config)
    checkpoint = torch.load(
        checkpoint_dir / f"{name}_seed{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def _reference_state(config: ExperimentConfig, stage: int):
    storage = (
        np.asarray(config.initial_storage)
        if stage == 0
        else config.planned_storage()[stage - 1]
    )
    previous = (
        np.zeros(config.n_reservoirs)
        if stage == 0
        else config.reference_release()[stage - 1]
    )
    queue = 0.25 * config.queue_capacity()
    return storage, previous, queue


@torch.no_grad()
def policy_slice(
    policies: dict[str, torch.nn.Module],
    config: ExperimentConfig,
    path: Path,
    neural_name: str,
) -> None:
    neural_policy = policies[neural_name]
    best = (-1.0, config.T // 2, 0, 0)
    for candidate_stage in range(0, config.T, max(config.T // 12, 1)):
        reference_storage, reference_previous, reference_queue = _reference_state(
            config, candidate_stage
        )
        for candidate_source, candidate_downstream in enumerate(
            config.downstream_index
        ):
            if candidate_downstream < 0:
                continue
            for source_fraction in (0.25, 0.40, 0.55, 0.70):
                candidate_storage = np.broadcast_to(
                    reference_storage, (2, config.n_reservoirs)
                ).copy()
                candidate_storage[:, candidate_source] = (
                    source_fraction * config.storage_capacity[candidate_source]
                )
                candidate_storage[0, candidate_downstream] = (
                    0.20 * config.storage_capacity[candidate_downstream]
                )
                candidate_storage[1, candidate_downstream] = (
                    0.90 * config.storage_capacity[candidate_downstream]
                )
                candidate_previous = np.broadcast_to(
                    reference_previous, candidate_storage.shape
                ).copy()
                candidate_queue = np.broadcast_to(
                    reference_queue,
                    (2, config.routing_delay, config.n_reservoirs),
                ).copy()
                storage_tensor = torch.as_tensor(
                    candidate_storage, dtype=torch.float32
                )
                previous_tensor = torch.as_tensor(
                    candidate_previous, dtype=torch.float32
                )
                queue_tensor = torch.as_tensor(
                    candidate_queue, dtype=torch.float32
                )
                proposal = neural_policy.propose(
                    candidate_stage,
                    storage_tensor,
                    previous_tensor,
                    queue_tensor,
                )
                candidate_action = project_action(
                    proposal, storage_tensor, previous_tensor, config
                )[:, candidate_source]
                sensitivity = float(
                    torch.abs(candidate_action[1] - candidate_action[0])
                    / config.release_capacity[candidate_source]
                )
                if sensitivity > best[0]:
                    best = (
                        sensitivity,
                        candidate_stage,
                        candidate_source,
                        candidate_downstream,
                    )
    downstream_sensitivity, stage, source, downstream = best
    storage, previous, queue = _reference_state(config, stage)
    fractions = np.linspace(0.15, 0.95, 61)
    source_grid, downstream_grid = np.meshgrid(fractions, fractions)
    n = source_grid.size
    storage_batch = np.broadcast_to(storage, (n, config.n_reservoirs)).copy()
    storage_batch[:, source] = (
        source_grid.ravel() * config.storage_capacity[source]
    )
    storage_batch[:, downstream] = (
        downstream_grid.ravel() * config.storage_capacity[downstream]
    )
    previous_batch = np.broadcast_to(previous, storage_batch.shape).copy()
    queue_batch = np.broadcast_to(
        queue, (n, config.routing_delay, config.n_reservoirs)
    ).copy()
    storage_tensor = torch.as_tensor(storage_batch, dtype=torch.float32)
    previous_tensor = torch.as_tensor(previous_batch, dtype=torch.float32)
    queue_tensor = torch.as_tensor(queue_batch, dtype=torch.float32)

    surfaces = {}
    for name, policy in policies.items():
        proposal = policy.propose(
            stage, storage_tensor, previous_tensor, queue_tensor
        )
        action = project_action(
            proposal, storage_tensor, previous_tensor, config
        )
        surfaces[name] = (
            action[:, source].numpy().reshape(source_grid.shape)
            / config.release_capacity[source]
        )

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    labels = {
        "projected_affine": "Affine",
        "projected_polynomial": "Polynomial",
        "projected_graph_neural": "Graph neural",
        "projected_graph_neural_global": "Global graph neural",
    }
    image = None
    displayed = (*BASE_POLICIES, neural_name)
    for axis, name in zip(axes[:3], displayed):
        image = axis.imshow(
            surfaces[name],
            origin="lower",
            extent=(0.15, 0.95, 0.15, 0.95),
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            cmap="viridis",
        )
        axis.set_title(labels[name])
        axis.set_xlabel("Source storage / capacity")
        axis.set_ylabel("Downstream storage / capacity")
    difference = surfaces[neural_name] - surfaces["projected_affine"]
    limit = max(float(np.abs(difference).max()), 1e-3)
    difference_image = axes[3].imshow(
        difference,
        origin="lower",
        extent=(0.15, 0.95, 0.15, 0.95),
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        cmap="coolwarm",
    )
    axes[3].set_title("Graph neural - affine")
    axes[3].set_xlabel("Source storage / capacity")
    axes[3].set_ylabel("Downstream storage / capacity")
    figure.colorbar(image, ax=axes[:3], label="Source release / capacity")
    figure.colorbar(difference_image, ax=axes[3], label="Release difference")
    figure.suptitle(
        f"Most downstream-sensitive slice: stage {stage + 1}, "
        f"reservoir {source} -> {downstream}, sensitivity={downstream_sensitivity:.3f}"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def efficiency_curves(config: ExperimentConfig, path: Path) -> None:
    source = next(
        index
        for index, downstream in enumerate(config.downstream_index)
        if downstream >= 0
    )
    fractions = np.linspace(0.0, 1.0, 500)
    capacity = np.asarray(config.storage_capacity)
    release = np.asarray(config.release_capacity)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for storage_fraction in (0.30, 0.60, 0.90):
        storage = np.broadcast_to(
            storage_fraction * capacity,
            (len(fractions), config.n_reservoirs),
        ).copy()
        action = np.zeros_like(storage)
        action[:, source] = fractions * release[source]
        efficiency = turbine_efficiency_np(storage, action, config)[:, source]
        axis.plot(
            fractions,
            efficiency,
            linewidth=2,
            label=f"storage={storage_fraction:.1f}C",
        )
    axis.set_xlabel("Release / release capacity")
    axis.set_ylabel("Turbine efficiency")
    axis.set_title("Storage-dependent turbine operating bands")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def affine_surrogate_diagnostic(
    graph_policy,
    config: ExperimentConfig,
    seed: int = 91,
    states_per_stage: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    capacity = np.asarray(config.storage_capacity)
    release = np.asarray(config.release_capacity)
    queue_capacity = config.queue_capacity()
    rows = []
    for stage in sorted({config.T // 4, config.T // 2, 3 * config.T // 4}):
        storage = rng.uniform(0.15, 0.95, (states_per_stage, config.n_reservoirs)) * capacity
        previous = rng.uniform(0.0, 0.85, (states_per_stage, config.n_reservoirs)) * release
        queue = rng.uniform(
            0.0,
            0.70,
            (states_per_stage, config.routing_delay, config.n_reservoirs),
        ) * queue_capacity
        storage_tensor = torch.as_tensor(storage, dtype=torch.float32)
        previous_tensor = torch.as_tensor(previous, dtype=torch.float32)
        queue_tensor = torch.as_tensor(queue, dtype=torch.float32)
        state = normalized_flat_state(
            config, storage_tensor, previous_tensor, queue_tensor
        ).numpy()
        features = np.concatenate((np.ones((states_per_stage, 1)), state), axis=1)
        proposal = graph_policy.propose(
            stage, storage_tensor, previous_tensor, queue_tensor
        )
        action = project_action(
            proposal, storage_tensor, previous_tensor, config
        )
        for mapping, target in (
            ("proposal", proposal.numpy()),
            ("projected_action", action.numpy()),
        ):
            split = states_per_stage // 2
            coefficients = np.linalg.lstsq(
                features[:split], target[:split], rcond=None
            )[0]
            prediction = features[split:] @ coefficients
            residual = target[split:] - prediction
            denominator = np.square(
                target[split:] - target[split:].mean(axis=0, keepdims=True)
            ).sum()
            r2 = 1.0 - float(np.square(residual).sum()) / max(float(denominator), 1e-12)
            normalized_rmse = float(np.sqrt(np.mean(np.square(residual)))) / float(
                release.mean()
            )
            rows.append(
                {
                    "stage": stage + 1,
                    "mapping": mapping,
                    "affine_surrogate_r2": r2,
                    "normalized_rmse": normalized_rmse,
                    "n_fit_states": split,
                    "n_test_states": states_per_stage - split,
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def operating_modes(
    policies: dict[str, torch.nn.Module],
    config: ExperimentConfig,
    data_config: DataConfig,
) -> pd.DataFrame:
    test = generate_dataset("test", data_config.n_test, data_config.test_seed, config)
    rows = []
    release = torch.as_tensor(config.release_capacity)
    capacity = torch.as_tensor(config.storage_capacity)
    center1 = torch.as_tensor(config.band1_center)
    center2 = torch.as_tensor(config.band2_center)
    shift1 = torch.as_tensor(config.band1_storage_shift)
    shift2 = torch.as_tensor(config.band2_storage_shift)
    for name, policy in policies.items():
        trace = simulate(policy, test, config)
        fraction = trace.actions / release
        storage_fraction = trace.storage_before_action / capacity
        low_center = center1 + shift1 * (1.0 - storage_fraction)
        high_center = center2 + shift2 * (1.0 - storage_fraction)
        off = fraction < 0.12
        low = (~off) & (torch.abs(fraction - low_center) <= 0.10)
        high = (~off) & (~low) & (torch.abs(fraction - high_center) <= 0.12)
        other = ~(off | low | high)
        for mode, indicator in (
            ("off", off),
            ("low_band", low),
            ("high_band", high),
            ("other", other),
        ):
            rows.append(
                {
                    "policy": name,
                    "mode": mode,
                    "share": float(indicator.float().mean()),
                }
            )
    return pd.DataFrame(rows)


def run(
    experiment_dir: str | Path,
    neural_name: str = "projected_graph_neural",
) -> None:
    experiment_dir = Path(experiment_dir)
    saved = json.loads((experiment_dir / "config.json").read_text())
    config = ExperimentConfig(**saved["experiment"])
    data_config = DataConfig(**saved["data"])
    seed = int(saved["training"]["train_seeds"][0])
    policies = {
        name: _load_policy(name, config, experiment_dir / "checkpoints", seed)
        for name in (*BASE_POLICIES, neural_name)
    }
    figures = experiment_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    efficiency_curves(config, figures / "efficiency_bands.png")
    policy_slice(
        policies, config, figures / "policy_slice.png", neural_name
    )
    affine_surrogate_diagnostic(
        policies[neural_name], config
    ).to_csv(experiment_dir / "nonlinearity_diagnostic.csv", index=False)
    operating_modes(policies, config, data_config).to_csv(
        experiment_dir / "operating_mode_shares.csv", index=False
    )
    print(f"Wrote policy diagnostics to {experiment_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--neural-policy",
        default="projected_graph_neural",
        choices=("projected_graph_neural", "projected_graph_neural_global"),
    )
    args = parser.parse_args()
    run(args.experiment_dir, args.neural_policy)


if __name__ == "__main__":
    main()
