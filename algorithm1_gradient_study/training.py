"""Validation-only tuning with matched path budgets for gradient experiments."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from efficiency_switching_hydro.config import ExperimentConfig
from efficiency_switching_hydro.data import Dataset
from efficiency_switching_hydro.simulator import simulate
from efficiency_switching_hydro.train import unused_batches

from .config import StudyProtocol


@dataclass
class CandidateFit:
    policy: torch.nn.Module
    learning_rate: float
    best_validation_cost: float
    selected_iteration: int
    fit_time_seconds: float
    history: list[dict[str, float | int | str]]
    checkpoint_path: str


@dataclass
class SelectedFit:
    policy: torch.nn.Module
    learning_rate: float
    best_validation_cost: float
    selected_iteration: int
    selected_fit_time_seconds: float
    total_tuning_time_seconds: float
    candidates: list[CandidateFit]
    checkpoint_path: str


@torch.no_grad()
def validation_cost(
    policy,
    validation: Dataset,
    config: ExperimentConfig,
    device: str,
) -> float:
    policy.eval()
    return float(simulate(policy, validation, config, device=device).mean_cost.cpu())


def fit_candidate(
    policy,
    fit: Dataset,
    validation: Dataset,
    config: ExperimentConfig,
    protocol: StudyProtocol,
    train_seed: int,
    learning_rate: float,
    gradient_mode: str,
    checkpoint_path: Path,
) -> CandidateFit:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    rng = np.random.default_rng(train_seed)
    policy.to(protocol.device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=protocol.weight_decay,
    )
    batches = unused_batches(
        fit.n,
        protocol.iterations,
        protocol.batch_size,
        rng,
    )
    best_state = copy.deepcopy(policy.state_dict())
    best_iteration = 0
    start = time.perf_counter()
    best_validation = validation_cost(policy, validation, config, protocol.device)
    history: list[dict[str, float | int | str]] = [
        {
            "train_seed": train_seed,
            "gradient_mode": gradient_mode,
            "learning_rate": learning_rate,
            "iteration": 0,
            "fit_trajectories_seen": 0,
            "gradient_updates": 0,
            "train_cost": float("nan"),
            "validation_cost": best_validation,
            "is_best": 1,
        }
    ]
    for iteration in range(1, protocol.iterations + 1):
        policy.train()
        batch = fit.take(batches[iteration - 1], name="fit_batch")
        optimizer.zero_grad(set_to_none=True)
        loss = simulate(
            policy,
            batch,
            config,
            device=protocol.device,
            gradient_mode=gradient_mode,
        ).mean_cost
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), protocol.gradient_clip)
        optimizer.step()
        should_validate = (
            iteration % protocol.validation_every == 0
            or iteration == protocol.iterations
        )
        current_validation = float("nan")
        is_best = 0
        if should_validate:
            current_validation = validation_cost(
                policy,
                validation,
                config,
                protocol.device,
            )
            if current_validation < best_validation:
                best_validation = current_validation
                best_iteration = iteration
                best_state = copy.deepcopy(policy.state_dict())
                is_best = 1
        history.append(
            {
                "train_seed": train_seed,
                "gradient_mode": gradient_mode,
                "learning_rate": learning_rate,
                "iteration": iteration,
                "fit_trajectories_seen": iteration * protocol.batch_size,
                "gradient_updates": iteration,
                "train_cost": float(loss.detach().cpu()),
                "validation_cost": current_validation,
                "is_best": is_best,
            }
        )
    fit_time = time.perf_counter() - start
    policy.load_state_dict(best_state)
    policy.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_seed": train_seed,
            "gradient_mode": gradient_mode,
            "learning_rate": learning_rate,
            "selected_iteration": best_iteration,
            "best_validation_cost": best_validation,
            "iterations": protocol.iterations,
            "batch_size": protocol.batch_size,
            "state_dict": best_state,
        },
        checkpoint_path,
    )
    return CandidateFit(
        policy=policy,
        learning_rate=learning_rate,
        best_validation_cost=best_validation,
        selected_iteration=best_iteration,
        fit_time_seconds=fit_time,
        history=history,
        checkpoint_path=str(checkpoint_path),
    )


def tune_and_select(
    policy_builder: Callable[[], torch.nn.Module],
    fit: Dataset,
    validation: Dataset,
    config: ExperimentConfig,
    protocol: StudyProtocol,
    train_seed: int,
    learning_rates: tuple[float, ...],
    gradient_mode: str,
    checkpoint_dir: Path,
    run_name: str,
) -> SelectedFit:
    candidates: list[CandidateFit] = []
    for candidate_index, learning_rate in enumerate(learning_rates):
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        policy = policy_builder()
        rate_tag = f"{learning_rate:.8g}".replace(".", "p")
        candidate = fit_candidate(
            policy,
            fit,
            validation,
            config,
            protocol,
            train_seed,
            learning_rate,
            gradient_mode,
            checkpoint_dir
            / f"{run_name}_candidate{candidate_index}_lr{rate_tag}_seed{train_seed}.pt",
        )
        candidates.append(candidate)
    selected = min(candidates, key=lambda candidate: candidate.best_validation_cost)
    selected_path = checkpoint_dir / f"{run_name}_selected_seed{train_seed}.pt"
    selected_payload = torch.load(
        selected.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    selected_payload["candidate_learning_rates"] = list(learning_rates)
    selected_payload["total_tuning_time_seconds"] = sum(
        candidate.fit_time_seconds for candidate in candidates
    )
    torch.save(selected_payload, selected_path)
    return SelectedFit(
        policy=selected.policy,
        learning_rate=selected.learning_rate,
        best_validation_cost=selected.best_validation_cost,
        selected_iteration=selected.selected_iteration,
        selected_fit_time_seconds=selected.fit_time_seconds,
        total_tuning_time_seconds=sum(
            candidate.fit_time_seconds for candidate in candidates
        ),
        candidates=candidates,
        checkpoint_path=str(selected_path),
    )
