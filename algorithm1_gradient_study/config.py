"""Pre-registered protocol settings for the Algorithm 1 mechanism study."""

from __future__ import annotations

from dataclasses import dataclass, replace

from efficiency_switching_hydro.config import DataConfig


@dataclass(frozen=True)
class StudyProtocol:
    profile: str = "decision_switching"
    n_reservoirs: int = 12
    horizon: int = 48
    shock_factors: int = 4
    study_ramp_fraction: float = 0.10
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    n_fit: int = 1000
    n_validation: int = 500
    n_test: int = 4096
    n_calibration: int = 500
    n_evaluation: int = 2048
    fit_seed: int = 4101
    validation_seed: int = 4202
    test_seed: int = 4303
    calibration_seed: int = 4404
    evaluation_seed: int = 4505

    iterations: int = 500
    batch_size: int = 2
    validation_every: int = 10
    ablation_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4)
    graph_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4)
    mlp_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4)
    weight_decay: float = 1.0e-6
    gradient_clip: float = 10.0
    mlp_width: int = 256
    graph_width: int = 64
    network_depth: int = 2

    gradient_paths: int = 32
    finite_difference_steps: tuple[float, ...] = (
        1.0e-2,
        3.0e-3,
        1.0e-3,
        3.0e-4,
        1.0e-4,
        3.0e-5,
        1.0e-5,
    )
    horizon_values: tuple[int, ...] = (6, 12, 24, 48, 96)
    graph_severities: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40)
    graph_topology_seeds: tuple[int, ...] = (7101, 7102, 7103, 7104, 7105)
    latency_batch_size: int = 256
    latency_repetitions: int = 100
    device: str = "cpu"
    smoke: bool = False

    def __post_init__(self) -> None:
        split_seeds = (
            self.fit_seed,
            self.validation_seed,
            self.test_seed,
            self.calibration_seed,
            self.evaluation_seed,
        )
        if len(set(split_seeds)) != len(split_seeds):
            raise ValueError("fit/validation/test/calibration/evaluation seeds must differ.")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("Training seeds must be unique.")
        if not self.smoke and len(self.train_seeds) != 5:
            raise ValueError("Formal runs require exactly five training seeds.")
        if self.iterations * self.batch_size > self.n_fit:
            raise ValueError(
                "Each candidate must use unique fit paths: iterations * batch_size "
                "cannot exceed n_fit."
            )
        positive = (
            self.n_fit,
            self.n_validation,
            self.n_test,
            self.n_calibration,
            self.n_evaluation,
            self.iterations,
            self.batch_size,
            self.validation_every,
            self.gradient_paths,
            self.mlp_width,
            self.graph_width,
            self.network_depth,
            self.latency_batch_size,
            self.latency_repetitions,
        )
        if any(value < 1 for value in positive):
            raise ValueError("All sample, iteration, network, and timing sizes must be positive.")
        if min(
            *self.ablation_learning_rates,
            *self.graph_learning_rates,
            *self.mlp_learning_rates,
            *self.finite_difference_steps,
        ) <= 0.0:
            raise ValueError("Learning rates and finite-difference steps must be positive.")
        if any(not 0.0 <= value <= 1.0 for value in self.graph_severities):
            raise ValueError("Graph severities must lie in [0, 1].")
        if not 0.0 < self.study_ramp_fraction <= 1.0:
            raise ValueError("study_ramp_fraction must lie in (0, 1].")

    @classmethod
    def smoke_protocol(cls, device: str = "cpu") -> "StudyProtocol":
        return cls(
            n_reservoirs=4,
            horizon=6,
            shock_factors=2,
            train_seeds=(0,),
            n_fit=16,
            n_validation=8,
            n_test=16,
            n_calibration=4,
            n_evaluation=4,
            iterations=4,
            batch_size=2,
            validation_every=2,
            ablation_learning_rates=(3.0e-4,),
            graph_learning_rates=(3.0e-4,),
            mlp_learning_rates=(3.0e-4,),
            mlp_width=16,
            graph_width=16,
            gradient_paths=4,
            finite_difference_steps=(1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4),
            horizon_values=(6, 9, 12),
            graph_severities=(0.0, 0.25),
            graph_topology_seeds=(7101,),
            latency_batch_size=8,
            latency_repetitions=5,
            device=device,
            smoke=True,
        )

    def with_device(self, device: str) -> "StudyProtocol":
        return replace(self, device=device)

    def data_config(self) -> DataConfig:
        return DataConfig(
            n_fit=self.n_fit,
            n_validation=self.n_validation,
            n_test=self.n_test,
            n_calibration=self.n_calibration,
            n_evaluation=self.n_evaluation,
            fit_seed=self.fit_seed,
            validation_seed=self.validation_seed,
            test_seed=self.test_seed,
            calibration_seed=self.calibration_seed,
            evaluation_seed=self.evaluation_seed,
        )
