"""Frozen protocols for the ICLR-focused Algorithm 1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from algorithm1_gradient_study.config import StudyProtocol
from efficiency_switching_hydro.config import DataConfig, ExperimentConfig, RLConfig


@dataclass(frozen=True)
class SeedBlock:
    fit: int
    validation: int
    test: int
    calibration: int
    evaluation: int

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if len(set(values)) != len(values):
            raise ValueError("Every data interface must use a distinct seed.")

    def data_config(
        self,
        n_fit: int,
        n_validation: int,
        n_test: int,
        n_calibration: int,
        n_evaluation: int,
    ) -> DataConfig:
        return DataConfig(
            n_fit=n_fit,
            n_validation=n_validation,
            n_test=n_test,
            n_calibration=n_calibration,
            n_evaluation=n_evaluation,
            fit_seed=self.fit,
            validation_seed=self.validation,
            test_seed=self.test,
            calibration_seed=self.calibration,
            evaluation_seed=self.evaluation,
        )


@dataclass(frozen=True)
class DirectTrainingView:
    iterations: int
    batch_size: int
    validation_every: int
    weight_decay: float
    gradient_clip: float
    device: str


@dataclass(frozen=True)
class ICLRProtocol:
    profile: str = "decision_switching"
    n_reservoirs: int = 12
    horizon: int = 48
    shock_factors: int = 4
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    mechanism_seeds: SeedBlock = SeedBlock(14_101, 14_202, 14_303, 14_404, 14_505)
    budget_seeds: SeedBlock = SeedBlock(15_101, 15_202, 15_303, 15_404, 15_505)
    scaling_seeds: SeedBlock = SeedBlock(16_101, 16_202, 16_303, 16_404, 16_505)
    transfer_seeds: SeedBlock = SeedBlock(17_101, 17_202, 17_303, 17_404, 17_505)

    n_validation: int = 500
    n_test: int = 4096
    n_calibration: int = 500
    n_evaluation: int = 2048
    fit_budgets: tuple[int, ...] = (100, 250, 500, 1000, 2500)
    base_n_fit: int = 1000
    batch_size: int = 2
    max_validation_checks: int = 50

    mlp_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4, 6.0e-4)
    graph_learning_rates: tuple[float, ...] = (3.0e-4, 6.0e-4, 1.0e-3)
    shac_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4, 6.0e-4)
    ppo_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4, 1.0e-3)
    sac_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4, 1.0e-3)
    weight_decay: float = 1.0e-6
    gradient_clip: float = 10.0
    network_depth: int = 2
    graph_width: int = 64

    shac_window: int = 8
    shac_critic_learning_rate: float = 1.0e-3
    shac_critic_epochs: int = 2
    shac_critic_tau: float = 0.05
    shac_cost_scale: float = 100_000.0

    rl_reward_scale: float = 1000.0
    rl_gamma: float = 1.0
    rl_validation_every_episodes: int = 50
    rl_network_width: int = 256
    rl_network_depth: int = 2
    ppo_rollout_episodes: int = 10
    ppo_epochs: int = 10
    ppo_batch_size: int = 256
    replay_buffer_size: int = 100_000
    learning_starts: int = 1000
    off_policy_batch_size: int = 256
    off_policy_gradient_steps: int = 1
    off_policy_n_steps: int = 3

    reservoir_values: tuple[int, ...] = (6, 12, 24, 48)
    horizon_values: tuple[int, ...] = (12, 24, 48, 96)
    routing_severities: tuple[float, ...] = (0.10, 0.25, 0.40)
    routing_topology_seeds: tuple[int, ...] = (27_101, 27_102, 27_103, 27_104, 27_105)

    latency_batch_size: int = 256
    latency_repetitions: int = 100
    device: str = "cpu"
    rl_device: str = "cpu"
    smoke: bool = False

    def __post_init__(self) -> None:
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("Training seeds must be unique.")
        if not self.smoke and len(self.train_seeds) != 5:
            raise ValueError("Formal experiments require five training seeds.")
        seed_values = []
        for block in (
            self.mechanism_seeds,
            self.budget_seeds,
            self.scaling_seeds,
            self.transfer_seeds,
        ):
            seed_values.extend(asdict(block).values())
        if len(seed_values) != len(set(seed_values)):
            raise ValueError("Data seeds must also be distinct across experiment families.")
        if tuple(sorted(set(self.fit_budgets))) != self.fit_budgets:
            raise ValueError("fit_budgets must be unique and increasing.")
        if self.base_n_fit not in self.fit_budgets:
            raise ValueError("base_n_fit must be one of the fit budgets.")
        if any(value % self.batch_size for value in self.fit_budgets):
            raise ValueError("Every fit budget must be divisible by batch_size.")
        positive = (
            self.n_reservoirs,
            self.horizon,
            self.shock_factors,
            self.n_validation,
            self.n_test,
            self.n_calibration,
            self.n_evaluation,
            self.batch_size,
            self.max_validation_checks,
            self.graph_width,
            self.network_depth,
            self.shac_window,
            self.shac_critic_epochs,
            self.latency_batch_size,
            self.latency_repetitions,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Dimensions, sample sizes, and iteration settings must be positive.")
        rates = (
            *self.mlp_learning_rates,
            *self.graph_learning_rates,
            *self.shac_learning_rates,
            *self.ppo_learning_rates,
            *self.sac_learning_rates,
            self.shac_critic_learning_rate,
        )
        if min(rates) <= 0.0:
            raise ValueError("Learning rates must be positive.")
        if not 0.0 < self.shac_critic_tau <= 1.0:
            raise ValueError("shac_critic_tau must lie in (0, 1].")

    @classmethod
    def smoke_protocol(cls, device: str = "cpu", rl_device: str = "cpu") -> "ICLRProtocol":
        return cls(
            n_reservoirs=4,
            horizon=6,
            shock_factors=2,
            train_seeds=(0,),
            n_validation=8,
            n_test=16,
            n_calibration=4,
            n_evaluation=4,
            fit_budgets=(8, 16),
            base_n_fit=16,
            batch_size=2,
            max_validation_checks=4,
            mlp_learning_rates=(3.0e-4,),
            graph_learning_rates=(6.0e-4,),
            shac_learning_rates=(3.0e-4,),
            ppo_learning_rates=(3.0e-4,),
            sac_learning_rates=(3.0e-4,),
            graph_width=16,
            shac_window=3,
            shac_critic_epochs=1,
            rl_validation_every_episodes=4,
            rl_network_width=16,
            ppo_rollout_episodes=4,
            ppo_epochs=2,
            ppo_batch_size=32,
            replay_buffer_size=256,
            learning_starts=4,
            off_policy_batch_size=16,
            reservoir_values=(4, 6),
            horizon_values=(6, 9),
            routing_severities=(0.25,),
            routing_topology_seeds=(27_101,),
            latency_batch_size=4,
            latency_repetitions=3,
            device=device,
            rl_device=rl_device,
            smoke=True,
        )

    def with_devices(self, device: str, rl_device: str) -> "ICLRProtocol":
        return replace(self, device=device, rl_device=rl_device)

    def hydro_config(
        self,
        n_reservoirs: int | None = None,
        horizon: int | None = None,
    ) -> ExperimentConfig:
        return ExperimentConfig.make(
            self.profile,
            n_reservoirs or self.n_reservoirs,
            horizon or self.horizon,
            self.shock_factors,
        )

    def iterations(self, n_fit: int) -> int:
        if n_fit % self.batch_size:
            raise ValueError("n_fit must be divisible by batch_size.")
        return n_fit // self.batch_size

    def validation_every(self, n_fit: int) -> int:
        return max(1, self.iterations(n_fit) // self.max_validation_checks)

    def direct_training(self, n_fit: int) -> DirectTrainingView:
        return DirectTrainingView(
            iterations=self.iterations(n_fit),
            batch_size=self.batch_size,
            validation_every=self.validation_every(n_fit),
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            device=self.device,
        )

    def study_protocol(self) -> StudyProtocol:
        from .policies import parameter_matched_mlp_width

        n_fit = self.base_n_fit
        matched_width, _, _ = parameter_matched_mlp_width(
            self.hydro_config(),
            graph_width=self.graph_width,
            depth=self.network_depth,
        )
        return StudyProtocol(
            profile=self.profile,
            n_reservoirs=self.n_reservoirs,
            horizon=self.horizon,
            shock_factors=self.shock_factors,
            train_seeds=self.train_seeds,
            n_fit=n_fit,
            n_validation=self.n_validation,
            n_test=self.n_test,
            n_calibration=self.n_calibration,
            n_evaluation=self.n_evaluation,
            fit_seed=self.mechanism_seeds.fit,
            validation_seed=self.mechanism_seeds.validation,
            test_seed=self.mechanism_seeds.test,
            calibration_seed=self.mechanism_seeds.calibration,
            evaluation_seed=self.mechanism_seeds.evaluation,
            iterations=self.iterations(n_fit),
            batch_size=self.batch_size,
            validation_every=self.validation_every(n_fit),
            ablation_learning_rates=self.mlp_learning_rates,
            graph_learning_rates=self.graph_learning_rates,
            mlp_learning_rates=self.mlp_learning_rates,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            mlp_width=matched_width,
            graph_width=self.graph_width,
            network_depth=self.network_depth,
            gradient_paths=32 if not self.smoke else 4,
            horizon_values=(12, 24, 48, 96) if not self.smoke else self.horizon_values,
            graph_severities=(0.0, *self.routing_severities),
            graph_topology_seeds=self.routing_topology_seeds,
            latency_batch_size=self.latency_batch_size,
            latency_repetitions=self.latency_repetitions,
            device=self.device,
            smoke=self.smoke,
        )

    def rl_config(self, algorithm: str, learning_rate: float, n_fit: int) -> RLConfig:
        if algorithm not in {"ppo", "sac"}:
            raise ValueError(f"Unsupported data-budget RL algorithm: {algorithm}")
        validation_every = min(
            self.rl_validation_every_episodes,
            max(1, n_fit // self.max_validation_checks),
        )
        return RLConfig(
            reward_scale=self.rl_reward_scale,
            gamma=self.rl_gamma,
            validation_every_episodes=validation_every,
            network_width=self.rl_network_width,
            network_depth=self.rl_network_depth,
            ppo_learning_rate=(learning_rate if algorithm == "ppo" else self.ppo_learning_rates[0]),
            ppo_rollout_episodes=min(self.ppo_rollout_episodes, n_fit),
            ppo_epochs=self.ppo_epochs,
            ppo_batch_size=self.ppo_batch_size,
            sac_learning_rate=(learning_rate if algorithm == "sac" else self.sac_learning_rates[0]),
            td3_learning_rate=1.0e-3,
            replay_buffer_size=self.replay_buffer_size,
            learning_starts=self.learning_starts,
            off_policy_batch_size=self.off_policy_batch_size,
            gradient_steps=self.off_policy_gradient_steps,
            off_policy_n_steps=self.off_policy_n_steps,
            td3_action_noise=0.10,
            device=self.rl_device,
        )

    @property
    def all_seed_values(self) -> tuple[int, ...]:
        values: list[int] = []
        for block in (
            self.mechanism_seeds,
            self.budget_seeds,
            self.scaling_seeds,
            self.transfer_seeds,
        ):
            values.extend(asdict(block).values())
        return tuple(values)
