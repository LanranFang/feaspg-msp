"""Pre-registered settings for the unified nonlinear-hydro comparison."""

from __future__ import annotations

from dataclasses import dataclass, replace

from efficiency_switching_hydro.config import (
    DataConfig,
    ExperimentConfig,
    RLConfig,
    SDDPConfig,
)


@dataclass(frozen=True)
class CFAConfig:
    """Settings for deterministic lookahead and its fitted inflow multiplier."""

    lookahead_horizon: int = 12
    iterations: int = 100
    batch_size: int = 10
    validation_every: int = 20
    learning_rates: tuple[float, ...] = (0.02, 0.05)
    smoothing_radius: float = 0.05
    rms_decay: float = 0.90
    theta_lower: float = 0.25
    theta_upper: float = 1.75

    def __post_init__(self) -> None:
        positive = (
            self.lookahead_horizon,
            self.iterations,
            self.batch_size,
            self.validation_every,
        )
        if any(value < 1 for value in positive):
            raise ValueError("CFA horizon and training sizes must be positive.")
        if min(*self.learning_rates, self.smoothing_radius) <= 0.0:
            raise ValueError("CFA learning rates and smoothing radius must be positive.")
        if not 0.0 <= self.rms_decay < 1.0:
            raise ValueError("CFA rms_decay must lie in [0, 1).")
        if not 0.0 < self.theta_lower < 1.0 < self.theta_upper:
            raise ValueError("CFA multiplier bounds must contain one.")


@dataclass(frozen=True)
class FullComparisonProtocol:
    """One protocol shared by every policy in the main comparison."""

    profile: str = "decision_switching"
    n_reservoirs: int = 12
    horizon: int = 48
    shock_factors: int = 4
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

    direct_iterations: int = 500
    direct_batch_size: int = 2
    direct_validation_every: int = 10
    affine_learning_rates: tuple[float, ...] = (5.0e-4, 2.0e-3)
    polynomial_learning_rates: tuple[float, ...] = (1.5e-4, 3.0e-4)
    mlp_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4)
    graph_learning_rates: tuple[float, ...] = (3.0e-4, 6.0e-4)
    weight_decay: float = 1.0e-6
    gradient_clip: float = 10.0
    mlp_width: int = 256
    graph_width: int = 64
    network_depth: int = 2

    rl_learning_rates_ppo: tuple[float, ...] = (1.0e-4, 3.0e-4)
    rl_learning_rates_sac: tuple[float, ...] = (1.0e-4, 3.0e-4)
    rl_learning_rates_td3: tuple[float, ...] = (3.0e-4, 1.0e-3)
    rl_reward_scale: float = 1000.0
    rl_gamma: float = 1.0
    rl_validation_every_episodes: int = 50
    rl_network_width: int = 256
    rl_network_depth: int = 2
    ppo_rollout_episodes: int = 10
    ppo_epochs: int = 10
    ppo_batch_size: int = 256
    replay_buffer_size: int = 100_000
    learning_starts: int = 1_000
    off_policy_batch_size: int = 256
    gradient_steps: int = 1
    off_policy_n_steps: int = 3
    td3_action_noise: float = 0.10

    sddp_iterations: int = 50
    sddp_validation_every: int = 10
    # Two scenarios keep every convex quadratic stage model below the
    # 200-variable limit of the reproducibility machine's Gurobi license.
    sddp_noise_scenarios: int = 2

    cfa_lookahead_horizon: int = 12
    cfa_iterations: int = 100
    cfa_batch_size: int = 10
    cfa_validation_every: int = 20
    cfa_learning_rates: tuple[float, ...] = (0.02, 0.05)
    cfa_smoothing_radius: float = 0.05
    cfa_rms_decay: float = 0.90
    cfa_theta_lower: float = 0.25
    cfa_theta_upper: float = 1.75

    latency_batch_size: int = 256
    latency_repetitions: int = 100
    device: str = "cpu"
    rl_device: str = "cpu"
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
            raise ValueError("Every data interface must have an independent seed.")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("Training seeds must be unique.")
        if not self.smoke and len(self.train_seeds) != 5:
            raise ValueError("Formal comparisons require exactly five training seeds.")
        if self.direct_iterations * self.direct_batch_size != self.n_fit:
            raise ValueError("Direct policies must consume every fit path exactly once.")
        if self.cfa_iterations * self.cfa_batch_size != self.n_fit:
            raise ValueError("CFA must consume every fit path exactly once per candidate.")
        positive = (
            self.n_reservoirs,
            self.horizon,
            self.shock_factors,
            self.n_fit,
            self.n_validation,
            self.n_test,
            self.n_calibration,
            self.n_evaluation,
            self.direct_iterations,
            self.direct_batch_size,
            self.direct_validation_every,
            self.mlp_width,
            self.graph_width,
            self.network_depth,
            self.sddp_iterations,
            self.sddp_validation_every,
            self.sddp_noise_scenarios,
            self.latency_batch_size,
            self.latency_repetitions,
        )
        if any(value < 1 for value in positive):
            raise ValueError("All dimensions, sample sizes, and iteration counts must be positive.")
        rates = (
            *self.affine_learning_rates,
            *self.polynomial_learning_rates,
            *self.mlp_learning_rates,
            *self.graph_learning_rates,
            *self.rl_learning_rates_ppo,
            *self.rl_learning_rates_sac,
            *self.rl_learning_rates_td3,
            *self.cfa_learning_rates,
        )
        if min(rates) <= 0.0:
            raise ValueError("All pre-registered learning rates must be positive.")

    @classmethod
    def smoke_protocol(
        cls,
        device: str = "cpu",
        rl_device: str = "cpu",
    ) -> "FullComparisonProtocol":
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
            direct_iterations=4,
            direct_batch_size=4,
            direct_validation_every=2,
            mlp_width=16,
            graph_width=16,
            rl_validation_every_episodes=4,
            rl_network_width=16,
            rl_network_depth=2,
            ppo_rollout_episodes=4,
            ppo_epochs=2,
            ppo_batch_size=32,
            replay_buffer_size=256,
            learning_starts=4,
            off_policy_batch_size=16,
            cfa_lookahead_horizon=3,
            cfa_iterations=2,
            cfa_batch_size=8,
            cfa_validation_every=1,
            sddp_iterations=2,
            sddp_validation_every=1,
            sddp_noise_scenarios=2,
            latency_batch_size=4,
            latency_repetitions=3,
            device=device,
            rl_device=rl_device,
            smoke=True,
        )

    def with_devices(self, device: str, rl_device: str) -> "FullComparisonProtocol":
        return replace(self, device=device, rl_device=rl_device)

    def experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig.make(
            self.profile,
            self.n_reservoirs,
            self.horizon,
            self.shock_factors,
        )

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

    def cfa_config(self) -> CFAConfig:
        return CFAConfig(
            lookahead_horizon=self.cfa_lookahead_horizon,
            iterations=self.cfa_iterations,
            batch_size=self.cfa_batch_size,
            validation_every=self.cfa_validation_every,
            learning_rates=self.cfa_learning_rates,
            smoothing_radius=self.cfa_smoothing_radius,
            rms_decay=self.cfa_rms_decay,
            theta_lower=self.cfa_theta_lower,
            theta_upper=self.cfa_theta_upper,
        )

    def sddp_config(self) -> SDDPConfig:
        return SDDPConfig(
            iterations=self.sddp_iterations,
            validation_every=self.sddp_validation_every,
            noise_scenarios=self.sddp_noise_scenarios,
        )

    def rl_config(self, algorithm: str, learning_rate: float) -> RLConfig:
        if algorithm not in {"ppo", "sac", "td3"}:
            raise ValueError(f"Unknown RL algorithm: {algorithm}")
        ppo_rate = (
            learning_rate if algorithm == "ppo" else self.rl_learning_rates_ppo[0]
        )
        sac_rate = (
            learning_rate if algorithm == "sac" else self.rl_learning_rates_sac[0]
        )
        td3_rate = (
            learning_rate if algorithm == "td3" else self.rl_learning_rates_td3[0]
        )
        return RLConfig(
            reward_scale=self.rl_reward_scale,
            gamma=self.rl_gamma,
            validation_every_episodes=self.rl_validation_every_episodes,
            network_width=self.rl_network_width,
            network_depth=self.rl_network_depth,
            ppo_learning_rate=ppo_rate,
            ppo_rollout_episodes=self.ppo_rollout_episodes,
            ppo_epochs=self.ppo_epochs,
            ppo_batch_size=self.ppo_batch_size,
            sac_learning_rate=sac_rate,
            td3_learning_rate=td3_rate,
            replay_buffer_size=self.replay_buffer_size,
            learning_starts=self.learning_starts,
            off_policy_batch_size=self.off_policy_batch_size,
            gradient_steps=self.gradient_steps,
            off_policy_n_steps=self.off_policy_n_steps,
            td3_action_noise=self.td3_action_noise,
            device=self.rl_device,
        )

    def direct_learning_rates(self, policy: str) -> tuple[float, ...]:
        return {
            "projected_affine": self.affine_learning_rates,
            "projected_polynomial": self.polynomial_learning_rates,
            "projected_mlp_pathwise": self.mlp_learning_rates,
            "projected_graph_neural_global": self.graph_learning_rates,
        }[policy]

    @property
    def iterations(self) -> int:
        """Compatibility with the shared direct-policy trainer."""

        return self.direct_iterations

    @property
    def batch_size(self) -> int:
        return self.direct_batch_size

    @property
    def validation_every(self) -> int:
        return self.direct_validation_every

    def rl_learning_rates(self, algorithm: str) -> tuple[float, ...]:
        return {
            "ppo": self.rl_learning_rates_ppo,
            "sac": self.rl_learning_rates_sac,
            "td3": self.rl_learning_rates_td3,
        }[algorithm]
