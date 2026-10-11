"""Configuration for the context-free efficiency-switching hydro experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PROFILES = (
    "linear",
    "single_band",
    "double_band",
    "double_band_network",
    "double_band_full",
    "decision_switching",
)

RL_ALGORITHMS = ("ppo", "sac", "td3")


def _routing_network(n_reservoirs: int, branched: bool) -> np.ndarray:
    routing = np.zeros((n_reservoirs, n_reservoirs), dtype=np.float64)
    if not branched:
        for source in range(n_reservoirs - 1):
            routing[source, source + 1] = 0.90 + 0.02 * (source % 3)
        return routing

    first_merge = n_reservoirs // 2
    merge_nodes = max(n_reservoirs - first_merge, 1)
    for source in range(first_merge):
        target = first_merge + min(source // 2, merge_nodes - 1)
        if target > source:
            routing[source, target] = 0.88 + 0.02 * (source % 3)
    for source in range(first_merge, n_reservoirs - 1):
        routing[source, source + 1] = 0.90 + 0.02 * (source % 3)
    return routing


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    T: int
    n_reservoirs: int
    shock_factors: int
    storage_capacity: tuple[float, ...]
    initial_storage: tuple[float, ...]
    terminal_target: tuple[float, ...]
    release_capacity: tuple[float, ...]
    generation_coefficient: tuple[float, ...]
    shared_release_capacity: float
    routing: tuple[tuple[float, ...], ...]
    downstream_index: tuple[int, ...]
    routing_delay: int
    spill_cost: tuple[float, ...]
    terminal_water_value: tuple[float, ...]
    base_inflow: tuple[float, ...]
    inflow_amplitude: tuple[float, ...]
    inflow_phase: tuple[float, ...]
    factor_loadings: tuple[tuple[float, ...], ...]
    demand_factor_loadings: tuple[float, ...]
    factor_std: float
    idiosyncratic_low: float
    idiosyncratic_high: float
    demand_noise_std: float
    base_demand: float
    demand_amplitude: float
    base_price: float
    price_amplitude: float
    head_base: tuple[float, ...]
    head_storage_gain: tuple[float, ...]
    head_downstream_loss: tuple[float, ...]
    head_release_loss: tuple[float, ...]
    efficiency_floor: tuple[float, ...]
    band1_amplitude: tuple[float, ...]
    band1_center: tuple[float, ...]
    band1_storage_shift: tuple[float, ...]
    band1_width: tuple[float, ...]
    band2_amplitude: tuple[float, ...]
    band2_center: tuple[float, ...]
    band2_storage_shift: tuple[float, ...]
    band2_width: tuple[float, ...]
    evaporation_rate: tuple[float, ...]
    ramp_fraction: float
    ramp_cost: float
    thermal_quadratic_cost: float
    peaker_threshold: float
    peaker_quadratic_cost: float
    flood_threshold: tuple[float, ...]
    low_storage_threshold: tuple[float, ...]
    flood_cost: float
    low_storage_cost: float
    storage_feedback_gain: float = 0.10

    @classmethod
    def make(
        cls,
        profile: str,
        n_reservoirs: int = 12,
        T: int = 48,
        shock_factors: int = 4,
    ) -> "ExperimentConfig":
        profile = profile.strip().lower()
        if profile not in PROFILES:
            raise ValueError(f"Unknown efficiency-switching profile: {profile}")
        if n_reservoirs < 2 or T < 2 or shock_factors < 1:
            raise ValueError("The network, horizon, and shock dimension must be positive.")

        r = np.arange(n_reservoirs, dtype=np.float64)
        position = r / max(n_reservoirs - 1, 1)
        storage = 165.0 + 72.0 * position + 14.0 * (r % 3.0)
        release = (0.22 + 0.035 * (r % 2.0)) * storage
        generation = 0.92 + 0.20 * (0.5 + 0.5 * np.cos(1.31 * r))
        nonlinear = profile != "linear"
        double_band = profile in {
            "double_band",
            "double_band_network",
            "double_band_full",
            "decision_switching",
        }
        network = profile in {
            "double_band_network",
            "double_band_full",
            "decision_switching",
        }
        full = profile == "double_band_full"
        switching = profile == "decision_switching"
        routing = _routing_network(n_reservoirs, branched=network)
        downstream = np.full(n_reservoirs, -1, dtype=np.int64)
        for source in range(n_reservoirs):
            targets = np.flatnonzero(routing[source] > 0.0)
            if len(targets):
                downstream[source] = int(targets[0])

        base_inflow = 25.0 - 12.0 * position + 2.0 * np.sin(1.17 * r)
        rated_release = float(generation @ release)
        shared_capacity = 0.72 * rated_release
        factor_index = np.arange(shock_factors, dtype=np.float64)
        loadings = np.empty((n_reservoirs, shock_factors), dtype=np.float64)
        for reservoir in range(n_reservoirs):
            row = (
                0.65 * np.cos(0.39 * (reservoir + 1) * (factor_index + 1))
                + 0.35 * np.sin(0.71 * (reservoir + 2) * (factor_index + 1))
            )
            loadings[reservoir] = (
                (1.0 - 0.40 * position[reservoir])
                * row
                / max(float(np.linalg.norm(row)), 1e-12)
            )
        demand_loading = -np.cos(0.57 * (factor_index + 1))
        demand_loading /= max(float(np.linalg.norm(demand_loading)), 1e-12)

        def values(array: np.ndarray) -> tuple[float, ...]:
            return tuple(float(value) for value in array)

        zeros = np.zeros(n_reservoirs, dtype=np.float64)
        ones = np.ones(n_reservoirs, dtype=np.float64)
        head_base = 0.74 + 0.04 * np.cos(0.83 * r) if nonlinear else ones
        head_storage = 0.46 + 0.05 * np.sin(0.61 * r) if nonlinear else zeros
        head_downstream = (
            0.10 + 0.025 * (r % 3.0) if network else zeros
        )
        head_release = 0.075 + 0.015 * (r % 2.0) if nonlinear else zeros
        efficiency_floor = 0.68 + 0.025 * np.sin(0.91 * r) if nonlinear else ones
        band1_amplitude = 0.22 + 0.02 * np.cos(0.77 * r) if nonlinear else zeros
        band2_amplitude = (
            0.20 + 0.025 * np.sin(0.69 * r) if double_band else zeros
        )
        if switching:
            head_base = 0.78 + 0.035 * np.cos(0.83 * r)
            head_storage = 0.50 + 0.06 * np.sin(0.61 * r)
            head_downstream = 0.28 + 0.04 * (r % 3.0)
            head_release = 0.08 + 0.015 * (r % 2.0)
            efficiency_floor = 0.36 + 0.02 * np.sin(0.91 * r)
            band1_amplitude = 0.54 + 0.025 * np.cos(0.77 * r)
            band2_amplitude = 0.50 + 0.03 * np.sin(0.69 * r)

        return cls(
            profile=profile,
            T=T,
            n_reservoirs=n_reservoirs,
            shock_factors=shock_factors,
            storage_capacity=values(storage),
            initial_storage=values((0.64 if switching else 0.68) * storage),
            terminal_target=values((0.62 if switching else 0.58) * storage),
            release_capacity=values(release),
            generation_coefficient=values(generation),
            shared_release_capacity=shared_capacity,
            routing=tuple(tuple(float(value) for value in row) for row in routing),
            downstream_index=tuple(int(value) for value in downstream),
            routing_delay=2 if network else 0,
            spill_cost=values(
                (0.70 + 0.30 * position)
                if switching
                else (0.16 + 0.12 * position)
            ),
            terminal_water_value=values(
                (19.0 - 6.0 * position)
                if switching
                else (15.0 - 5.5 * position)
            ),
            base_inflow=values(base_inflow),
            inflow_amplitude=values(0.45 * base_inflow),
            inflow_phase=values(1.35 * position),
            factor_loadings=tuple(tuple(float(value) for value in row) for row in loadings),
            demand_factor_loadings=values(demand_loading),
            factor_std=8.5 if switching else 5.5,
            idiosyncratic_low=-6.0 if switching else -3.5,
            idiosyncratic_high=6.0 if switching else 3.5,
            demand_noise_std=(0.060 if switching else 0.045) * rated_release,
            base_demand=(0.43 if switching else 0.54) * rated_release,
            demand_amplitude=(0.19 if switching else 0.15) * rated_release,
            base_price=9.0,
            price_amplitude=2.2,
            head_base=values(head_base),
            head_storage_gain=values(head_storage),
            head_downstream_loss=values(head_downstream),
            head_release_loss=values(head_release),
            efficiency_floor=values(efficiency_floor),
            band1_amplitude=values(band1_amplitude),
            band1_center=values(
                (0.25 if switching else 0.30) + 0.025 * np.sin(0.73 * r)
            ),
            band1_storage_shift=values(
                (0.13 if switching else 0.10) + 0.015 * np.cos(0.51 * r)
            ),
            band1_width=values(
                (0.060 + 0.008 * (r % 2.0))
                if switching
                else (0.105 + 0.015 * (r % 2.0))
            ),
            band2_amplitude=values(band2_amplitude),
            band2_center=values(
                (0.79 if switching else 0.72) + 0.025 * np.cos(0.67 * r)
            ),
            band2_storage_shift=values(
                (-0.15 if switching else -0.12) + 0.015 * np.sin(0.47 * r)
            ),
            band2_width=values(
                (0.070 + 0.010 * (r % 3.0) / 2.0)
                if switching
                else (0.12 + 0.015 * (r % 3.0) / 2.0)
            ),
            evaporation_rate=values(
                0.012 + 0.008 * position
                if switching
                else (0.020 + 0.015 * position if full else zeros)
            ),
            ramp_fraction=0.50 if switching else (0.35 if full else 1.0),
            ramp_cost=1.5 if switching else (3.0 if full else 0.0),
            thermal_quadratic_cost=0.003 if switching else (0.004 if full else 0.0),
            peaker_threshold=(0.10 if switching else 0.12) * rated_release
            if (switching or full)
            else rated_release,
            peaker_quadratic_cost=0.020 if switching else (0.025 if full else 0.0),
            flood_threshold=values((0.76 if switching else 0.86) * storage),
            low_storage_threshold=values((0.30 if switching else 0.24) * storage),
            flood_cost=0.080 if switching else (0.012 if full else 0.0),
            low_storage_cost=0.040 if switching else (0.010 if full else 0.0),
        )

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError("Invalid efficiency-switching profile.")
        reservoir_fields = (
            self.storage_capacity,
            self.initial_storage,
            self.terminal_target,
            self.release_capacity,
            self.generation_coefficient,
            self.routing,
            self.downstream_index,
            self.spill_cost,
            self.terminal_water_value,
            self.base_inflow,
            self.inflow_amplitude,
            self.inflow_phase,
            self.factor_loadings,
            self.head_base,
            self.head_storage_gain,
            self.head_downstream_loss,
            self.head_release_loss,
            self.efficiency_floor,
            self.band1_amplitude,
            self.band1_center,
            self.band1_storage_shift,
            self.band1_width,
            self.band2_amplitude,
            self.band2_center,
            self.band2_storage_shift,
            self.band2_width,
            self.evaporation_rate,
            self.flood_threshold,
            self.low_storage_threshold,
        )
        if any(len(values) != self.n_reservoirs for values in reservoir_fields):
            raise ValueError("Every reservoir-specific field must match n_reservoirs.")
        if any(len(row) != self.n_reservoirs for row in self.routing):
            raise ValueError("routing must be square.")
        if any(len(row) != self.shock_factors for row in self.factor_loadings):
            raise ValueError("Factor loadings must match shock_factors.")
        if len(self.demand_factor_loadings) != self.shock_factors:
            raise ValueError("Demand factor loadings must match shock_factors.")
        capacity = np.asarray(self.storage_capacity)
        initial = np.asarray(self.initial_storage)
        release = np.asarray(self.release_capacity)
        if np.any(capacity <= 0.0) or np.any(release <= 0.0):
            raise ValueError("Storage and release capacities must be positive.")
        if np.any(initial < 0.0) or np.any(initial > capacity):
            raise ValueError("Initial storage is infeasible.")
        if not 0.0 < self.ramp_fraction <= 1.0:
            raise ValueError("ramp_fraction must lie in (0, 1].")
        if np.any(np.asarray(self.band1_width) <= 0.0) or np.any(
            np.asarray(self.band2_width) <= 0.0
        ):
            raise ValueError("Efficiency-band widths must be positive.")

    def inflow_mean(self) -> np.ndarray:
        stages = np.arange(self.T, dtype=np.float64)[:, None]
        return np.maximum(
            0.0,
            np.asarray(self.base_inflow)[None, :]
            + np.asarray(self.inflow_amplitude)[None, :]
            * np.sin(2.0 * np.pi * stages / self.T + np.asarray(self.inflow_phase)[None, :]),
        )

    def demand_mean(self) -> np.ndarray:
        stages = np.arange(self.T, dtype=np.float64)
        return np.maximum(
            20.0,
            self.base_demand
            + self.demand_amplitude * np.sin(2.0 * np.pi * stages / self.T + 0.55),
        )

    def price(self) -> np.ndarray:
        stages = np.arange(self.T, dtype=np.float64)
        return np.maximum(
            1.0,
            self.base_price
            + self.price_amplitude * np.cos(2.0 * np.pi * stages / self.T - 0.30),
        )

    def planned_storage(self) -> np.ndarray:
        fraction = (np.arange(self.T, dtype=np.float64) + 1.0) / self.T
        initial = np.asarray(self.initial_storage)
        target = np.asarray(self.terminal_target)
        return initial[None, :] + fraction[:, None] * (target - initial)[None, :]

    def reference_release(self) -> np.ndarray:
        release = np.asarray(self.release_capacity)
        generation = np.asarray(self.generation_coefficient)
        weights = release / float(generation @ release)
        effective_productivity = 1.0 if self.profile == "linear" else 0.82
        proposal = self.demand_mean()[:, None] * weights[None, :] / effective_productivity
        proposal = np.minimum(proposal, release)
        rated = proposal @ generation
        scale = np.minimum(1.0, self.shared_release_capacity / np.maximum(rated, 1e-12))
        return proposal * scale[:, None]

    def queue_capacity(self) -> np.ndarray:
        if self.routing_delay == 0:
            return np.zeros((0, self.n_reservoirs), dtype=np.float64)
        routed = np.asarray(self.routing).T @ np.asarray(self.release_capacity)
        return np.broadcast_to(routed, (self.routing_delay, self.n_reservoirs)).copy()


@dataclass(frozen=True)
class DataConfig:
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


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 500
    batch_size: int = 2
    validation_every: int = 10
    train_seeds: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2))
    lr_affine: float = 5e-4
    lr_polynomial: float = 1.5e-4
    lr_neural: float = 3e-4
    lr_mlp: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 10.0
    device: str = "cpu"


@dataclass(frozen=True)
class RLConfig:
    """Training settings shared by the model-free RL comparators."""

    reward_scale: float = 1000.0
    gamma: float = 1.0
    validation_every_episodes: int = 50
    network_width: int = 256
    network_depth: int = 2
    ppo_learning_rate: float = 3e-4
    ppo_rollout_episodes: int = 10
    ppo_epochs: int = 10
    ppo_batch_size: int = 256
    sac_learning_rate: float = 3e-4
    td3_learning_rate: float = 1e-3
    replay_buffer_size: int = 100_000
    learning_starts: int = 1_000
    off_policy_batch_size: int = 256
    gradient_steps: int = 1
    off_policy_n_steps: int = 3
    td3_action_noise: float = 0.10
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1].")
        positive_integers = (
            self.validation_every_episodes,
            self.network_width,
            self.network_depth,
            self.ppo_rollout_episodes,
            self.ppo_epochs,
            self.ppo_batch_size,
            self.replay_buffer_size,
            self.learning_starts,
            self.off_policy_batch_size,
            self.gradient_steps,
            self.off_policy_n_steps,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("RL iteration, network, and batch settings must be positive.")
        if min(
            self.ppo_learning_rate,
            self.sac_learning_rate,
            self.td3_learning_rate,
        ) <= 0.0:
            raise ValueError("RL learning rates must be positive.")
        if self.td3_action_noise < 0.0:
            raise ValueError("td3_action_noise must be nonnegative.")


@dataclass(frozen=True)
class SDDPConfig:
    iterations: int = 50
    validation_every: int = 10
    noise_scenarios: int = 3

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.validation_every < 1:
            raise ValueError("SDDP iteration settings must be positive.")
        if self.noise_scenarios < 1:
            raise ValueError("noise_scenarios must be positive.")
