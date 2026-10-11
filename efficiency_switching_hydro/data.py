"""Independent context-free data interfaces for efficiency-switching hydro."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DataConfig, ExperimentConfig


@dataclass(frozen=True)
class Dataset:
    name: str
    seed: int
    factors: np.ndarray
    idiosyncratic: np.ndarray
    demand_noise: np.ndarray
    inflow: np.ndarray
    demand: np.ndarray
    price: np.ndarray

    @property
    def n(self) -> int:
        return int(self.inflow.shape[0])

    def take(self, indices: np.ndarray, name: str = "fit_batch") -> "Dataset":
        return Dataset(
            name=name,
            seed=self.seed,
            factors=self.factors[indices],
            idiosyncratic=self.idiosyncratic[indices],
            demand_noise=self.demand_noise[indices],
            inflow=self.inflow[indices],
            demand=self.demand[indices],
            price=self.price[indices],
        )

    def to_torch(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        return {
            "inflow": torch.as_tensor(self.inflow, dtype=dtype, device=device),
            "demand": torch.as_tensor(self.demand, dtype=dtype, device=device),
            "price": torch.as_tensor(self.price, dtype=dtype, device=device),
        }


@dataclass(frozen=True)
class DataSplits:
    fit: Dataset
    validation: Dataset
    test: Dataset
    calibration: Dataset
    evaluation: Dataset

    def as_dict(self) -> dict[str, Dataset]:
        return {
            "fit": self.fit,
            "validation": self.validation,
            "test": self.test,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
        }


def generate_dataset(
    name: str,
    n: int,
    seed: int,
    config: ExperimentConfig,
) -> Dataset:
    rng = np.random.default_rng(seed)
    standard_factors = np.clip(
        rng.standard_normal((n, config.T, config.shock_factors)), -4.0, 4.0
    )
    factors = config.factor_std * standard_factors
    idiosyncratic = rng.uniform(
        config.idiosyncratic_low,
        config.idiosyncratic_high,
        size=(n, config.T, config.n_reservoirs),
    )
    demand_noise = config.demand_noise_std * np.clip(
        rng.standard_normal((n, config.T)), -4.0, 4.0
    )
    inflow_factor = np.einsum(
        "ntf,rf->ntr", factors, np.asarray(config.factor_loadings)
    )
    demand_factor = np.einsum(
        "ntf,f->nt", factors, np.asarray(config.demand_factor_loadings)
    )
    inflow = np.maximum(
        0.0,
        config.inflow_mean()[None, :, :] + inflow_factor + idiosyncratic,
    )
    demand = np.maximum(
        0.0,
        config.demand_mean()[None, :] + demand_factor + demand_noise,
    )
    price = np.broadcast_to(config.price(), (n, config.T)).copy()
    return Dataset(
        name=name,
        seed=seed,
        factors=factors.astype(np.float32),
        idiosyncratic=idiosyncratic.astype(np.float32),
        demand_noise=demand_noise.astype(np.float32),
        inflow=inflow.astype(np.float32),
        demand=demand.astype(np.float32),
        price=price.astype(np.float32),
    )


def generate_splits(data: DataConfig, config: ExperimentConfig) -> DataSplits:
    return DataSplits(
        fit=generate_dataset("fit", data.n_fit, data.fit_seed, config),
        validation=generate_dataset(
            "validation", data.n_validation, data.validation_seed, config
        ),
        test=generate_dataset("test", data.n_test, data.test_seed, config),
        calibration=generate_dataset(
            "calibration", data.n_calibration, data.calibration_seed, config
        ),
        evaluation=generate_dataset(
            "evaluation", data.n_evaluation, data.evaluation_seed, config
        ),
    )
