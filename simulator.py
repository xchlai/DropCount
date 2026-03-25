from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from config import ExperimentConfig


ArrayLike = np.ndarray
DistributionCallable = Callable[[np.random.Generator, int, float], ArrayLike]


@dataclass
class Sample:
    volume_fractions: np.ndarray
    labels: np.ndarray
    true_total_copies: int
    counts: np.ndarray
    metadata: Dict[str, Any]
    false_positive_rate: float


def _lognormal_sigma_from_cv(cv: float) -> float:
    if cv <= 0:
        return 0.0
    return math.sqrt(math.log1p(cv * cv))


class VolumeSampler:
    def __init__(self, positive_floor: float = 1e-6, binomial_k: int = 10) -> None:
        self.positive_floor = positive_floor
        self.binomial_k = binomial_k
        self.custom_callables: Dict[str, DistributionCallable] = {}

    def register(self, name: str, fn: DistributionCallable) -> None:
        self.custom_callables[name] = fn

    def sample(self, rng: np.random.Generator, n: int, distribution: str, cv: float) -> np.ndarray:
        if distribution == "custom_callable":
            if distribution not in self.custom_callables:
                raise ValueError("custom_callable requested but no callable has been registered")
            volumes = self.custom_callables[distribution](rng, n, cv)
        elif distribution == "monodisperse":
            volumes = np.ones(n, dtype=np.float64)
        elif distribution == "lognormal":
            sigma = _lognormal_sigma_from_cv(cv)
            mu = -0.5 * sigma * sigma
            volumes = rng.lognormal(mean=mu, sigma=sigma, size=n)
        elif distribution == "gamma":
            if cv <= 0:
                volumes = np.ones(n, dtype=np.float64)
            else:
                shape = max(1.0 / (cv * cv), 1e-3)
                scale = 1.0 / shape
                volumes = rng.gamma(shape=shape, scale=scale, size=n)
        elif distribution == "truncated_normal":
            sigma = max(cv, 1e-6)
            volumes = self._sample_truncated_normal(rng, n=n, mean=1.0, std=sigma)
        elif distribution == "uniform":
            if cv <= 0:
                volumes = np.ones(n, dtype=np.float64)
            else:
                half_width = min(math.sqrt(3.0) * cv, 0.95)
                low = max(self.positive_floor, 1.0 - half_width)
                high = 1.0 + half_width
                volumes = rng.uniform(low, high, size=n)
        elif distribution == "two_point":
            if cv <= 0:
                volumes = np.ones(n, dtype=np.float64)
            else:
                delta = min(cv, 0.95)
                choices = np.array([1.0 - delta, 1.0 + delta], dtype=np.float64)
                volumes = rng.choice(choices, size=n)
        elif distribution == "binomial_mapped":
            p = 0.5
            k = self.binomial_k
            k_samples = rng.binomial(k, p, size=n)
            mean_k = k * p
            var_k = k * p * (1.0 - p)
            if cv <= 0 or var_k == 0:
                a, b = 1.0, 0.0
            else:
                b = cv / math.sqrt(var_k)
                a = max(self.positive_floor, 1.0 - b * mean_k)
                if a <= self.positive_floor:
                    b = 0.5 / max(mean_k, 1.0)
                    a = max(self.positive_floor, 1.0 - b * mean_k)
            volumes = a + b * k_samples.astype(np.float64)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        volumes = np.asarray(volumes, dtype=np.float64)
        if np.any(volumes <= 0):
            raise ValueError(f"Distribution {distribution} produced non-positive volumes")
        return volumes / volumes.sum()

    def _sample_truncated_normal(
        self,
        rng: np.random.Generator,
        n: int,
        mean: float,
        std: float,
        max_tries: int = 100,
    ) -> np.ndarray:
        samples = rng.normal(loc=mean, scale=std, size=n)
        mask = samples <= self.positive_floor
        tries = 0
        while np.any(mask) and tries < max_tries:
            samples[mask] = rng.normal(loc=mean, scale=std, size=int(mask.sum()))
            mask = samples <= self.positive_floor
            tries += 1
        samples = np.maximum(samples, self.positive_floor)
        return samples


class DropletDigitalPCRSimulator:
    def __init__(self, config: ExperimentConfig, custom_sampler: Optional[VolumeSampler] = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.simulation.random_seed)
        self.volume_sampler = custom_sampler or VolumeSampler(
            positive_floor=config.simulation.distributions.positive_floor,
            binomial_k=config.simulation.distributions.binomial_k,
        )

    def sample_false_positive_rate(self, rng: Optional[np.random.Generator] = None) -> float:
        rng = rng or self.rng
        lo, hi = self.config.simulation.false_positive_rate_range
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-12:
            return lo
        return float(rng.uniform(lo, hi))

    def sample_true_total_copies(self, rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or self.rng
        lo, hi = self.config.simulation.true_copy_range
        mode = self.config.simulation.copy_sampling_mode
        if mode == "uniform_integer":
            return int(rng.integers(lo, hi + 1))
        if mode == "log_uniform_integer":
            if hi <= 0:
                return 0
            shifted = 1
            low = math.log(max(lo + shifted, 1))
            high = math.log(hi + shifted)
            value = math.exp(rng.uniform(low, high)) - shifted
            return int(np.clip(round(value), lo, hi))
        if mode == "custom":
            probs = np.linspace(1, 2, hi - lo + 1, dtype=np.float64)
            probs /= probs.sum()
            return int(rng.choice(np.arange(lo, hi + 1), p=probs))
        raise ValueError(f"Unsupported copy sampling mode: {mode}")

    def generate_volume_fractions(
        self,
        n_droplets: Optional[int] = None,
        distribution_name: Optional[str] = None,
        cv: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or self.rng
        n_droplets = n_droplets or self.config.simulation.n_droplets
        distribution_name = distribution_name or self.config.simulation.distributions.name
        cv = self.config.simulation.distributions.cv if cv is None else cv
        return self.volume_sampler.sample(rng, n_droplets, distribution_name, cv)

    def allocate_counts_fixed_total(
        self,
        n_true: int,
        f: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or self.rng
        return rng.multinomial(n_true, f)

    def allocate_counts_poisson(
        self,
        n_true: int,
        f: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or self.rng
        lam = np.asarray(f, dtype=np.float64) * float(n_true)
        return rng.poisson(lam)

    def simulate_sample(
        self,
        n_true: Optional[int] = None,
        n_droplets: Optional[int] = None,
        distribution_name: Optional[str] = None,
        cv: Optional[float] = None,
        simulation_mode: Optional[str] = None,
        false_positive_rate: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Sample:
        rng = rng or self.rng
        n_true = self.sample_true_total_copies(rng) if n_true is None else int(n_true)
        f = self.generate_volume_fractions(n_droplets=n_droplets, distribution_name=distribution_name, cv=cv, rng=rng)
        sim_mode = simulation_mode or self.config.simulation.simulation_mode
        if sim_mode == "fixed_total_multinomial":
            counts = self.allocate_counts_fixed_total(n_true, f, rng)
        elif sim_mode == "poisson_loading":
            counts = self.allocate_counts_poisson(n_true, f, rng)
        else:
            raise ValueError(f"Unsupported simulation mode: {sim_mode}")
        fp_rate = self.sample_false_positive_rate(rng) if false_positive_rate is None else float(false_positive_rate)
        fp_rate = float(np.clip(fp_rate, 0.0, 1.0))
        labels = (counts > 0).astype(np.float32)
        if fp_rate > 0:
            negative_mask = counts == 0
            false_positives = rng.random(len(counts)) < fp_rate
            labels = np.where(negative_mask & false_positives, 1.0, labels).astype(np.float32)
        metadata = {
            "n_droplets": int(len(f)),
            "distribution_name": distribution_name or self.config.simulation.distributions.name,
            "cv": float(self.config.simulation.distributions.cv if cv is None else cv),
            "positive_ratio": float(labels.mean()),
            "saturation_ratio": float(labels.mean()),
            "simulation_mode": sim_mode,
            "false_positive_rate": fp_rate,
        }
        return Sample(
            volume_fractions=f.astype(np.float32),
            labels=labels.astype(np.float32),
            true_total_copies=n_true,
            counts=counts.astype(np.int64),
            metadata=metadata,
            false_positive_rate=fp_rate,
        )


def sample_to_tensor_dict(sample: Sample) -> Dict[str, Any]:
    return {
        "volume_fractions": torch.tensor(sample.volume_fractions, dtype=torch.float32),
        "labels": torch.tensor(sample.labels, dtype=torch.float32),
        "true_total_copies": torch.tensor(sample.true_total_copies, dtype=torch.float32),
        "counts": torch.tensor(sample.counts, dtype=torch.int64),
        "false_positive_rate": torch.tensor(sample.false_positive_rate, dtype=torch.float32),
        "mask": torch.ones_like(torch.tensor(sample.labels, dtype=torch.float32), dtype=torch.bool),
        "metadata": sample.metadata,
    }


class OnlineSimulationDataset(IterableDataset):
    def __init__(
        self,
        simulator: DropletDigitalPCRSimulator,
        num_samples: int,
        distributions: Optional[Sequence[str]] = None,
        cv_values: Optional[Sequence[float]] = None,
        n_droplets_choices: Optional[Sequence[int]] = None,
        seed_offset: int = 0,
    ) -> None:
        self.simulator = simulator
        self.num_samples = num_samples
        self.distributions = list(distributions or simulator.config.simulation.distributions.train_names)
        self.cv_values = list(cv_values or simulator.config.simulation.distributions.cv_values)
        self.n_droplets_choices = list(n_droplets_choices or [simulator.config.simulation.n_droplets])
        self.seed_offset = seed_offset

    def __iter__(self):
        if not hasattr(self, "_epoch_counter"):
            self._epoch_counter = 0
        self._epoch_counter += 1
        worker = torch.utils.data.get_worker_info()
        worker_seed = self.simulator.config.simulation.random_seed + self.seed_offset + self._epoch_counter * 1_000_003
        if worker is not None:
            worker_seed += worker.id
        rng = np.random.default_rng(worker_seed)
        for _ in range(self.num_samples):
            distribution = self.distributions[int(rng.integers(0, len(self.distributions)))]
            cv = float(self.cv_values[int(rng.integers(0, len(self.cv_values)))])
            n_droplets = int(self.n_droplets_choices[int(rng.integers(0, len(self.n_droplets_choices)))])
            sample = self.simulator.simulate_sample(
                n_droplets=n_droplets,
                distribution_name=distribution,
                cv=cv,
                rng=rng,
            )
            yield sample_to_tensor_dict(sample)


class FixedSimulationDataset(Dataset):
    def __init__(
        self,
        simulator: DropletDigitalPCRSimulator,
        num_samples: int,
        distributions: Optional[Sequence[str]] = None,
        cv_values: Optional[Sequence[float]] = None,
        n_droplets_choices: Optional[Sequence[int]] = None,
        seed_offset: int = 1000,
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        distributions = list(distributions or simulator.config.simulation.distributions.eval_names)
        cv_values = list(cv_values or simulator.config.simulation.distributions.cv_values)
        n_droplets_choices = list(n_droplets_choices or [simulator.config.simulation.n_droplets])
        rng = np.random.default_rng(simulator.config.simulation.random_seed + seed_offset)
        combos: List[tuple[str, float, int]] = [
            (dist, cv, n) for dist in distributions for cv in cv_values for n in n_droplets_choices
        ]
        for idx in range(num_samples):
            dist, cv, n_drop = combos[idx % len(combos)]
            sample = simulator.simulate_sample(
                n_droplets=n_drop,
                distribution_name=dist,
                cv=cv,
                rng=rng,
            )
            tensor_dict = sample_to_tensor_dict(sample)
            tensor_dict["metadata"] = {**tensor_dict["metadata"], "combo_index": idx % len(combos)}
            self.samples.append(tensor_dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def collate_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(item["volume_fractions"].shape[0] for item in batch)
    bsz = len(batch)
    volume_fractions = torch.zeros(bsz, max_len, dtype=torch.float32)
    labels = torch.zeros(bsz, max_len, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    counts = torch.zeros(bsz, max_len, dtype=torch.int64)
    true_total_copies = torch.zeros(bsz, dtype=torch.float32)
    false_positive_rate = torch.zeros(bsz, dtype=torch.float32)
    metadata: List[Dict[str, Any]] = []
    for i, item in enumerate(batch):
        n = item["volume_fractions"].shape[0]
        volume_fractions[i, :n] = item["volume_fractions"]
        labels[i, :n] = item["labels"]
        counts[i, :n] = item["counts"]
        mask[i, :n] = item.get("mask", torch.ones(n, dtype=torch.bool))
        true_total_copies[i] = item["true_total_copies"]
        false_positive_rate[i] = item["false_positive_rate"]
        metadata.append(item["metadata"])
    return {
        "volume_fractions": volume_fractions,
        "labels": labels,
        "counts": counts,
        "mask": mask,
        "true_total_copies": true_total_copies,
        "false_positive_rate": false_positive_rate,
        "metadata": metadata,
    }
