from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DistributionConfig:
    name: str = "lognormal"
    cv: float = 0.2
    train_names: List[str] = field(default_factory=lambda: ["monodisperse", "lognormal", "gamma", "two_point"])
    eval_names: List[str] = field(
        default_factory=lambda: [
            "monodisperse",
            "lognormal",
            "gamma",
            "truncated_normal",
            "two_point",
            "binomial_mapped",
        ]
    )
    cv_values: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    lognormal_sigma_cap: float = 2.0
    binomial_k: int = 10
    positive_floor: float = 1e-6


@dataclass
class SimulationConfig:
    n_droplets: int = 1024
    true_copy_range: List[int] = field(default_factory=lambda: [0, 200000])
    copy_sampling_mode: str = "log_uniform_integer"
    simulation_mode: str = "fixed_total_multinomial"
    distributions: DistributionConfig = field(default_factory=DistributionConfig)
    random_seed: int = 123
    false_positive_rate_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
    train_samples_per_epoch: int = 1024
    val_samples: int = 256
    test_samples_per_combo: int = 64
    test_n_droplets: List[int] = field(default_factory=lambda: [1000, 5000])
    copy_bins: List[int] = field(default_factory=lambda: [0, 10, 100, 1000, 10000, 100000, 200000])


@dataclass
class ModelConfig:
    model_type: str = "perceiver"
    input_dim: int = 10
    hidden_dim: int = 128
    latent_dim: int = 128
    num_latents: int = 16
    num_heads: int = 4
    num_self_attn_layers: int = 3
    dropout: float = 0.1
    fourier_features: int = 2
    use_rmsnorm: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    loss_name: str = "huber_log"
    linear_loss_weight: float = 0.05
    huber_delta: float = 0.5
    early_stopping_patience: int = 5
    num_workers: int = 0
    amp: bool = True
    device: str = "auto"


@dataclass
class OutputConfig:
    run_root: str = "outputs"
    run_name: str = "default_run"
    save_validation_dataset: bool = True


@dataclass
class BaselineConfig:
    max_copy_cap: float = 1e6
    eps: float = 1e-12
    mle_search_upper_multiplier: float = 20.0


@dataclass
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output.run_root) / self.output.run_name


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    config = ExperimentConfig()
    merged = asdict(config)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        merged = _deep_update(merged, loaded)
    if overrides:
        merged = _deep_update(merged, overrides)
    return ExperimentConfig(
        simulation=SimulationConfig(
            **{k: v for k, v in merged["simulation"].items() if k != "distributions"},
            distributions=DistributionConfig(**merged["simulation"]["distributions"]),
        ),
        model=ModelConfig(**merged["model"]),
        training=TrainingConfig(**merged["training"]),
        output=OutputConfig(**merged["output"]),
        baselines=BaselineConfig(**merged["baselines"]),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
