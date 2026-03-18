from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stable_log1mexp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    log2 = math.log(2.0)
    small = x <= log2
    out[small] = np.log(-np.expm1(-x[small]))
    out[~small] = np.log1p(-np.exp(-x[~small]))
    return out


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    weights = mask.float()
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)
    summed = (x * weights).sum(dim=dim)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return summed / denom


def masked_std(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int, eps: float = 1e-6) -> torch.Tensor:
    mean = masked_mean(x, mask, dim)
    centered = x - mean.unsqueeze(dim)
    if mask is None:
        var = centered.pow(2).mean(dim=dim)
    else:
        weights = mask.float()
        while weights.ndim < x.ndim:
            weights = weights.unsqueeze(-1)
        var = (centered.pow(2) * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(var + eps)


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)


def as_serializable_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")


def plot_training_curves(history: List[Dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_loss" in df:
        ax.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df:
        ax.plot(df["epoch"], df["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metrics_tables(metrics: Dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    for name, df in metrics.items():
        df.to_csv(Path(out_dir) / f"{name}.csv", index=False)
