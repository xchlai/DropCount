from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from utils import stable_log1mexp


@dataclass
class BaselineEstimate:
    estimate: float
    saturation: bool
    success: bool
    details: Dict[str, float]


def naive_equal_volume_estimate(labels: np.ndarray, max_copy_cap: float = 1e6, eps: float = 1e-12) -> BaselineEstimate:
    labels = np.asarray(labels, dtype=np.float64)
    n = labels.size
    k = float(labels.sum())
    z = n - k
    if k <= 0:
        return BaselineEstimate(estimate=0.0, saturation=False, success=True, details={"positive_count": k})
    if z <= 0:
        return BaselineEstimate(
            estimate=float(max_copy_cap),
            saturation=True,
            success=True,
            details={"positive_count": k, "negative_fraction": 0.0},
        )
    negative_fraction = max(z / n, eps)
    estimate = -n * np.log(negative_fraction)
    estimate = float(min(estimate, max_copy_cap))
    return BaselineEstimate(
        estimate=estimate,
        saturation=estimate >= max_copy_cap,
        success=True,
        details={"positive_count": k, "negative_fraction": negative_fraction},
    )


def volume_aware_log_likelihood(n_total: float, f: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    if n_total < 0:
        return -np.inf
    f = np.asarray(f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.clip(n_total * f, 0.0, 1e12)
    positive_term = y * stable_log1mexp(np.maximum(x, eps))
    negative_term = (1.0 - y) * (-x)
    return float(np.sum(positive_term + negative_term))


def volume_aware_mle_estimate(
    volume_fractions: np.ndarray,
    labels: np.ndarray,
    max_copy_cap: float = 1e6,
    eps: float = 1e-12,
    search_upper_multiplier: float = 20.0,
    return_curve: bool = False,
) -> BaselineEstimate:
    f = np.asarray(volume_fractions, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(f)
    positive_count = float(y.sum())
    if positive_count <= 0:
        return BaselineEstimate(estimate=0.0, saturation=False, success=True, details={"positive_count": 0.0})
    if positive_count >= n:
        return BaselineEstimate(
            estimate=float(max_copy_cap),
            saturation=True,
            success=True,
            details={"positive_count": positive_count},
        )

    naive_seed = -n * np.log(max((n - positive_count) / n, eps))
    upper = max(max_copy_cap / 100.0, naive_seed * search_upper_multiplier, 10.0)
    upper = min(max_copy_cap, upper)

    def objective(x: float) -> float:
        return -volume_aware_log_likelihood(x, f, y, eps=eps)

    result = minimize_scalar(objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-6})
    estimate = float(np.clip(result.x, 0.0, max_copy_cap))
    curvature = _approximate_curvature(estimate, f, y, eps=eps)
    details: Dict[str, float] = {
        "positive_count": positive_count,
        "log_likelihood": float(-result.fun),
        "curvature": float(curvature),
        "approx_std": float(np.sqrt(1.0 / max(curvature, eps))),
    }
    if return_curve:
        grid = np.linspace(0.0, upper, 128)
        ll = [volume_aware_log_likelihood(v, f, y, eps=eps) for v in grid]
        details["curve_min_x"] = float(grid[int(np.argmax(ll))])
    return BaselineEstimate(
        estimate=estimate,
        saturation=estimate >= max_copy_cap,
        success=bool(result.success),
        details=details,
    )


def _approximate_curvature(n_total: float, f: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    h = max(1e-3, 1e-3 * max(1.0, n_total))
    center = volume_aware_log_likelihood(n_total, f, y, eps=eps)
    left = volume_aware_log_likelihood(max(0.0, n_total - h), f, y, eps=eps)
    right = volume_aware_log_likelihood(n_total + h, f, y, eps=eps)
    second = (left - 2.0 * center + right) / (h * h)
    return max(-second, eps)


def log_likelihood_curve(volume_fractions: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.asarray([volume_aware_log_likelihood(x, volume_fractions, labels) for x in grid], dtype=np.float64)
