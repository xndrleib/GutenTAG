"""Numerical helpers shared by capability analysis modules."""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np


def robust_scale(values: np.ndarray, *, fallback: float = 1.0) -> float:
    """Return a robust positive scale for one-dimensional or matrix data."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float(max(abs(fallback), 1e-8))
    flat = array.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float(max(abs(fallback), 1e-8))
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    std = float(np.std(finite))
    ptp = float(np.ptp(finite))
    return float(max(mad, std, 0.25 * ptp, abs(float(fallback)), 1e-8))


def safe_variance(values: np.ndarray) -> float:
    """Return a finite sample variance with lower bound."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return 1e-8
    var = float(np.var(array, ddof=1))
    if not math.isfinite(var):
        return 1e-8
    return float(max(var, 1e-8))


def safe_corrcoef(values: np.ndarray) -> np.ndarray:
    """Return a finite correlation matrix for rows as observations."""

    matrix = _as_2d(values)
    channels = int(matrix.shape[1])
    if matrix.shape[0] < 3 or channels < 2:
        return np.eye(channels, dtype=np.float64)
    scale = np.std(matrix, axis=0)
    active = scale > 1e-8
    corr = np.eye(channels, dtype=np.float64)
    if int(active.sum()) >= 2:
        active_matrix = matrix[:, active]
        sub = np.corrcoef(active_matrix, rowvar=False)
        sub = np.asarray(sub, dtype=np.float64)
        sub[~np.isfinite(sub)] = 0.0
        indices = np.flatnonzero(active)
        for i, src_i in enumerate(indices):
            for j, src_j in enumerate(indices):
                corr[int(src_i), int(src_j)] = float(sub[i, j])
    return corr


def safe_covariance(values: np.ndarray) -> np.ndarray:
    """Return a finite covariance matrix for rows as observations."""

    matrix = _as_2d(values)
    channels = int(matrix.shape[1])
    if matrix.shape[0] < 2:
        return np.eye(channels, dtype=np.float64) * 1e-8
    cov = np.cov(matrix, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov[~np.isfinite(cov)] = 0.0
    return cov.reshape(channels, channels)


def channel_subsets(channels: int | Sequence[int], max_size: int) -> list[tuple[int, ...]]:
    """Return deterministic channel subsets up to ``max_size``."""

    if isinstance(channels, int):
        universe = tuple(range(int(channels)))
    else:
        universe = tuple(sorted({int(channel) for channel in channels}))
    subsets: list[tuple[int, ...]] = []
    limit = max(1, min(int(max_size), len(universe)))
    for size in range(1, limit + 1):
        subsets.extend(tuple(combo) for combo in itertools.combinations(universe, size))
    return subsets


def contiguous_windows(length: int, window_length: int, *, max_windows: int, stride_fraction: float) -> list[tuple[int, int]]:
    """Return a bounded deterministic scan library of contiguous windows."""

    n = int(length)
    w = max(1, min(int(window_length), n))
    if w >= n:
        return [(0, n)]
    stride = max(1, int(round(w * float(stride_fraction))))
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] != n - w:
        starts.append(n - w)
    if len(starts) <= int(max_windows):
        return [(int(start), int(start + w)) for start in starts]
    indices = np.linspace(0, len(starts) - 1, int(max_windows), dtype=int)
    selected = [starts[int(index)] for index in np.unique(indices)]
    return [(int(start), int(start + w)) for start in selected]


def context_indices(
    length: int,
    start: int,
    end: int,
    *,
    multiplier: int = 4,
    min_points: int = 8,
) -> np.ndarray:
    """Return indices around a support interval, excluding that interval."""

    n = int(length)
    left = max(0, int(start))
    right = min(n, int(end))
    width = max(1, right - left)
    radius = max(int(min_points), int(multiplier) * width)
    left_idx = np.arange(max(0, left - radius), left, dtype=int)
    right_idx = np.arange(right, min(n, right + radius), dtype=int)
    idx = np.concatenate([left_idx, right_idx])
    if idx.size >= int(min_points):
        return idx
    mask = np.ones(n, dtype=bool)
    mask[left:right] = False
    return np.flatnonzero(mask)


def segment(values: np.ndarray, start: int, end: int, channels: Sequence[int]) -> np.ndarray:
    """Extract a time/channel segment as a 2D matrix."""

    matrix = _as_2d(values)
    return matrix[int(start) : int(end), list(channels)]


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Expected 1D or 2D numeric array")
    return array


def finite_float(value: float | np.floating, *, default: float = 0.0) -> float:
    """Normalize non-finite values for CSV/JSON output."""

    number = float(value)
    return number if math.isfinite(number) else float(default)


def bootstrap_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    samples: int,
    statistic: str = "mean",
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a simple percentile bootstrap confidence interval."""

    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1 or int(samples) <= 0:
        return (float(arr[0]), float(arr[0]))
    boot = np.empty(int(samples), dtype=np.float64)
    for idx in range(int(samples)):
        draw = arr[rng.integers(0, arr.size, size=arr.size)]
        if statistic == "median":
            boot[idx] = float(np.median(draw))
        else:
            boot[idx] = float(np.mean(draw))
    alpha = (1.0 - float(confidence)) / 2.0
    return (float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha)))


def log2_comb(n: int, k: int) -> float:
    """Return log2 binomial coefficient without materializing combinations."""

    n_i = int(n)
    k_i = int(k)
    if k_i < 0 or k_i > n_i:
        return 0.0
    return float((math.lgamma(n_i + 1) - math.lgamma(k_i + 1) - math.lgamma(n_i - k_i + 1)) / math.log(2.0))
