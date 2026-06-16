"""Variance repair operators."""

from __future__ import annotations

import numpy as np

from .pattern import affine_channel_repair


def match_variance(
    clean_segment: np.ndarray, anomalous_segment: np.ndarray
) -> np.ndarray:
    """Match local scale using the best monotonic oracle repair candidate."""

    if clean_segment.size == 0 or anomalous_segment.size == 0:
        return np.asarray(anomalous_segment, dtype=np.float64)
    original = np.asarray(anomalous_segment, dtype=np.float64)
    clean = np.asarray(clean_segment, dtype=np.float64)
    clean_mean = np.mean(clean_segment, axis=0, keepdims=True)
    anomalous_mean = np.mean(anomalous_segment, axis=0, keepdims=True)
    clean_std = np.std(clean_segment, axis=0, keepdims=True) + 1e-8
    anomalous_std = np.std(anomalous_segment, axis=0, keepdims=True) + 1e-8
    rescaled = (original - anomalous_mean) * (clean_std / anomalous_std) + clean_mean
    candidates = (
        original,
        rescaled,
        affine_channel_repair(clean, original),
    )
    return min(candidates, key=lambda candidate: _rmse(candidate, clean))


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    residual = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))
