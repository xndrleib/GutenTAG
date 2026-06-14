"""Pattern and shape repair operators."""

from __future__ import annotations

import numpy as np


def affine_channel_repair(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Fit a per-channel affine map from anomalous to clean values."""

    if clean_segment.size == 0 or anomalous_segment.size == 0:
        return np.asarray(anomalous_segment, dtype=np.float64)
    repaired = np.empty_like(anomalous_segment, dtype=np.float64)
    for channel in range(anomalous_segment.shape[1]):
        source = np.asarray(anomalous_segment[:, channel], dtype=np.float64)
        target = np.asarray(clean_segment[:, channel], dtype=np.float64)
        source_centered = source - float(np.mean(source))
        target_centered = target - float(np.mean(target))
        denom = float(np.dot(source_centered, source_centered))
        if denom <= 1e-12:
            repaired[:, channel] = float(np.mean(target))
        else:
            beta = float(np.dot(source_centered, target_centered) / denom)
            alpha = float(np.mean(target) - beta * np.mean(source))
            repaired[:, channel] = alpha + beta * source
    return repaired


def match_pattern(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Align a local template anomaly using the best monotonic repair candidate."""

    if clean_segment.size == 0 or anomalous_segment.size == 0:
        return np.asarray(anomalous_segment, dtype=np.float64)
    clean = np.asarray(clean_segment, dtype=np.float64)
    original = np.asarray(anomalous_segment, dtype=np.float64)
    candidates = [original, affine_channel_repair(clean, original)]
    for lag in _candidate_lags(original.shape[0]):
        shifted = _shift_with_edge_fill(original, lag)
        candidates.append(affine_channel_repair(clean, shifted))
    return min(candidates, key=lambda candidate: _rmse(candidate, clean))


def _candidate_lags(length: int) -> tuple[int, ...]:
    if length <= 1:
        return ()
    max_lag = max(1, min(12, int(length) // 3))
    return tuple(lag for lag in range(-max_lag, max_lag + 1) if lag != 0)


def _shift_with_edge_fill(values: np.ndarray, lag: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if lag == 0 or source.shape[0] == 0:
        return source.copy()
    shifted = np.empty_like(source, dtype=np.float64)
    length = source.shape[0]
    if abs(lag) >= length:
        shifted[:] = source[0 if lag > 0 else -1]
        return shifted
    if lag > 0:
        shifted[lag:] = source[:-lag]
        shifted[:lag] = source[0]
        return shifted
    offset = -lag
    shifted[:-offset] = source[offset:]
    shifted[-offset:] = source[-1]
    return shifted


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    residual = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))
