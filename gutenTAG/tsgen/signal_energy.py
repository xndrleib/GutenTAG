"""Window-energy helpers shared by planning and parameter policies."""

from __future__ import annotations

import numpy as np

from .signal_ops import local_centerline


def window_rms_from_prefix(sq_prefix: np.ndarray, start: int, end: int) -> float:
    """Compute RMS for a window from a squared-value prefix sum."""

    length = max(1, int(end - start))
    sum_sq = float(sq_prefix[end] - sq_prefix[start])
    mean_sq = max(sum_sq / float(length), 0.0)
    return float(np.sqrt(mean_sq))


def window_residual_scale(values: np.ndarray, center_mode: str) -> float:
    """Return residual scale after removing a local centerline."""

    window = np.asarray(values, dtype=np.float64)
    if window.size == 0:
        return 0.0
    residual = window - local_centerline(window, center_mode)
    return float(max(float(np.std(residual)), 0.25 * float(np.ptp(residual))))


def rms_values_for_channel_length(
    sq_prefix: np.ndarray,
    length: int,
) -> np.ndarray:
    """Return per-start RMS values for a fixed-length window."""

    sums = sq_prefix[length:] - sq_prefix[:-length]
    means = np.maximum(sums / float(length), 0.0)
    return np.sqrt(means)


def peak_values_for_channel_length(
    abs_values: np.ndarray,
    length: int,
) -> np.ndarray:
    """Return per-start absolute peak values for a fixed-length window."""

    windows = np.lib.stride_tricks.sliding_window_view(
        abs_values,
        window_shape=length,
    )
    return np.max(windows, axis=1)


def residual_stats_values(
    values: np.ndarray,
    length: int,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual scale and peak values for every fixed-length window."""

    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(values, dtype=np.float64),
        window_shape=int(length),
    )
    normalized_mode = str(center_mode or "linear").lower()
    if normalized_mode in {"zero", "origin"}:
        residual = windows.astype(np.float64, copy=False)
    elif normalized_mode in {"mean", "constant"} or int(length) <= 2:
        residual = windows - np.mean(windows, axis=1, keepdims=True)
    elif normalized_mode in {"median", "robust"}:
        residual = windows - np.median(windows, axis=1, keepdims=True)
    else:
        degree = (
            2 if normalized_mode in {"quadratic", "poly2"} and int(length) >= 5 else 1
        )
        x_values = np.linspace(-1.0, 1.0, int(length), dtype=np.float64)
        columns = [np.ones(int(length), dtype=np.float64)]
        if degree >= 1:
            columns.insert(0, x_values)
        if degree >= 2:
            columns.insert(0, x_values * x_values)
        design = np.column_stack(columns)
        coefficients = windows @ np.linalg.pinv(design).T
        residual = windows - coefficients @ design.T
    residual_scale = np.maximum(
        np.std(residual, axis=1),
        0.25 * np.ptp(residual, axis=1),
    )
    residual_peak = np.max(np.abs(residual), axis=1)
    return residual_scale.astype(np.float64), residual_peak.astype(np.float64)


__all__ = [
    "peak_values_for_channel_length",
    "residual_stats_values",
    "rms_values_for_channel_length",
    "window_residual_scale",
    "window_rms_from_prefix",
]
