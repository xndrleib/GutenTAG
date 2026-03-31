from __future__ import annotations

from typing import Any

import numpy as np


def compose_observed_window(base: np.ndarray, bo: Any, channel: int, start: int, end: int) -> np.ndarray:
    """Compose a single observed channel window from base and fixed variations."""
    result = np.asarray(base[start:end, channel], dtype=np.float64).copy()
    if getattr(bo, "noise", None) is not None:
        result = result + np.asarray(bo.noise[start:end], dtype=np.float64)
    if getattr(bo, "trend_series", None) is not None:
        result = result + np.asarray(bo.trend_series[start:end], dtype=np.float64)
    if getattr(bo, "offset", None) is not None:
        result = result + float(bo.offset)
    return result.astype(np.float64, copy=False)


def replace_observed_window(
    base: np.ndarray,
    bo: Any,
    channel: int,
    start: int,
    end: int,
    target_observed: np.ndarray,
) -> None:
    """Rewrite base so that the final observed window matches `target_observed`."""
    target = np.asarray(target_observed, dtype=np.float64)
    candidate = np.array(target, dtype=np.float64, copy=True)
    if getattr(bo, "noise", None) is not None:
        candidate = candidate - np.asarray(bo.noise[start:end], dtype=np.float64)
    if getattr(bo, "trend_series", None) is not None:
        candidate = candidate - np.asarray(bo.trend_series[start:end], dtype=np.float64)
    if getattr(bo, "offset", None) is not None:
        candidate = candidate - float(bo.offset)
    base[start:end, channel] = candidate.astype(np.float64, copy=False)


def _safe_standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    series = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(series))
    centered = series - mean
    std = float(np.std(centered))
    if std <= 1e-12:
        return np.zeros_like(series), mean, 1.0
    return centered / std, mean, std


def matched_coupling_window(
    reference: np.ndarray,
    anchor: np.ndarray,
    coupling_strength: float,
) -> np.ndarray:
    """Create a channel window with matched scale but altered coupling to an anchor."""
    z_ref, mean_ref, std_ref = _safe_standardize(reference)
    z_anchor, _, _ = _safe_standardize(anchor)
    strength = float(np.clip(coupling_strength, -0.999, 0.999))
    mixed = np.sqrt(max(0.0, 1.0 - strength**2)) * z_ref + strength * z_anchor
    mixed_z, _, _ = _safe_standardize(mixed)
    return (mean_ref + std_ref * mixed_z).astype(np.float64)


def matched_rewiring_windows(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap local structure while preserving each channel's local mean/scale."""
    z_first, mean_first, std_first = _safe_standardize(first)
    z_second, mean_second, std_second = _safe_standardize(second)
    rewired_first = mean_first + std_first * z_second
    rewired_second = mean_second + std_second * z_first
    return rewired_first.astype(np.float64), rewired_second.astype(np.float64)


def lag_shift_window(
    series: np.ndarray,
    start: int,
    end: int,
    lag_steps: int,
) -> tuple[np.ndarray, int]:
    """Return a lag-shifted window and the realized lag after boundary clipping."""
    full = np.asarray(series, dtype=np.float64)
    n = int(full.shape[0])
    lag = int(lag_steps)
    if end <= start or n <= 1:
        return full[start:end].copy(), 0
    if lag == 0:
        return full[start:end].copy(), 0
    low = -int(n - end)
    high = int(start)
    realized_lag = int(np.clip(lag, low, high))
    shifted_start = int(start - realized_lag)
    shifted_end = int(end - realized_lag)
    return full[shifted_start:shifted_end].copy(), realized_lag
