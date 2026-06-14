"""Witness functionals for paired observability and single-series detection."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .numerics import (
    context_indices,
    finite_float,
    robust_scale,
    safe_corrcoef,
    safe_covariance,
    safe_variance,
    segment,
)


def paired_witness_scores(
    *,
    clean: np.ndarray,
    anomalous: np.ndarray,
    start: int,
    end: int,
    channels: Sequence[int],
    length: int,
    context_multiplier: int,
    min_context_points: int,
) -> dict[str, float]:
    """Compute paired clean-vs-anomalous witness scores on a support interval."""

    if int(end) <= int(start):
        return {}
    subset = tuple(int(channel) for channel in channels)
    clean_seg = segment(clean, start, end, subset)
    anom_seg = segment(anomalous, start, end, subset)
    idx = context_indices(
        length,
        start,
        end,
        multiplier=context_multiplier,
        min_points=min_context_points,
    )
    clean_context = np.asarray(clean, dtype=np.float64)[idx[:, None], list(subset)] if idx.size else clean_seg
    delta_seg = anom_seg - clean_seg
    delta_all = np.asarray(anomalous, dtype=np.float64)[:, list(subset)] - np.asarray(clean, dtype=np.float64)[:, list(subset)]
    scale = robust_scale(clean_context, fallback=robust_scale(clean_seg))
    raw_norm = float(np.sqrt(np.mean(np.square(delta_seg)))) if delta_seg.size else 0.0
    scores: dict[str, float] = {
        "energy_delta": finite_float(raw_norm / scale),
        "support_concentration_l2": finite_float(
            float(np.sum(np.square(delta_seg))) / max(float(np.sum(np.square(delta_all))), 1e-12)
        ),
    }
    mean_delta = np.mean(anom_seg, axis=0) - np.mean(clean_seg, axis=0)
    scores["mean_delta"] = finite_float(float(np.linalg.norm(mean_delta)) / (scale * math.sqrt(len(subset))))
    clean_var = np.var(clean_seg, axis=0) + 1e-8
    anom_var = np.var(anom_seg, axis=0) + 1e-8
    scores["variance_delta"] = finite_float(float(np.linalg.norm(np.log(anom_var / clean_var))) / math.sqrt(len(subset)))
    scores["shape_residual"] = finite_float(_shape_residual(clean_seg, anom_seg))
    scores["spectral_delta"] = finite_float(_spectral_delta(clean_seg, anom_seg))
    if len(subset) >= 2:
        clean_corr = safe_corrcoef(clean_seg)
        anom_corr = safe_corrcoef(anom_seg)
        corr_norm = float(np.linalg.norm(anom_corr - clean_corr, ord="fro"))
        denom_corr = math.sqrt(max(1, len(subset) * (len(subset) - 1)))
        scores["correlation_delta"] = finite_float(corr_norm / denom_corr)
        clean_cov = safe_covariance(clean_seg)
        anom_cov = safe_covariance(anom_seg)
        scores["covariance_delta"] = finite_float(
            float(np.linalg.norm(anom_cov - clean_cov, ord="fro")) / max(float(np.linalg.norm(clean_cov, ord="fro")), 1e-8)
        )
        scores["lag_correlation_delta"] = finite_float(_lag_correlation_delta(clean_seg, anom_seg))
    return scores


def detection_window_scores(
    *,
    series: np.ndarray,
    start: int,
    end: int,
    channels: Sequence[int],
    context_multiplier: int,
    min_context_points: int,
) -> dict[str, float]:
    """Compute single-series local-window anomaly scores against context."""

    matrix = np.asarray(series, dtype=np.float64)
    if int(end) <= int(start):
        return {}
    subset = tuple(int(channel) for channel in channels)
    window = segment(matrix, start, end, subset)
    idx = context_indices(
        matrix.shape[0],
        start,
        end,
        multiplier=context_multiplier,
        min_points=min_context_points,
    )
    context = matrix[idx[:, None], list(subset)] if idx.size else matrix[:, list(subset)]
    scale = robust_scale(context, fallback=robust_scale(window))
    mean_shift = np.mean(window, axis=0) - np.mean(context, axis=0)
    scores: dict[str, float] = {
        "mean_z": finite_float(float(np.linalg.norm(mean_shift)) / (scale * math.sqrt(len(subset)))),
        "local_energy_z": finite_float(_local_energy_against_context(window, context) / scale),
    }
    window_var = np.var(window, axis=0) + 1e-8
    context_var = np.var(context, axis=0) + 1e-8
    scores["variance_log_ratio"] = finite_float(float(np.linalg.norm(np.log(window_var / context_var))) / math.sqrt(len(subset)))
    if len(subset) >= 2:
        window_corr = safe_corrcoef(window)
        context_corr = safe_corrcoef(context)
        scores["correlation_shift"] = finite_float(
            float(np.linalg.norm(window_corr - context_corr, ord="fro")) / math.sqrt(max(1, len(subset) * (len(subset) - 1)))
        )
        window_cov = safe_covariance(window)
        context_cov = safe_covariance(context)
        scores["covariance_shift"] = finite_float(
            float(np.linalg.norm(window_cov - context_cov, ord="fro")) / max(float(np.linalg.norm(context_cov, ord="fro")), 1e-8)
        )
    return scores


def _shape_residual(clean_seg: np.ndarray, anom_seg: np.ndarray) -> float:
    """Residual after the best per-channel affine alignment."""

    if clean_seg.size == 0 or anom_seg.size == 0:
        return 0.0
    raw = float(np.sqrt(np.mean(np.square(anom_seg - clean_seg))))
    if raw <= 1e-12:
        return 0.0
    repaired = np.empty_like(anom_seg)
    for channel in range(anom_seg.shape[1]):
        x = anom_seg[:, channel]
        y = clean_seg[:, channel]
        x_centered = x - float(np.mean(x))
        y_centered = y - float(np.mean(y))
        denom = float(np.dot(x_centered, x_centered))
        if denom <= 1e-12:
            repaired[:, channel] = np.mean(y)
        else:
            beta = float(np.dot(x_centered, y_centered) / denom)
            alpha = float(np.mean(y) - beta * np.mean(x))
            repaired[:, channel] = alpha + beta * x
    residual = float(np.sqrt(np.mean(np.square(repaired - clean_seg))))
    return residual / max(raw, 1e-12)


def _spectral_delta(clean_seg: np.ndarray, anom_seg: np.ndarray) -> float:
    """Compare normalized amplitude spectra."""

    if clean_seg.shape[0] < 4 or anom_seg.shape[0] < 4:
        return 0.0
    clean_centered = clean_seg - np.mean(clean_seg, axis=0, keepdims=True)
    anom_centered = anom_seg - np.mean(anom_seg, axis=0, keepdims=True)
    clean_fft = np.abs(np.fft.rfft(clean_centered, axis=0))
    anom_fft = np.abs(np.fft.rfft(anom_centered, axis=0))
    clean_dist = clean_fft / np.maximum(np.sum(clean_fft, axis=0, keepdims=True), 1e-8)
    anom_dist = anom_fft / np.maximum(np.sum(anom_fft, axis=0, keepdims=True), 1e-8)
    return float(np.mean(np.abs(anom_dist - clean_dist)))


def _lag_correlation_delta(clean_seg: np.ndarray, anom_seg: np.ndarray, max_lag: int = 8) -> float:
    """Compare cross-correlation lag profiles for the first two projected channels."""

    if clean_seg.shape[1] < 2 or clean_seg.shape[0] < 4:
        return 0.0
    lag_limit = min(int(max_lag), max(1, clean_seg.shape[0] // 3))
    clean_profile = _lag_profile(clean_seg[:, 0], clean_seg[:, 1], lag_limit)
    anom_profile = _lag_profile(anom_seg[:, 0], anom_seg[:, 1], lag_limit)
    return float(np.linalg.norm(anom_profile - clean_profile) / math.sqrt(clean_profile.size))


def _lag_profile(x: np.ndarray, y: np.ndarray, lag_limit: int) -> np.ndarray:
    values: list[float] = []
    for lag in range(-int(lag_limit), int(lag_limit) + 1):
        if lag < 0:
            a, b = x[:lag], y[-lag:]
        elif lag > 0:
            a, b = x[lag:], y[:-lag]
        else:
            a, b = x, y
        if a.size < 3 or b.size < 3 or safe_variance(a) <= 1e-8 or safe_variance(b) <= 1e-8:
            values.append(0.0)
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
            values.append(corr if math.isfinite(corr) else 0.0)
    return np.asarray(values, dtype=np.float64)


def _local_energy_against_context(window: np.ndarray, context: np.ndarray) -> float:
    """RMS distance of window samples to context center."""

    center = np.mean(context, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.square(window - center))))


def select_scores(scores: Mapping[str, float], witnesses: Sequence[str]) -> dict[str, float]:
    """Select finite scores by witness name."""

    return {name: finite_float(scores[name]) for name in witnesses if name in scores}
