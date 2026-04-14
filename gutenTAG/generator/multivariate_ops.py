from __future__ import annotations

from typing import Any

import numpy as np


def compose_observed_window(
    base: np.ndarray, bo: Any, channel: int, start: int, end: int
) -> np.ndarray:
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


def has_shared_noise_decomposition(bo: Any, start: int, end: int) -> bool:
    """Return whether a channel exposes latent shared-noise components."""
    idio = getattr(bo, "_idio_noise_component", None)
    shared = getattr(bo, "_shared_noise_component", None)
    mean = getattr(bo, "_noise_mean", None)
    if idio is None or shared is None or mean is None:
        return False
    if getattr(bo, "noise", None) is None:
        return False
    n = int(np.asarray(bo.noise).shape[0])
    return 0 <= int(start) <= int(end) <= n


def shared_noise_window_from_components(
    *,
    idiosyncratic_component: np.ndarray,
    shared_component: np.ndarray,
    noise_mean: float,
    target_shared_weight: float,
) -> np.ndarray:
    """Recompose a channel-noise window with a changed shared component weight.

    The construction preserves the channel's local noise scale by keeping the
    squared sum of the idiosyncratic and shared weights equal to one.
    """
    idio = np.asarray(idiosyncratic_component, dtype=np.float64)
    shared = np.asarray(shared_component, dtype=np.float64)
    if idio.shape != shared.shape:
        raise ValueError("Noise components must have the same shape.")
    shared_weight = float(np.clip(target_shared_weight, -0.999, 0.999))
    idio_weight = float(np.sqrt(max(0.0, 1.0 - shared_weight**2)))
    return (float(noise_mean) + idio_weight * idio + shared_weight * shared).astype(
        np.float64
    )


def _deterministic_decorrelated_component(
    values: np.ndarray, variant: int = 0
) -> np.ndarray:
    """Build a same-shape surrogate component with weak correlation to `values`.

    Different ``variant`` values choose different decorrelated surrogates. This is
    useful for pair-level structural anomalies where every affected channel should
    lose the shared factor in a slightly different way instead of inheriting the
    exact same replacement component.
    """
    series = np.asarray(values, dtype=np.float64)
    n = int(series.shape[0])
    if n <= 1:
        return np.array(series, dtype=np.float64, copy=True)
    centered = series - float(np.mean(series))
    if n <= 3:
        return centered[::-1].copy()
    candidates: list[np.ndarray] = []
    candidates.append(np.roll(centered, max(1, n // 2)))
    candidates.append(centered[::-1].copy())
    even_odd_indices = np.concatenate((np.arange(0, n, 2), np.arange(1, n, 2)))
    candidates.append(centered[even_odd_indices])
    for step in range(max(2, n // 2 - 2), 1, -1):
        if np.gcd(step, n) != 1:
            continue
        indices = (np.arange(n, dtype=int) * step + 1) % n
        candidates.append(centered[indices])
        if len(candidates) >= 6:
            break

    scored_candidates: list[tuple[float, np.ndarray]] = []
    for candidate in candidates:
        corr = float(np.corrcoef(centered, candidate)[0, 1]) if n > 2 else 0.0
        score = abs(corr) if np.isfinite(corr) else float("inf")
        scored_candidates.append((score, candidate))

    scored_candidates.sort(key=lambda item: item[0])
    selected = scored_candidates[int(variant) % len(scored_candidates)][1]
    surrogate_z, _, _ = _safe_standardize(selected)
    shared_scale = float(np.std(centered))
    return (shared_scale * surrogate_z).astype(np.float64)


def mixed_shared_noise_window_from_components(
    *,
    idiosyncratic_component: np.ndarray,
    shared_component: np.ndarray,
    noise_mean: float,
    current_shared_weight: float,
    target_alignment: float,
    surrogate_variant: int = 0,
) -> np.ndarray:
    """Recompose noise with preserved shared-energy magnitude but changed alignment.

    `target_alignment=1` keeps the original shared component, `0` replaces it with a
    decorrelated surrogate, and `-1` flips it while preserving local scale.
    """
    idio = np.asarray(idiosyncratic_component, dtype=np.float64)
    shared = np.asarray(shared_component, dtype=np.float64)
    if idio.shape != shared.shape:
        raise ValueError("Noise components must have the same shape.")
    magnitude = float(np.clip(abs(current_shared_weight), 0.0, 0.999))
    residual_weight = float(np.sqrt(max(0.0, 1.0 - magnitude**2)))
    shared_centered = shared - float(np.mean(shared))
    shared_z, _, _ = _safe_standardize(shared_centered)
    surrogate = _deterministic_decorrelated_component(
        shared_centered, variant=surrogate_variant
    )
    surrogate_z, _, _ = _safe_standardize(surrogate)
    alignment = float(np.clip(target_alignment, -0.999, 0.999))
    orth_weight = float(np.sqrt(max(0.0, 1.0 - alignment**2)))
    mixed_shared = alignment * shared_z + orth_weight * surrogate_z
    mixed_shared_z, _, _ = _safe_standardize(mixed_shared)
    shared_scale = float(np.std(shared_centered))
    shared_candidate = shared_scale * mixed_shared_z
    return (
        float(noise_mean) + residual_weight * idio + magnitude * shared_candidate
    ).astype(np.float64)


def scaled_shared_noise_window_from_components(
    *,
    idiosyncratic_component: np.ndarray,
    shared_component: np.ndarray,
    noise_mean: float,
    current_shared_weight: float,
    shared_factor_scale: float,
) -> np.ndarray:
    """Recompose noise with a scaled version of the original shared factor.

    ``shared_factor_scale=1`` keeps the current shared-factor magnitude,
    ``0`` removes it completely, and negative values flip its sign while
    attenuating or preserving its magnitude.
    """
    idio = np.asarray(idiosyncratic_component, dtype=np.float64)
    shared = np.asarray(shared_component, dtype=np.float64)
    if idio.shape != shared.shape:
        raise ValueError("Noise components must have the same shape.")
    base_weight = float(np.clip(current_shared_weight, -0.999, 0.999))
    scale = float(np.clip(shared_factor_scale, -1.0, 1.0))
    target_shared_weight = float(np.clip(base_weight * scale, -0.999, 0.999))
    residual_weight = float(np.sqrt(max(0.0, 1.0 - target_shared_weight**2)))
    return (
        float(noise_mean) + residual_weight * idio + target_shared_weight * shared
    ).astype(np.float64)


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


def _local_linear_trend(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    n = int(series.shape[0])
    if n <= 2:
        return np.full_like(series, float(np.mean(series)))
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    degree = 2 if n >= 5 else 1
    coeff = np.polyfit(x, series, deg=degree)
    return np.polyval(coeff, x).astype(np.float64)


def residualized_matched_coupling_window(
    reference: np.ndarray,
    anchor: np.ndarray,
    coupling_strength: float,
) -> np.ndarray:
    """Change relation structure on residuals while keeping the local trend."""
    ref = np.asarray(reference, dtype=np.float64)
    anc = np.asarray(anchor, dtype=np.float64)
    if ref.size == 0 or anc.size == 0:
        return np.array(ref, dtype=np.float64, copy=True)
    ref_trend = _local_linear_trend(ref)
    anc_trend = _local_linear_trend(anc)
    ref_residual = ref - ref_trend
    anc_residual = anc - anc_trend
    if float(np.std(ref_residual)) <= 1e-8 or float(np.std(anc_residual)) <= 1e-8:
        return matched_coupling_window(ref, anc, coupling_strength)
    candidate_residual = matched_coupling_window(
        ref_residual,
        anc_residual,
        coupling_strength,
    )
    return (ref_trend + candidate_residual).astype(np.float64)


def residualized_correlation_flip_window(
    reference: np.ndarray,
    anchor: np.ndarray,
    target_correlation: float | None = -0.85,
) -> np.ndarray:
    """Flip relation structure on residuals while keeping the local trend."""
    ref = np.asarray(reference, dtype=np.float64)
    anc = np.asarray(anchor, dtype=np.float64)
    if ref.size == 0 or anc.size == 0:
        return np.array(ref, dtype=np.float64, copy=True)
    ref_trend = _local_linear_trend(ref)
    anc_trend = _local_linear_trend(anc)
    ref_residual = ref - ref_trend
    anc_residual = anc - anc_trend
    if float(np.std(ref_residual)) <= 1e-8 or float(np.std(anc_residual)) <= 1e-8:
        from .correlation_geometry import correlation_flip_window

        return correlation_flip_window(ref, anc, target_correlation)
    from .correlation_geometry import correlation_flip_window

    candidate_residual = correlation_flip_window(
        ref_residual,
        anc_residual,
        target_correlation,
    )
    return (ref_trend + candidate_residual).astype(np.float64)


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


def rotated_pair_windows(
    first: np.ndarray,
    second: np.ndarray,
    rotation_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a standardized channel pair to perturb relation structure smoothly."""
    z_first, mean_first, std_first = _safe_standardize(first)
    z_second, mean_second, std_second = _safe_standardize(second)
    theta = np.deg2rad(float(rotation_degrees))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    mixed_first = cos_theta * z_first - sin_theta * z_second
    mixed_second = sin_theta * z_first + cos_theta * z_second
    mixed_first_z, _, _ = _safe_standardize(mixed_first)
    mixed_second_z, _, _ = _safe_standardize(mixed_second)
    rewired_first = mean_first + std_first * mixed_first_z
    rewired_second = mean_second + std_second * mixed_second_z
    return rewired_first.astype(np.float64), rewired_second.astype(np.float64)


def break_shared_factor_window(
    reference: np.ndarray,
    anchor: np.ndarray,
    shared_factor_scale: float,
) -> np.ndarray:
    """Attenuate or flip the shared factor of a channel relative to its anchor."""
    z_ref, mean_ref, std_ref = _safe_standardize(reference)
    z_anchor, _, _ = _safe_standardize(anchor)
    denom = float(np.dot(z_anchor, z_anchor))
    if denom <= 1e-12:
        return np.array(reference, dtype=np.float64, copy=True)
    projection = (float(np.dot(z_ref, z_anchor)) / denom) * z_anchor
    residual = z_ref - projection
    residual_z, _, _ = _safe_standardize(residual)
    scale = float(shared_factor_scale)
    candidate = residual_z + scale * projection
    candidate_z, _, _ = _safe_standardize(candidate)
    return (mean_ref + std_ref * candidate_z).astype(np.float64)


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
