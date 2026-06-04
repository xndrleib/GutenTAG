"""Signal-shaping utilities shared by anomaly operators."""

from __future__ import annotations

from typing import Optional

import numpy as np


def robust_scale(values: np.ndarray, fallback: float = 0.0) -> float:
    """Return a deterministic robust local scale estimate."""
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return float(max(abs(fallback), 1e-6))
    median = float(np.median(series))
    mad = float(1.4826 * np.median(np.abs(series - median)))
    std = float(np.std(series))
    ptp = float(np.ptp(series))
    return float(max(mad, std, 0.25 * ptp, abs(float(fallback)), 1e-6))


def local_centerline(values: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Estimate a local baseline for residual-preserving amplitude transforms."""
    series = np.asarray(values, dtype=np.float64)
    n = int(series.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    normalized_mode = str(mode or "linear").lower()
    if normalized_mode in {"zero", "origin"}:
        return np.zeros(n, dtype=np.float64)
    if normalized_mode in {"mean", "constant"}:
        return np.full(n, float(np.mean(series)), dtype=np.float64)
    if normalized_mode in {"median", "robust"}:
        return np.full(n, float(np.median(series)), dtype=np.float64)
    if n <= 2:
        return np.full(n, float(np.mean(series)), dtype=np.float64)
    degree = 2 if normalized_mode in {"quadratic", "poly2"} and n >= 5 else 1
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    try:
        coeff = np.polyfit(x, series, deg=degree)
        return np.polyval(coeff, x).astype(np.float64)
    except (np.linalg.LinAlgError, ValueError):
        return np.full(n, float(np.median(series)), dtype=np.float64)


def symmetric_envelope(length: int, transition_length: Optional[int]) -> np.ndarray:
    """Build a deterministic [0, 1] envelope with neutral endpoints."""
    n = int(length)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    if transition_length is None:
        transition = max(1, min(int(round(n * 0.2)), n // 2))
    else:
        transition = max(0, min(int(transition_length), n // 2))
    if transition == 0:
        envelope = np.ones(n, dtype=np.float64)
    else:
        plateau_len = max(0, n - 2 * transition)
        phase = np.linspace(0.0, np.pi / 2.0, transition, dtype=np.float64)
        left = np.sin(phase) ** 2
        right = left[::-1]
        envelope = np.concatenate([left, np.ones(plateau_len, dtype=np.float64), right])
    envelope[0] = 0.0
    envelope[-1] = 0.0
    return envelope.astype(np.float64, copy=False)


def match_boundary_value_and_slope(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match candidate to reference at both interval boundaries in C0/C1 form."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.size == 0 or cand.size == 0:
        return cand.astype(np.float64, copy=True)
    n = min(int(ref.shape[0]), int(cand.shape[0]))
    ref = ref[:n]
    result = cand[:n].astype(np.float64, copy=True)
    if n == 1:
        result[0] = ref[0]
        return result
    left_delta = float(ref[0] - result[0])
    right_delta = float(ref[-1] - result[-1])
    if n < 4:
        alpha = np.linspace(0.0, 1.0, n, dtype=np.float64)
        result += (1.0 - alpha) * left_delta + alpha * right_delta
        result[0] = ref[0]
        result[-1] = ref[-1]
        return result.astype(np.float64, copy=False)

    ref_left_slope = float(ref[1] - ref[0])
    ref_right_slope = float(ref[-1] - ref[-2])
    cand_left_slope = float(result[1] - result[0])
    cand_right_slope = float(result[-1] - result[-2])
    m0 = float((ref_left_slope - cand_left_slope) * (n - 1))
    m1 = float((ref_right_slope - cand_right_slope) * (n - 1))
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    a = left_delta
    b = m0
    c = 3.0 * (right_delta - left_delta) - 2.0 * m0 - m1
    d = 2.0 * (left_delta - right_delta) + m0 + m1
    correction = a + b * t + c * t * t + d * t * t * t
    local_scale = max(float(np.ptp(ref)), float(np.std(ref)), 1e-6)
    if float(np.max(np.abs(correction))) > 3.0 * local_scale:
        alpha = np.linspace(0.0, 1.0, n, dtype=np.float64)
        correction = (1.0 - alpha) * left_delta + alpha * right_delta
    result += correction
    result[0] = ref[0]
    result[-1] = ref[-1]
    return result.astype(np.float64, copy=False)


def enforce_min_effect_delta(
    *,
    baseline: np.ndarray,
    candidate: np.ndarray,
    target_effect: float,
    transition_length: Optional[int] = None,
    allow_zero_delta_fallback: bool = False,
) -> np.ndarray:
    """Scale an operator delta until local peak effect reaches a floor."""
    target = max(0.0, float(target_effect))
    cand = np.asarray(candidate, dtype=np.float64)
    if target <= 0.0:
        return cand
    base = np.asarray(baseline, dtype=np.float64)
    if base.size == 0 or cand.size == 0:
        return cand.astype(np.float64, copy=True)
    n = min(int(base.shape[0]), int(cand.shape[0]))
    base = base[:n]
    result = cand[:n].astype(np.float64, copy=True)

    def scale_once(current: np.ndarray) -> np.ndarray:
        delta = current - base
        peak = float(np.max(np.abs(delta))) if delta.size else 0.0
        if peak >= target:
            return current
        if peak > 1e-12:
            return base + delta * (target / peak)
        if not allow_zero_delta_fallback:
            raise ValueError(
                "Cannot enforce min_effect_delta without changing anomaly family "
                "semantics because generated delta is zero."
            )
        envelope = symmetric_envelope(n, transition_length)
        if not np.any(envelope > 0.0):
            envelope = np.ones(n, dtype=np.float64)
        return base + target * envelope

    result = scale_once(result)
    if transition_length is not None:
        result = match_boundary_value_and_slope(result, base)
        result = scale_once(result)
    if result.size > 0:
        result[0] = base[0]
        result[-1] = base[-1]
    return result.astype(np.float64, copy=False)
