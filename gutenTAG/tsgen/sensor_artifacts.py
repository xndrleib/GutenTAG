"""Sensor-level anomaly transforms for v13 synthetic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorArtifactResult:
    values: np.ndarray
    observed_mask: np.ndarray
    metadata: dict[str, float | int | str]


def apply_sensor_artifact(
    values: np.ndarray,
    *,
    kind: str,
    rng: np.random.Generator,
    severity: float = 0.5,
) -> SensorArtifactResult:
    """Apply a realistic observation-layer fault to a one-dimensional window."""
    source = np.asarray(values, dtype=np.float64)
    target = np.array(source, copy=True)
    observed = np.ones(source.shape[0], dtype=np.int8)
    severity = float(np.clip(severity, 0.0, 1.0))
    if source.size == 0:
        return SensorArtifactResult(target, observed, {"kind": str(kind), "severity": severity})
    kind = str(kind)
    if kind == "clipping":
        center = float(np.median(source))
        scale = max(float(np.std(source)), 1e-8)
        bound = max(0.25, 2.5 * (1.0 - severity)) * scale
        target = np.clip(source, center - bound, center + bound)
        meta = {"kind": kind, "severity": severity, "clip_bound": float(bound)}
    elif kind == "quantization":
        levels = max(4, int(round(64.0 * (1.0 - severity))))
        lo, hi = float(np.min(source)), float(np.max(source))
        if hi > lo:
            step = (hi - lo) / float(levels - 1)
            target = lo + np.round((source - lo) / step) * step
        else:
            step = 0.0
        meta = {"kind": kind, "severity": severity, "levels": int(levels), "step": float(step)}
    elif kind == "stuck-at":
        anchor = int(rng.integers(0, source.size))
        target[:] = float(source[anchor])
        meta = {"kind": kind, "severity": severity, "anchor": anchor}
    elif kind == "dropout":
        missing_share = 0.15 + 0.75 * severity
        missing = rng.random(source.size) < missing_share
        observed[missing] = 0
        target[missing] = np.nan
        meta = {"kind": kind, "severity": severity, "missing_share": float(np.mean(missing))}
    elif kind == "timestamp-jitter":
        x = np.arange(source.size, dtype=np.float64)
        jitter = rng.normal(0.0, max(0.05, severity), size=source.size)
        warped = np.clip(x + jitter, 0.0, max(0.0, source.size - 1.0))
        warped.sort()
        target = np.interp(x, warped, source)
        meta = {"kind": kind, "severity": severity, "jitter_std": max(0.05, severity)}
    else:
        raise ValueError(f"Unsupported sensor artifact kind: {kind}")
    return SensorArtifactResult(target.astype(np.float64), observed, meta)
