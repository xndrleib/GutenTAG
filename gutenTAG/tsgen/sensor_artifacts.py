"""Observation faults with explicit timestamps, missingness and null controls."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SENSOR_KINDS = ("clipping", "quantization", "stuck-at", "dropout", "missing-block",
                "timestamp-jitter")


@dataclass(frozen=True)
class SensorArtifactResult:
    values: np.ndarray
    observed_mask: np.ndarray
    metadata: dict[str, float | int | str]
    timestamps: np.ndarray | None = None


def apply_sensor_artifact(values: np.ndarray, *, kind: str,
                          rng: np.random.Generator, severity: float = 0.5,
                          timestamps: np.ndarray | None = None,
                          previous_value: float | None = None,
                          reference_center: float | None = None,
                          reference_scale: float | None = None) -> SensorArtifactResult:
    """Apply a one-channel sensor fault; severity zero is exactly the identity.

    Positive-severity clipping/quantization use explicit reference calibration
    when supplied. Otherwise their window-relative calibration is recorded as
    an offline diagnostic, not a causal sensor model. Jitter returns sorted
    (timestamp, measurement) PAIRS; it does not silently interpolate samples.
    """
    source = np.asarray(values, dtype=float)
    if source.ndim != 1 or np.isinf(source).any():
        raise ValueError("values must be one-dimensional, with no infinities")
    if kind not in SENSOR_KINDS:
        raise ValueError(f"Unsupported sensor artifact kind: {kind}")
    if not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("severity must be finite and in [0, 1]")
    times = np.arange(source.size, dtype=float) if timestamps is None else np.asarray(timestamps, dtype=float).copy()
    if times.shape != source.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite, aligned and strictly increasing")
    target = source.copy()
    observed = np.isfinite(target).astype(np.int8)
    meta: dict[str, float | int | str] = {"kind": kind, "severity": float(severity)}
    if severity == 0 or not source.size:
        return SensorArtifactResult(target, observed, meta, times)
    finite = source[np.isfinite(source)]
    if not finite.size:
        return SensorArtifactResult(target, observed, meta, times)
    center = float(np.median(finite) if reference_center is None else reference_center)
    scale = float(np.std(finite) if reference_scale is None else reference_scale)
    if not np.isfinite([center, scale]).all() or scale < 0:
        raise ValueError("reference center/scale must be finite; scale nonnegative")
    meta["calibration"] = "reference" if reference_scale is not None else "offline_window"
    scale = max(scale, np.finfo(float).eps)
    if kind == "clipping":
        bound = (3.0 - 2.75 * severity) * scale
        target = np.clip(source, center-bound, center+bound)
        meta["clip_bound"] = bound
    elif kind == "quantization":
        step = scale * severity
        target = center + np.round((source-center)/step)*step
        meta["step"] = step
    elif kind == "stuck-at":
        anchor = source[0] if previous_value is None else previous_value
        if not np.isfinite(anchor):
            raise ValueError("stuck-at requires a finite last available measurement")
        target[:] = float(anchor)
        observed[:] = 1
        meta["anchor_policy"] = "previous" if previous_value is not None else "first"
    elif kind == "dropout":
        missing = rng.random(source.size) < severity
        target[missing] = np.nan
    elif kind == "missing-block":
        n = max(1, int(np.ceil(severity*source.size)))
        start = int(rng.integers(0, source.size-n+1))
        target[start:start+n] = np.nan
        meta.update(block_start=start, block_length=n)
    else:
        dt = float(np.min(np.diff(times))) if times.size > 1 else 1.0
        times += rng.uniform(-0.45, 0.45, times.size) * dt * severity
        order = np.argsort(times, kind="stable")
        times, target = times[order], target[order]
        meta["max_jitter"] = 0.45*dt*severity
    observed = np.isfinite(target).astype(np.int8)
    return SensorArtifactResult(target, observed, meta, times)
