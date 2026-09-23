"""Finite-feature Gaussian laws with explicit kernels and angular frequencies.

Random spectra belong to the law; Gaussian feature coefficients belong to the
realization. Paired sine/cosine features give exactly unit pointwise variance
conditional on a spectrum. No trajectory-dependent normalization is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

KERNELS = ("rbf", "matern32", "rq", "periodic", "locally-periodic")


@dataclass(frozen=True)
class KernelSpec:
    """Stationary kernel parameters in the units of the supplied time grid."""

    kind: str = "rbf"
    length_scale: float = 1.0
    period: float = 1.0
    alpha: float = 1.0
    periodic_smoothness: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KERNELS:
            raise ValueError(f"Unknown kernel: {self.kind}")
        for name in ("length_scale", "period", "alpha", "periodic_smoothness"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def covariance(self, lag: np.ndarray) -> np.ndarray:
        """Evaluate the exact, unit-variance target covariance."""
        lag = np.asarray(lag, dtype=float)
        r = np.abs(lag) / self.length_scale
        if self.kind == "rbf":
            return np.exp(-0.5 * r * r)
        if self.kind == "matern32":
            u = np.sqrt(3.0) * r
            return (1.0 + u) * np.exp(-u)
        if self.kind == "rq":
            return (1.0 + r * r / (2.0 * self.alpha)) ** (-self.alpha)
        periodic = np.exp(
            -2.0 * np.sin(np.pi * lag / self.period) ** 2 / self.periodic_smoothness**2
        )
        return periodic if self.kind == "periodic" else periodic * np.exp(-0.5 * r * r)

    def spectrum(self, rng: np.random.Generator, features: int) -> np.ndarray:
        """Sample angular frequencies for this kernel (not cycles/time)."""
        if isinstance(features, bool) or int(features) != features or features < 1:
            raise ValueError("features must be a positive integer")
        if self.kind == "rbf":
            return rng.normal(size=features) / self.length_scale
        if self.kind == "matern32":
            return rng.standard_t(3, size=features) / self.length_scale
        if self.kind == "rq":
            precision = rng.gamma(self.alpha, 1.0 / self.alpha, size=features)
            return rng.normal(size=features) * np.sqrt(precision) / self.length_scale
        k = 0.5 / self.periodic_smoothness**2
        harmonics = rng.poisson(k, features) - rng.poisson(k, features)
        omega = (2.0 * np.pi / self.period) * harmonics
        if self.kind == "locally-periodic":
            omega = omega + rng.normal(size=features) / self.length_scale
        return np.asarray(omega, dtype=float)


def feature_sample(
    omega: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    *,
    block_size: int = 2048,
) -> np.ndarray:
    """Draw one realization with O(block_size * features) temporary memory."""
    omega = np.asarray(omega, dtype=float)
    times = np.asarray(times, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or not np.isfinite(omega).all():
        raise ValueError("omega must be a finite, nonempty vector")
    if times.ndim != 1 or not np.isfinite(times).all() or block_size < 1:
        raise ValueError("times must be a finite vector; block_size must be positive")
    # Draw once, outside the blocks: changing block_size must not change the law.
    a, b = rng.normal(size=(2, omega.size)) / np.sqrt(omega.size)
    result = np.empty(times.size)
    for start in range(0, times.size, block_size):
        phase = times[start : start + block_size, None] * omega[None, :]
        result[start : start + block_size] = np.cos(phase) @ a + np.sin(phase) @ b
    return result
