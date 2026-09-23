"""Immutable system laws and independently seeded paired realizations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from scipy.signal import lfilter

from .kernels import KernelSpec, feature_sample, KERNELS


def readonly(value: np.ndarray) -> np.ndarray:
    """Own an immutable copy, safe to share between clean/anomalous branches."""
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def require_spd(value: np.ndarray, *, correlation: bool = False) -> np.ndarray:
    """Validate rather than silently project a requested scientific law."""
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] < 1:
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.isfinite(x).all() or not np.allclose(x, x.T, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if correlation and not np.allclose(np.diag(x), 1.0, atol=1e-12):
        raise ValueError("correlation diagonal must be one")
    np.linalg.cholesky(x)
    return readonly(x)


@dataclass(frozen=True)
class ProcessLaw:
    """One system, with parameters/spectra fixed across all its realizations.

    ``diagonal-ar`` supports exact covariance-only innovation changes with
    unchanged complete one-channel trajectory laws. ``gp-lmc`` shares temporal
    factors. Its loading changes are NOT advertised as trajectory-marginal pure.
    """

    family: str
    mean: np.ndarray
    coefficients: np.ndarray
    covariance: np.ndarray
    loadings: np.ndarray
    noise_std: np.ndarray
    spectra: tuple[np.ndarray, ...] = ()
    kernel_specs: tuple[KernelSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in {"diagonal-ar", "gp-lmc", "independent-gp"}:
            raise ValueError(f"Unsupported process family: {self.family}")
        for name in ("mean", "coefficients", "loadings", "noise_std"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, readonly(arr))
        object.__setattr__(self, "covariance", require_spd(self.covariance))
        object.__setattr__(self, "spectra", tuple(readonly(x) for x in self.spectra))
        c = self.mean.size
        if self.mean.ndim != 1 or c < 1:
            raise ValueError("mean must be a nonempty vector")
        if self.coefficients.shape != (c,) or np.any(np.abs(self.coefficients) >= 1):
            raise ValueError("AR coefficients must have shape C and absolute value < 1")
        if self.covariance.shape != (c, c) or self.noise_std.shape != (c,):
            raise ValueError("law dimensions disagree")
        if np.any(self.noise_std < 0) or self.loadings.shape != (c, len(self.spectra)):
            raise ValueError("invalid noise or loading dimensions")
        if len(self.kernel_specs) != len(self.spectra):
            raise ValueError("kernel specifications must identify every spectrum")

    def __deepcopy__(self, memo: dict) -> ProcessLaw:
        return self

    @property
    def channels(self) -> int:
        return self.mean.size

    @property
    def marginal_std(self) -> np.ndarray:
        if self.family == "diagonal-ar":
            return np.sqrt(np.diag(self.covariance)/(1-self.coefficients**2))
        return np.sqrt(np.sum(self.loadings**2, axis=1)+self.noise_std**2)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete finite-feature law, not just a seed."""
        return {"family": self.family, "mean": self.mean.tolist(),
                "coefficients": self.coefficients.tolist(),
                "covariance": self.covariance.tolist(),
                "loadings": self.loadings.tolist(), "noise_std": self.noise_std.tolist(),
                "spectra": [x.tolist() for x in self.spectra],
                "kernel_specs": [vars(x) for x in self.kernel_specs]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProcessLaw:
        v = dict(value)
        v["spectra"] = tuple(np.asarray(x) for x in v["spectra"])
        v["kernel_specs"] = tuple(KernelSpec(**x) for x in v["kernel_specs"])
        return cls(**v)

    @property
    def law_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()
        return "law:" + hashlib.sha256(payload).hexdigest()

    def sample(self, length: int, seed: int) -> Realization:
        """Generate a new independent realization of this fixed law."""
        if isinstance(length, bool) or length < 1:
            raise ValueError("length must be positive")
        rng = np.random.default_rng(seed)
        if self.family == "diagonal-ar":
            stationary = self.covariance/(1-self.coefficients[:, None]*self.coefficients[None, :])
            initial = rng.multivariate_normal(np.zeros(self.channels), stationary)
            z = rng.normal(size=(length, self.channels))
            noise = z @ np.linalg.cholesky(self.covariance).T
            signal = ar_filter(noise, self.coefficients, initial)
            return Realization(self, seed, signal+self.mean, signal, noise, z, initial,
                               np.empty((length, 0)))
        t = np.arange(length, dtype=float)
        latent = np.column_stack([feature_sample(w, t, rng) for w in self.spectra])
        signal = latent @ self.loadings.T
        z = rng.normal(size=(length, self.channels))
        noise = z*self.noise_std
        return Realization(self, seed, signal+noise+self.mean, signal, noise, z,
                           np.zeros(self.channels), latent)


@dataclass(frozen=True)
class Realization:
    law: ProcessLaw
    seed: int
    values: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    standard_noise: np.ndarray
    initial: np.ndarray
    latents: np.ndarray

    @property
    def realization_id(self) -> str:
        return f"{self.law.law_id}:realization:{self.seed}"


def ar_filter(innovations: np.ndarray, coefficients: np.ndarray,
              initial: np.ndarray) -> np.ndarray:
    """Run diagonal recurrences with explicit pre-sample state."""
    return np.column_stack([
        lfilter([1.0], [1.0, -a], innovations[:, c], zi=[a*initial[c]])[0]
        for c, a in enumerate(coefficients)
    ])


def sample_law(family: str, channels: int, seed: int, *, length_scale: float = 16,
               memory: float = 0.5, rank: int = 3, features: int = 48,
               noise_std: float = 0.2, sparsity: float = 0.25) -> ProcessLaw:
    """Sample a system prior independently of anomaly mechanism and event support."""
    if channels < 1 or rank < 1 or features < 1 or not 0 <= memory < 1:
        raise ValueError("invalid system dimensions or memory")
    if not 0 <= sparsity < 1 or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("invalid sparsity/noise_std")
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.5, channels)
    if family == "diagonal-ar":
        b = rng.normal(size=(channels, min(rank, channels)))
        cov = b @ b.T + 0.5*np.eye(channels)
        sd = np.sqrt(np.diag(cov))
        cov /= np.outer(sd, sd)
        a = rng.uniform(-memory, memory, channels)
        return ProcessLaw(family, mean, a, cov, np.empty((channels, 0)),
                          np.ones(channels))
    if family not in {"gp-lmc", "independent-gp"}:
        raise ValueError(f"Unknown family {family}")
    q = channels if family == "independent-gp" else min(rank, channels)
    b = np.eye(channels) if family == "independent-gp" else rng.normal(size=(channels, q))
    if family == "gp-lmc":
        b *= rng.random(b.shape) >= sparsity
        for c in range(channels):
            if np.linalg.norm(b[c]) < 1e-12:
                b[c, c % q] = 1.0
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    specs = tuple(KernelSpec(kind=str(rng.choice(KERNELS)), length_scale=length_scale,
                             period=4*length_scale, alpha=float(rng.uniform(0.5, 2)))
                  for _ in range(q))
    spectra = tuple(k.spectrum(rng, features) for k in specs)
    return ProcessLaw(family, mean, np.zeros(channels), np.eye(channels), b,
                      np.full(channels, noise_std), spectra, specs)
