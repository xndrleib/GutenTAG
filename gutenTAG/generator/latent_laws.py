"""White-noise LMC compatibility adapter; temporal LMC lives in processes.laws."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from ..tsgen.processes.laws import readonly, require_spd


@dataclass(frozen=True)
class NoiseLawState:
    """Shared immutable state, stored once rather than copied into C channels."""
    correlation: np.ndarray
    standardized: np.ndarray
    scales: np.ndarray
    loadings: np.ndarray

    def __deepcopy__(self, memo: dict) -> NoiseLawState:
        return self


def nearest_correlation(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Eigenvalue-floor normalization (NOT a nearest-matrix optimizer).

    Retained compatibility name. Requested benchmark covariances use
    require_spd and fail validation instead of being silently projected.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or not x.size or not np.isfinite(x).all():
        raise ValueError("finite nonempty square matrix required")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    vals, vecs = np.linalg.eigh((x+x.T)/2)
    y = (vecs*np.maximum(vals, eps)) @ vecs.T
    sd = np.sqrt(np.diag(y)); y /= np.outer(sd, sd)
    return require_spd(y, correlation=True).copy()


def sample_lmc_correlation(channels: int, rng: np.random.Generator, *, rank: int,
                           sparsity: float, idiosyncratic: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample signed sparse factors; positive noise makes covariance SPD."""
    if channels < 1 or rank < 1 or not 0 <= sparsity < 1:
        raise ValueError("invalid LMC dimension/sparsity")
    if not np.isfinite(idiosyncratic) or idiosyncratic <= 0:
        raise ValueError("idiosyncratic variance must be finite and positive")
    b = rng.normal(size=(channels, min(rank, channels)))
    b *= rng.random(b.shape) >= sparsity
    for c in range(channels):
        if np.linalg.norm(b[c]) < 1e-12:
            b[c, c % b.shape[1]] = float(rng.choice([-1, 1]))
    cov = b @ b.T + idiosyncratic*np.eye(channels)
    sd = np.sqrt(np.diag(cov))
    return b, require_spd(cov/np.outer(sd, sd), correlation=True).copy()


def apply_lmc_noise_correlation(*, channel_bos: Sequence[Any], seed: int,
                                config: Mapping[str, Any]) -> bool:
    """Apply the explicitly white-noise LMC mode using low-rank sampling."""
    if config.get("mode", "shared_noise") != "lmc":
        return False
    if len(channel_bos) < 1:
        raise ValueError("LMC requires channels")
    noises = [np.asarray(bo.noise, dtype=float) for bo in channel_bos]
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in noises):
        raise ValueError("LMC requires finite vector noises")
    n = len(noises[0]); c = len(noises)
    if n < 1 or any(len(x) != n for x in noises):
        raise ValueError("LMC noises must have equal positive lengths")
    rng = np.random.default_rng(seed)
    kappa = float(config.get("lmc_idiosyncratic", 0.25))
    b, corr = sample_lmc_correlation(c, rng, rank=int(config.get("lmc_rank", min(3, c))),
                                    sparsity=float(config.get("lmc_sparsity", 0.25)), idiosyncratic=kappa)
    d = np.sqrt(np.sum(b*b, axis=1)+kappa)
    z = (rng.normal(size=(n, b.shape[1])) @ b.T + np.sqrt(kappa)*rng.normal(size=(n, c)))/d
    scales = np.asarray([abs(float(bo.variance)*float(bo.amplitude))
                         if hasattr(bo, "variance") and hasattr(bo, "amplitude")
                         else float(np.std(x)) for bo, x in zip(channel_bos, noises)])
    state = NoiseLawState(readonly(corr), readonly(z), readonly(scales), readonly(b/d[:, None]))
    for i, bo in enumerate(channel_bos):
        bo._process_state = state
        bo._normal_law_channel = i
        bo._normal_law_kind = "white_noise_lmc"
        bo.noise = z[:, i]*scales[i]
    return True


def recouple_lmc_target(*, bo: Any, anchor: int, start: int, end: int,
                         target_correlation: float | None, transition: int) -> np.ndarray | None:
    """Recouple a target to an anchor in innovation space with unit variance.

    The anchor and original Gaussian conditional residual are independent.
    Rotating their coefficients on the unit circle avoids crossfade variance
    dips. Changes are relative to the stored normal law, not fitted to query.
    """
    state = getattr(bo, "_process_state", None)
    if not isinstance(state, NoiseLawState):
        return None
    target = int(bo._normal_law_channel)
    rho0 = float(state.correlation[target, anchor])
    rho1 = -rho0 if target_correlation is None else float(target_correlation)
    if not np.isfinite(rho1) or abs(rho1) >= 1:
        raise ValueError("LMC target correlation must be strictly between -1 and 1")
    anchor_z = state.standardized[start:end, anchor]
    residual = (state.standardized[start:end, target]-rho0*anchor_z)/np.sqrt(1-rho0*rho0)
    g = np.ones(end-start)
    w = min(max(0, transition), (len(g)-1)//2)
    if w:
        ramp = np.linspace(0, 1, w+2)[1:-1]
        g[:w], g[-w:] = ramp, ramp[::-1]
    theta = (1-g)*np.arccos(rho0)+g*np.arccos(rho1)
    return state.scales[target]*(np.cos(theta)*anchor_z+np.sin(theta)*residual)
