"""Latent-law helpers for v13 multivariate synthetic processes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def nearest_correlation(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to a numerically positive-definite correlation matrix."""
    value = np.asarray(matrix, dtype=np.float64)
    value = 0.5 * (value + value.T)
    eigvals, eigvecs = np.linalg.eigh(value)
    eigvals = np.maximum(eigvals, float(eps))
    projected = (eigvecs * eigvals) @ eigvecs.T
    scale = np.sqrt(np.maximum(np.diag(projected), float(eps)))
    corr = projected / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float64)


def sample_lmc_correlation(
    channels: int,
    rng: np.random.Generator,
    *,
    rank: int,
    sparsity: float,
    idiosyncratic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an LMC loading matrix and its implied correlation matrix."""
    channels = int(channels)
    rank = max(1, min(int(rank), channels))
    loadings = rng.normal(0.0, 1.0, size=(channels, rank))
    if sparsity > 0.0:
        mask = rng.random(size=loadings.shape) >= float(np.clip(sparsity, 0.0, 0.95))
        loadings *= mask
    for channel in range(channels):
        if float(np.linalg.norm(loadings[channel])) <= 1e-8:
            loadings[channel, int(rng.integers(0, rank))] = float(rng.choice([-1.0, 1.0]))
    covariance = loadings @ loadings.T
    covariance += max(float(idiosyncratic), 1e-4) * np.eye(channels)
    return loadings.astype(np.float64), nearest_correlation(covariance)


def apply_lmc_noise_correlation(
    *,
    channel_bos: Sequence[Any],
    seed: int,
    config: Mapping[str, Any],
) -> bool:
    """Recompose channel noise using a sampled LMC correlation law."""
    if len(channel_bos) <= 1:
        return False
    mode = str(config.get("mode", "shared_noise"))
    if mode != "lmc":
        return False
    noises = [np.asarray(bo.noise, dtype=np.float64) for bo in channel_bos]
    if not noises or any(noise.ndim != 1 for noise in noises):
        return False
    length = int(noises[0].shape[0])
    if any(int(noise.shape[0]) != length for noise in noises):
        raise ValueError("LMC noise correlation requires equal channel lengths")
    rng = np.random.default_rng(int(seed))
    rank = int(config.get("lmc_rank", min(3, len(channel_bos))))
    sparsity = float(config.get("lmc_sparsity", 0.25))
    idiosyncratic = float(config.get("lmc_idiosyncratic", 0.25))
    loadings, corr = sample_lmc_correlation(
        len(channel_bos), rng, rank=rank, sparsity=sparsity, idiosyncratic=idiosyncratic
    )
    latent = rng.normal(0.0, 1.0, size=(length, corr.shape[0]))
    chol = np.linalg.cholesky(corr)
    correlated = latent @ chol.T
    for channel, bo in enumerate(channel_bos):
        original = noises[channel]
        mean = float(np.mean(original))
        std = float(np.std(original))
        target = mean + max(std, 1e-8) * correlated[:, channel]
        bo._normal_law_kind = "lmc"
        bo._normal_law_correlation = np.array(corr, copy=True)
        bo._normal_law_loadings = np.array(loadings, copy=True)
        bo._normal_law_channel = int(channel)
        bo.noise = target.astype(np.float64)
    return True
