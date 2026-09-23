"""Shared relation-anomaly policies for group operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .multivariate_ops import has_shared_noise_decomposition


def latent_shared_noise_attrs(
    bo: Any,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    """Return latent shared-noise components for an aligned window."""
    if not has_shared_noise_decomposition(bo, start, end):
        return None
    idio = np.asarray(getattr(bo, "_idio_noise_component")[start:end], dtype=np.float64)
    shared = np.asarray(
        getattr(bo, "_shared_noise_component")[start:end], dtype=np.float64
    )
    mean = float(getattr(bo, "_noise_mean"))
    shared_weight = float(getattr(bo, "_shared_noise_weight"))
    residual_weight_raw = getattr(
        bo,
        "_residual_noise_weight",
        np.sqrt(max(0.0, 1.0 - shared_weight**2)),
    )
    residual_weight = float(
        residual_weight_raw
        if residual_weight_raw is not None
        else np.sqrt(max(0.0, 1.0 - shared_weight**2))
    )
    return idio, shared, mean, shared_weight, residual_weight


def effective_latent_transition_length(bo: Any, transition_length: int) -> int:
    """Return transition length for latent-noise rewrites."""
    return int(max(0, transition_length))


def effective_relation_target(
    *,
    bo: Any,
    target_correlation: float | None,
) -> float | None:
    """Return target correlation after base-specific relation policy."""
    if target_correlation is None:
        return None
    if not np.isfinite(target_correlation) or abs(target_correlation) > 1:
        raise ValueError("target_correlation must be finite and in [-1, 1]")
    return float(target_correlation)


def effective_coupling_strength(
    *,
    bo: Any,
    coupling_strength: float,
) -> float:
    """Return coupling strength after base-specific relation policy."""
    if not np.isfinite(coupling_strength) or abs(coupling_strength) > 1:
        raise ValueError("coupling_strength must be finite and in [-1, 1]")
    return float(coupling_strength)


def prefer_observed_relation_rewrite(bo: Any) -> bool:
    """Return whether relation operators should rewrite observed windows."""
    kind = str(bo.get_base_oscillation_kind())
    return kind == "shared-noise-sine"
