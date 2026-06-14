"""Mode-alignment repair operators."""

from __future__ import annotations

import numpy as np

from .correlation import match_covariance
from .pattern import affine_channel_repair


def restore_mode_agreement(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Restore collective-mode agreement with a deterministic oracle repair."""

    covariance_repair = match_covariance(clean_segment, anomalous_segment)
    signed_affine_repair = affine_channel_repair(clean_segment, anomalous_segment)
    if _rmse(signed_affine_repair, clean_segment) < _rmse(covariance_repair, clean_segment):
        return signed_affine_repair
    return covariance_repair


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    residual = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))
