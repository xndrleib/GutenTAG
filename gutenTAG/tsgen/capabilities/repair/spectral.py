"""Spectral repair operators."""

from __future__ import annotations

import numpy as np

from .pattern import affine_channel_repair


def match_spectral_profile(
    clean_segment: np.ndarray, anomalous_segment: np.ndarray
) -> np.ndarray:
    """Approximate spectral repair with per-channel affine shape alignment."""

    return affine_channel_repair(clean_segment, anomalous_segment)
