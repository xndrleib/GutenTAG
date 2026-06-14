"""Lag-profile repair operators."""

from __future__ import annotations

import numpy as np

from .pattern import affine_channel_repair


def restore_lag_profile(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Approximate lag repair with a stable affine channel alignment."""

    return affine_channel_repair(clean_segment, anomalous_segment)
