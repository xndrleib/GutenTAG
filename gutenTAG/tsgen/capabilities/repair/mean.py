"""Mean repair operators."""

from __future__ import annotations

import numpy as np


def match_mean(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Shift each anomalous channel to the clean segment mean."""

    if clean_segment.size == 0 or anomalous_segment.size == 0:
        return np.asarray(anomalous_segment, dtype=np.float64)
    clean_mean = np.mean(clean_segment, axis=0, keepdims=True)
    anomalous_mean = np.mean(anomalous_segment, axis=0, keepdims=True)
    return np.asarray(anomalous_segment, dtype=np.float64) - anomalous_mean + clean_mean
