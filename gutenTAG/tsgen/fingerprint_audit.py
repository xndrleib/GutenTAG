"""Nuisance diagnostics with explicit group holdout and training-only scaling."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
from .capabilities.grouped_statistics import grouped_centroid_accuracy

NUISANCE_FEATURES = ("source_length", "effective_length", "boundary_jump",
                     "derivative_jump", "realized_density", "transition_length")


def nuisance_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Require complete finite features; missing measurements are not zeroes."""
    result = np.empty((len(rows), len(NUISANCE_FEATURES)))
    for i, row in enumerate(rows):
        for j, key in enumerate(NUISANCE_FEATURES):
            if key not in row or row[key] is None:
                raise ValueError(f"Missing nuisance measurement: {key}")
            result[i, j] = float(row[key])
    if not np.isfinite(result).all():
        raise ValueError("Nuisance measurements must be finite")
    return result


def nearest_centroid_fingerprint_accuracy(rows: Sequence[Mapping[str, object]],
                                         labels: Sequence[str],
                                         groups: Sequence[str] | None = None) -> float:
    """Return balanced held-out accuracy, not a shortcut-absence certificate.

    Supply system IDs for correlated replicas/windows. With groups=None this
    is explicitly the IID-row diagnostic retained for backwards compatibility.
    """
    if len(rows) != len(labels):
        raise ValueError("rows and labels must align")
    g = np.arange(len(rows)) if groups is None else np.asarray(groups)
    return grouped_centroid_accuracy(nuisance_matrix(rows), np.asarray(labels), g)
