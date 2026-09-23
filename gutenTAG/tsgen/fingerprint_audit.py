"""Generator-fingerprint diagnostics for synthetic benchmark shortcuts."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


NUISANCE_FEATURES = (
    "source_length",
    "effective_length",
    "boundary_jump",
    "derivative_jump",
    "realized_density",
    "transition_length",
)


def nuisance_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Build a finite nuisance-only feature matrix without signal semantics."""
    matrix = np.zeros((len(rows), len(NUISANCE_FEATURES)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, key in enumerate(NUISANCE_FEATURES):
            raw = row.get(key, 0.0)
            try:
                value = float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                value = 0.0
            matrix[row_idx, col_idx] = value if np.isfinite(value) else 0.0
    return matrix


def nearest_centroid_fingerprint_accuracy(
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[str],
) -> float:
    """Estimate label leakage from nuisance features with leave-one-out centroids.

    This deliberately simple classifier is a diagnostic, not a benchmark model.
    Accuracy far above the majority baseline means generation artifacts encode
    anomaly identity even without the time-series values.
    """
    if len(rows) != len(labels) or len(rows) < 2:
        return float("nan")
    x = nuisance_matrix(rows)
    scale = np.std(x, axis=0)
    scale[scale <= 1e-12] = 1.0
    x = (x - np.mean(x, axis=0)) / scale
    labels_array = np.asarray([str(label) for label in labels], dtype=object)
    classes = sorted(set(labels_array.tolist()))
    correct = 0
    scored = 0
    for idx in range(len(rows)):
        candidates: list[tuple[float, str]] = []
        for label in classes:
            mask = (labels_array == label)
            mask[idx] = False
            if not np.any(mask):
                continue
            centroid = np.mean(x[mask], axis=0)
            candidates.append((float(np.linalg.norm(x[idx] - centroid)), label))
        if not candidates:
            continue
        predicted = min(candidates, key=lambda item: item[0])[1]
        correct += int(predicted == labels_array[idx])
        scored += 1
    return float(correct / scored) if scored else float("nan")
