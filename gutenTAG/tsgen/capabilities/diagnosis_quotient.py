"""Epsilon-neighborhood quotient metrics for descriptor diagnosis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .identifiability import DESCRIPTOR_COLUMNS


def _epsilon_quotient(embeddings: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    if embeddings.empty:
        return pd.DataFrame()
    epsilon = _epsilon_from_matrix(matrix)
    rows: list[dict[str, object]] = []
    zero_feature_neighbors = matrix.shape[1] == 0
    radius_neighbors = _radius_neighbors(matrix, epsilon)
    for descriptor in DESCRIPTOR_COLUMNS:
        labels = embeddings[descriptor].astype(str).to_numpy()
        classes = sorted(set(labels))
        if zero_feature_neighbors:
            rows.extend(_zero_feature_rows(descriptor, labels, classes, epsilon))
            continue
        for label in classes:
            rows.append(
                _label_neighborhood_row(
                    descriptor,
                    label,
                    labels,
                    radius_neighbors,
                    epsilon,
                )
            )
    return pd.DataFrame(rows)


def _radius_neighbors(
    matrix: np.ndarray,
    epsilon: float,
) -> list[np.ndarray] | None:
    if matrix.shape[1] == 0 or matrix.shape[0] == 0:
        return None
    neighbors = NearestNeighbors(
        radius=epsilon + 1e-12, algorithm="auto", metric="euclidean"
    )
    neighbors.fit(matrix)
    return list(neighbors.radius_neighbors(matrix, return_distance=False))


def _zero_feature_rows(
    descriptor: str,
    labels: np.ndarray,
    classes: list[str],
    epsilon: float,
) -> list[dict[str, object]]:
    global_counts = {
        candidate: int(np.sum(labels == candidate)) for candidate in classes
    }
    global_majority = max(global_counts.values()) if global_counts else 0
    global_impurity = float(1.0 - global_majority / max(len(labels), 1))
    global_size = float(len(labels))
    return [
        {
            "descriptor": descriptor,
            "label": label,
            "event_count": int(np.sum(labels == label)),
            "epsilon": epsilon,
            "mean_local_impurity": global_impurity,
            "max_local_impurity": global_impurity,
            "mean_quotient_cell_size": global_size,
            "quotient_status": _quotient_status(global_impurity),
        }
        for label in classes
    ]


def _label_neighborhood_row(
    descriptor: str,
    label: str,
    labels: np.ndarray,
    radius_neighbors: list[np.ndarray] | None,
    epsilon: float,
) -> dict[str, object]:
    local_impurities, local_sizes = _local_neighborhood_metrics(
        label,
        labels,
        radius_neighbors,
    )
    mean_impurity = (
        float(np.mean(local_impurities)) if local_impurities else float("nan")
    )
    return {
        "descriptor": descriptor,
        "label": label,
        "event_count": int(np.sum(labels == label)),
        "epsilon": epsilon,
        "mean_local_impurity": mean_impurity,
        "max_local_impurity": (
            float(np.max(local_impurities)) if local_impurities else float("nan")
        ),
        "mean_quotient_cell_size": (
            float(np.mean(local_sizes)) if local_sizes else float("nan")
        ),
        "quotient_status": _quotient_status(mean_impurity),
    }


def _local_neighborhood_metrics(
    label: str,
    labels: np.ndarray,
    radius_neighbors: list[np.ndarray] | None,
) -> tuple[list[float], list[int]]:
    local_impurities: list[float] = []
    local_sizes: list[int] = []
    for idx, current_label in enumerate(labels):
        if current_label != label:
            continue
        neighbor_indices = _neighbor_indices(radius_neighbors, idx)
        neighbor_labels = labels[neighbor_indices]
        majority = max(
            int(np.sum(neighbor_labels == candidate))
            for candidate in set(neighbor_labels)
        )
        local_impurities.append(float(1.0 - majority / max(len(neighbor_labels), 1)))
        local_sizes.append(int(len(neighbor_labels)))
    return local_impurities, local_sizes


def _neighbor_indices(
    radius_neighbors: list[np.ndarray] | None,
    idx: int,
) -> np.ndarray:
    if radius_neighbors is None:
        return np.asarray([idx], dtype=int)
    neighbor_indices = radius_neighbors[idx]
    if neighbor_indices.size == 0:
        return np.asarray([idx], dtype=int)
    return neighbor_indices


def _epsilon_from_matrix(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return 0.0
    neighbors = NearestNeighbors(
        n_neighbors=min(2, matrix.shape[0]),
        algorithm="auto",
        metric="euclidean",
    )
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix, return_distance=True)
    nearest: list[float] = []
    for row_idx, row_indices in enumerate(indices):
        for distance, candidate_idx in zip(distances[row_idx], row_indices):
            if int(candidate_idx) != row_idx:
                nearest.append(float(distance))
                break
    finite = np.asarray(
        [value for value in nearest if math.isfinite(value)], dtype=np.float64
    )
    if finite.size == 0:
        return 0.0
    return float(max(np.median(finite), 1e-12))


def _quotient_status(impurity: float) -> str:
    if not math.isfinite(impurity):
        return "not_estimable"
    if impurity <= 0.10:
        return "low_impurity"
    if impurity <= 0.30:
        return "mixed_neighborhoods"
    return "high_impurity"


__all__ = [
    "_epsilon_quotient",
]
