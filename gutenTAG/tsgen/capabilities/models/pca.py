"""PCA reconstruction residual model."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class PCAResidualModel:
    """Low-dimensional PCA reconstruction residual."""

    model_id = "pca_residual_v1"
    family = "structural.pca_residual"

    def __init__(self, rank: int = 1) -> None:
        self.rank = max(1, int(rank))
        self.effective_rank_: int = self.rank
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.scale_: float = 1.0

    def fit(self, clean_instances: Sequence[np.ndarray]) -> None:
        matrix = _stack(clean_instances)
        if matrix.size == 0:
            self.mean_ = np.zeros(1, dtype=np.float64)
            self.components_ = np.eye(1, dtype=np.float64)
            self.scale_ = 1.0
            return
        self.mean_ = np.mean(matrix, axis=0)
        centered = matrix - self.mean_
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(self.rank, vt.shape[0], max(1, vt.shape[1] - 1))
        self.effective_rank_ = int(rank)
        self.components_ = vt[:rank]
        residual = self._point_scores(matrix)
        self.scale_ = _robust_positive_scale(residual)

    def score_series(self, series: np.ndarray) -> np.ndarray:
        values = self._point_scores(np.asarray(series, dtype=np.float64))
        return values / max(self.scale_, 1e-8)

    def score_windows(self, series: np.ndarray, windows: np.ndarray) -> np.ndarray:
        point_scores = self.score_series(series)
        return _window_mean(point_scores, windows)

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "rank": int(self.rank),
            "effective_rank": int(self.effective_rank_),
            "scale": float(self.scale_),
            "channels": int(self.mean_.size) if self.mean_ is not None else 0,
        }

    def _point_scores(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        values = np.asarray(matrix, dtype=np.float64)
        centered = values - self.mean_
        reconstructed = centered @ self.components_.T @ self.components_ + self.mean_
        return np.sqrt(np.mean(np.square(values - reconstructed), axis=1))


def _stack(clean_instances: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [
        np.asarray(item, dtype=np.float64)
        for item in clean_instances
        if np.asarray(item).size
    ]
    return np.vstack(arrays) if arrays else np.empty((0, 0), dtype=np.float64)


def _robust_positive_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    std = float(np.std(finite))
    return max(mad, std, median, 1e-8)


def _window_mean(values: np.ndarray, windows: np.ndarray) -> np.ndarray:
    result: list[float] = []
    for start, end in np.asarray(windows, dtype=int):
        lo = max(0, min(int(start), values.size))
        hi = max(lo, min(int(end), values.size))
        result.append(float(np.mean(values[lo:hi])) if hi > lo else 0.0)
    return np.asarray(result, dtype=np.float64)
