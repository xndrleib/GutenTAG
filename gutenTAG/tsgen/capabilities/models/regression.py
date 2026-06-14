"""Target-from-others regression residual model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .pca import _robust_positive_scale, _stack, _window_mean


class TargetRegressionResidualModel:
    """Predict each channel from the others and score the largest residual."""

    model_id = "target_from_others_regression_v2"
    family = "structural.target_regression_residual"

    def __init__(self) -> None:
        self.coefs_: list[np.ndarray] = []
        self.scales_: np.ndarray = np.ones(0, dtype=np.float64)
        self.scale_: float = 1.0
        self.channels_: int = 0

    def fit(self, clean_instances: Sequence[np.ndarray]) -> None:
        matrix = _stack(clean_instances)
        self.channels_ = int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0
        if self.channels_ < 2:
            self.coefs_ = []
            self.scales_ = np.ones(max(0, self.channels_), dtype=np.float64)
            self.scale_ = 1.0
            return
        self.coefs_ = []
        scales: list[float] = []
        for target_index in range(self.channels_):
            design = _design(matrix, target_index)
            target = matrix[:, target_index]
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            self.coefs_.append(np.asarray(coef, dtype=np.float64))
            scales.append(_robust_positive_scale(np.abs(target - design @ coef)))
        self.scales_ = np.asarray(scales, dtype=np.float64)
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
            "channels": int(self.channels_),
            "target_policy": "all_targets_max_standardized_residual",
            "scale": float(self.scale_),
        }

    def _point_scores(self, matrix: np.ndarray) -> np.ndarray:
        if not self.coefs_ and self.channels_ >= 2:
            raise RuntimeError("Model must be fitted before scoring")
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 2 or len(self.coefs_) != values.shape[1]:
            return np.zeros(values.shape[0], dtype=np.float64)
        residuals: list[np.ndarray] = []
        for target_index, coef in enumerate(self.coefs_):
            prediction = _design(values, target_index) @ coef
            scale = max(float(self.scales_[target_index]), 1e-8)
            residuals.append(np.abs(values[:, target_index] - prediction) / scale)
        return np.max(np.vstack(residuals), axis=0)


def _design(matrix: np.ndarray, target_index: int) -> np.ndarray:
    context = [index for index in range(matrix.shape[1]) if index != int(target_index)]
    return np.column_stack([np.ones(matrix.shape[0]), matrix[:, context]])
