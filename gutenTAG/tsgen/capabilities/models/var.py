"""Linear VAR-style forecasting residual model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .pca import _robust_positive_scale, _window_mean


class VARResidualModel:
    """First-order linear forecasting residual."""

    model_id = "var1_forecast_residual_v1"
    family = "structural.var_forecast_residual"

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.scale_: float = 1.0
        self.channels_: int = 0

    def fit(self, clean_instances: Sequence[np.ndarray]) -> None:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for item in clean_instances:
            values = np.asarray(item, dtype=np.float64)
            if values.ndim == 2 and values.shape[0] >= 2:
                x_parts.append(values[:-1])
                y_parts.append(values[1:])
        if not x_parts:
            self.coef_ = np.zeros((1, 1), dtype=np.float64)
            self.channels_ = 0
            self.scale_ = 1.0
            return
        x = np.vstack(x_parts)
        y = np.vstack(y_parts)
        self.channels_ = int(y.shape[1])
        design = np.column_stack([np.ones(x.shape[0]), x])
        self.coef_, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = self._point_scores(np.vstack([x_parts[0][:1], y]))
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
            "order": 1,
            "channels": int(self.channels_),
            "scale": float(self.scale_),
        }

    def _point_scores(self, matrix: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        values = np.asarray(matrix, dtype=np.float64)
        scores = np.zeros(values.shape[0], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or self.coef_.shape[0] <= 1:
            return scores
        design = np.column_stack([np.ones(values.shape[0] - 1), values[:-1]])
        prediction = design @ self.coef_
        residual = values[1:] - prediction
        scores[1:] = np.sqrt(np.mean(np.square(residual), axis=1))
        return scores
