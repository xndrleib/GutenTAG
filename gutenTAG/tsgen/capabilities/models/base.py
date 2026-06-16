"""Base protocol for structural capability models."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class CapabilityModel(Protocol):
    """Minimal interface for structural model-zoo scoring."""

    model_id: str
    family: str

    def fit(self, clean_instances: Sequence[np.ndarray]) -> None:
        """Fit model state from clean trajectories."""
        ...

    def score_series(self, series: np.ndarray) -> np.ndarray:
        """Return one score per time point."""
        ...

    def score_windows(self, series: np.ndarray, windows: np.ndarray) -> np.ndarray:
        """Return one score per ``[start, end)`` window."""
        ...

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible model metadata."""
        ...
