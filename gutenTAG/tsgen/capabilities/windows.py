"""Reusable scan-window libraries for capability profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numerics import contiguous_windows


@dataclass(frozen=True)
class WindowSpec:
    """Specification for a deterministic scan-window library."""

    series_length: int
    window_length: int
    max_windows: int
    stride_fraction: float
    boundary_policy: str = "clipped"


class WindowLibrary:
    """Cache deterministic contiguous-window arrays by specification."""

    def __init__(self) -> None:
        self._cache: dict[WindowSpec, np.ndarray] = {}

    def get(self, spec: WindowSpec) -> np.ndarray:
        """Return an ``[n_windows, 2]`` integer array of start/end windows."""

        if spec not in self._cache:
            windows = contiguous_windows(
                spec.series_length,
                spec.window_length,
                max_windows=spec.max_windows,
                stride_fraction=spec.stride_fraction,
            )
            self._cache[spec] = np.asarray(windows, dtype=np.int64)
        return self._cache[spec]
