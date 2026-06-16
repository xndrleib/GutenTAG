"""Period-boundary geometry helpers for period-locked planners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ...base_oscillations.utils.math_func_support import SAMPLING_F


def compute_period_boundaries(
    base_frequency: Any,
    *,
    series_length: int,
) -> np.ndarray | None:
    """Compute approximate period boundaries for a sampled oscillator."""

    if not isinstance(base_frequency, (int, float)):
        return None
    frequency = float(base_frequency)
    length = int(series_length)
    if frequency <= 0 or length <= 1:
        return None
    # Base oscillators are sampled with np.linspace(..., endpoint=True), so the
    # effective period in sample indices is slightly shorter than SAMPLING_F / f.
    period_size = (float(SAMPLING_F) * float(length - 1)) / (frequency * float(length))
    if not np.isfinite(period_size) or period_size <= 1:
        return None
    max_periods = int(np.floor(length / period_size))
    if max_periods <= 0:
        return None
    boundaries = np.round(
        np.arange(max_periods + 1, dtype=np.float64) * period_size
    ).astype(int)
    boundaries = np.clip(boundaries, 0, length)
    boundaries = np.unique(boundaries)
    if boundaries.size == 0 or boundaries[0] != 0:
        boundaries = np.concatenate([[0], boundaries])
    if boundaries[-1] != length:
        boundaries = np.concatenate([boundaries, [length]])
    boundaries = np.unique(boundaries)
    if boundaries.size < 2:
        return None
    return boundaries


def sanitize_period_boundaries(
    period_boundaries: Iterable[int] | None,
    *,
    series_length: int,
) -> np.ndarray | None:
    """Validate and normalize period boundaries for a generated series."""

    if period_boundaries is None:
        return None
    try:
        boundaries = np.array([int(value) for value in period_boundaries], dtype=int)
    except TypeError:
        return None
    if boundaries.size == 0:
        return None
    length = int(series_length)
    boundaries = np.clip(boundaries, 0, length)
    boundaries = np.unique(boundaries)
    if boundaries.size == 0 or boundaries[0] != 0:
        boundaries = np.concatenate([[0], boundaries])
    if boundaries[-1] != length:
        boundaries = np.concatenate([boundaries, [length]])
    boundaries = np.unique(boundaries)
    if boundaries.size < 2:
        return None
    if np.any(np.diff(boundaries) <= 0):
        return None
    return boundaries


def resolve_channel_boundaries(
    *,
    series_length: int,
    channels: int,
    base_period_size: int | None,
    period_boundaries: Iterable[int] | None,
    period_boundaries_by_channel: Mapping[int, Iterable[int] | None] | None,
) -> dict[int, np.ndarray]:
    """Return normalized period boundaries for each eligible channel."""

    channel_boundaries: dict[int, np.ndarray] = {}
    if period_boundaries_by_channel is not None:
        for channel in range(int(channels)):
            raw_boundaries = period_boundaries_by_channel.get(channel)
            sanitized = sanitize_period_boundaries(
                raw_boundaries,
                series_length=series_length,
            )
            if sanitized is not None and sanitized.shape[0] >= 2:
                channel_boundaries[channel] = sanitized
    if len(channel_boundaries) > 0:
        return channel_boundaries

    boundaries = sanitize_period_boundaries(
        period_boundaries,
        series_length=series_length,
    )
    if boundaries is None:
        if base_period_size is None or int(base_period_size) <= 1:
            raise ValueError(
                "period_locked_frequency planner requires either period boundaries "
                "or period_size > 1."
            )
        period_size = int(base_period_size)
        boundaries = np.arange(0, int(series_length) + 1, period_size, dtype=int)
        if boundaries[-1] != int(series_length):
            boundaries = np.append(boundaries, int(series_length))
    return {
        channel: np.array(boundaries, copy=True) for channel in range(int(channels))
    }


__all__ = [
    "compute_period_boundaries",
    "resolve_channel_boundaries",
    "sanitize_period_boundaries",
]
