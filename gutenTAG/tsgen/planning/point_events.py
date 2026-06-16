"""Point-event segment planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import is_slot_available, occupy_slot
from .types import SegmentPlan


@dataclass
class _PointEventPlacementState:
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    segments: list[SegmentPlan]


def sample_point_event_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    overlap_policy: str,
    planner_cfg: Mapping[str, Any],
    segment_count_range: Sequence[int] | None = None,
    density_range: Sequence[float] | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample one-point anomaly segments from density or count policy."""

    length = int(series_length)
    n_channels = int(channels)
    n_points = _sample_point_count(
        rng=rng,
        target_density=target_density,
        length=length,
        segment_count_range=segment_count_range,
        density_range=density_range,
    )
    timestamps = _sample_point_timestamps(
        rng=rng,
        length=length,
        n_points=n_points,
        unique_timestamps=bool(planner_cfg.get("unique_timestamps", True)),
    )
    state = _new_point_event_state(length=length, channels=n_channels)
    for timestamp in timestamps.tolist():
        _place_point_event(
            rng=rng,
            timestamp=int(timestamp),
            n_channels=n_channels,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        )
    state.segments.sort(
        key=lambda segment: (segment.start, segment.channel, segment.length)
    )
    return state.segments


def _sample_point_count(
    *,
    rng: np.random.Generator,
    target_density: float,
    length: int,
    segment_count_range: Sequence[int] | None,
    density_range: Sequence[float] | None,
) -> int:
    if segment_count_range is not None:
        low, high = _ordered_count_bounds(segment_count_range)
        n_points = int(rng.integers(low, high + 1))
    else:
        density = _clamped_density(target_density, density_range)
        n_points = int(np.floor(density * length))
    return max(1, min(n_points, length))


def _ordered_count_bounds(segment_count_range: Sequence[int]) -> tuple[int, int]:
    low, high = int(segment_count_range[0]), int(segment_count_range[1])
    if low > high:
        low, high = high, low
    return low, high


def _clamped_density(
    target_density: float,
    density_range: Sequence[float] | None,
) -> float:
    density = float(target_density)
    if density_range is None:
        return density
    return float(
        np.clip(
            density,
            float(density_range[0]),
            float(density_range[1]),
        )
    )


def _sample_point_timestamps(
    *,
    rng: np.random.Generator,
    length: int,
    n_points: int,
    unique_timestamps: bool,
) -> np.ndarray:
    if unique_timestamps:
        timestamps = rng.choice(length, size=n_points, replace=False).astype(int)
    else:
        timestamps = rng.integers(0, length, size=n_points, dtype=int)
    return np.sort(timestamps)


def _new_point_event_state(length: int, channels: int) -> _PointEventPlacementState:
    return _PointEventPlacementState(
        occupied_global=np.zeros(length, dtype=np.int8),
        occupied_per_channel=np.zeros((channels, length), dtype=np.int8),
        segments=[],
    )


def _place_point_event(
    *,
    rng: np.random.Generator,
    timestamp: int,
    n_channels: int,
    overlap_policy: str,
    state: _PointEventPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> None:
    channel = int(rng.integers(0, n_channels))
    start = int(timestamp)
    end = start + 1
    channel = _available_point_channel(
        rng=rng,
        start=start,
        end=end,
        channel=channel,
        n_channels=n_channels,
        overlap_policy=overlap_policy,
        state=state,
    )
    if channel is None:
        return
    occupy_slot(
        start,
        end,
        channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    )
    state.segments.append(
        segment_factory(start=start, end=end, length=1, channel=channel)
    )


def _available_point_channel(
    *,
    rng: np.random.Generator,
    start: int,
    end: int,
    channel: int,
    n_channels: int,
    overlap_policy: str,
    state: _PointEventPlacementState,
) -> int | None:
    if _point_slot_available(start, end, channel, overlap_policy, state):
        return channel
    for fallback_channel in rng.permutation(n_channels).tolist():
        candidate = int(fallback_channel)
        if _point_slot_available(start, end, candidate, overlap_policy, state):
            return candidate
    return None


def _point_slot_available(
    start: int,
    end: int,
    channel: int,
    overlap_policy: str,
    state: _PointEventPlacementState,
) -> bool:
    return is_slot_available(
        start,
        end,
        channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    )
