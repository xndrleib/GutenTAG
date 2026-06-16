"""Uniform segment planning."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .common import is_slot_available, occupy_slot, sample_segment_lengths
from .types import SegmentPlan


@dataclass
class _UniformPlacementState:
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    segments: list[SegmentPlan]


def sample_uniform_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_count_range: Sequence[int],
    min_segment_length: int,
    logger: logging.Logger | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample uniformly placed anomaly segments.

    Parameters
    ----------
    rng, target_density, series_length, channels
        Randomness, target support density, and generated-series shape.
    max_placement_attempts, overlap_policy
        Random-placement retry budget and temporal overlap policy.
    segment_count_range, min_segment_length, logger, segment_factory
        Count/length bounds, optional warning logger, and segment factory.

    Returns
    -------
    list[SegmentPlan]
        Sorted segment plan.
    """

    length = int(series_length)
    n_channels = int(channels)
    n_segments = _sample_uniform_segment_count(rng, segment_count_range)
    if n_segments > length:
        raise ValueError(
            f"Requested n_segments={n_segments} exceeds series length={length}."
        )
    target_points = _uniform_target_points(target_density, length, n_segments)
    n_segments = _fit_uniform_segment_count(
        n_segments=n_segments,
        target_points=target_points,
        min_segment_length=int(min_segment_length),
        logger=logger,
    )
    lengths = sample_segment_lengths(
        rng,
        target_points,
        n_segments,
        min_segment_length=int(min_segment_length),
    )
    state = _place_uniform_segments(
        rng=rng,
        lengths=lengths,
        series_length=length,
        channels=n_channels,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    )
    state.segments.sort(
        key=lambda segment: (segment.start, segment.channel, segment.length)
    )
    return state.segments


def _sample_uniform_segment_count(
    rng: np.random.Generator,
    segment_count_range: Sequence[int],
) -> int:
    low, high = int(segment_count_range[0]), int(segment_count_range[1])
    if low > high:
        low, high = high, low
    return int(rng.integers(low, high + 1))


def _uniform_target_points(
    target_density: float,
    series_length: int,
    n_segments: int,
) -> int:
    target_points = int(round(float(target_density) * int(series_length)))
    target_points = max(target_points, int(n_segments))
    return min(target_points, int(series_length))


def _fit_uniform_segment_count(
    *,
    n_segments: int,
    target_points: int,
    min_segment_length: int,
    logger: logging.Logger | None,
) -> int:
    max_segments_for_min_length = max(
        1,
        int(target_points) // max(1, int(min_segment_length)),
    )
    if n_segments <= max_segments_for_min_length:
        return n_segments
    if logger is not None:
        logger.warning(
            "Reducing n_segments from %s to %s to satisfy "
            "min_segment_length=%s for target_points=%s.",
            n_segments,
            max_segments_for_min_length,
            min_segment_length,
            target_points,
        )
    return max_segments_for_min_length


def _new_uniform_placement_state(
    *,
    series_length: int,
    channels: int,
) -> _UniformPlacementState:
    return _UniformPlacementState(
        occupied_global=np.zeros(int(series_length), dtype=np.int8),
        occupied_per_channel=np.zeros(
            (int(channels), int(series_length)),
            dtype=np.int8,
        ),
        segments=[],
    )


def _place_uniform_segments(
    *,
    rng: np.random.Generator,
    lengths: Sequence[int],
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> _UniformPlacementState:
    state = _new_uniform_placement_state(
        series_length=series_length,
        channels=channels,
    )
    for length in lengths:
        _place_uniform_segment(
            rng=rng,
            length=int(length),
            series_length=series_length,
            channels=channels,
            max_placement_attempts=max_placement_attempts,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        )
    return state


def _place_uniform_segment(
    *,
    rng: np.random.Generator,
    length: int,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    state: _UniformPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> None:
    max_start = int(series_length) - int(length)
    if _try_random_uniform_placement(
        rng=rng,
        length=length,
        max_start=max_start,
        channels=channels,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        state=state,
        segment_factory=segment_factory,
    ):
        return
    if _try_fallback_uniform_placement(
        rng=rng,
        length=length,
        max_start=max_start,
        channels=channels,
        overlap_policy=overlap_policy,
        state=state,
        segment_factory=segment_factory,
    ):
        return
    raise ValueError(f"Failed to place segment of length {length} without overlap.")


def _try_random_uniform_placement(
    *,
    rng: np.random.Generator,
    length: int,
    max_start: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    state: _UniformPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    for _ in range(int(max_placement_attempts)):
        channel = int(rng.integers(0, int(channels)))
        start = int(rng.integers(0, max_start + 1))
        if _place_if_available(
            start=start,
            end=start + int(length),
            channel=channel,
            length=length,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        ):
            return True
    return False


def _try_fallback_uniform_placement(
    *,
    rng: np.random.Generator,
    length: int,
    max_start: int,
    channels: int,
    overlap_policy: str,
    state: _UniformPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    channel_order = rng.permutation(int(channels)).tolist()
    for channel in channel_order:
        for start in range(max_start + 1):
            if _place_if_available(
                start=start,
                end=start + int(length),
                channel=int(channel),
                length=length,
                overlap_policy=overlap_policy,
                state=state,
                segment_factory=segment_factory,
            ):
                return True
    return False


def _place_if_available(
    *,
    start: int,
    end: int,
    channel: int,
    length: int,
    overlap_policy: str,
    state: _UniformPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    if not is_slot_available(
        start,
        end,
        channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    ):
        return False
    occupy_slot(
        start,
        end,
        channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    )
    state.segments.append(
        segment_factory(
            start=start,
            end=end,
            length=int(length),
            channel=channel,
        )
    )
    return True
