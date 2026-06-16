"""Period-locked frequency segment planning."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import (
    sample_bounded_integer_lengths,
)
from .period_geometry import (
    compute_period_boundaries,
    resolve_channel_boundaries as _resolve_channel_boundaries,
    sanitize_period_boundaries,
)
from .period_placement import place_period_locked_segments
from .types import SegmentPlan

__all__ = [
    "compute_period_boundaries",
    "sample_period_locked_frequency_segments",
    "sanitize_period_boundaries",
]


@dataclass(frozen=True)
class _PeriodLockedGeometry:
    length: int
    n_channels: int


@dataclass(frozen=True)
class _PeriodCountContext:
    channel_period_counts: dict[int, int]
    max_whole_periods: int
    avg_period_size: float


@dataclass(frozen=True)
class _PeriodRange:
    min_periods: int
    max_periods: int


@dataclass(frozen=True)
class _PeriodTarget:
    total_period_points: int
    total_points: int
    effective_min_periods: int
    effective_max_periods: int


def sample_period_locked_frequency_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    planner_cfg: Mapping[str, Any],
    base_period_size: int | None,
    period_boundaries: Iterable[int] | None,
    period_boundaries_by_channel: Mapping[int, Iterable[int] | None] | None,
    segment_count_range: Sequence[int],
    logger: logging.Logger | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample frequency segments aligned to oscillator periods.

    Parameters
    ----------
    rng, target_density, series_length, channels
        Randomness, target support density, and generated-series shape.
    max_placement_attempts, overlap_policy, planner_cfg
        Placement retry budget, overlap policy, and period planner options.
    base_period_size, period_boundaries, period_boundaries_by_channel
        Fallback, shared, or channel-specific period boundary sources.
    segment_count_range, logger, segment_factory
        Segment-count bounds, optional warning logger, and segment factory.

    Returns
    -------
    list[SegmentPlan]
        Sorted period-locked segment plan.
    """

    geometry = _period_locked_geometry(series_length, channels)
    channel_boundaries = _resolve_channel_boundaries(
        series_length=geometry.length,
        channels=geometry.n_channels,
        base_period_size=base_period_size,
        period_boundaries=period_boundaries,
        period_boundaries_by_channel=period_boundaries_by_channel,
    )
    count_context = _period_count_context(channel_boundaries, geometry.length)
    periods_per_segment = _sample_periods_per_segment(
        rng=rng,
        target_density=target_density,
        segment_count_range=segment_count_range,
        planner_cfg=planner_cfg,
        count_context=count_context,
        geometry=geometry,
        logger=logger,
    )
    align_to_period_start = bool(planner_cfg.get("align_to_period_start", True))
    segments = place_period_locked_segments(
        rng=rng,
        periods_per_segment=periods_per_segment,
        channel_boundaries=channel_boundaries,
        series_length=geometry.length,
        channels=geometry.n_channels,
        max_placement_attempts=max_placement_attempts,
        align_to_period_start=align_to_period_start,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    )
    segments.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
    return segments


def _period_locked_geometry(
    series_length: int,
    channels: int,
) -> _PeriodLockedGeometry:
    return _PeriodLockedGeometry(length=int(series_length), n_channels=int(channels))


def _period_count_context(
    channel_boundaries: Mapping[int, np.ndarray],
    length: int,
) -> _PeriodCountContext:
    channel_period_counts = {
        channel: int(boundaries.shape[0] - 1)
        for channel, boundaries in channel_boundaries.items()
    }
    channel_period_counts = {
        channel: count for channel, count in channel_period_counts.items() if count > 0
    }
    if len(channel_period_counts) == 0:
        raise ValueError(
            "period_locked_frequency planner requires at least one channel "
            "with period boundaries."
        )
    return _PeriodCountContext(
        channel_period_counts=channel_period_counts,
        max_whole_periods=int(max(channel_period_counts.values())),
        avg_period_size=float(
            np.median(
                [
                    float(length) / float(period_count)
                    for period_count in channel_period_counts.values()
                ]
            )
        ),
    )


def _sample_periods_per_segment(
    *,
    rng: np.random.Generator,
    target_density: float,
    segment_count_range: Sequence[int],
    planner_cfg: Mapping[str, Any],
    count_context: _PeriodCountContext,
    geometry: _PeriodLockedGeometry,
    logger: logging.Logger | None,
) -> list[int]:
    period_range = _periods_per_segment_range(
        planner_cfg,
        max_whole_periods=count_context.max_whole_periods,
    )
    target = _period_target(
        target_density=target_density,
        period_range=period_range,
        count_context=count_context,
        geometry=geometry,
    )
    low, high = _feasible_segment_count_range(segment_count_range, target, logger)
    n_segments = int(rng.integers(low, high + 1))
    return sample_bounded_integer_lengths(
        rng=rng,
        total_points=target.total_period_points,
        n_segments=n_segments,
        min_value=target.effective_min_periods,
        max_value=target.effective_max_periods,
    )


def _periods_per_segment_range(
    planner_cfg: Mapping[str, Any],
    *,
    max_whole_periods: int,
) -> _PeriodRange:
    periods_per_segment_range = _parse_pair_int(
        planner_cfg.get("periods_per_segment_range", [2, 4]),
        "periods_per_segment_range",
    )
    if periods_per_segment_range[0] > periods_per_segment_range[1]:
        periods_per_segment_range = (
            periods_per_segment_range[1],
            periods_per_segment_range[0],
        )
    min_periods = max(1, int(periods_per_segment_range[0]))
    max_periods = max(min_periods, int(periods_per_segment_range[1]))
    max_periods = min(max_periods, max_whole_periods)
    if min_periods > max_periods:
        raise ValueError(
            "periods_per_segment_range is infeasible for available period "
            "boundaries."
        )
    return _PeriodRange(min_periods=min_periods, max_periods=max_periods)


def _period_target(
    *,
    target_density: float,
    period_range: _PeriodRange,
    count_context: _PeriodCountContext,
    geometry: _PeriodLockedGeometry,
) -> _PeriodTarget:
    target_points = int(round(float(target_density) * geometry.length))
    target_points = max(1, min(target_points, geometry.length))
    total_period_points = int(
        np.clip(
            int(round(target_points / count_context.avg_period_size)),
            1,
            count_context.max_whole_periods,
        )
    )
    effective_min_periods = min(period_range.min_periods, total_period_points)
    effective_max_periods = min(period_range.max_periods, total_period_points)
    if effective_min_periods > effective_max_periods:
        effective_min_periods = effective_max_periods
    return _PeriodTarget(
        total_period_points=total_period_points,
        total_points=int(round(total_period_points * count_context.avg_period_size)),
        effective_min_periods=effective_min_periods,
        effective_max_periods=effective_max_periods,
    )


def _feasible_segment_count_range(
    segment_count_range: Sequence[int],
    target: _PeriodTarget,
    logger: logging.Logger | None,
) -> tuple[int, int]:
    low_count, high_count = int(segment_count_range[0]), int(segment_count_range[1])
    if low_count > high_count:
        low_count, high_count = high_count, low_count

    min_required_segments = int(
        np.ceil(target.total_period_points / float(target.effective_max_periods))
    )
    max_allowed_segments = int(
        np.floor(target.total_period_points / float(target.effective_min_periods))
    )
    low = max(low_count, min_required_segments)
    high = min(high_count, max_allowed_segments)
    if low <= high:
        return low, high

    low = min_required_segments
    high = max_allowed_segments
    if logger is not None:
        logger.warning(
            "period_locked_frequency planner adjusted segment_count_range "
            "from requested=%s to feasible=%s for target_points=%s.",
            [low_count, high_count],
            [low, high],
            target.total_points,
        )
    return low, high


def _parse_pair_int(raw: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"'{field_name}' must be a list/tuple with exactly two values."
        )
    return int(raw[0]), int(raw[1])
