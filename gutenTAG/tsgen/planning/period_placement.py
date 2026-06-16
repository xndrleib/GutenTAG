"""Period-locked segment placement mechanics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .common import is_slot_available, occupy_slot
from .types import SegmentPlan


@dataclass
class _PeriodPlacementState:
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    segments: list[SegmentPlan]


def place_period_locked_segments(
    *,
    rng: np.random.Generator,
    periods_per_segment: Sequence[int],
    channel_boundaries: Mapping[int, np.ndarray],
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    align_to_period_start: bool,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Place period-count segments against period boundaries."""

    state = _new_period_placement_state(
        series_length=series_length,
        channels=channels,
    )
    for period_count in periods_per_segment:
        _place_period_locked_segment(
            rng=rng,
            period_count=int(period_count),
            channel_boundaries=channel_boundaries,
            series_length=int(series_length),
            max_placement_attempts=max_placement_attempts,
            align_to_period_start=align_to_period_start,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        )
    return state.segments


def _new_period_placement_state(
    *,
    series_length: int,
    channels: int,
) -> _PeriodPlacementState:
    return _PeriodPlacementState(
        occupied_global=np.zeros(int(series_length), dtype=np.int8),
        occupied_per_channel=np.zeros(
            (int(channels), int(series_length)),
            dtype=np.int8,
        ),
        segments=[],
    )


def _place_period_locked_segment(
    *,
    rng: np.random.Generator,
    period_count: int,
    channel_boundaries: Mapping[int, np.ndarray],
    series_length: int,
    max_placement_attempts: int,
    align_to_period_start: bool,
    overlap_policy: str,
    state: _PeriodPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> None:
    eligible_channels = _eligible_period_channels(channel_boundaries, period_count)
    if len(eligible_channels) == 0:
        raise ValueError(
            "No channels can satisfy period_count=%s under period-locked "
            "planner." % period_count
        )
    placed = _place_period_segment_random(
        rng=rng,
        period_count=period_count,
        eligible_channels=eligible_channels,
        channel_boundaries=channel_boundaries,
        series_length=series_length,
        max_placement_attempts=max_placement_attempts,
        align_to_period_start=align_to_period_start,
        overlap_policy=overlap_policy,
        state=state,
        segment_factory=segment_factory,
    )
    if not placed:
        placed = _place_period_segment_fallback(
            rng=rng,
            period_count=period_count,
            eligible_channels=eligible_channels,
            channel_boundaries=channel_boundaries,
            series_length=series_length,
            align_to_period_start=align_to_period_start,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        )
    if not placed:
        raise ValueError(
            "Failed to place period-locked segment without overlap "
            f"(period_count={period_count})."
        )


def _eligible_period_channels(
    channel_boundaries: Mapping[int, np.ndarray],
    period_count: int,
) -> list[int]:
    return [
        channel
        for channel, boundaries in channel_boundaries.items()
        if int(boundaries.shape[0] - 1) >= int(period_count)
    ]


def _place_period_segment_random(
    *,
    rng: np.random.Generator,
    period_count: int,
    eligible_channels: Sequence[int],
    channel_boundaries: Mapping[int, np.ndarray],
    series_length: int,
    max_placement_attempts: int,
    align_to_period_start: bool,
    overlap_policy: str,
    state: _PeriodPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    for _ in range(int(max_placement_attempts)):
        channel = int(eligible_channels[int(rng.integers(0, len(eligible_channels)))])
        boundaries = channel_boundaries[channel]
        candidate_start_period_idxs = candidate_period_start_indices(
            boundaries,
            period_count,
        )
        if candidate_start_period_idxs.size == 0:
            continue
        if align_to_period_start:
            start_period_idx = int(
                candidate_start_period_idxs[
                    int(rng.integers(0, candidate_start_period_idxs.size))
                ]
            )
            start = int(boundaries[start_period_idx])
            end = int(boundaries[start_period_idx + period_count])
        else:
            candidate = _sample_unaligned_period_start(
                rng=rng,
                boundaries=boundaries,
                period_count=period_count,
                series_length=series_length,
            )
            if candidate is None:
                continue
            start, end = candidate
        if _place_if_available(
            start=start,
            end=end,
            channel=channel,
            period_count=period_count,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        ):
            return True
    return False


def _place_period_segment_fallback(
    *,
    rng: np.random.Generator,
    period_count: int,
    eligible_channels: Sequence[int],
    channel_boundaries: Mapping[int, np.ndarray],
    series_length: int,
    align_to_period_start: bool,
    overlap_policy: str,
    state: _PeriodPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    channel_order = rng.permutation(eligible_channels).tolist()
    for channel in channel_order:
        boundaries = channel_boundaries[int(channel)]
        candidate_start_period_idxs = candidate_period_start_indices(
            boundaries,
            period_count,
        )
        if candidate_start_period_idxs.size == 0:
            continue
        if align_to_period_start:
            start_order = candidate_start_period_idxs[
                rng.permutation(candidate_start_period_idxs.size)
            ]
            candidate_values = [int(value) for value in start_order.tolist()]
            for start_candidate in candidate_values:
                start = int(boundaries[start_candidate])
                end = int(boundaries[start_candidate + period_count])
                if _place_if_available(
                    start=start,
                    end=end,
                    channel=int(channel),
                    period_count=period_count,
                    overlap_policy=overlap_policy,
                    state=state,
                    segment_factory=segment_factory,
                ):
                    return True
        else:
            candidate_values = _unaligned_candidate_starts(
                rng=rng,
                boundaries=boundaries,
                period_count=period_count,
                series_length=series_length,
            )
            if candidate_values is None:
                continue
            period_size = int(np.median(np.diff(boundaries)))
            for start_candidate in candidate_values:
                start = int(start_candidate)
                end = int(start + period_count * period_size)
                if _place_if_available(
                    start=start,
                    end=end,
                    channel=int(channel),
                    period_count=period_count,
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
    period_count: int,
    overlap_policy: str,
    state: _PeriodPlacementState,
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
            length=end - start,
            channel=channel,
            attrs={"period_count": int(period_count)},
        )
    )
    return True


def candidate_period_start_indices(
    boundaries: np.ndarray,
    period_count: int,
) -> np.ndarray:
    """Return period-index starts that can cover ``period_count`` periods."""

    return np.arange(
        0,
        int(boundaries.shape[0] - 1) - int(period_count) + 1,
        dtype=int,
    )


def _sample_unaligned_period_start(
    *,
    rng: np.random.Generator,
    boundaries: np.ndarray,
    period_count: int,
    series_length: int,
) -> tuple[int, int] | None:
    period_size = int(np.median(np.diff(boundaries)))
    if period_size <= 0:
        return None
    segment_length = int(period_count * period_size)
    max_start = int(series_length) - segment_length
    if max_start < 0:
        return None
    candidate_starts = np.arange(0, max_start + 1, dtype=int)
    start = int(candidate_starts[int(rng.integers(0, candidate_starts.size))])
    return start, start + segment_length


def _unaligned_candidate_starts(
    *,
    rng: np.random.Generator,
    boundaries: np.ndarray,
    period_count: int,
    series_length: int,
) -> list[int] | None:
    period_size = int(np.median(np.diff(boundaries)))
    if period_size <= 0:
        return None
    segment_length = int(period_count * period_size)
    max_start = int(series_length) - segment_length
    if max_start < 0:
        return None
    candidate_starts = np.arange(0, max_start + 1, dtype=int)
    start_order = candidate_starts[rng.permutation(candidate_starts.size)]
    return [int(value) for value in start_order.tolist()]


__all__ = [
    "candidate_period_start_indices",
    "place_period_locked_segments",
]
