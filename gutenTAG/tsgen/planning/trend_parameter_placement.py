"""Slot placement for trend-parameter-aware segment planning."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import occupy_slot
from .energy_selection import sample_uniform_slot
from .types import SegmentPlan


@dataclass
class _TrendPlacementState:
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    segments: list[SegmentPlan]


def place_trend_segments(
    *,
    rng: np.random.Generator,
    lengths: Sequence[int],
    attrs: Sequence[Mapping[str, Any]],
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Place trend-aware segments while preserving supplied segment metadata."""

    state = _new_trend_placement_state(
        series_length=series_length,
        channels=channels,
    )
    for segment_length, segment_attrs in zip(lengths, attrs):
        _place_trend_segment(
            rng=rng,
            segment_length=int(segment_length),
            attrs=segment_attrs,
            series_length=series_length,
            channels=channels,
            max_placement_attempts=max_placement_attempts,
            overlap_policy=overlap_policy,
            state=state,
            segment_factory=segment_factory,
        )
    return state.segments


def _new_trend_placement_state(
    *,
    series_length: int,
    channels: int,
) -> _TrendPlacementState:
    return _TrendPlacementState(
        occupied_global=np.zeros(int(series_length), dtype=np.int8),
        occupied_per_channel=np.zeros(
            (int(channels), int(series_length)),
            dtype=np.int8,
        ),
        segments=[],
    )


def _place_trend_segment(
    *,
    rng: np.random.Generator,
    segment_length: int,
    attrs: Mapping[str, Any],
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    state: _TrendPlacementState,
    segment_factory: Callable[..., SegmentPlan],
) -> None:
    selected = sample_uniform_slot(
        rng=rng,
        length=segment_length,
        series_length=series_length,
        channels=channels,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        occupied_global=state.occupied_global,
        occupied_per_channel=state.occupied_per_channel,
        segment_factory=segment_factory,
    )
    if selected is None:
        raise ValueError(
            "Failed to place a trend_parameter_aware segment without overlap "
            f"(length={segment_length})."
        )
    occupy_slot(
        selected.start,
        selected.end,
        selected.channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    )
    selected.attrs.update(copy.deepcopy(dict(attrs)))
    state.segments.append(selected)


__all__ = ["place_trend_segments"]
