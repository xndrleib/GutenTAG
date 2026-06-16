"""Mode-grid segment placement mechanics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import is_slot_available, occupy_slot
from .mode_grid_types import ModeGridGeometry, ModeGridOptions
from .types import SegmentPlan


@dataclass(frozen=True)
class _ModeGridCandidate:
    channel: int
    start_block: int
    end_block: int
    block_length: int
    start: int
    end: int
    slot_start: int
    slot_end: int


@dataclass
class _ModeGridPlacementState:
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    segments: list[SegmentPlan]


def place_mode_grid_segments(
    *,
    rng: np.random.Generator,
    block_lengths: Sequence[int],
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Place mode-grid-aligned segment supports."""

    state = _new_placement_state(geometry)
    for block_length in block_lengths:
        _place_mode_grid_segment(
            rng=rng,
            block_length=int(block_length),
            geometry=geometry,
            options=options,
            state=state,
            max_placement_attempts=max_placement_attempts,
            overlap_policy=overlap_policy,
            segment_factory=segment_factory,
        )
    return state.segments


def _new_placement_state(geometry: ModeGridGeometry) -> _ModeGridPlacementState:
    return _ModeGridPlacementState(
        occupied_global=np.zeros(geometry.length, dtype=np.int8),
        occupied_per_channel=np.zeros(
            (geometry.n_channels, geometry.length),
            dtype=np.int8,
        ),
        segments=[],
    )


def _place_mode_grid_segment(
    *,
    rng: np.random.Generator,
    block_length: int,
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
    state: _ModeGridPlacementState,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> None:
    if _try_random_placement(
        rng=rng,
        block_length=block_length,
        geometry=geometry,
        options=options,
        state=state,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    ):
        return
    if _try_fallback_placement(
        rng=rng,
        block_length=block_length,
        geometry=geometry,
        options=options,
        state=state,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    ):
        return
    raise ValueError(
        "Failed to place a mode_grid segment of target block length " f"{block_length}."
    )


def _try_random_placement(
    *,
    rng: np.random.Generator,
    block_length: int,
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
    state: _ModeGridPlacementState,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    max_start_block = max(0, geometry.n_blocks - int(block_length))
    for _ in range(int(max_placement_attempts)):
        candidate = mode_grid_candidate(
            channel=int(rng.integers(0, geometry.n_channels)),
            start_block=int(rng.integers(0, max_start_block + 1)),
            block_length=block_length,
            geometry=geometry,
            options=options,
        )
        if _place_candidate_if_available(
            candidate,
            state=state,
            overlap_policy=overlap_policy,
            segment_factory=segment_factory,
            geometry=geometry,
            options=options,
        ):
            return True
    return False


def _try_fallback_placement(
    *,
    rng: np.random.Generator,
    block_length: int,
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
    state: _ModeGridPlacementState,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    max_start_block = max(0, geometry.n_blocks - int(block_length))
    channel_order = rng.permutation(geometry.n_channels).tolist()
    for channel in channel_order:
        for start_block in range(max_start_block + 1):
            candidate = mode_grid_candidate(
                channel=int(channel),
                start_block=int(start_block),
                block_length=block_length,
                geometry=geometry,
                options=options,
            )
            if _place_candidate_if_available(
                candidate,
                state=state,
                overlap_policy=overlap_policy,
                segment_factory=segment_factory,
                geometry=geometry,
                options=options,
            ):
                return True
    return False


def mode_grid_candidate(
    *,
    channel: int,
    start_block: int,
    block_length: int,
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
) -> _ModeGridCandidate:
    """Return a mode-grid placement candidate with padded occupancy slot."""

    end_block = int(start_block) + int(block_length)
    start = int(start_block * geometry.block_size)
    end = int(min(geometry.length, end_block * geometry.block_size))
    return _ModeGridCandidate(
        channel=int(channel),
        start_block=int(start_block),
        end_block=int(end_block),
        block_length=int(block_length),
        start=start,
        end=end,
        slot_start=max(0, start - options.min_gap_points),
        slot_end=min(geometry.length, end + options.min_gap_points),
    )


def _place_candidate_if_available(
    candidate: _ModeGridCandidate,
    *,
    state: _ModeGridPlacementState,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
) -> bool:
    if candidate.end <= candidate.start:
        return False
    if not is_slot_available(
        candidate.slot_start,
        candidate.slot_end,
        candidate.channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    ):
        return False
    occupy_slot(
        candidate.slot_start,
        candidate.slot_end,
        candidate.channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    )
    state.segments.append(
        _make_mode_grid_segment(
            candidate,
            segment_factory=segment_factory,
            geometry=geometry,
            options=options,
        )
    )
    return True


def _make_mode_grid_segment(
    candidate: _ModeGridCandidate,
    *,
    segment_factory: Callable[..., SegmentPlan],
    geometry: ModeGridGeometry,
    options: ModeGridOptions,
) -> SegmentPlan:
    return segment_factory(
        start=candidate.start,
        end=candidate.end,
        length=int(candidate.end - candidate.start),
        channel=candidate.channel,
        attrs=mode_grid_attrs(
            block_size=geometry.block_size,
            start_block=candidate.start_block,
            end_block=candidate.end_block,
            block_length=candidate.block_length,
            min_gap_blocks=options.min_gap_blocks,
        ),
    )


def mode_grid_attrs(
    *,
    block_size: int,
    start_block: int,
    end_block: int,
    block_length: int,
    min_gap_blocks: int,
) -> dict[str, Any]:
    """Return standard mode-grid segment metadata."""

    return {
        "planner": "mode_grid_segments",
        "mode_grid_aligned": True,
        "mode_change_aligned": False,
        "support_independent_of_realized_mode_state": True,
        "mode_grid_block_size": int(block_size),
        "mode_grid_start_block": int(start_block),
        "mode_grid_end_block": int(end_block),
        "mode_grid_block_length": int(block_length),
        "mode_grid_min_gap_blocks": int(min_gap_blocks),
    }


__all__ = [
    "mode_grid_attrs",
    "mode_grid_candidate",
    "place_mode_grid_segments",
]
