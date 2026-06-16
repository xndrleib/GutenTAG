"""Mode-boundary segment placement mechanics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .common import is_slot_available, occupy_slot
from .mode_boundary_types import ModeBoundaryGeometry
from .types import SegmentPlan


@dataclass(frozen=True)
class _ModeBoundaryCandidate:
    start: int
    end: int
    channel: int


@dataclass
class _ModeBoundaryPlacementState:
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    segments: list[SegmentPlan]


def place_mode_boundary_segments(
    *,
    rng: np.random.Generator,
    lengths: Sequence[int],
    change_boundaries: np.ndarray,
    geometry: ModeBoundaryGeometry,
    min_segment_length: int,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Place segments on realized mode-change boundaries."""

    state = _new_placement_state(geometry)
    for segment_length in lengths:
        _place_mode_boundary_segment(
            rng=rng,
            segment_length=int(segment_length),
            change_boundaries=change_boundaries,
            geometry=geometry,
            min_segment_length=min_segment_length,
            state=state,
            max_placement_attempts=max_placement_attempts,
            overlap_policy=overlap_policy,
            segment_factory=segment_factory,
        )
    return state.segments


def _new_placement_state(
    geometry: ModeBoundaryGeometry,
) -> _ModeBoundaryPlacementState:
    return _ModeBoundaryPlacementState(
        occupied_global=np.zeros(geometry.length, dtype=np.int8),
        occupied_per_channel=np.zeros(
            (geometry.n_channels, geometry.length),
            dtype=np.int8,
        ),
        segments=[],
    )


def _place_mode_boundary_segment(
    *,
    rng: np.random.Generator,
    segment_length: int,
    change_boundaries: np.ndarray,
    geometry: ModeBoundaryGeometry,
    min_segment_length: int,
    state: _ModeBoundaryPlacementState,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> None:
    if _try_random_boundary_placement(
        rng=rng,
        segment_length=segment_length,
        change_boundaries=change_boundaries,
        geometry=geometry,
        min_segment_length=min_segment_length,
        state=state,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    ):
        return
    if _try_fallback_boundary_placement(
        rng=rng,
        segment_length=segment_length,
        change_boundaries=change_boundaries,
        geometry=geometry,
        min_segment_length=min_segment_length,
        state=state,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    ):
        return
    raise ValueError(
        "Failed to place a mode_boundary segment of target length " f"{segment_length}."
    )


def _try_random_boundary_placement(
    *,
    rng: np.random.Generator,
    segment_length: int,
    change_boundaries: np.ndarray,
    geometry: ModeBoundaryGeometry,
    min_segment_length: int,
    state: _ModeBoundaryPlacementState,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    for _ in range(int(max_placement_attempts)):
        channel = int(rng.integers(0, geometry.n_channels))
        start = int(change_boundaries[int(rng.integers(0, len(change_boundaries) - 1))])
        candidate = mode_boundary_candidate(
            channel=channel,
            start=start,
            segment_length=segment_length,
            change_boundaries=change_boundaries,
            min_segment_length=min_segment_length,
        )
        if _place_candidate_if_available(
            candidate,
            state=state,
            overlap_policy=overlap_policy,
            segment_factory=segment_factory,
        ):
            return True
    return False


def _try_fallback_boundary_placement(
    *,
    rng: np.random.Generator,
    segment_length: int,
    change_boundaries: np.ndarray,
    geometry: ModeBoundaryGeometry,
    min_segment_length: int,
    state: _ModeBoundaryPlacementState,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    channel_order = rng.permutation(geometry.n_channels).tolist()
    for channel in channel_order:
        for start in change_boundaries[:-1]:
            candidate = mode_boundary_candidate(
                channel=int(channel),
                start=int(start),
                segment_length=segment_length,
                change_boundaries=change_boundaries,
                min_segment_length=min_segment_length,
            )
            if _place_candidate_if_available(
                candidate,
                state=state,
                overlap_policy=overlap_policy,
                segment_factory=segment_factory,
            ):
                return True
    return False


def mode_boundary_candidate(
    *,
    channel: int,
    start: int,
    segment_length: int,
    change_boundaries: np.ndarray,
    min_segment_length: int,
) -> _ModeBoundaryCandidate | None:
    """Return a boundary-aligned candidate closest to requested length."""

    end = nearest_valid_end(
        change_boundaries,
        start=start,
        target_length=int(segment_length),
        min_segment_length=min_segment_length,
    )
    if end is None:
        return None
    return _ModeBoundaryCandidate(start=int(start), end=int(end), channel=int(channel))


def _place_candidate_if_available(
    candidate: _ModeBoundaryCandidate | None,
    *,
    state: _ModeBoundaryPlacementState,
    overlap_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> bool:
    if candidate is None:
        return False
    if not is_slot_available(
        candidate.start,
        candidate.end,
        candidate.channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    ):
        return False
    occupy_slot(
        candidate.start,
        candidate.end,
        candidate.channel,
        overlap_policy,
        state.occupied_global,
        state.occupied_per_channel,
    )
    state.segments.append(
        make_mode_boundary_segment(
            segment_factory,
            start=candidate.start,
            end=candidate.end,
            channel=candidate.channel,
        )
    )
    return True


def nearest_valid_end(
    change_boundaries: np.ndarray,
    *,
    start: int,
    target_length: int,
    min_segment_length: int,
) -> int | None:
    """Return the nearest boundary end that satisfies minimum length."""

    end_candidates = change_boundaries[
        change_boundaries >= int(start) + max(1, int(min_segment_length))
    ]
    if end_candidates.size == 0:
        return None
    candidate_lengths = end_candidates - int(start)
    return int(
        end_candidates[int(np.argmin(np.abs(candidate_lengths - int(target_length))))]
    )


def make_mode_boundary_segment(
    segment_factory: Callable[..., SegmentPlan],
    *,
    start: int,
    end: int,
    channel: int,
) -> SegmentPlan:
    """Build a standard mode-boundary segment plan."""

    return segment_factory(
        start=int(start),
        end=int(end),
        length=int(end - start),
        channel=int(channel),
        attrs={
            "planner": "mode_boundary_segments",
            "mode_change_aligned": True,
        },
    )


__all__ = [
    "make_mode_boundary_segment",
    "mode_boundary_candidate",
    "nearest_valid_end",
    "place_mode_boundary_segments",
]
