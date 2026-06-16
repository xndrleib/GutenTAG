import numpy as np

from gutenTAG.tsgen.planning.mode_boundary_placement import (
    mode_boundary_candidate,
    nearest_valid_end,
    place_mode_boundary_segments,
)
from gutenTAG.tsgen.planning.mode_boundary_types import ModeBoundaryGeometry


def test_nearest_valid_end_prefers_closest_valid_boundary() -> None:
    boundaries = np.asarray([10, 20, 35, 50], dtype=int)

    assert (
        nearest_valid_end(
            boundaries,
            start=10,
            target_length=18,
            min_segment_length=5,
        )
        == 35
    )


def test_mode_boundary_candidate_requires_valid_end() -> None:
    boundaries = np.asarray([10, 20], dtype=int)

    assert (
        mode_boundary_candidate(
            channel=0,
            start=10,
            segment_length=20,
            change_boundaries=boundaries,
            min_segment_length=15,
        )
        is None
    )

    candidate = mode_boundary_candidate(
        channel=1,
        start=10,
        segment_length=8,
        change_boundaries=boundaries,
        min_segment_length=5,
    )

    assert candidate is not None
    assert candidate.start == 10
    assert candidate.end == 20
    assert candidate.channel == 1


def test_place_mode_boundary_segments_uses_fallback_without_overlap() -> None:
    segments = place_mode_boundary_segments(
        rng=np.random.default_rng(51),
        lengths=[10, 10],
        change_boundaries=np.asarray([10, 20, 30, 40], dtype=int),
        geometry=ModeBoundaryGeometry(length=50, n_channels=1),
        min_segment_length=5,
        max_placement_attempts=0,
        overlap_policy="global",
    )

    assert len(segments) == 2
    occupied = np.zeros(50, dtype=np.int8)
    for segment in segments:
        assert segment.attrs["planner"] == "mode_boundary_segments"
        assert segment.attrs["mode_change_aligned"] is True
        assert int(occupied[segment.start : segment.end].sum()) == 0
        occupied[segment.start : segment.end] = 1
