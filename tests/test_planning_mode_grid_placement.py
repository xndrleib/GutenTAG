import numpy as np

from gutenTAG.tsgen.planning.mode_grid_placement import (
    mode_grid_candidate,
    place_mode_grid_segments,
)
from gutenTAG.tsgen.planning.mode_grid_types import ModeGridGeometry, ModeGridOptions


def test_mode_grid_candidate_expands_occupancy_slot_by_gap_points() -> None:
    candidate = mode_grid_candidate(
        channel=1,
        start_block=2,
        block_length=3,
        geometry=ModeGridGeometry(length=100, n_channels=2, block_size=10, n_blocks=10),
        options=ModeGridOptions(
            min_blocks=1,
            max_blocks=3,
            min_gap_blocks=1,
            min_gap_points=10,
        ),
    )

    assert candidate.start == 20
    assert candidate.end == 50
    assert candidate.slot_start == 10
    assert candidate.slot_end == 60


def test_place_mode_grid_segments_uses_fallback_and_records_metadata() -> None:
    segments = place_mode_grid_segments(
        rng=np.random.default_rng(41),
        block_lengths=[2, 2],
        geometry=ModeGridGeometry(length=80, n_channels=1, block_size=8, n_blocks=10),
        options=ModeGridOptions(
            min_blocks=2,
            max_blocks=2,
            min_gap_blocks=1,
            min_gap_points=8,
        ),
        max_placement_attempts=0,
        overlap_policy="global",
    )

    assert len(segments) == 2
    occupied = np.zeros(80, dtype=np.int8)
    for segment in segments:
        assert segment.channel == 0
        assert segment.length == 16
        assert segment.attrs["planner"] == "mode_grid_segments"
        assert segment.attrs["mode_grid_aligned"] is True
        assert segment.attrs["mode_grid_block_size"] == 8
        assert segment.attrs["mode_grid_block_length"] == 2
        slot_start = max(0, segment.start - 8)
        slot_end = min(80, segment.end + 8)
        assert int(occupied[slot_start:slot_end].sum()) == 0
        occupied[slot_start:slot_end] = 1
