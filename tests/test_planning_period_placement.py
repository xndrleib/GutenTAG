import numpy as np
import pytest

from gutenTAG.tsgen.planning.period_placement import (
    candidate_period_start_indices,
    place_period_locked_segments,
)


def test_candidate_period_start_indices_cover_valid_start_periods() -> None:
    boundaries = np.asarray([0, 10, 20, 30, 40], dtype=int)

    np.testing.assert_array_equal(
        candidate_period_start_indices(boundaries, period_count=2),
        np.asarray([0, 1, 2], dtype=int),
    )


def test_place_period_locked_segments_uses_fallback_without_overlap() -> None:
    segments = place_period_locked_segments(
        rng=np.random.default_rng(31),
        periods_per_segment=[2, 2],
        channel_boundaries={0: np.arange(0, 101, 10, dtype=int)},
        series_length=100,
        channels=1,
        max_placement_attempts=0,
        align_to_period_start=True,
        overlap_policy="global",
    )

    assert len(segments) == 2
    occupied = np.zeros(100, dtype=np.int8)
    for segment in segments:
        assert segment.channel == 0
        assert segment.length == 20
        assert segment.attrs["period_count"] == 2
        assert int(occupied[segment.start : segment.end].sum()) == 0
        occupied[segment.start : segment.end] = 1


def test_place_period_locked_segments_rejects_infeasible_period_count() -> None:
    with pytest.raises(ValueError, match="No channels can satisfy"):
        place_period_locked_segments(
            rng=np.random.default_rng(32),
            periods_per_segment=[4],
            channel_boundaries={0: np.asarray([0, 10, 20], dtype=int)},
            series_length=20,
            channels=1,
            max_placement_attempts=0,
            align_to_period_start=True,
            overlap_policy="global",
        )
