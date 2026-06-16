import numpy as np

from gutenTAG.tsgen.planning.energy_selection import (
    available_start_mask,
    sample_uniform_slot,
    select_energy_candidate,
)
from gutenTAG.tsgen.planning.types import SegmentPlan


def test_available_start_mask_marks_only_empty_windows() -> None:
    occupied = np.asarray([0, 1, 0, 0, 1], dtype=np.int8)

    np.testing.assert_array_equal(
        available_start_mask(occupied, length=2),
        np.asarray([False, False, True, False]),
    )


def test_select_energy_candidate_filters_by_threshold_and_availability() -> None:
    selected = select_energy_candidate(
        rng=np.random.default_rng(1),
        length=2,
        channels=2,
        metric_mode="rms",
        weighted_sampling=False,
        min_residual_scale=0.0,
        availability_by_channel={
            0: np.asarray([False, True, True], dtype=bool),
            1: np.asarray([False, False, False], dtype=bool),
        },
        rms_cache={
            (0, 2): np.asarray([0.1, 3.0, 0.5], dtype=np.float64),
            (1, 2): np.asarray([10.0, 10.0, 10.0], dtype=np.float64),
        },
        peak_cache={},
        residual_scale_cache={},
        thresholds_rms={0: 2.0, 1: 9.0},
        thresholds_peak={},
        segment_factory=SegmentPlan,
    )

    assert selected is not None
    assert selected.channel == 0
    assert selected.start == 1
    assert selected.end == 3


def test_sample_uniform_slot_uses_exhaustive_fallback_when_random_attempts_exhausted() -> (
    None
):
    selected = sample_uniform_slot(
        rng=np.random.default_rng(2),
        length=2,
        series_length=5,
        channels=1,
        max_placement_attempts=0,
        overlap_policy="global",
        occupied_global=np.asarray([1, 1, 0, 0, 1], dtype=np.int8),
        occupied_per_channel=np.asarray([[1, 1, 0, 0, 1]], dtype=np.int8),
        segment_factory=SegmentPlan,
    )

    assert selected is not None
    assert selected.channel == 0
    assert selected.start == 2
    assert selected.end == 4
