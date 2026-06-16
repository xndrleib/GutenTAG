import unittest

import numpy as np

from gutenTAG.tsgen.planning import (
    compute_period_boundaries,
    sample_period_locked_frequency_segments,
)


class TestPeriodLockedPlanning(unittest.TestCase):
    def test_compute_period_boundaries_matches_sampling_grid(self) -> None:
        boundaries = compute_period_boundaries(5.0, series_length=1200)

        self.assertIsNotNone(boundaries)
        assert boundaries is not None
        self.assertEqual(int(boundaries[0]), 0)
        self.assertEqual(int(boundaries[-1]), 1200)
        self.assertTrue(np.all(np.diff(boundaries) > 0))
        self.assertAlmostEqual(float(np.median(np.diff(boundaries))), 20.0, delta=1.0)

    def test_samples_period_aligned_segments(self) -> None:
        boundaries = np.arange(0, 101, 10, dtype=int)

        segments = sample_period_locked_frequency_segments(
            rng=np.random.default_rng(19),
            target_density=0.40,
            series_length=100,
            channels=2,
            max_placement_attempts=30,
            overlap_policy="global",
            planner_cfg={
                "planner": "period_locked_frequency",
                "periods_per_segment_range": [2, 2],
                "align_to_period_start": True,
            },
            base_period_size=None,
            period_boundaries=boundaries,
            period_boundaries_by_channel=None,
            segment_count_range=(2, 2),
        )

        self.assertEqual(len(segments), 2)
        occupied = np.zeros(100, dtype=np.int8)
        for segment in segments:
            self.assertEqual(segment.attrs["period_count"], 2)
            self.assertEqual(segment.start % 10, 0)
            self.assertEqual(segment.end % 10, 0)
            self.assertEqual(segment.length, 20)
            self.assertEqual(int(occupied[segment.start : segment.end].sum()), 0)
            occupied[segment.start : segment.end] = 1

    def test_uses_base_period_size_when_boundaries_are_missing(self) -> None:
        segments = sample_period_locked_frequency_segments(
            rng=np.random.default_rng(20),
            target_density=0.30,
            series_length=100,
            channels=1,
            max_placement_attempts=30,
            overlap_policy="global",
            planner_cfg={
                "planner": "period_locked_frequency",
                "periods_per_segment_range": [3, 3],
                "align_to_period_start": True,
            },
            base_period_size=10,
            period_boundaries=None,
            period_boundaries_by_channel=None,
            segment_count_range=(1, 1),
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start % 10, 0)
        self.assertEqual(segments[0].length, 30)
        self.assertEqual(segments[0].attrs["period_count"], 3)

    def test_requires_boundaries_or_base_period_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "period boundaries"):
            sample_period_locked_frequency_segments(
                rng=np.random.default_rng(21),
                target_density=0.30,
                series_length=100,
                channels=1,
                max_placement_attempts=30,
                overlap_policy="global",
                planner_cfg={"planner": "period_locked_frequency"},
                base_period_size=None,
                period_boundaries=None,
                period_boundaries_by_channel=None,
                segment_count_range=(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
