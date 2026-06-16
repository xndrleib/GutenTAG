import unittest

import numpy as np

from gutenTAG.tsgen.planning import sample_mode_boundary_segments


class TestModeBoundaryPlanning(unittest.TestCase):
    def test_samples_segments_on_realized_mode_boundaries(self) -> None:
        clean = np.zeros((50, 2), dtype=np.float64)
        clean[:, 0] = np.repeat([1.0, -1.0, 1.0, -1.0, 1.0], 10)
        clean[:, 1] = clean[:, 0]

        segments = sample_mode_boundary_segments(
            rng=np.random.default_rng(12),
            target_density=0.40,
            series_length=50,
            channels=2,
            max_placement_attempts=20,
            overlap_policy="global",
            planner_cfg={"planner": "mode_boundary_segments"},
            anomaly_type="mode-correlation",
            clean_values=clean,
            segment_count_range=(2, 2),
            min_segment_length=5,
        )

        boundaries = {10, 20, 30, 40}
        self.assertEqual(len(segments), 2)
        occupied = np.zeros(50, dtype=np.int8)
        for segment in segments:
            self.assertIn(segment.start, boundaries)
            self.assertIn(segment.end, boundaries)
            self.assertGreater(segment.end, segment.start)
            self.assertTrue(segment.attrs["mode_change_aligned"])
            self.assertEqual(segment.attrs["planner"], "mode_boundary_segments")
            self.assertEqual(int(occupied[segment.start : segment.end].sum()), 0)
            occupied[segment.start : segment.end] = 1

    def test_falls_back_to_uniform_when_mode_changes_are_insufficient(self) -> None:
        clean = np.ones((50, 2), dtype=np.float64)

        segments = sample_mode_boundary_segments(
            rng=np.random.default_rng(13),
            target_density=0.20,
            series_length=50,
            channels=2,
            max_placement_attempts=20,
            overlap_policy="global",
            planner_cfg={"planner": "mode_boundary_segments"},
            anomaly_type="mode-correlation",
            clean_values=clean,
            segment_count_range=(2, 2),
            min_segment_length=4,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(sum(segment.length for segment in segments), 10)
        for segment in segments:
            self.assertNotIn("mode_change_aligned", segment.attrs)
            self.assertEqual(segment.end - segment.start, segment.length)

    def test_requires_mode_correlation_and_clean_values(self) -> None:
        clean = np.ones((20, 2), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "mode-correlation"):
            sample_mode_boundary_segments(
                rng=np.random.default_rng(14),
                target_density=0.20,
                series_length=20,
                channels=2,
                max_placement_attempts=20,
                overlap_policy="global",
                planner_cfg={"planner": "mode_boundary_segments"},
                anomaly_type="mean",
                clean_values=clean,
                segment_count_range=(1, 1),
                min_segment_length=4,
            )

        with self.assertRaisesRegex(ValueError, "clean_values"):
            sample_mode_boundary_segments(
                rng=np.random.default_rng(15),
                target_density=0.20,
                series_length=20,
                channels=2,
                max_placement_attempts=20,
                overlap_policy="global",
                planner_cfg={"planner": "mode_boundary_segments"},
                anomaly_type="mode-correlation",
                clean_values=None,
                segment_count_range=(1, 1),
                min_segment_length=4,
            )


if __name__ == "__main__":
    unittest.main()
