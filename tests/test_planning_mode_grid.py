import unittest

import numpy as np

from gutenTAG.tsgen.planning import sample_mode_grid_segments


class TestModeGridPlanning(unittest.TestCase):
    def test_samples_segments_on_complete_rmj_grid(self) -> None:
        segments = sample_mode_grid_segments(
            rng=np.random.default_rng(5),
            target_density=0.25,
            series_length=130,
            channels=3,
            max_placement_attempts=20,
            overlap_policy="global",
            planner_cfg={"planner": "mode_grid_segments"},
            anomaly_policy={},
            anomaly_type="mode-correlation",
            base_period_size=8,
            density_range=(0.0, 1.0),
            segment_count_range=(2, 2),
            min_segment_length=5,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(sum(segment.length // 8 for segment in segments), 4)
        occupied = np.zeros(130, dtype=np.int8)
        for segment in segments:
            self.assertEqual(segment.start % 8, 0)
            self.assertEqual(segment.end % 8, 0)
            self.assertLessEqual(segment.end, 128)
            self.assertTrue(segment.attrs["mode_grid_aligned"])
            self.assertFalse(segment.attrs["mode_change_aligned"])
            self.assertTrue(segment.attrs["support_independent_of_realized_mode_state"])
            self.assertEqual(
                segment.start,
                segment.attrs["mode_grid_start_block"] * 8,
            )
            self.assertEqual(segment.end, segment.attrs["mode_grid_end_block"] * 8)
            self.assertEqual(int(occupied[segment.start : segment.end].sum()), 0)
            occupied[segment.start : segment.end] = 1

    def test_density_range_clamps_target_blocks(self) -> None:
        segments = sample_mode_grid_segments(
            rng=np.random.default_rng(6),
            target_density=0.80,
            series_length=128,
            channels=2,
            max_placement_attempts=20,
            overlap_policy="global",
            planner_cfg={"planner": "mode_grid_segments"},
            anomaly_policy={},
            anomaly_type="mode-correlation",
            base_period_size=8,
            density_range=(0.10, 0.20),
            segment_count_range=(1, 1),
            min_segment_length=5,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].attrs["mode_grid_block_length"], 3)
        self.assertEqual(segments[0].length, 24)

    def test_requires_mode_correlation_and_positive_block_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode-correlation"):
            sample_mode_grid_segments(
                rng=np.random.default_rng(7),
                target_density=0.10,
                series_length=128,
                channels=2,
                max_placement_attempts=20,
                overlap_policy="global",
                planner_cfg={"planner": "mode_grid_segments"},
                anomaly_policy={},
                anomaly_type="mean",
                base_period_size=8,
                density_range=(0.0, 1.0),
                segment_count_range=(1, 1),
                min_segment_length=5,
            )

        with self.assertRaisesRegex(ValueError, "base_period_size"):
            sample_mode_grid_segments(
                rng=np.random.default_rng(8),
                target_density=0.10,
                series_length=128,
                channels=2,
                max_placement_attempts=20,
                overlap_policy="global",
                planner_cfg={"planner": "mode_grid_segments"},
                anomaly_policy={},
                anomaly_type="mode-correlation",
                base_period_size=None,
                density_range=(0.0, 1.0),
                segment_count_range=(1, 1),
                min_segment_length=5,
            )


if __name__ == "__main__":
    unittest.main()
