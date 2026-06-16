import unittest

import numpy as np

from gutenTAG.tsgen.planning import sample_uniform_segments


class TestUniformPlanning(unittest.TestCase):
    def test_samples_requested_total_points_without_global_overlap(self) -> None:
        segments = sample_uniform_segments(
            rng=np.random.default_rng(2),
            target_density=0.30,
            series_length=50,
            channels=3,
            max_placement_attempts=20,
            overlap_policy="global",
            segment_count_range=(3, 3),
            min_segment_length=3,
        )

        self.assertEqual(len(segments), 3)
        self.assertEqual(sum(segment.length for segment in segments), 15)
        self.assertEqual(
            segments,
            sorted(segments, key=lambda item: (item.start, item.channel, item.length)),
        )
        occupied = np.zeros(50, dtype=np.int8)
        for segment in segments:
            self.assertGreaterEqual(segment.length, 3)
            self.assertEqual(segment.end - segment.start, segment.length)
            self.assertEqual(int(occupied[segment.start : segment.end].sum()), 0)
            occupied[segment.start : segment.end] = 1

    def test_reduces_segment_count_to_satisfy_min_length(self) -> None:
        segments = sample_uniform_segments(
            rng=np.random.default_rng(3),
            target_density=0.10,
            series_length=100,
            channels=2,
            max_placement_attempts=10,
            overlap_policy="per-channel",
            segment_count_range=(5, 5),
            min_segment_length=4,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(sum(segment.length for segment in segments), 10)
        self.assertTrue(all(segment.length >= 4 for segment in segments))

    def test_rejects_more_segments_than_series_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds series length"):
            sample_uniform_segments(
                rng=np.random.default_rng(4),
                target_density=1.0,
                series_length=10,
                channels=2,
                max_placement_attempts=10,
                overlap_policy="global",
                segment_count_range=(11, 11),
                min_segment_length=1,
            )


if __name__ == "__main__":
    unittest.main()
