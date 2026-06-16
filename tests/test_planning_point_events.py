import unittest

import numpy as np

from gutenTAG.tsgen.planning import sample_point_event_segments


class TestPointEventPlanning(unittest.TestCase):
    def test_segment_count_range_controls_point_count(self) -> None:
        segments = sample_point_event_segments(
            rng=np.random.default_rng(9),
            target_density=0.90,
            series_length=100,
            channels=3,
            overlap_policy="global",
            planner_cfg={"planner": "point_events_from_density"},
            segment_count_range=(5, 5),
            density_range=(0.0, 0.1),
        )

        self.assertEqual(len(segments), 5)
        self.assertEqual(
            segments,
            sorted(segments, key=lambda item: (item.start, item.channel, item.length)),
        )
        starts = [segment.start for segment in segments]
        self.assertEqual(len(starts), len(set(starts)))
        for segment in segments:
            self.assertEqual(segment.end, segment.start + 1)
            self.assertEqual(segment.length, 1)
            self.assertGreaterEqual(segment.channel, 0)
            self.assertLess(segment.channel, 3)

    def test_density_range_clamps_point_count(self) -> None:
        segments = sample_point_event_segments(
            rng=np.random.default_rng(10),
            target_density=0.90,
            series_length=50,
            channels=2,
            overlap_policy="global",
            planner_cfg={"planner": "point_events_from_density"},
            density_range=(0.10, 0.20),
        )

        self.assertEqual(len(segments), 10)

    def test_density_policy_keeps_at_least_one_point(self) -> None:
        segments = sample_point_event_segments(
            rng=np.random.default_rng(11),
            target_density=0.0,
            series_length=50,
            channels=2,
            overlap_policy="global",
            planner_cfg={"planner": "point_events_from_density"},
        )

        self.assertEqual(len(segments), 1)


if __name__ == "__main__":
    unittest.main()
