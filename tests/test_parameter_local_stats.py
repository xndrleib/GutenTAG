import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gutenTAG.tsgen.parameters import (
    annotate_segment_local_stats,
    window_residual_scale,
)


@dataclass
class Segment:
    start: int
    end: int
    channel: int
    attrs: dict[str, Any] = field(default_factory=dict)


class TestParameterLocalStats(unittest.TestCase):
    def test_annotate_segment_local_stats_attaches_window_metrics(self) -> None:
        clean = np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 8.0],
            ],
            dtype=np.float64,
        )
        segments = [Segment(start=1, end=4, channel=1)]

        annotate_segment_local_stats(segments, clean)

        attrs = segments[0].attrs
        expected_rms = np.sqrt((2.0**2 + 4.0**2 + 8.0**2) / 3)
        self.assertAlmostEqual(attrs["window_rms"], expected_rms)
        self.assertEqual(attrs["window_peak"], 8.0)
        self.assertEqual(attrs["window_median"], 4.0)
        self.assertIn("window_robust_scale", attrs)
        self.assertIn("window_residual_scale", attrs)

    def test_annotate_segment_local_stats_skips_invalid_windows(self) -> None:
        segments = [
            Segment(start=0, end=2, channel=3),
            Segment(start=2, end=1, channel=0),
        ]

        annotate_segment_local_stats(segments, np.ones((3, 1), dtype=np.float64))

        self.assertEqual(segments[0].attrs, {})
        self.assertEqual(segments[1].attrs, {})

    def test_window_residual_scale_handles_empty_and_nonempty_windows(self) -> None:
        self.assertEqual(window_residual_scale(np.array([]), "linear"), 0.0)
        self.assertGreater(
            window_residual_scale(np.array([0.0, 2.0, 0.0]), "mean"),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
