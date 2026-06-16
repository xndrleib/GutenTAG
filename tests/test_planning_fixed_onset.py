import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gutenTAG.tsgen.planning.fixed_onset import (
    event_extra_from_segment_attrs,
    inject_fixed_onset_split_context,
    requested_pre_context_from_split,
    sample_fixed_first_onset_segments,
    temporal_union_density,
)


@dataclass
class Segment:
    start: int
    end: int
    length: int
    channel: int
    attrs: dict[str, Any] = field(default_factory=dict)


class TestFixedOnsetPlanning(unittest.TestCase):
    def test_requested_pre_context_from_split(self) -> None:
        self.assertEqual(requested_pre_context_from_split("ctx003"), 3)
        self.assertEqual(requested_pre_context_from_split("ctx120"), 120)
        self.assertIsNone(requested_pre_context_from_split("train"))
        self.assertIsNone(requested_pre_context_from_split("ctxabc"))

    def test_injects_split_context_only_for_fixed_onset_planner(self) -> None:
        uniform = inject_fixed_onset_split_context(
            {"planner": "uniform_segments"},
            "ctx010",
        )
        self.assertEqual(uniform, {"planner": "uniform_segments"})

        fixed = inject_fixed_onset_split_context(
            {"planner": "fixed_first_onset_segments"},
            "ctx010",
        )
        self.assertEqual(fixed["requested_pre_context"], 10)
        self.assertEqual(fixed["onset_bucket"], "ctx010")

        explicit = inject_fixed_onset_split_context(
            {
                "planner": "fixed_first_onset_segments",
                "requested_pre_context": 4,
                "onset_bucket": "custom",
            },
            "ctx010",
        )
        self.assertEqual(explicit["requested_pre_context"], 4)
        self.assertEqual(explicit["onset_bucket"], "custom")

    def test_sample_fixed_first_onset_exact_segment(self) -> None:
        segments = sample_fixed_first_onset_segments(
            rng=np.random.default_rng(1),
            anomaly_type="mean",
            planner_cfg={
                "planner": "fixed_first_onset_segments",
                "requested_pre_context": 10,
                "length": 16,
                "alignment_strategy": "exact",
            },
            series_length=128,
            channels=3,
            segment_factory=Segment,
        )

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.start, 10)
        self.assertEqual(segment.end, 26)
        self.assertEqual(segment.length, 16)
        self.assertEqual(segment.attrs["alignment_strategy"], "exact")
        self.assertEqual(segment.attrs["alignment_error"], 0)

    def test_sample_fixed_first_onset_mode_grid_alignment(self) -> None:
        segments = sample_fixed_first_onset_segments(
            rng=np.random.default_rng(1),
            anomaly_type="mode-correlation",
            planner_cfg={
                "planner": "fixed_first_onset_segments",
                "requested_pre_context": 10,
                "length": 15,
                "alignment_strategy": "mode_grid",
            },
            series_length=128,
            channels=3,
            segment_factory=Segment,
            base_period_size=8,
        )

        segment = segments[0]
        self.assertEqual(segment.start, 16)
        self.assertEqual(segment.end, 32)
        self.assertEqual(segment.length, 16)
        self.assertEqual(segment.attrs["alignment_error"], 6)
        self.assertTrue(segment.attrs["mode_grid_aligned"])
        self.assertTrue(segment.attrs["support_independent_of_realized_mode_state"])

    def test_extremum_fixed_onset_is_one_point(self) -> None:
        segments = sample_fixed_first_onset_segments(
            rng=np.random.default_rng(1),
            anomaly_type="extremum",
            planner_cfg={
                "planner": "fixed_first_onset_segments",
                "requested_pre_context": 3,
                "length": 16,
            },
            series_length=128,
            channels=3,
            segment_factory=Segment,
        )

        self.assertEqual(segments[0].start, 3)
        self.assertEqual(segments[0].end, 4)
        self.assertEqual(segments[0].length, 1)

    def test_temporal_union_density_does_not_double_count_channels(self) -> None:
        segments = [
            Segment(start=0, end=10, length=10, channel=0),
            Segment(start=0, end=10, length=10, channel=1),
            Segment(start=20, end=30, length=10, channel=0),
        ]

        self.assertAlmostEqual(temporal_union_density(segments, 100), 0.2)

    def test_event_extra_from_segment_attrs_adds_realized_onset(self) -> None:
        extra = event_extra_from_segment_attrs(
            {
                "requested_pre_context": 10,
                "onset_bucket": "ctx010",
                "alignment_strategy": "exact",
                "alignment_error": 2,
                "unrelated": "ignored",
            },
            source_start=12,
            support_start=13,
        )

        self.assertEqual(extra["requested_pre_context"], 10)
        self.assertEqual(extra["actual_source_start"], 12)
        self.assertEqual(extra["actual_support_start"], 13)
        self.assertNotIn("unrelated", extra)


if __name__ == "__main__":
    unittest.main()
