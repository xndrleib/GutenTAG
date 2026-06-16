import unittest
from dataclasses import dataclass

from gutenTAG.generator.anomaly_objects import (
    build_anomalies,
    build_single_anomaly_kind,
)


@dataclass
class Segment:
    start: int
    length: int
    channel: int


class TestAnomalyObjectBuilders(unittest.TestCase):
    def test_build_anomalies_rejects_mismatched_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have equal length"):
            build_anomalies(
                anomaly_type="mean",
                anomaly_parameters_per_segment=[],
                segment_plan=[Segment(start=1, length=5, channel=0)],
            )

    def test_builds_builtin_anomaly_object_for_single_channel_type(self) -> None:
        anomalies = build_anomalies(
            anomaly_type="mean",
            anomaly_parameters_per_segment=[{"offset": 1.0}],
            segment_plan=[Segment(start=3, length=7, channel=1)],
        )

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].exact_position, 3)
        self.assertEqual(anomalies[0].anomaly_length, 7)
        self.assertEqual(anomalies[0].channel, 1)
        self.assertEqual(len(anomalies[0].anomaly_kinds), 1)

    def test_group_level_anomaly_object_defers_operator_kind(self) -> None:
        anomalies = build_anomalies(
            anomaly_type="mode-correlation",
            anomaly_parameters_per_segment=[{}],
            segment_plan=[Segment(start=3, length=7, channel=1)],
        )

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(len(anomalies[0].anomaly_kinds), 0)

    def test_builds_trend_anomaly_kind_from_oscillation_parameters(self) -> None:
        anomaly_kind = build_single_anomaly_kind(
            "trend",
            {
                "oscillation": {"kind": "sine", "frequency": 2.0, "amplitude": 1.0},
                "transition_length": 2,
                "boundary_mode": "inside_window_zero_endpoints",
                "envelope_kind": "sine2",
                "min_effect_delta": 0.2,
            },
            anomaly_length=12,
        )

        self.assertIsNotNone(anomaly_kind)


if __name__ == "__main__":
    unittest.main()
