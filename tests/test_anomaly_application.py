import unittest
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from gutenTAG.generator.anomaly_application import (
    AnomalyApplicationRuntime,
    apply_anomalies,
)


@dataclass
class Segment:
    start: int
    length: int
    channel: int
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Protocol:
    start: int
    end: int
    subsequences: list[np.ndarray]


@dataclass
class FakeAnomaly:
    channel: int
    exact_position: int
    anomaly_length: int
    values: np.ndarray

    def generate(self, _ctx: Any) -> Protocol:
        return Protocol(
            start=self.exact_position,
            end=self.exact_position + self.anomaly_length,
            subsequences=[self.values],
        )


class FakeBaseOscillation:
    pass


def runtime() -> AnomalyApplicationRuntime:
    def compose_window(**kwargs: Any) -> np.ndarray:
        base = kwargs["base"]
        channel = int(kwargs["channel"])
        start = int(kwargs["start"])
        end = int(kwargs["end"])
        return np.array(base[start:end, channel], copy=True)

    def resolve_label_bounds(**kwargs: Any) -> tuple[int, int]:
        return int(kwargs["protocol_start"]), int(kwargs["protocol_end"])

    def to_builtin(value: Any) -> Any:
        if isinstance(value, Mapping):
            return dict(value)
        return value

    return AnomalyApplicationRuntime(
        compose_window=compose_window,
        replace_window=lambda **_kwargs: None,
        compose_noise=lambda **_kwargs: np.array([], dtype=np.float64),
        replace_noise=lambda **_kwargs: None,
        resolve_label_bounds=resolve_label_bounds,
        normalize_subsequence=lambda subsequence, _length: subsequence,
        to_builtin=to_builtin,
    )


class TestAnomalyApplication(unittest.TestCase):
    def test_applies_single_anomaly_and_builds_event(self) -> None:
        base = np.zeros((8, 1), dtype=np.float64)

        labels, events = apply_anomalies(
            anomaly_objects=[
                FakeAnomaly(
                    channel=0,
                    exact_position=2,
                    anomaly_length=3,
                    values=np.array([1.0, 2.0, 3.0]),
                )
            ],
            segment_plan=[
                Segment(
                    start=2,
                    length=3,
                    channel=0,
                    attrs={"group_channels": [0], "anomaly_object": "mean"},
                )
            ],
            base=base,
            channel_bos=[FakeBaseOscillation()],
            anomaly_seed=41,
            anomaly_type="mean",
            anomaly_parameters_per_segment=[{"offset": 1.0}],
            series_length=8,
            channels=1,
            runtime=runtime(),
        )

        np.testing.assert_array_equal(base[2:5, 0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(labels[:, 0], np.array([0, 0, 1, 1, 1, 0, 0, 0]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start"], 2)
        self.assertEqual(events[0]["end"], 5)
        self.assertEqual(events[0]["params"], {"offset": 1.0})
        self.assertEqual(events[0]["source_start"], 2)
        self.assertEqual(events[0]["source_end"], 5)

    def test_events_are_sorted_by_support_and_channel(self) -> None:
        base = np.zeros((10, 1), dtype=np.float64)

        _labels, events = apply_anomalies(
            anomaly_objects=[
                FakeAnomaly(
                    channel=0,
                    exact_position=6,
                    anomaly_length=2,
                    values=np.array([6.0, 7.0]),
                ),
                FakeAnomaly(
                    channel=0,
                    exact_position=1,
                    anomaly_length=2,
                    values=np.array([1.0, 2.0]),
                ),
            ],
            segment_plan=[
                Segment(start=6, length=2, channel=0),
                Segment(start=1, length=2, channel=0),
            ],
            base=base,
            channel_bos=[FakeBaseOscillation()],
            anomaly_seed=42,
            anomaly_type="mean",
            anomaly_parameters_per_segment=[{"offset": 6.0}, {"offset": 1.0}],
            series_length=10,
            channels=1,
            runtime=runtime(),
        )

        self.assertEqual([event["start"] for event in events], [1, 6])


if __name__ == "__main__":
    unittest.main()
