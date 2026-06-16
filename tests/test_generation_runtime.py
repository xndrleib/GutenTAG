from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gutenTAG.tsgen.config import TSGeneratorConfig
from gutenTAG.tsgen.generation_runtime import apply_runtime_anomalies
from gutenTAG.tsgen.planning import SegmentPlan


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
    noise = None
    trend_series = None
    offset = None


def test_apply_runtime_anomalies_uses_config_bound_callbacks() -> None:
    config = TSGeneratorConfig(
        output_root=Path("/tmp/unused"),
        master_seed=1,
        length=8,
        channels=1,
        support_label_mode="strict_segment",
    )
    base = np.zeros((8, 1), dtype=np.float64)

    labels, events = apply_runtime_anomalies(
        config=config,
        anomaly_objects=[
            FakeAnomaly(
                channel=0,
                exact_position=2,
                anomaly_length=3,
                values=np.asarray([1.0, 2.0, 3.0]),
            )
        ],
        segment_plan=[
            SegmentPlan(
                start=2,
                end=5,
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
    )

    np.testing.assert_array_equal(base[2:5, 0], np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(labels[:, 0], np.asarray([0, 0, 1, 1, 1, 0, 0, 0]))
    assert events[0]["params"] == {"offset": 1.0}
    assert events[0]["source_start"] == 2
    assert events[0]["source_end"] == 5
