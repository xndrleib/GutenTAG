from dataclasses import dataclass

import numpy as np

from gutenTAG.generator.group_lag_synchronization import (
    apply_lag_synchronization_group,
)


@dataclass
class Segment:
    start: int
    end: int
    length: int
    channel: int
    attrs: dict


class BaseChannel:
    def get_base_oscillation_kind(self) -> str:
        return "sine"


class Runtime:
    def compose_window(self, *, base, bo, channel, start, end):
        return np.asarray(base[start:end, channel], dtype=np.float64).copy()

    def replace_window(self, *, base, bo, channel, start, end, target_observed):
        base[start:end, channel] = np.asarray(target_observed, dtype=np.float64)

    def compose_noise(self, *, bo, start, end):
        return np.zeros(int(end) - int(start), dtype=np.float64)

    def replace_noise(self, *, bo, start, end, target_noise):
        raise AssertionError("lag-synchronization should not rewrite latent noise")

    def resolve_label_bounds(
        self, *, protocol_start, protocol_end, delta, anomaly_type
    ):
        assert anomaly_type == "lag-synchronization"
        assert np.any(np.asarray(delta, dtype=np.float64) > 0.0)
        return protocol_start, protocol_end

    def to_builtin(self, payload):
        return dict(payload)


def _segments() -> list[Segment]:
    attrs = {"group_id": 11, "group_channels": [0, 1]}
    return [
        Segment(start=2, end=7, length=5, channel=0, attrs=dict(attrs)),
        Segment(start=2, end=7, length=5, channel=1, attrs=dict(attrs)),
    ]


def test_lag_synchronization_shifts_targets_and_records_realized_lag() -> None:
    base = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 5.0],
            [2.0, 1.0],
            [3.0, 7.0],
            [4.0, 3.0],
            [5.0, 9.0],
            [6.0, 4.0],
            [7.0, 8.0],
            [8.0, 6.0],
        ],
        dtype=np.float64,
    )
    before = base.copy()
    labels = np.zeros_like(base, dtype=int)
    used_positions = {0: [], 1: []}

    events = apply_lag_synchronization_group(
        group_indices=[0, 1],
        segment_plan=_segments(),
        base=base,
        channel_bos=[BaseChannel(), BaseChannel()],
        labels=labels,
        used_positions=used_positions,
        anomaly_type="lag-synchronization",
        anomaly_parameters_per_segment=[
            {"role": "anchor"},
            {"lag_steps": 2, "transition_length": 1},
        ],
        runtime=Runtime(),
    )

    assert len(events) == 1
    np.testing.assert_allclose(base[:, 0], before[:, 0])
    assert not np.allclose(base[2:7, 1], before[2:7, 1])
    assert labels[:, 0].tolist() == [0] * 9
    assert labels[:, 1].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert used_positions == {0: [], 1: [(2, 7)]}
    assert events[0]["anomaly_object"] == "lag_synchronization_shift"
    assert events[0]["purity_hint"] == "not_pure_local"
    assert events[0]["channel_visible"] is True
    assert events[0]["anchor_channel"] == 0
    assert events[0]["realized_lag_steps"] == 2


def test_lag_synchronization_ignores_single_channel_groups() -> None:
    base = np.ones((5, 1), dtype=np.float64)
    labels = np.zeros_like(base, dtype=int)

    events = apply_lag_synchronization_group(
        group_indices=[0],
        segment_plan=[
            Segment(
                start=1,
                end=4,
                length=3,
                channel=0,
                attrs={"group_id": 1, "group_channels": [0]},
            )
        ],
        base=base,
        channel_bos=[BaseChannel()],
        labels=labels,
        used_positions={0: []},
        anomaly_type="lag-synchronization",
        anomaly_parameters_per_segment=[{}],
        runtime=Runtime(),
    )

    assert events == []
    np.testing.assert_allclose(base, np.ones((5, 1), dtype=np.float64))
    np.testing.assert_array_equal(labels, np.zeros_like(labels))
