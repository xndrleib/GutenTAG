from dataclasses import dataclass

import numpy as np

from gutenTAG.generator.group_channel_rewiring import apply_channel_rewiring_group


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
        raise AssertionError("observed-window test should not rewrite latent noise")

    def resolve_label_bounds(
        self, *, protocol_start, protocol_end, delta, anomaly_type
    ):
        assert anomaly_type == "channel-rewiring"
        assert np.any(np.asarray(delta, dtype=np.float64) > 0.0)
        return protocol_start, protocol_end

    def to_builtin(self, payload):
        return dict(payload)


def _segments() -> list[Segment]:
    attrs = {"group_id": 9, "group_channels": [0, 1]}
    return [
        Segment(start=1, end=6, length=5, channel=0, attrs=dict(attrs)),
        Segment(start=1, end=6, length=5, channel=1, attrs=dict(attrs)),
    ]


def test_channel_rewiring_rotates_pair_and_records_per_channel_events() -> None:
    base = np.asarray(
        [
            [0.0, 6.0],
            [1.0, 5.0],
            [2.0, 4.0],
            [4.0, 6.0],
            [7.0, 3.0],
            [11.0, 2.0],
            [12.0, 1.0],
        ],
        dtype=np.float64,
    )
    before = base.copy()
    labels = np.zeros_like(base, dtype=int)
    used_positions = {0: [], 1: []}

    events = apply_channel_rewiring_group(
        group_indices=[0, 1],
        segment_plan=_segments(),
        base=base,
        channel_bos=[BaseChannel(), BaseChannel()],
        labels=labels,
        used_positions=used_positions,
        anomaly_type="channel-rewiring",
        anomaly_parameters_per_segment=[
            {"rotation_degrees": 45.0, "transition_length": 1},
            {},
        ],
        runtime=Runtime(),
    )

    assert len(events) == 2
    assert labels[:, 0].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert labels[:, 1].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert used_positions == {0: [(1, 6)], 1: [(1, 6)]}
    assert not np.allclose(base[1:6, 0], before[1:6, 0])
    assert not np.allclose(base[1:6, 1], before[1:6, 1])
    for event in events:
        assert event["anomaly_object"] == "paired_structure_rotation"
        assert event["purity_hint"] == "not_pure_local"
        assert event["channel_visible"] is True
        assert event["rewired_pair"] == [0, 1]
        assert event["rotation_degrees"] == 45.0
        assert event["injection_level"] == "observed_window"


def test_channel_rewiring_ignores_single_channel_groups() -> None:
    base = np.ones((5, 1), dtype=np.float64)
    labels = np.zeros_like(base, dtype=int)

    events = apply_channel_rewiring_group(
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
        anomaly_type="channel-rewiring",
        anomaly_parameters_per_segment=[{}],
        runtime=Runtime(),
    )

    assert events == []
    np.testing.assert_allclose(base, np.ones((5, 1), dtype=np.float64))
    np.testing.assert_array_equal(labels, np.zeros_like(labels))
