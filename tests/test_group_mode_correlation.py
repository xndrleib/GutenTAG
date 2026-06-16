from dataclasses import dataclass

import numpy as np

from gutenTAG.base_oscillations import RandomModeJump
from gutenTAG.generator.group_mode_correlation import apply_mode_correlation_group


@dataclass
class Segment:
    start: int
    end: int
    length: int
    channel: int
    attrs: dict


class BaseChannel:
    def __init__(self, kind: str = RandomModeJump.KIND) -> None:
        self.kind = kind

    def get_base_oscillation_kind(self) -> str:
        return self.kind


class Runtime:
    def compose_window(self, *, base, bo, channel, start, end):
        return np.asarray(base[start:end, channel], dtype=np.float64).copy()

    def resolve_label_bounds(
        self, *, protocol_start, protocol_end, delta, anomaly_type
    ):
        assert anomaly_type == "mode-correlation"
        assert np.any(delta > 0)
        return protocol_start, protocol_end

    def to_builtin(self, payload):
        return dict(payload)


def _segments() -> list[Segment]:
    attrs = {
        "group_id": 3,
        "group_channels": [0, 1, 2],
        "mode_change_aligned": True,
        "mode_grid_aligned": True,
        "support_independent_of_realized_mode_state": True,
        "mode_grid_block_size": 4,
        "mode_grid_start_block": 1,
        "mode_grid_end_block": 2,
        "mode_grid_block_length": 3,
        "mode_grid_min_gap_blocks": 1,
    }
    return [
        Segment(start=1, end=4, length=3, channel=0, attrs=dict(attrs)),
        Segment(start=1, end=4, length=3, channel=1, attrs=dict(attrs)),
        Segment(start=1, end=4, length=3, channel=2, attrs=dict(attrs)),
    ]


def test_mode_correlation_flips_non_anchor_rmj_channels_and_records_events() -> None:
    base = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0],
        ]
    )
    labels = np.zeros_like(base, dtype=int)
    used_positions = {0: [], 1: [], 2: []}

    events = apply_mode_correlation_group(
        group_indices=[0, 1, 2],
        segment_plan=_segments(),
        base=base,
        channel_bos=[BaseChannel(), BaseChannel(), BaseChannel()],
        labels=labels,
        used_positions=used_positions,
        anomaly_type="mode-correlation",
        anomaly_parameters_per_segment=[
            {"role": "anchor"},
            {"role": "first_flip"},
            {"role": "second_flip"},
        ],
        runtime=Runtime(),
    )

    np.testing.assert_allclose(base[1:4, 0], np.asarray([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(base[1:4, 1], np.asarray([-20.0, -30.0, -40.0]))
    np.testing.assert_allclose(base[1:4, 2], np.asarray([-200.0, -300.0, -400.0]))
    assert labels[:, 0].tolist() == [0, 0, 0, 0, 0]
    assert labels[:, 1].tolist() == [0, 1, 1, 1, 0]
    assert labels[:, 2].tolist() == [0, 1, 1, 1, 0]
    assert used_positions == {0: [], 1: [(1, 4)], 2: [(1, 4)]}
    assert [event["channel"] for event in events] == [1, 2]
    assert all(event["anomaly_object"] == "relation_sign_flip" for event in events)
    assert all(event["anchor_channel"] == 0 for event in events)
    assert all(event["flipped_channels"] == [1, 2] for event in events)
    assert all(event["latent_mode_flip"] for event in events)
    assert events[0]["params"] == {"role": "first_flip"}
    assert events[1]["params"] == {"role": "second_flip"}
    assert events[0]["mode_grid_block_size"] == 4


def test_mode_correlation_ignores_non_rmj_groups() -> None:
    base = np.ones((5, 3), dtype=np.float64)

    events = apply_mode_correlation_group(
        group_indices=[0, 1, 2],
        segment_plan=_segments(),
        base=base,
        channel_bos=[BaseChannel(), BaseChannel("sine"), BaseChannel()],
        labels=np.zeros_like(base, dtype=int),
        used_positions={0: [], 1: [], 2: []},
        anomaly_type="mode-correlation",
        anomaly_parameters_per_segment=[{}, {}, {}],
        runtime=Runtime(),
    )

    assert events == []
    np.testing.assert_allclose(base, np.ones((5, 3), dtype=np.float64))
