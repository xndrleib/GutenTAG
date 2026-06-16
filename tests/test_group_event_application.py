import numpy as np

from gutenTAG.generator.group_event_application import apply_group_channel_effect


class DummyRuntime:
    def compose_window(self, *, base, bo, channel, start, end):
        return np.asarray(base[start:end, channel], dtype=np.float64)

    def resolve_label_bounds(
        self, *, protocol_start, protocol_end, delta, anomaly_type
    ):
        assert anomaly_type == "mode-correlation"
        assert np.any(delta > 0)
        return protocol_start, protocol_end

    def to_builtin(self, payload):
        return dict(payload)


def test_apply_group_channel_effect_updates_labels_used_positions_and_event() -> None:
    base = np.asarray([[0.0], [2.0], [2.0], [0.0]])
    labels = np.zeros((4, 1), dtype=int)
    used_positions = {0: []}

    event = apply_group_channel_effect(
        base=base,
        channel_bos=[object()],
        labels=labels,
        used_positions=used_positions,
        runtime=DummyRuntime(),
        before_window=np.asarray([0.0, 0.0]),
        source_start=1,
        source_end=3,
        channel=0,
        anomaly_type="mode-correlation",
        group_id=7,
        group_channels=(0, 1),
        intervention_channels=(0,),
        anomaly_object="relation_sign_flip",
        channel_visible=False,
        purity_hint="relation_change",
        params={"strength": 1.0},
        extra={"anchor_channel": 1},
    )

    assert labels[:, 0].tolist() == [0, 1, 1, 0]
    assert used_positions == {0: [(1, 3)]}
    assert event["start"] == 1
    assert event["end"] == 3
    assert event["channel"] == 0
    assert event["group_id"] == 7
    assert event["group_channels"] == [0, 1]
    assert event["intervention_channels"] == [0]
    assert event["anomaly_object"] == "relation_sign_flip"
    assert event["purity_hint"] == "relation_change"
    assert event["params"] == {"strength": 1.0}
    assert event["anchor_channel"] == 1
