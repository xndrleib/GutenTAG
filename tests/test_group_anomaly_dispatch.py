import numpy as np
import pytest

from gutenTAG.generator import group_anomalies
from gutenTAG.generator.group_anomalies import GroupAnomalyRuntime, apply_group_anomaly


def _runtime() -> GroupAnomalyRuntime:
    return GroupAnomalyRuntime(
        compose_window=lambda **_: np.array([]),
        replace_window=lambda **_: None,
        compose_noise=lambda **_: np.array([]),
        replace_noise=lambda **_: None,
        resolve_label_bounds=lambda **_: (0, 0),
        to_builtin=lambda value: value,
    )


def test_apply_group_anomaly_dispatches_through_handler_table(monkeypatch) -> None:
    captured = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return [{"event": "ok"}]

    monkeypatch.setitem(
        group_anomalies._GROUP_ANOMALY_HANDLERS,
        "mode-correlation",
        handler,
    )
    runtime = _runtime()

    events = apply_group_anomaly(
        anomaly_type="mode-correlation",
        group_indices=[0, 1],
        segment_plan=[],
        base=np.zeros((2, 8)),
        channel_bos=[],
        labels=np.zeros((2, 8), dtype=int),
        used_positions={},
        anomaly_parameters_per_segment=[],
        runtime=runtime,
    )

    assert events == [{"event": "ok"}]
    assert captured["anomaly_type"] == "mode-correlation"
    assert captured["group_indices"] == [0, 1]
    assert captured["runtime"] is runtime


def test_apply_group_anomaly_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported group-level anomaly type"):
        apply_group_anomaly(
            anomaly_type="unknown-group-op",
            group_indices=[],
            segment_plan=[],
            base=np.zeros((1, 4)),
            channel_bos=[],
            labels=np.zeros((1, 4), dtype=int),
            used_positions={},
            anomaly_parameters_per_segment=[],
            runtime=_runtime(),
        )
