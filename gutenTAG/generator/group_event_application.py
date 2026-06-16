"""Shared event application plumbing for group-level anomalies."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from .event_metadata import build_event_record


def apply_group_channel_effect(
    *,
    base: np.ndarray,
    channel_bos: Sequence[Any],
    labels: np.ndarray,
    used_positions: MutableMapping[int, list[tuple[int, int]]],
    runtime: Any,
    before_window: np.ndarray,
    source_start: int,
    source_end: int,
    channel: int,
    anomaly_type: str,
    group_id: int,
    group_channels: Sequence[int],
    intervention_channels: Sequence[int],
    anomaly_object: str,
    channel_visible: bool,
    purity_hint: str,
    params: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Update labels/used spans and return an event record for one channel."""

    active_channel = int(channel)
    after_window = runtime.compose_window(
        base=base,
        bo=channel_bos[active_channel],
        channel=active_channel,
        start=source_start,
        end=source_end,
    )
    effective_delta = np.abs(after_window - before_window)
    label_start, label_end = runtime.resolve_label_bounds(
        protocol_start=source_start,
        protocol_end=source_end,
        delta=effective_delta,
        anomaly_type=anomaly_type,
    )
    labels[label_start:label_end, active_channel] = 1
    used_positions[active_channel].append((int(source_start), int(source_end)))
    return build_event_record(
        start=int(label_start),
        end=int(label_end),
        channel=active_channel,
        anomaly_type=anomaly_type,
        group_id=int(group_id),
        group_channels=[int(ch) for ch in group_channels],
        intervention_channels=[int(ch) for ch in intervention_channels],
        anomaly_object=anomaly_object,
        channel_visible=bool(channel_visible),
        purity_hint=purity_hint,
        params=runtime.to_builtin(params),
        source_start=int(source_start),
        source_end=int(source_end),
        extra=dict(extra),
    )


__all__ = ["apply_group_channel_effect"]
