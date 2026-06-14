from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def build_event_record(
    *,
    start: int,
    end: int,
    channel: int,
    anomaly_type: str,
    params: Mapping[str, Any],
    source_start: int,
    source_end: int,
    group_id: int,
    group_channels: Sequence[int],
    intervention_channels: Optional[Sequence[int]] = None,
    anomaly_object: Optional[str] = None,
    channel_visible: Optional[bool] = None,
    purity_hint: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create a normalized event record for `events.json`."""
    normalized_group_channels = [int(ch) for ch in group_channels]
    normalized_intervention = (
        [int(ch) for ch in intervention_channels]
        if intervention_channels is not None
        else [int(channel)]
    )
    context_channels = sorted(
        set(normalized_group_channels + normalized_intervention + [int(channel)])
    )
    if extra is not None and "anchor_channel" in extra:
        context_channels = sorted(
            set(context_channels + [int(extra["anchor_channel"])])
        )
    event_scope = (
        "relation"
        if len(normalized_group_channels) > 1 and channel_visible is False
        else "group" if len(normalized_group_channels) > 1 else "channel"
    )
    record: dict[str, Any] = {
        "start": int(start),
        "end": int(end),
        "channel": int(channel),
        "anomaly_type": str(anomaly_type),
        "group_id": int(group_id),
        "group_channels": normalized_group_channels,
        "intervention_channels": normalized_intervention,
        "operator_target_channels": normalized_intervention,
        "perturbed_channels": normalized_intervention,
        "context_channels": context_channels,
        "event_scope": event_scope,
        "params": params,
        "length": int(end - start),
        "source_start": int(source_start),
        "source_end": int(source_end),
        "anomaly_object": str(anomaly_object or anomaly_type),
        "channel_visible": True if channel_visible is None else bool(channel_visible),
        "purity_hint": str(purity_hint or "channel_visible"),
    }
    if extra is not None:
        for key, value in extra.items():
            record[key] = value
    return record
