"""Per-instance event-derived summary statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class SegmentLike(Protocol):
    """Minimal segment contract required for summary metadata."""

    attrs: Mapping[str, Any]


def paired_event_stats(
    *,
    events: Sequence[Mapping[str, Any]],
    segment_plan: Sequence[SegmentLike],
    channels: int,
) -> dict[str, Any]:
    """Compute event-derived fields for paired instance summaries."""

    segment_lengths = [int(event["length"]) for event in events]
    return {
        "segment_lengths": segment_lengths,
        "source_segment_lengths": [source_segment_length(event) for event in events],
        "effective_support_shrink_count": int(
            sum(1 for event in events if has_effective_support_shrink(event))
        ),
        "energy_fallback_count": int(
            sum(
                1
                for segment in segment_plan
                if bool(segment.attrs.get("energy_fallback", False))
            )
        ),
        "per_channel_counts": {
            str(channel): int(sum(1 for event in events if event["channel"] == channel))
            for channel in range(int(channels))
        },
        "unique_group_ids": sorted(
            {int(event.get("group_id", idx)) for idx, event in enumerate(events)}
        ),
        "first_event": first_source_event(events),
    }


def source_segment_length(event: Mapping[str, Any]) -> int:
    """Return event source-support length."""

    return int(event.get("source_end", event["end"])) - int(
        event.get("source_start", event["start"])
    )


def has_effective_support_shrink(event: Mapping[str, Any]) -> bool:
    """Return whether labeled support is smaller than source support."""

    return int(event.get("source_start", event["start"])) != int(event["start"]) or int(
        event.get("source_end", event["end"])
    ) != int(event["end"])


def first_source_event(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the earliest event by source start and channel."""

    return min(
        events,
        key=lambda event: (
            int(event.get("source_start", event["start"])),
            int(event["channel"]),
        ),
    )


__all__ = [
    "SegmentLike",
    "first_source_event",
    "has_effective_support_shrink",
    "paired_event_stats",
    "source_segment_length",
]
