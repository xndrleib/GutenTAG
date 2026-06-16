from dataclasses import dataclass
from typing import Any

from gutenTAG.tsgen.output.instance_events import (
    first_source_event,
    has_effective_support_shrink,
    paired_event_stats,
    source_segment_length,
)


@dataclass
class Segment:
    attrs: dict[str, Any]


def test_source_segment_length_uses_source_bounds_when_present() -> None:
    assert (
        source_segment_length(
            {"start": 4, "end": 7, "source_start": 2, "source_end": 10}
        )
        == 8
    )


def test_has_effective_support_shrink_detects_trimmed_support() -> None:
    assert has_effective_support_shrink(
        {"start": 4, "end": 7, "source_start": 2, "source_end": 10}
    )
    assert not has_effective_support_shrink({"start": 4, "end": 7})


def test_paired_event_stats_counts_channels_groups_and_fallbacks() -> None:
    events = [
        {
            "start": 4,
            "end": 7,
            "length": 3,
            "source_start": 2,
            "source_end": 10,
            "channel": 1,
            "group_id": 3,
        },
        {
            "start": 12,
            "end": 14,
            "length": 2,
            "channel": 0,
            "group_id": 4,
        },
    ]

    stats = paired_event_stats(
        events=events,
        segment_plan=[Segment(attrs={"energy_fallback": True}), Segment(attrs={})],
        channels=2,
    )

    assert stats["segment_lengths"] == [3, 2]
    assert stats["source_segment_lengths"] == [8, 2]
    assert stats["effective_support_shrink_count"] == 1
    assert stats["energy_fallback_count"] == 1
    assert stats["per_channel_counts"] == {"0": 1, "1": 1}
    assert stats["unique_group_ids"] == [3, 4]
    assert first_source_event(events) is events[0]
