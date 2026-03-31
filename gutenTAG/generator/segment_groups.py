from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, List, Optional

import numpy as np


def expand_segments_by_channel_policy(
    segments: Iterable[Any],
    rng: np.random.Generator,
    n_channels: int,
    segment_factory: Callable[..., Any],
    channel_policy: str = "single-random",
) -> List[Any]:
    """Expand segment plans according to the configured channel policy."""
    items = list(segments)
    if channel_policy == "single-random":
        expanded: List[Any] = []
        for group_id, segment in enumerate(items):
            attrs = copy.deepcopy(dict(getattr(segment, "attrs", {})))
            attrs.setdefault("group_id", int(group_id))
            attrs.setdefault("group_channels", [int(segment.channel)])
            expanded.append(
                segment_factory(
                    start=int(segment.start),
                    end=int(segment.end),
                    length=int(segment.length),
                    channel=int(segment.channel),
                    attrs=attrs,
                )
            )
        return expanded

    expanded = []
    all_channels = list(range(int(n_channels)))
    for group_id, segment in enumerate(items):
        if channel_policy == "all-channels":
            group_channels = all_channels
        elif channel_policy == "paired-random":
            remaining = [ch for ch in all_channels if ch != int(segment.channel)]
            partner = int(remaining[int(rng.integers(0, len(remaining)))])
            group_channels = sorted({int(segment.channel), partner})
        else:
            raise ValueError(
                "channel_policy must be one of {'single-random','paired-random','all-channels'}"
            )
        for channel in group_channels:
            attrs = copy.deepcopy(dict(getattr(segment, "attrs", {})))
            attrs["group_id"] = int(group_id)
            attrs["group_channels"] = [int(ch) for ch in group_channels]
            expanded.append(
                segment_factory(
                    start=int(segment.start),
                    end=int(segment.end),
                    length=int(segment.length),
                    channel=int(channel),
                    attrs=attrs,
                )
            )

    expanded.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
    return expanded


def group_indices_by_id(segments: Iterable[Any]) -> dict[int, list[int]]:
    """Collect segment indices by group id."""
    grouped: dict[int, list[int]] = {}
    for idx, segment in enumerate(segments):
        group_id = int(getattr(segment, "attrs", {}).get("group_id", idx))
        grouped.setdefault(group_id, []).append(int(idx))
    return grouped


def collect_group_channels(segments: Iterable[Any]) -> list[int]:
    """Collect the declared group channels from a segment set."""
    channels = {
        int(ch)
        for segment in segments
        for ch in getattr(segment, "attrs", {}).get(
            "group_channels", [int(getattr(segment, "channel"))]
        )
    }
    return sorted(channels)


def group_source_bounds(segments: Iterable[Any]) -> tuple[int, int]:
    """Return inclusive group source bounds."""
    items = list(segments)
    if len(items) == 0:
        return 0, 0
    return int(min(segment.start for segment in items)), int(
        max(segment.end for segment in items)
    )


def resolve_channel_policy(
    default_policy: str,
    special_policy: Optional[dict[str, Any]] = None,
) -> str:
    """Resolve the active channel policy for an anomaly type."""
    special = special_policy or {}
    return str(special.get("channel_policy", default_policy))
