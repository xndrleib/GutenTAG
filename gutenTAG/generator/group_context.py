"""Shared context resolution for group-level anomaly operators."""

from __future__ import annotations

from typing import Any

from .segment_groups import collect_group_channels, group_source_bounds


def resolve_group_context(
    group_indices: list[int],
    segment_plan: list[Any],
) -> tuple[list[Any], int, int, int, list[int], dict[int, int]]:
    """Resolve common group bounds, channels, and segment index lookups."""

    group_segments = [segment_plan[idx] for idx in group_indices]
    source_start, source_end = group_source_bounds(group_segments)
    group_id = int(group_segments[0].attrs.get("group_id", group_indices[0]))
    group_channels = collect_group_channels(group_segments)
    segment_idx_by_channel = {
        int(segment_plan[idx].channel): int(idx) for idx in group_indices
    }
    return (
        group_segments,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    )


__all__ = ["resolve_group_context"]
