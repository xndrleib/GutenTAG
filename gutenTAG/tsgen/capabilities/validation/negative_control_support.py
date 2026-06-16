"""Support-window and candidate context helpers for negative controls."""

from __future__ import annotations

from collections.abc import Sequence

from ..dataset import EventGroup, InstanceRecord
from ..numerics import channel_subsets
from ..ontology import witness_requires_projection_size
from ..protocol import CapabilityProtocol
from .negative_control_types import NegativeControlEventContext


def negative_control_event_context(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    protocol: CapabilityProtocol,
) -> NegativeControlEventContext:
    """Resolve candidate subsets, witnesses, and shifted support for an event."""

    candidates = _candidate_channels(instance, group)
    subsets = tuple(channel_subsets(candidates, protocol.max_projection_size))
    witnesses = _admissible_witnesses(subsets, protocol.detection_witnesses)
    relation_like = is_relation_like(group)
    clean_witnesses = _relation_witnesses(witnesses) if relation_like else witnesses
    shifted_start, shifted_end = shifted_support(instance, group, protocol)
    marginal_subsets = tuple(subset for subset in subsets if len(subset) == 1)
    return NegativeControlEventContext(
        subsets=subsets,
        witnesses=witnesses,
        relation_like=relation_like,
        clean_control_witnesses=clean_witnesses,
        shifted_start=shifted_start,
        shifted_end=shifted_end,
        marginal_subsets=marginal_subsets,
    )


def shifted_support(
    instance: InstanceRecord,
    group: EventGroup,
    protocol: CapabilityProtocol,
) -> tuple[int, int]:
    """Return an event-length support window outside declared event contexts."""

    length = int(instance.length)
    width = max(1, group.length)
    protected = tuple(
        _declared_support_bounds(other, length) for other in instance.event_groups
    )
    for start in _shifted_support_candidates(group, width, length):
        end = start + width
        context_start, context_end = _context_bounds(start, end, length, protocol)
        if not _overlaps_any(start, end, protected) and not _overlaps_any(
            context_start,
            context_end,
            protected,
        ):
            return start, end
    if int(group.end) + width <= length:
        return int(group.end), int(group.end) + width
    if int(group.start) - width >= 0:
        return int(group.start) - width, int(group.start)
    return 0, min(length, width)


def is_relation_like(group: EventGroup) -> bool:
    """Return whether an event should be evaluated with relation witnesses."""

    return (
        len(group.group_channels) >= 2
        or len(group.context_channels) >= 2
        or "relation" in str(group.semantic_scope)
        or "dependence" in str(group.constraint_tag)
    )


def _candidate_channels(instance: InstanceRecord, group: EventGroup) -> tuple[int, ...]:
    channels = tuple(
        sorted(
            set(group.group_channels)
            | set(group.context_channels)
            | set(group.intervention_channels)
        )
    )
    return channels if channels else tuple(range(instance.channels))


def _admissible_witnesses(
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
) -> tuple[str, ...]:
    max_size = max((len(subset) for subset in subsets), default=1)
    return tuple(
        witness
        for witness in witnesses
        if witness_requires_projection_size(witness) <= max_size
    )


def _relation_witnesses(witnesses: Sequence[str]) -> tuple[str, ...]:
    relation_witnesses = tuple(
        witness
        for witness in witnesses
        if witness in {"correlation_shift", "covariance_shift"}
    )
    return relation_witnesses if relation_witnesses else tuple(witnesses)


def _shifted_support_candidates(
    group: EventGroup,
    width: int,
    length: int,
) -> tuple[int, ...]:
    starts: list[int] = []
    for multiplier in range(1, max(2, length // max(1, width)) + 2):
        starts.extend(
            (
                int(group.end) + (multiplier - 1) * width,
                int(group.start) - multiplier * width,
            )
        )
    starts.extend(range(0, max(1, length - width + 1), max(1, width)))
    deduped: list[int] = []
    seen: set[int] = set()
    for start in starts:
        clipped = max(0, min(int(start), max(0, length - width)))
        if clipped in seen:
            continue
        seen.add(clipped)
        deduped.append(clipped)
    return tuple(deduped)


def _declared_support_bounds(group: EventGroup, length: int) -> tuple[int, int]:
    start = group.source_start if group.source_end > group.source_start else group.start
    end = group.source_end if group.source_end > group.source_start else group.end
    lo = max(0, min(int(start), int(length)))
    hi = max(lo, min(int(end), int(length)))
    return lo, hi


def _context_bounds(
    start: int,
    end: int,
    length: int,
    protocol: CapabilityProtocol,
) -> tuple[int, int]:
    width = max(1, int(end) - int(start))
    radius = max(
        int(protocol.min_context_points),
        int(protocol.context_window_multiplier) * width,
    )
    return max(0, int(start) - radius), min(int(length), int(end) + radius)


def _overlaps_any(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> bool:
    for other_start, other_end in intervals:
        if int(start) < int(other_end) and int(end) > int(other_start):
            return True
    return False


__all__ = [
    "is_relation_like",
    "negative_control_event_context",
    "shifted_support",
]
