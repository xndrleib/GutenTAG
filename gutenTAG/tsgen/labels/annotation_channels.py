"""Annotation channel masks derived from event-level truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .masks import build_label_masks


@dataclass(frozen=True)
class AnnotationChannel:
    """A concrete annotation mask and its release semantics."""

    name: str
    values: np.ndarray
    columns: tuple[str, ...]
    reference_channel: str
    semantics: str


CHANNEL_ORDER: tuple[str, ...] = (
    "labels_oracle_any",
    "labels_oracle_intervention",
    "labels_oracle_context",
    "labels_event_only",
    "labels_delayed",
    "labels_weak_point",
    "labels_visible_only",
    "labels_noisy_boundary",
    "labels_censored",
)


def build_annotation_channels(
    *,
    length: int,
    channels: int,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, AnnotationChannel]:
    """Build deterministic v12 annotation channels for one instance."""

    n = int(length)
    c = int(channels)
    masks = build_label_masks(length=n, channels=c, events=events)
    event_only = _event_interval_mask(n, events)
    result = {
        "labels_oracle_any": AnnotationChannel(
            name="labels_oracle_any",
            values=masks.labels_any,
            columns=("label_any",),
            reference_channel="labels_oracle_any",
            semantics="Exact event interval union across all event channels.",
        ),
        "labels_oracle_intervention": AnnotationChannel(
            name="labels_oracle_intervention",
            values=masks.labels_intervention,
            columns=tuple(f"label-{idx}" for idx in range(c)),
            reference_channel="labels_oracle_intervention",
            semantics="Exact event support on intervention/operator-target channels.",
        ),
        "labels_oracle_context": AnnotationChannel(
            name="labels_oracle_context",
            values=masks.labels_context,
            columns=tuple(f"label-{idx}" for idx in range(c)),
            reference_channel="labels_oracle_context",
            semantics="Exact event support on channels needed to interpret the event.",
        ),
        "labels_event_only": AnnotationChannel(
            name="labels_event_only",
            values=event_only,
            columns=("label_event_only",),
            reference_channel="labels_oracle_any",
            semantics="Temporal event labels without intervention/context channel roles.",
        ),
        "labels_delayed": AnnotationChannel(
            name="labels_delayed",
            values=_delayed_mask(n, events),
            columns=("label_delayed",),
            reference_channel="labels_oracle_any",
            semantics="Delayed-positive temporal labels for latency-sensitive evaluation.",
        ),
        "labels_weak_point": AnnotationChannel(
            name="labels_weak_point",
            values=_weak_point_mask(n, events),
            columns=("label_weak_point",),
            reference_channel="labels_oracle_any",
            semantics="Sparse point-level labels at event centers.",
        ),
        "labels_visible_only": AnnotationChannel(
            name="labels_visible_only",
            values=_visible_only_mask(n, events),
            columns=("label_visible_only",),
            reference_channel="labels_oracle_any",
            semantics="Temporal labels for events marked channel-visible by the generator.",
        ),
        "labels_noisy_boundary": AnnotationChannel(
            name="labels_noisy_boundary",
            values=_noisy_boundary_mask(n, events),
            columns=("label_noisy_boundary",),
            reference_channel="labels_oracle_any",
            semantics="Temporal labels with deterministic boundary imprecision.",
        ),
        "labels_censored": AnnotationChannel(
            name="labels_censored",
            values=_censored_mask(n, events),
            columns=("label_censored",),
            reference_channel="labels_oracle_any",
            semantics="Partial temporal labels with event boundaries censored.",
        ),
    }
    return {name: result[name] for name in CHANNEL_ORDER}


def annotation_channel_manifest() -> dict[str, Any]:
    """Return stable manifest metadata for the supported annotation channels."""

    return {
        "annotation_channel_manifest_version": "synthgen.annotation_channels.v2",
        "channels": [
            {
                "name": "labels_oracle_any",
                "reference_channel": "labels_oracle_any",
                "scope": "temporal",
                "semantics": "Exact event interval union across all event channels.",
            },
            {
                "name": "labels_oracle_intervention",
                "reference_channel": "labels_oracle_intervention",
                "scope": "channel",
                "semantics": "Exact event support on intervention/operator-target channels.",
            },
            {
                "name": "labels_oracle_context",
                "reference_channel": "labels_oracle_context",
                "scope": "channel",
                "semantics": "Exact event support on context/interpreting channels.",
            },
            {
                "name": "labels_event_only",
                "reference_channel": "labels_oracle_any",
                "scope": "temporal",
                "semantics": "Event interval labels without channel roles.",
            },
            {
                "name": "labels_delayed",
                "reference_channel": "labels_oracle_any",
                "scope": "temporal",
                "semantics": "Labels begin after a deterministic event-dependent delay.",
            },
            {
                "name": "labels_weak_point",
                "reference_channel": "labels_oracle_any",
                "scope": "point",
                "semantics": "One positive point near the center of each event.",
            },
            {
                "name": "labels_visible_only",
                "reference_channel": "labels_oracle_any",
                "scope": "temporal",
                "semantics": "Only events with channel_visible truth metadata are labeled.",
            },
            {
                "name": "labels_noisy_boundary",
                "reference_channel": "labels_oracle_any",
                "scope": "temporal",
                "semantics": "Oracle event intervals with deterministic boundary noise.",
            },
            {
                "name": "labels_censored",
                "reference_channel": "labels_oracle_any",
                "scope": "temporal",
                "semantics": "Central event support only; boundaries are censored.",
            },
        ],
    }


def _event_interval_mask(
    length: int, events: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    values = np.zeros((int(length), 1), dtype=np.int8)
    for event in events:
        start, end = _event_bounds(event, length)
        if end > start:
            values[start:end, 0] = 1
    return values


def _delayed_mask(length: int, events: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.zeros((int(length), 1), dtype=np.int8)
    for event in events:
        start, end = _event_bounds(event, length)
        width = end - start
        if width <= 0:
            continue
        delay = max(1, min(width - 1, int(round(0.25 * width)))) if width > 1 else 0
        delayed_start = min(end, start + delay)
        if delayed_start < end:
            values[delayed_start:end, 0] = 1
    return values


def _weak_point_mask(length: int, events: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.zeros((int(length), 1), dtype=np.int8)
    for event in events:
        start, end = _event_bounds(event, length)
        if end > start:
            values[min(length - 1, start + (end - start) // 2), 0] = 1
    return values


def _visible_only_mask(length: int, events: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.zeros((int(length), 1), dtype=np.int8)
    for event in events:
        if not _is_visible_event(event):
            continue
        start, end = _event_bounds(event, length)
        if end > start:
            values[start:end, 0] = 1
    return values


def _noisy_boundary_mask(
    length: int, events: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    values = np.zeros((int(length), 1), dtype=np.int8)
    for index, event in enumerate(events):
        start, end = _event_bounds(event, length)
        width = end - start
        if width <= 0:
            continue
        noise = max(1, int(round(0.10 * width)))
        left_shift = -noise if index % 2 == 0 else noise
        right_shift = noise if index % 2 == 0 else -noise
        noisy_start = _clip_index(start + left_shift, length)
        noisy_end = _clip_index(end + right_shift, length)
        if noisy_end <= noisy_start:
            noisy_start, noisy_end = start, end
        values[noisy_start:noisy_end, 0] = 1
    return values


def _censored_mask(length: int, events: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.zeros((int(length), 1), dtype=np.int8)
    for event in events:
        start, end = _event_bounds(event, length)
        width = end - start
        if width <= 0:
            continue
        censor = int(round(0.25 * width))
        censored_start = min(end, start + censor)
        censored_end = max(censored_start, end - censor)
        if censored_end > censored_start:
            values[censored_start:censored_end, 0] = 1
    return values


def _event_bounds(event: Mapping[str, Any], length: int) -> tuple[int, int]:
    start = _clip_index(event.get("start", 0), length)
    end = _clip_index(event.get("end", start), length)
    return start, end


def _clip_index(value: Any, length: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return max(0, min(number, int(length)))


def _is_visible_event(event: Mapping[str, Any]) -> bool:
    if "channel_visible" in event:
        return bool(event["channel_visible"])
    hint = str(event.get("purity_hint", "")).lower()
    return "visible" in hint and "not" not in hint
