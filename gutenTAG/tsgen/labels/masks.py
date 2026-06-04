"""Event-derived label masks for TS dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelMasks:
    """Collection of v11 label masks derived from event-level truth."""

    labels_any: np.ndarray
    labels_affected: np.ndarray
    labels_context: np.ndarray


def build_label_masks(
    *,
    length: int,
    channels: int,
    events: Sequence[Mapping[str, Any]],
) -> LabelMasks:
    """Build target-agnostic v11 label masks from events.

    ``labels_affected`` follows operator targets/perturbed channels. For
    relation-only anomalies this is not necessarily the final supervised target;
    ``labels_context`` records the channels needed to interpret the event.
    """
    labels_any = np.zeros((int(length), 1), dtype=np.int8)
    labels_affected = np.zeros((int(length), int(channels)), dtype=np.int8)
    labels_context = np.zeros((int(length), int(channels)), dtype=np.int8)
    for event in events:
        start = _clip_index(event.get("start", 0), length)
        end = _clip_index(event.get("end", start), length)
        if end <= start:
            continue
        labels_any[start:end, 0] = 1
        affected_channels = _event_channels(
            event,
            preferred_keys=("operator_target_channels", "affected_channels", "channel"),
        )
        context_channels = _event_channels(
            event,
            preferred_keys=(
                "context_channels",
                "group_channels",
                "anchor_channel",
                "channel",
            ),
        )
        for channel in affected_channels:
            if 0 <= channel < channels:
                labels_affected[start:end, channel] = 1
        for channel in context_channels:
            if 0 <= channel < channels:
                labels_context[start:end, channel] = 1
    return LabelMasks(
        labels_any=labels_any,
        labels_affected=labels_affected,
        labels_context=labels_context,
    )


def write_label_masks(path: Path, masks: LabelMasks) -> None:
    """Write v11 label masks next to legacy labels."""
    _write_label_csv(path / "labels_any.csv", masks.labels_any, ["label_any"])
    _write_label_csv(
        path / "labels_affected.csv",
        masks.labels_affected,
        [f"label-{i}" for i in range(masks.labels_affected.shape[1])],
    )
    _write_label_csv(
        path / "labels_context.csv",
        masks.labels_context,
        [f"label-{i}" for i in range(masks.labels_context.shape[1])],
    )


def _event_channels(
    event: Mapping[str, Any],
    *,
    preferred_keys: Sequence[str],
) -> list[int]:
    channels: list[int] = []
    for key in preferred_keys:
        if key not in event:
            continue
        value = event[key]
        if isinstance(value, (list, tuple)):
            channels.extend(int(ch) for ch in value)
        else:
            channels.append(int(value))
        if channels:
            break
    return sorted(set(channels))


def _clip_index(value: Any, length: int) -> int:
    return max(0, min(int(value), int(length)))


def _write_label_csv(path: Path, values: np.ndarray, columns: Sequence[str]) -> None:
    df = pd.DataFrame(values.astype(np.int8), columns=list(columns))
    df.to_csv(path, index=False)
