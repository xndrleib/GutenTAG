"""Support-integrity audit for generated events."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..array_store import ArrayStore
from ..dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid, read_timeseries_csv
from ..numerics import finite_float


def compute_support_integrity(
    dataset: DatasetIndex,
    *,
    arrays: ArrayStore | None = None,
) -> pd.DataFrame:
    """Compute event-level localization of injected residual energy."""

    rows: list[dict[str, object]] = []
    for instance in dataset.instances:
        clean = arrays.get(instance, "clean") if arrays is not None else read_timeseries_csv(instance.clean_path)
        anomalous = arrays.get(instance, "anomalous") if arrays is not None else read_timeseries_csv(instance.anomalous_path)
        delta = np.asarray(anomalous, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
        for group in instance.event_groups:
            rows.append(_support_row(instance, group, delta))
    return pd.DataFrame(rows)


def _support_row(
    instance: InstanceRecord,
    group: EventGroup,
    delta: np.ndarray,
) -> dict[str, object]:
    channels = _audit_channels(group, instance.channels)
    energy = np.sum(np.square(delta[:, channels]), axis=1) if channels else np.sum(np.square(delta), axis=1)
    other_support_mask = _other_event_support_mask(instance, group)
    near_start, near_end = _near_field_bounds(group, instance.length)
    known_other_mask = other_support_mask.copy()
    known_other_mask[_slice_bounds(group.start, group.end, energy.size)] = False
    known_other_mask[_slice_bounds(near_start, near_end, energy.size)] = False
    known_other = float(np.sum(energy[known_other_mask]))
    adjusted_energy = energy.copy()
    adjusted_energy[known_other_mask] = 0.0
    total = float(np.sum(adjusted_energy))
    raw_total = float(np.sum(energy))
    inside = _interval_mass(energy, group.start, group.end)
    outside = max(0.0, total - inside)
    near = max(0.0, _interval_mass(adjusted_energy, near_start, near_end) - inside)
    far = max(0.0, outside - near)
    local = inside + near
    inside_share = inside / max(total, 1e-12)
    local_share = local / max(total, 1e-12)
    far_share = far / max(total, 1e-12)
    known_other_share = known_other / max(raw_total, 1e-12)
    width80 = _concentration_width(adjusted_energy, 0.80)
    width90 = _concentration_width(adjusted_energy, 0.90)
    center = _center_of_mass(adjusted_energy)
    return {
        **_common(instance, group),
        "inside_l2_mass": finite_float(inside),
        "outside_l2_mass": finite_float(outside),
        "near_field_l2_mass": finite_float(near),
        "far_field_l2_mass": finite_float(far),
        "known_other_event_l2_mass": finite_float(known_other),
        "known_other_event_mass_share": finite_float(known_other_share),
        "inside_mass_share": finite_float(inside_share),
        "support_or_near_field_mass_share": finite_float(local_share),
        "far_field_mass_share": finite_float(far_share),
        "residual_center_of_mass": finite_float(center, default=math.nan),
        "support_concentration_80_width": finite_float(width80, default=math.nan),
        "support_concentration_90_width": finite_float(width90, default=math.nan),
        "canonical_inside_share": finite_float(inside_share),
        "support_status": _support_status(group, total, inside_share, local_share, far_share),
    }


def _support_status(
    group: EventGroup,
    total: float,
    inside_share: float,
    local_share: float,
    far_share: float,
) -> str:
    if group.end <= group.start:
        return "support_mismatch"
    if total <= 1e-12:
        return "insufficient_effect"
    if inside_share >= 0.80 and far_share <= 0.10:
        return "valid"
    if local_share >= 0.95 and far_share <= 0.10:
        return "boundary_uncertain"
    if inside_share >= 0.55 and far_share <= 0.25:
        return "boundary_uncertain"
    return "leaky"


def _audit_channels(group: EventGroup, channels: int) -> tuple[int, ...]:
    selected = sorted(set(group.intervention_channels) | set(group.group_channels) | set(group.context_channels))
    return tuple(channel for channel in selected if 0 <= int(channel) < int(channels))


def _interval_mass(energy: np.ndarray, start: int, end: int) -> float:
    lo = max(0, min(int(start), energy.size))
    hi = max(lo, min(int(end), energy.size))
    return float(np.sum(energy[lo:hi]))


def _near_field_bounds(group: EventGroup, length: int) -> tuple[int, int]:
    width = max(group.length, 1)
    return max(0, int(group.start) - width), min(int(length), int(group.end) + width)


def _other_event_support_mask(instance: InstanceRecord, group: EventGroup) -> np.ndarray:
    mask = np.zeros(int(instance.length), dtype=bool)
    for other in instance.event_groups:
        if other.group_id == group.group_id:
            continue
        start, end = _declared_support_bounds(other)
        mask[_slice_bounds(start, end, instance.length)] = True
    return mask


def _declared_support_bounds(group: EventGroup) -> tuple[int, int]:
    start = group.source_start if group.source_end > group.source_start else group.start
    end = group.source_end if group.source_end > group.source_start else group.end
    return int(start), int(end)


def _slice_bounds(start: int, end: int, length: int) -> slice:
    lo = max(0, min(int(start), int(length)))
    hi = max(lo, min(int(end), int(length)))
    return slice(lo, hi)


def _concentration_width(energy: np.ndarray, share: float) -> float:
    total = float(np.sum(energy))
    if total <= 1e-12:
        return math.nan
    cumulative = np.cumsum(energy) / total
    lo = int(np.searchsorted(cumulative, (1.0 - share) / 2.0, side="left"))
    hi = int(np.searchsorted(cumulative, 1.0 - (1.0 - share) / 2.0, side="left"))
    return float(max(0, hi - lo + 1))


def _center_of_mass(energy: np.ndarray) -> float:
    total = float(np.sum(energy))
    if total <= 1e-12:
        return math.nan
    indices = np.arange(energy.size, dtype=np.float64)
    return float(np.sum(indices * energy) / total)


def _common(instance: InstanceRecord, group: EventGroup) -> dict[str, object]:
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "start": group.start,
        "end": group.end,
        "source_start": group.source_start,
        "source_end": group.source_end,
        "intervention_channels": _format_channels(group.intervention_channels),
        "context_channels": _format_channels(group.context_channels),
        "group_channels": _format_channels(group.group_channels),
    }


def _format_channels(channels: Sequence[int]) -> str:
    return "|".join(str(int(channel)) for channel in channels)
