"""Repair maturity profile computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..array_store import ArrayStore
from ..dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid, read_timeseries_csv
from ..numerics import finite_float, segment
from ..protocol import CapabilityProtocol
from .operators import operator_family, repair_segment


@dataclass(frozen=True)
class RepairProfileResult:
    """Repair profile output tables."""

    repair_profile: pd.DataFrame


def compute_repair_profiles(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    *,
    arrays: ArrayStore | None = None,
) -> RepairProfileResult:
    """Compute event-level repair gains for canonical repair operators."""

    del protocol
    rows: list[dict[str, object]] = []
    for instance in dataset.instances:
        clean = arrays.get(instance, "clean") if arrays is not None else read_timeseries_csv(instance.clean_path)
        anomalous = arrays.get(instance, "anomalous") if arrays is not None else read_timeseries_csv(instance.anomalous_path)
        for group in instance.event_groups:
            rows.append(_repair_row(instance, group, clean, anomalous))
    return RepairProfileResult(repair_profile=pd.DataFrame(rows))


def _repair_row(
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
) -> dict[str, object]:
    channels = _repair_channels(instance, group)
    clean_segment = segment(clean, group.start, group.end, channels)
    anomalous_segment = segment(anomalous, group.start, group.end, channels)
    raw_rmse = _rmse(anomalous_segment, clean_segment)
    repaired = repair_segment(
        clean_segment,
        anomalous_segment,
        constraint_tag=group.constraint_tag,
        repair_operator=group.repair_operator,
    )
    repaired_rmse = _rmse(repaired, clean_segment)
    repair_gain = 1.0 if raw_rmse <= 1e-12 else 1.0 - repaired_rmse / raw_rmse
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "base_oscillation": instance.base_oscillation,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "operator_family": operator_family(group.constraint_tag),
        "semantic_scope": group.semantic_scope,
        "repair_operator": group.repair_operator,
        "support_start": group.start,
        "support_end": group.end,
        "support_length": group.length,
        "repair_channels": "|".join(str(channel) for channel in channels),
        "repair_channel_count": len(channels),
        "raw_residual_rmse": finite_float(raw_rmse),
        "repaired_residual_rmse": finite_float(repaired_rmse),
        "repair_gain": finite_float(max(-1.0, min(1.0, repair_gain))),
        "repair_status": _repair_status(raw_rmse, repair_gain),
    }


def _repair_channels(instance: InstanceRecord, group: EventGroup) -> tuple[int, ...]:
    channels = tuple(sorted(set(group.group_channels) | set(group.context_channels) | set(group.intervention_channels)))
    if channels:
        return channels
    return tuple(range(instance.channels))


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))))


def _repair_status(raw_rmse: float, repair_gain: float) -> str:
    if raw_rmse <= 1e-12:
        return "no_effect_to_repair"
    if repair_gain >= 0.70:
        return "repair_effective"
    if repair_gain >= 0.20:
        return "repair_partial"
    if repair_gain >= 0.0:
        return "repair_weak"
    return "repair_harmful"
