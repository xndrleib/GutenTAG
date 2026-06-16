"""Boundary-artifact audit for generated events."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ...contracts import ContractRegistry
from ..array_store import ArrayStore
from ..dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
    event_uid,
    read_timeseries_csv,
)
from ..numerics import finite_float


def compute_boundary_audit(
    dataset: DatasetIndex,
    *,
    arrays: ArrayStore | None = None,
    registry: ContractRegistry | None = None,
) -> pd.DataFrame:
    """Compute boundary-vs-interior residual evidence per event."""

    active_registry = registry or ContractRegistry.from_resource_defaults()
    rows: list[dict[str, object]] = []
    for instance in dataset.instances:
        clean = (
            arrays.get(instance, "clean")
            if arrays is not None
            else read_timeseries_csv(instance.clean_path)
        )
        anomalous = (
            arrays.get(instance, "anomalous")
            if arrays is not None
            else read_timeseries_csv(instance.anomalous_path)
        )
        delta = np.asarray(anomalous, dtype=np.float64) - np.asarray(
            clean, dtype=np.float64
        )
        for group in instance.event_groups:
            rows.append(
                _boundary_row(instance, group, clean, anomalous, delta, active_registry)
            )
    return pd.DataFrame(rows)


def _boundary_row(
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    delta: np.ndarray,
    registry: ContractRegistry,
) -> dict[str, object]:
    channels = _audit_channels(group, instance.channels)
    if not channels:
        channels = tuple(range(instance.channels))
    audit_start, audit_end = _boundary_audit_bounds(group)
    boundary_mask, interior_mask = _boundary_masks_for_bounds(
        audit_start, audit_end, instance.length
    )
    metrics = _boundary_energy_metrics(
        delta,
        boundary_mask=boundary_mask,
        interior_mask=interior_mask,
        channels=channels,
    )
    status = _boundary_status(group, registry, metrics)
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "start": int(group.start),
        "end": int(group.end),
        "source_start": int(group.source_start),
        "source_end": int(group.source_end),
        "boundary_audit_start": int(audit_start),
        "boundary_audit_end": int(audit_end),
        "left_value_jump": finite_float(
            _boundary_jump(clean, anomalous, audit_start, channels)
        ),
        "right_value_jump": finite_float(
            _boundary_jump(clean, anomalous, audit_end, channels)
        ),
        "left_derivative_jump": finite_float(
            _derivative_jump(clean, anomalous, audit_start, channels)
        ),
        "right_derivative_jump": finite_float(
            _derivative_jump(clean, anomalous, audit_end, channels)
        ),
        "boundary_energy": finite_float(metrics["boundary_energy"]),
        "interior_energy": finite_float(metrics["interior_energy"]),
        "boundary_energy_share": finite_float(metrics["boundary_share"]),
        "boundary_witness_score": finite_float(metrics["boundary_score"]),
        "canonical_interior_score": finite_float(metrics["interior_score"]),
        "boundary_to_canonical_ratio": finite_float(metrics["ratio"]),
        "boundary_primary_allowed": status["allowed"],
        "boundary_status": status["status"],
        "boundary_primary_detection_cause": status["primary"] and not status["allowed"],
    }


def _boundary_energy_metrics(
    delta: np.ndarray,
    *,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
    channels: tuple[int, ...],
) -> dict[str, float]:
    boundary_energy = (
        float(np.sum(np.square(delta[boundary_mask][:, channels])))
        if boundary_mask.any()
        else 0.0
    )
    interior_energy = (
        float(np.sum(np.square(delta[interior_mask][:, channels])))
        if interior_mask.any()
        else 0.0
    )
    support_energy = boundary_energy + interior_energy
    boundary_score = math.sqrt(max(boundary_energy, 0.0)) / math.sqrt(
        max(len(channels), 1)
    )
    interior_score = math.sqrt(max(interior_energy, 0.0)) / math.sqrt(
        max(len(channels), 1)
    )
    return {
        "boundary_energy": boundary_energy,
        "interior_energy": interior_energy,
        "boundary_share": boundary_energy / max(support_energy, 1e-12),
        "boundary_score": boundary_score,
        "interior_score": interior_score,
        "ratio": boundary_score / max(interior_score, 1e-12),
    }


def _boundary_status(
    group: EventGroup,
    registry: ContractRegistry,
    metrics: dict[str, float],
) -> dict[str, object]:
    contract = registry.get_by_anomaly(group.anomaly_type)
    boundary_allowed = bool(
        contract is not None
        and contract.support_policy.get("boundary_primary_allowed", False) is True
    )
    primary_boundary = bool(
        metrics["boundary_share"] >= 0.60 and metrics["ratio"] >= 1.50
    )
    if primary_boundary and boundary_allowed:
        status = "valid_boundary_primary"
    elif primary_boundary:
        status = "boundary_primary_detection_cause"
    else:
        status = "valid_interior_or_mixed"
    return {"allowed": boundary_allowed, "primary": primary_boundary, "status": status}


def _audit_channels(group: EventGroup, channels: int) -> tuple[int, ...]:
    selected = sorted(
        set(group.intervention_channels)
        | set(group.group_channels)
        | set(group.context_channels)
    )
    return tuple(channel for channel in selected if 0 <= int(channel) < int(channels))


def _boundary_masks(group: EventGroup, length: int) -> tuple[np.ndarray, np.ndarray]:
    audit_start, audit_end = _boundary_audit_bounds(group)
    return _boundary_masks_for_bounds(audit_start, audit_end, length)


def _boundary_audit_bounds(group: EventGroup) -> tuple[int, int]:
    if int(group.source_end) > int(group.source_start):
        return int(group.source_start), int(group.source_end)
    return int(group.start), int(group.end)


def _boundary_masks_for_bounds(
    start: int, end: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(int(length), dtype=bool)
    interior = np.zeros(int(length), dtype=bool)
    start = max(0, min(int(start), int(length)))
    end = max(start, min(int(end), int(length)))
    if end <= start:
        return mask, interior
    edge_width = max(1, min(3, max(1, (end - start) // 4)))
    mask[start : min(end, start + edge_width)] = True
    mask[max(start, end - edge_width) : end] = True
    interior[start:end] = True
    interior[mask] = False
    return mask, interior


def _boundary_jump(
    clean: np.ndarray,
    anomalous: np.ndarray,
    index: int,
    channels: tuple[int, ...],
) -> float:
    idx = int(index)
    if idx <= 0 or idx >= clean.shape[0]:
        return 0.0
    clean_jump = clean[idx, list(channels)] - clean[idx - 1, list(channels)]
    anomalous_jump = anomalous[idx, list(channels)] - anomalous[idx - 1, list(channels)]
    return float(np.sqrt(np.mean(np.square(anomalous_jump - clean_jump))))


def _derivative_jump(
    clean: np.ndarray,
    anomalous: np.ndarray,
    index: int,
    channels: tuple[int, ...],
) -> float:
    idx = int(index)
    if idx <= 1 or idx + 1 >= clean.shape[0]:
        return 0.0
    clean_left = clean[idx, list(channels)] - clean[idx - 1, list(channels)]
    clean_right = clean[idx + 1, list(channels)] - clean[idx, list(channels)]
    anomalous_left = anomalous[idx, list(channels)] - anomalous[idx - 1, list(channels)]
    anomalous_right = (
        anomalous[idx + 1, list(channels)] - anomalous[idx, list(channels)]
    )
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    (anomalous_right - anomalous_left) - (clean_right - clean_left)
                )
            )
        )
    )
