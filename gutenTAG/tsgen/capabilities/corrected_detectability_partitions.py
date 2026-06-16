"""Partition execution for corrected detectability."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .array_store import ArrayStore
from .cache import CacheStore
from .calibration import (
    EmpiricalCalibrator,
)
from .corrected_detectability_blind_scan import cached_blind_scan_rows
from .corrected_detectability_blind_scores import (
    BlindScanScoreBlock,
    scores_for_window,
)
from .corrected_detectability_candidates import (
    CandidateSpec,
    candidate_specs,
    format_projection,
)
from .corrected_detectability_null_bundle import (
    build_null_bundle,
    clean_scan_score_rows,
)
from .corrected_detectability_nulls import CorrectedNullBundle
from .corrected_detectability_oracle import (
    OracleDetectionSummary,
    calibration_resolution_row,
    candidate_p_values_for_scores,
    oracle_frontier_row,
    summarize_oracle_detection,
)
from .corrected_detectability_tables import (
    frontier_columns,
    ordered_rows,
    resolution_columns,
)
from .dataset import (
    EventGroup,
    InstanceRecord,
    event_uid,
    read_timeseries_csv,
)
from .partitions import PartitionSpec
from .protocol import CapabilityProtocol, window_length_bin
from .scan_scores import CleanWindowScoreCache
from .windows import WindowLibrary


@dataclass(frozen=True)
class CorrectedPartitionResult:
    """Corrected detectability output rows for one variant partition."""

    rows: tuple[dict[str, object], ...]
    blind_rows: tuple[dict[str, object], ...]
    resolution_rows: tuple[dict[str, object], ...]
    candidate_manifest_rows: Mapping[str, dict[str, object]]
    scan_manifest_rows: Mapping[str, dict[str, object]]


@dataclass
class _CorrectedPartitionState:
    calibrator: EmpiricalCalibrator
    clean_cache: dict[str, np.ndarray]
    anom_cache: dict[str, np.ndarray]
    null_cache: dict[tuple[object, ...], CorrectedNullBundle]
    score_cache: CleanWindowScoreCache
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock]
    rows: list[dict[str, object]]
    blind_rows: list[dict[str, object]]
    resolution_rows: list[dict[str, object]]
    candidate_manifest_rows: dict[str, dict[str, object]]
    scan_manifest_rows: dict[str, dict[str, object]]


def cached_or_compute_partition(
    *,
    partition: PartitionSpec[InstanceRecord],
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
) -> CorrectedPartitionResult:
    """Return a corrected-detectability partition result, using cache if present."""

    if cache is None:
        return compute_partition(
            instances=partition.items,
            null_instances=null_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            cache=cache,
        )
    fingerprint = cache.fingerprint(
        profile_name="corrected_detectability",
        partition_id=partition.partition_id,
        extra={
            "corrected_detectability_partition_version": "v3",
            "variant_id": partition.partition_id,
            "instance_count": len(partition.items),
            "event_group_count": int(
                sum(len(instance.event_groups) for instance in partition.items)
            ),
            "alpha_grid": list(protocol.alpha_grid),
            "detection_witnesses": list(protocol.detection_witnesses),
            "max_projection_size": int(protocol.max_projection_size),
            "max_scan_windows_per_length": int(protocol.max_scan_windows_per_length),
            "window_length_policy_mode": protocol.window_length_policy_mode,
            "window_length_bins": list(protocol.window_length_bins),
            "calibration_split": protocol.calibration_split,
        },
    )
    payload = cache.get_or_compute_json(
        profile_name="corrected_detectability",
        partition_id=partition.partition_id,
        fingerprint=fingerprint,
        compute=lambda: partition_to_payload(
            compute_partition(
                instances=partition.items,
                null_instances=null_instances,
                protocol=protocol,
                arrays=arrays,
                windows=windows,
                cache=cache,
            )
        ),
    )
    return partition_from_payload(payload)


def compute_partition(
    *,
    instances: Sequence[InstanceRecord],
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
) -> CorrectedPartitionResult:
    """Compute corrected-detectability rows for one variant partition."""

    state = _new_corrected_partition_state()
    for instance in instances:
        _compute_instance_partition(
            instance=instance,
            null_instances=null_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            cache=cache,
            state=state,
        )
    return _partition_result_from_state(state)


def _new_corrected_partition_state() -> _CorrectedPartitionState:
    return _CorrectedPartitionState(
        calibrator=EmpiricalCalibrator(),
        clean_cache={},
        anom_cache={},
        null_cache={},
        score_cache=CleanWindowScoreCache(),
        blind_scan_cache={},
        rows=[],
        blind_rows=[],
        resolution_rows=[],
        candidate_manifest_rows={},
        scan_manifest_rows={},
    )


def _compute_instance_partition(
    *,
    instance: InstanceRecord,
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
    state: _CorrectedPartitionState,
) -> None:
    state.anom_cache[_instance_key(instance)] = _read_array(
        instance,
        "anomalous",
        arrays,
    )
    for group in instance.event_groups:
        _compute_event_partition(
            instance=instance,
            group=group,
            null_instances=null_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            cache=cache,
            state=state,
        )


def _compute_event_partition(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
    state: _CorrectedPartitionState,
) -> None:
    candidates = candidate_specs(instance, group, protocol)
    if not candidates:
        return
    scan_length = window_length_bin(max(1, group.length), protocol)
    null_key = _partition_null_key(instance, candidates, scan_length, protocol)
    nulls = _nulls_for_event(
        instance=instance,
        null_instances=null_instances,
        candidates=candidates,
        scan_length=scan_length,
        null_key=null_key,
        arrays=arrays,
        windows=windows,
        protocol=protocol,
        state=state,
    )
    event_scores = event_candidate_scores(
        series=state.anom_cache[_instance_key(instance)],
        group=group,
        candidates=candidates,
        protocol=protocol,
    )
    candidate_p_values = candidate_p_values_for_scores(
        event_scores=event_scores,
        nulls=nulls,
        calibrator=state.calibrator,
    )
    if not candidate_p_values:
        return
    _append_corrected_event_rows(
        instance=instance,
        group=group,
        candidates=candidates,
        nulls=nulls,
        null_key=null_key,
        event_scores=event_scores,
        candidate_p_values=candidate_p_values,
        protocol=protocol,
        windows=windows,
        cache=cache,
        state=state,
    )


def _partition_null_key(
    instance: InstanceRecord,
    candidates: Sequence[CandidateSpec],
    scan_length: int,
    protocol: CapabilityProtocol,
) -> tuple[object, ...]:
    return (
        instance.variant_id,
        scan_length,
        tuple(candidate.key for candidate in candidates),
        protocol.max_scan_windows_per_length,
        protocol.window_length_policy_mode,
        tuple(protocol.window_length_bins),
        protocol.calibration_split,
        protocol.clean_window_stride_fraction,
        protocol.context_window_multiplier,
        protocol.min_context_points,
    )


def _nulls_for_event(
    *,
    instance: InstanceRecord,
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    candidates: Sequence[CandidateSpec],
    scan_length: int,
    null_key: tuple[object, ...],
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    protocol: CapabilityProtocol,
    state: _CorrectedPartitionState,
) -> CorrectedNullBundle:
    if null_key not in state.null_cache:
        state.null_cache[null_key] = build_null_bundle(
            clean_instances=null_instances.get(instance.variant_id, [instance]),
            clean_cache=state.clean_cache,
            score_cache=state.score_cache,
            arrays=arrays,
            windows=windows,
            event_length=scan_length,
            candidates=candidates,
            protocol=protocol,
            null_id=_null_id(instance.variant_id, scan_length, null_key),
            calibrator=state.calibrator,
        )
    nulls = state.null_cache[null_key]
    _record_null_manifest_rows(nulls, state)
    return nulls


def _record_null_manifest_rows(
    nulls: CorrectedNullBundle,
    state: _CorrectedPartitionState,
) -> None:
    for item in nulls.candidate_manifest:
        state.candidate_manifest_rows[str(item["candidate_null_id"])] = item
    state.scan_manifest_rows[str(nulls.scan_manifest["scan_null_id"])] = (
        nulls.scan_manifest
    )


def _append_corrected_event_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    null_key: tuple[object, ...],
    event_scores: Mapping[str, float],
    candidate_p_values: Mapping[str, float],
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
    state: _CorrectedPartitionState,
) -> None:
    oracle_summary = summarize_oracle_detection(
        group=group,
        candidates=candidates,
        event_scores=event_scores,
        candidate_p_values=candidate_p_values,
        nulls=nulls,
        calibrator=state.calibrator,
    )
    event_blind_rows, false_alerts_by_alpha, blind_window_count = (
        cached_blind_scan_rows(
            instance=instance,
            group=group,
            series=state.anom_cache[_instance_key(instance)],
            candidates=candidates,
            nulls=nulls,
            null_key=null_key,
            protocol=protocol,
            windows=windows,
            calibrator=state.calibrator,
            cache=cache,
            blind_scan_cache=state.blind_scan_cache,
        )
    )
    state.blind_rows.extend(event_blind_rows)
    for alpha in protocol.alpha_grid:
        row = _oracle_row_for_alpha(
            instance=instance,
            group=group,
            summary=oracle_summary,
            nulls=nulls,
            alpha=float(alpha),
            false_alert_count=int(false_alerts_by_alpha.get(float(alpha), 0)),
            blind_window_count=blind_window_count,
            protocol=protocol,
            state=state,
        )
        state.rows.append(row)
        state.resolution_rows.append(calibration_resolution_row(row))


def _oracle_row_for_alpha(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    summary: OracleDetectionSummary,
    nulls: CorrectedNullBundle,
    alpha: float,
    false_alert_count: int,
    blind_window_count: int,
    protocol: CapabilityProtocol,
    state: _CorrectedPartitionState,
) -> dict[str, object]:
    threshold = state.calibrator.scan_threshold(nulls.scan_null, alpha)
    return oracle_frontier_row(
        common=_common(instance, group),
        summary=summary,
        nulls=nulls,
        alpha=alpha,
        threshold=threshold,
        false_alert_count=false_alert_count,
        blind_window_count=blind_window_count,
        protocol=protocol,
    )


def _partition_result_from_state(
    state: _CorrectedPartitionState,
) -> CorrectedPartitionResult:
    return CorrectedPartitionResult(
        rows=tuple(state.rows),
        blind_rows=tuple(state.blind_rows),
        resolution_rows=tuple(state.resolution_rows),
        candidate_manifest_rows=dict(state.candidate_manifest_rows),
        scan_manifest_rows=dict(state.scan_manifest_rows),
    )


def partition_to_payload(result: CorrectedPartitionResult) -> dict[str, Any]:
    """Serialize a partition result to cache payload."""

    return {
        "corrected_detectability_partition_version": "synthgen.capability.corrected.partition.v1",
        "rows": list(result.rows),
        "blind_rows": list(result.blind_rows),
        "resolution_rows": list(result.resolution_rows),
        "candidate_manifest_rows": list(result.candidate_manifest_rows.values()),
        "scan_manifest_rows": list(result.scan_manifest_rows.values()),
    }


def partition_from_payload(payload: Mapping[str, Any]) -> CorrectedPartitionResult:
    """Deserialize a partition result from cache payload."""

    candidate_rows = tuple(
        dict(row) for row in payload.get("candidate_manifest_rows", ())
    )
    scan_rows = tuple(dict(row) for row in payload.get("scan_manifest_rows", ()))
    return CorrectedPartitionResult(
        rows=ordered_rows(payload.get("rows", ()), frontier_columns()),
        blind_rows=tuple(dict(row) for row in payload.get("blind_rows", ())),
        resolution_rows=ordered_rows(
            payload.get("resolution_rows", ()), resolution_columns()
        ),
        candidate_manifest_rows={
            str(row.get("candidate_null_id")): row
            for row in candidate_rows
            if row.get("candidate_null_id") is not None
        },
        scan_manifest_rows={
            str(row.get("scan_null_id")): row
            for row in scan_rows
            if row.get("scan_null_id") is not None
        },
    )


def event_candidate_scores(
    *,
    series: np.ndarray,
    group: EventGroup,
    candidates: Sequence[CandidateSpec],
    protocol: CapabilityProtocol,
) -> dict[str, float]:
    """Score corrected-detectability candidates on the oracle event window."""

    return scores_for_window(
        series=series,
        start=group.start,
        end=group.end,
        candidates=candidates,
        protocol=protocol,
    )


def calibration_null_instances(
    instances: Sequence[InstanceRecord],
    protocol: CapabilityProtocol,
) -> tuple[InstanceRecord, ...]:
    """Return instances used for calibration nulls for one variant."""

    if protocol.calibration_split:
        preferred = tuple(
            instance
            for instance in instances
            if instance.split == protocol.calibration_split
        )
        if preferred:
            return preferred
    return tuple(instances)


def _common(instance: InstanceRecord, group: EventGroup) -> dict[str, object]:
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "base_oscillation": instance.base_oscillation,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "start": group.start,
        "end": group.end,
        "length": group.length,
        "group_channels": format_projection(group.group_channels),
    }


def _instance_key(instance: InstanceRecord) -> str:
    return f"{instance.variant_id}/{instance.split}/{instance.instance_id}"


def _null_id(variant_id: str, event_length: int, key: tuple[object, ...]) -> str:
    payload = repr(key).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{variant_id}__len{int(event_length)}__{digest}"


def _read_array(
    instance: InstanceRecord, kind: str, arrays: ArrayStore | None
) -> np.ndarray:
    if arrays is not None and kind == "clean":
        return arrays.get(instance, "clean")
    if arrays is not None and kind == "anomalous":
        return arrays.get(instance, "anomalous")
    if kind == "clean":
        return read_timeseries_csv(instance.clean_path)
    return read_timeseries_csv(instance.anomalous_path)


__all__ = [
    "CorrectedPartitionResult",
    "build_null_bundle",
    "cached_or_compute_partition",
    "calibration_null_instances",
    "clean_scan_score_rows",
    "compute_partition",
    "event_candidate_scores",
    "partition_from_payload",
    "partition_to_payload",
]
