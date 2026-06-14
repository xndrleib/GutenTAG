"""Corrected min-p detectability frontier."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .array_store import ArrayStore
from .cache import CacheStore
from .calibration import (
    CandidateNull,
    EmpiricalCalibrator,
    ScanNull,
    calibration_status,
    min_empirical_p,
    scan_statistic_from_candidate_p_values,
)
from .dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
    event_uid,
    instances_by_variant,
    read_timeseries_csv,
)
from .numerics import channel_subsets, contiguous_windows, finite_float
from .partitions import PartitionSpec, run_partitions
from .ontology import witness_requires_projection_size
from .protocol import CapabilityProtocol, window_length_bin
from .scan_scores import CleanWindowScoreBlock, CleanWindowScoreCache
from .windows import WindowLibrary, WindowSpec
from .witnesses import detection_window_scores


WITNESS_FAMILY: Mapping[str, str] = {
    "mean_z": "location.mean",
    "variance_log_ratio": "scale.variance",
    "local_energy_z": "local.energy",
    "correlation_shift": "dependence.correlation",
    "covariance_shift": "dependence.covariance",
}


@dataclass(frozen=True)
class CorrectedDetectabilityResult:
    """Corrected detectability tables and null manifests."""

    frontier: pd.DataFrame
    oracle_window_diagnostic: pd.DataFrame
    blind_scan_events: pd.DataFrame
    calibration_resolution: pd.DataFrame
    candidate_nulls_manifest: dict[str, object]
    scan_nulls_manifest: dict[str, object]


@dataclass(frozen=True)
class CandidateSpec:
    """Single witness/projection candidate."""

    witness: str
    projection: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.witness}@{_format_projection(self.projection)}"

    @property
    def family(self) -> str:
        return WITNESS_FAMILY.get(self.witness, "unknown")


@dataclass(frozen=True)
class CorrectedNullBundle:
    """Reusable nulls for one variant/window/candidate setup."""

    candidate_nulls: Mapping[str, CandidateNull]
    scan_null: ScanNull
    candidate_manifest: tuple[dict[str, object], ...]
    scan_manifest: dict[str, object]


@dataclass(frozen=True)
class CorrectedPartitionResult:
    """Corrected detectability output rows for one variant partition."""

    rows: tuple[dict[str, object], ...]
    blind_rows: tuple[dict[str, object], ...]
    resolution_rows: tuple[dict[str, object], ...]
    candidate_manifest_rows: Mapping[str, dict[str, object]]
    scan_manifest_rows: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class BlindScanScoreBlock:
    """Reusable blind-scan scores for one instance/candidate/null setup."""

    windows: np.ndarray
    best_keys: tuple[str, ...]
    best_families: tuple[str, ...]
    best_witnesses: tuple[str, ...]
    best_projections: tuple[str, ...]
    raw_scores: np.ndarray
    candidate_p_values: np.ndarray
    scan_statistics: np.ndarray
    scan_p_values: np.ndarray
    window_count_total: int


def compute_corrected_detectability_frontier(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
    windows: WindowLibrary | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
) -> CorrectedDetectabilityResult:
    """Compute corrected min-p detectability frontiers."""

    variant_instances = instances_by_variant(dataset.instances)
    null_instances = {
        variant_id: _calibration_null_instances(instances, protocol)
        for variant_id, instances in variant_instances.items()
    }
    partitions = tuple(
        PartitionSpec(partition_id=str(variant_id), items=tuple(instances))
        for variant_id, instances in variant_instances.items()
    )
    partition_results = run_partitions(
        partitions,
        lambda partition: _cached_or_compute_partition(
            partition=partition,
            null_instances=null_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            cache=cache,
        ),
        n_jobs=n_jobs,
    )
    rows: list[dict[str, object]] = []
    blind_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    candidate_manifest_rows: dict[str, dict[str, object]] = {}
    scan_manifest_rows: dict[str, dict[str, object]] = {}
    for partition_result in partition_results:
        result = partition_result.value
        rows.extend(result.rows)
        blind_rows.extend(result.blind_rows)
        resolution_rows.extend(result.resolution_rows)
        candidate_manifest_rows.update(result.candidate_manifest_rows)
        scan_manifest_rows.update(result.scan_manifest_rows)
    frontier = pd.DataFrame(rows, columns=_frontier_columns())
    diagnostic = frontier.copy()
    blind_scan = pd.DataFrame(blind_rows, columns=_blind_scan_columns())
    resolution = pd.DataFrame(resolution_rows, columns=_resolution_columns())
    return CorrectedDetectabilityResult(
        frontier=frontier,
        oracle_window_diagnostic=diagnostic,
        blind_scan_events=blind_scan,
        calibration_resolution=resolution,
        candidate_nulls_manifest={
            "candidate_nulls_manifest_version": "synthgen.capability.candidate_nulls.v1",
            "candidate_nulls": list(candidate_manifest_rows.values()),
        },
        scan_nulls_manifest={
            "scan_nulls_manifest_version": "synthgen.capability.scan_nulls.v1",
            "scan_nulls": list(scan_manifest_rows.values()),
        },
    )


def _cached_or_compute_partition(
    *,
    partition: PartitionSpec[InstanceRecord],
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
) -> CorrectedPartitionResult:
    if cache is None:
        return _compute_partition(
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
            "event_group_count": int(sum(len(instance.event_groups) for instance in partition.items)),
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
        compute=lambda: _partition_to_payload(
            _compute_partition(
                instances=partition.items,
                null_instances=null_instances,
                protocol=protocol,
                arrays=arrays,
                windows=windows,
                cache=cache,
            )
        ),
    )
    return _partition_from_payload(payload)


def _compute_partition(
    *,
    instances: Sequence[InstanceRecord],
    null_instances: Mapping[str, Sequence[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    cache: CacheStore | None,
) -> CorrectedPartitionResult:
    calibrator = EmpiricalCalibrator()
    clean_cache: dict[str, np.ndarray] = {}
    anom_cache: dict[str, np.ndarray] = {}
    null_cache: dict[tuple[object, ...], CorrectedNullBundle] = {}
    score_cache = CleanWindowScoreCache()
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] = {}
    rows: list[dict[str, object]] = []
    blind_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    candidate_manifest_rows: dict[str, dict[str, object]] = {}
    scan_manifest_rows: dict[str, dict[str, object]] = {}
    for instance in instances:
        anom_cache[_instance_key(instance)] = _read_array(instance, "anomalous", arrays)
        for group in instance.event_groups:
            candidates = _candidate_specs(instance, group, protocol)
            if not candidates:
                continue
            event_length = max(1, group.length)
            scan_length = window_length_bin(event_length, protocol)
            null_key = (
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
            if null_key not in null_cache:
                null_cache[null_key] = _build_null_bundle(
                    clean_instances=null_instances.get(instance.variant_id, [instance]),
                    clean_cache=clean_cache,
                    score_cache=score_cache,
                    arrays=arrays,
                    windows=windows,
                    event_length=scan_length,
                    candidates=candidates,
                    protocol=protocol,
                    null_id=_null_id(instance.variant_id, scan_length, null_key),
                    calibrator=calibrator,
                )
            nulls = null_cache[null_key]
            for item in nulls.candidate_manifest:
                candidate_manifest_rows[str(item["candidate_null_id"])] = item
            scan_manifest_rows[str(nulls.scan_manifest["scan_null_id"])] = nulls.scan_manifest
            event_scores = _event_candidate_scores(
                series=anom_cache[_instance_key(instance)],
                group=group,
                candidates=candidates,
                protocol=protocol,
            )
            candidate_p_values = {
                key: calibrator.candidate_p_value(nulls.candidate_nulls[key], score)
                for key, score in event_scores.items()
                if key in nulls.candidate_nulls
            }
            if not candidate_p_values:
                continue
            best_key = min(candidate_p_values, key=candidate_p_values.get)
            best_spec = _spec_by_key(candidates, best_key)
            canonical_key = _best_canonical_key(group, candidates, candidate_p_values)
            best_raw_score = float(event_scores.get(best_key, math.nan))
            best_candidate_p = float(candidate_p_values[best_key])
            canonical_p = float(candidate_p_values[canonical_key]) if canonical_key is not None else math.nan
            canonical_spec = _spec_by_key(candidates, canonical_key) if canonical_key is not None else None
            canonical_scan_stat = (
                scan_statistic_from_candidate_p_values(np.asarray([canonical_p], dtype=np.float64))
                if math.isfinite(canonical_p)
                else math.nan
            )
            scan_stat = scan_statistic_from_candidate_p_values(
                np.asarray(list(candidate_p_values.values()), dtype=np.float64)
            )
            scan_p = calibrator.scan_p_value(nulls.scan_null, scan_stat)
            event_blind_rows, false_alerts_by_alpha, blind_window_count = _cached_blind_scan_rows(
                instance=instance,
                group=group,
                series=anom_cache[_instance_key(instance)],
                candidates=candidates,
                nulls=nulls,
                null_key=null_key,
                protocol=protocol,
                windows=windows,
                calibrator=calibrator,
                cache=cache,
                blind_scan_cache=blind_scan_cache,
            )
            blind_rows.extend(event_blind_rows)
            for alpha in protocol.alpha_grid:
                threshold = calibrator.scan_threshold(nulls.scan_null, float(alpha))
                detected = bool(scan_stat >= threshold) if math.isfinite(threshold) else False
                canonical_detected = (
                    bool(canonical_scan_stat >= threshold)
                    if math.isfinite(canonical_scan_stat) and math.isfinite(threshold)
                    else False
                )
                false_alert_count = int(false_alerts_by_alpha.get(float(alpha), 0))
                min_scan_p = min_empirical_p(len(nulls.scan_null.scan_statistics))
                row = {
                    **_common(instance, group),
                    "scope": "oracle_window",
                    "family": best_spec.family if best_spec is not None else "unknown",
                    "witness_or_model": best_spec.witness if best_spec is not None else "none",
                    "projection": _format_projection(best_spec.projection) if best_spec is not None else "",
                    "alpha": float(alpha),
                    "raw_score": finite_float(best_raw_score, default=math.nan),
                    "candidate_level_p_value": finite_float(best_candidate_p, default=math.nan),
                    "scan_statistic": finite_float(scan_stat, default=math.nan),
                    "scan_level_p_value": finite_float(scan_p, default=math.nan),
                    "scan_threshold": finite_float(threshold, default=math.nan),
                    "detected": detected,
                    "latency": 0,
                    "false_alert_count": false_alert_count,
                    "false_alert_rate": finite_float(
                        false_alert_count / max(float(blind_window_count), 1.0),
                        default=math.nan,
                    ),
                    "candidate_null_count": int(nulls.candidate_nulls[best_key].scores.size),
                    "scan_null_count": int(nulls.scan_null.scan_statistics.size),
                    "raw_candidate_count": int(nulls.scan_null.raw_candidate_count),
                    "effective_candidate_count": finite_float(nulls.scan_null.effective_candidate_count, default=math.nan),
                    "calibration_resolution_min_p": finite_float(min_scan_p, default=math.nan),
                    "calibration_status": calibration_status(
                        len(nulls.scan_null.scan_statistics),
                        float(alpha),
                        protocol.calibration_min_clean_scan_count_for_alpha,
                    ),
                    "best_detection_witness": best_key,
                    "best_canonical_witness": canonical_key or "",
                    "best_canonical_family": canonical_spec.family if canonical_spec is not None else "",
                    "best_canonical_projection": (
                        _format_projection(canonical_spec.projection) if canonical_spec is not None else ""
                    ),
                    "best_canonical_raw_score": finite_float(
                        event_scores.get(canonical_key, math.nan) if canonical_key is not None else math.nan,
                        default=math.nan,
                    ),
                    "best_canonical_candidate_p_value": finite_float(canonical_p, default=math.nan),
                    "best_canonical_scan_statistic": finite_float(canonical_scan_stat, default=math.nan),
                    "best_canonical_detected": canonical_detected,
                    "candidate_count": len(candidate_p_values),
                }
                rows.append(row)
                resolution_rows.append(
                    {
                        "event_id": row["event_id"],
                        "variant_id": row["variant_id"],
                        "split": row["split"],
                        "instance_id": row["instance_id"],
                        "anomaly_type": row["anomaly_type"],
                        "alpha": float(alpha),
                        "scope": "oracle_window",
                        "scan_null_count": int(nulls.scan_null.scan_statistics.size),
                        "candidate_null_count": int(nulls.candidate_nulls[best_key].scores.size),
                        "raw_candidate_count": int(nulls.scan_null.raw_candidate_count),
                        "effective_candidate_count": finite_float(nulls.scan_null.effective_candidate_count, default=math.nan),
                        "calibration_resolution_min_p": finite_float(min_scan_p, default=math.nan),
                        "calibration_status": calibration_status(
                            len(nulls.scan_null.scan_statistics),
                            float(alpha),
                            protocol.calibration_min_clean_scan_count_for_alpha,
                        ),
                    }
                )
    return CorrectedPartitionResult(
        rows=tuple(rows),
        blind_rows=tuple(blind_rows),
        resolution_rows=tuple(resolution_rows),
        candidate_manifest_rows=dict(candidate_manifest_rows),
        scan_manifest_rows=dict(scan_manifest_rows),
    )


def _partition_to_payload(result: CorrectedPartitionResult) -> dict[str, Any]:
    return {
        "corrected_detectability_partition_version": "synthgen.capability.corrected.partition.v1",
        "rows": list(result.rows),
        "blind_rows": list(result.blind_rows),
        "resolution_rows": list(result.resolution_rows),
        "candidate_manifest_rows": list(result.candidate_manifest_rows.values()),
        "scan_manifest_rows": list(result.scan_manifest_rows.values()),
    }


def _partition_from_payload(payload: Mapping[str, Any]) -> CorrectedPartitionResult:
    candidate_rows = tuple(dict(row) for row in payload.get("candidate_manifest_rows", ()))
    scan_rows = tuple(dict(row) for row in payload.get("scan_manifest_rows", ()))
    return CorrectedPartitionResult(
        rows=_ordered_rows(payload.get("rows", ()), _frontier_columns()),
        blind_rows=tuple(dict(row) for row in payload.get("blind_rows", ())),
        resolution_rows=_ordered_rows(payload.get("resolution_rows", ()), _resolution_columns()),
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


def _build_null_bundle(
    *,
    clean_instances: Sequence[InstanceRecord],
    clean_cache: dict[str, np.ndarray],
    score_cache: CleanWindowScoreCache,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    event_length: int,
    candidates: Sequence[CandidateSpec],
    protocol: CapabilityProtocol,
    null_id: str,
    calibrator: EmpiricalCalibrator,
) -> CorrectedNullBundle:
    candidate_scores: dict[str, list[float]] = {candidate.key: [] for candidate in candidates}
    raw_candidate_count = 0
    blocks: dict[tuple[int, ...], CleanWindowScoreBlock] = {}
    for candidate in candidates:
        if candidate.projection not in blocks:
            blocks[candidate.projection] = score_cache.get(
                clean_instances=clean_instances,
                clean_cache=clean_cache,
                arrays=arrays,
                windows=windows,
                event_length=int(event_length),
                subset=candidate.projection,
                protocol=protocol,
            )
        block = blocks[candidate.projection]
        for values in block.scores_by_witness.get(candidate.witness, ()):
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            raw_candidate_count += int(finite.size)
            candidate_scores[candidate.key].extend(float(value) for value in finite)
    candidate_nulls: dict[str, CandidateNull] = {}
    candidate_manifest: list[dict[str, object]] = []
    for candidate in candidates:
        scores = np.asarray(candidate_scores[candidate.key], dtype=np.float64)
        null = CandidateNull(
            family=candidate.family,
            witness_or_model=candidate.witness,
            projection_policy=_format_projection(candidate.projection),
            window_length_bin=int(event_length),
            scores=scores,
            candidate_count=int(scores.size),
        )
        candidate_nulls[candidate.key] = null
        candidate_manifest.append(
            {
                "candidate_null_id": f"{null_id}__{candidate.key}",
                "family": null.family,
                "witness_or_model": null.witness_or_model,
                "projection_policy": null.projection_policy,
                "window_length_bin": null.window_length_bin,
                "candidate_count": null.candidate_count,
                "score_min": finite_float(np.min(scores), default=math.nan) if scores.size else math.nan,
                "score_median": finite_float(np.median(scores), default=math.nan) if scores.size else math.nan,
                "score_max": finite_float(np.max(scores), default=math.nan) if scores.size else math.nan,
            }
        )
    scan_stats: list[float] = []
    for scores in _clean_scan_score_rows(candidates, blocks):
        p_values = []
        for candidate_key, score in scores.items():
            p_values.append(calibrator.candidate_p_value(candidate_nulls[candidate_key], score))
        scan_stats.append(
            scan_statistic_from_candidate_p_values(np.asarray(p_values, dtype=np.float64))
        )
    scan_arr = np.asarray(scan_stats, dtype=np.float64)
    effective_count = float(raw_candidate_count) / max(float(scan_arr.size), 1.0)
    scan_null = ScanNull(
        scope="oracle_window",
        family="all",
        aggregation="min_candidate_p",
        window_length_bin=int(event_length),
        scan_statistics=scan_arr,
        raw_candidate_count=int(raw_candidate_count),
        effective_candidate_count=finite_float(effective_count, default=math.nan),
    )
    scan_manifest = {
        "scan_null_id": null_id,
        "scope": scan_null.scope,
        "family": scan_null.family,
        "aggregation": scan_null.aggregation,
        "window_length_bin": scan_null.window_length_bin,
        "scan_null_count": int(scan_null.scan_statistics.size),
        "raw_candidate_count": int(scan_null.raw_candidate_count),
        "effective_candidate_count": finite_float(scan_null.effective_candidate_count, default=math.nan),
        "scan_stat_min": finite_float(np.min(scan_arr), default=math.nan) if scan_arr.size else math.nan,
        "scan_stat_median": finite_float(np.median(scan_arr), default=math.nan) if scan_arr.size else math.nan,
        "scan_stat_max": finite_float(np.max(scan_arr), default=math.nan) if scan_arr.size else math.nan,
    }
    return CorrectedNullBundle(
        candidate_nulls=candidate_nulls,
        scan_null=scan_null,
        candidate_manifest=tuple(candidate_manifest),
        scan_manifest=scan_manifest,
    )


def _clean_scan_score_rows(
    candidates: Sequence[CandidateSpec],
    blocks: Mapping[tuple[int, ...], CleanWindowScoreBlock],
) -> list[dict[str, float]]:
    if not candidates:
        return []
    first = blocks.get(candidates[0].projection)
    if first is None:
        return []
    rows: list[dict[str, float]] = []
    for instance_index, window_count in enumerate(first.window_counts):
        for window_index in range(int(window_count)):
            scores: dict[str, float] = {}
            for candidate in candidates:
                block = blocks.get(candidate.projection)
                if block is None or instance_index >= len(block.window_counts):
                    continue
                values_by_instance = block.scores_by_witness.get(candidate.witness, ())
                if instance_index >= len(values_by_instance):
                    continue
                values = values_by_instance[instance_index]
                if window_index >= len(values):
                    continue
                score = float(values[window_index])
                if math.isfinite(score):
                    scores[candidate.key] = score
            if scores:
                rows.append(scores)
    return rows


def _event_candidate_scores(
    *,
    series: np.ndarray,
    group: EventGroup,
    candidates: Sequence[CandidateSpec],
    protocol: CapabilityProtocol,
) -> dict[str, float]:
    return _scores_for_window(
        series=series,
        start=group.start,
        end=group.end,
        candidates=candidates,
        protocol=protocol,
    )


def _blind_scan_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None = None,
    score_cache_key: tuple[object, ...] | None = None,
) -> tuple[list[dict[str, object]], dict[float, int], int]:
    scan_length = window_length_bin(max(1, group.length), protocol)
    block = _cached_blind_scan_score_block(
        series=series,
        scan_length=scan_length,
        candidates=candidates,
        nulls=nulls,
        protocol=protocol,
        windows=windows,
        calibrator=calibrator,
        blind_scan_cache=blind_scan_cache,
        score_cache_key=score_cache_key,
    )
    rows: list[dict[str, object]] = []
    false_alerts_by_alpha = {float(alpha): 0 for alpha in protocol.alpha_grid}
    thresholds = {
        float(alpha): calibrator.scan_threshold(nulls.scan_null, float(alpha))
        for alpha in protocol.alpha_grid
    }
    for index, (start, end) in enumerate(block.windows):
        scan_stat = float(block.scan_statistics[index])
        if not math.isfinite(scan_stat):
            continue
        overlaps_event = _overlaps(int(start), int(end), group.start, group.end)
        for alpha in protocol.alpha_grid:
            alpha_value = float(alpha)
            threshold = thresholds[alpha_value]
            detected = bool(scan_stat >= threshold) if math.isfinite(threshold) else False
            if not detected:
                continue
            false_alert = not overlaps_event
            if false_alert:
                false_alerts_by_alpha[alpha_value] += 1
            rows.append(
                {
                    "event_id": event_uid(instance, group),
                    "variant_id": instance.variant_id,
                    "split": instance.split,
                    "instance_id": instance.instance_id,
                    "window_start": int(start),
                    "window_end": int(end),
                    "overlaps_oracle_event": overlaps_event,
                    "false_alert": false_alert,
                    "scope": "blind_scan",
                    "family": block.best_families[index],
                    "witness_or_model": block.best_witnesses[index],
                    "projection": block.best_projections[index],
                    "alpha": alpha_value,
                    "raw_score": finite_float(block.raw_scores[index], default=math.nan),
                    "candidate_level_p_value": finite_float(block.candidate_p_values[index], default=math.nan),
                    "scan_statistic": finite_float(scan_stat, default=math.nan),
                    "scan_level_p_value": finite_float(block.scan_p_values[index], default=math.nan),
                    "scan_threshold": finite_float(threshold, default=math.nan),
                    "detected": detected,
                    "candidate_null_count": int(nulls.candidate_nulls[block.best_keys[index]].scores.size),
                    "scan_null_count": int(nulls.scan_null.scan_statistics.size),
                    "raw_candidate_count": int(nulls.scan_null.raw_candidate_count),
                    "effective_candidate_count": finite_float(nulls.scan_null.effective_candidate_count, default=math.nan),
                    "calibration_resolution_min_p": finite_float(
                        min_empirical_p(len(nulls.scan_null.scan_statistics)),
                        default=math.nan,
                    ),
                    "calibration_status": calibration_status(
                        len(nulls.scan_null.scan_statistics),
                        alpha_value,
                        protocol.calibration_min_clean_scan_count_for_alpha,
                    ),
                }
            )
    return rows, false_alerts_by_alpha, block.window_count_total


def _cached_blind_scan_score_block(
    *,
    series: np.ndarray,
    scan_length: int,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None,
    score_cache_key: tuple[object, ...] | None,
) -> BlindScanScoreBlock:
    key = score_cache_key or (
        int(series.shape[0]),
        int(scan_length),
        tuple(candidate.key for candidate in candidates),
        id(nulls),
    )
    if blind_scan_cache is not None and key in blind_scan_cache:
        return blind_scan_cache[key]
    block = _blind_scan_score_block(
        series=series,
        scan_length=scan_length,
        candidates=candidates,
        nulls=nulls,
        protocol=protocol,
        windows=windows,
        calibrator=calibrator,
    )
    if blind_scan_cache is not None:
        blind_scan_cache[key] = block
    return block


def _blind_scan_score_block(
    *,
    series: np.ndarray,
    scan_length: int,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
) -> BlindScanScoreBlock:
    scan_windows = np.asarray(
        _scan_windows(series.shape[0], scan_length, protocol, windows),
        dtype=np.int64,
    )
    best_keys: list[str] = []
    best_families: list[str] = []
    best_witnesses: list[str] = []
    best_projections: list[str] = []
    raw_scores: list[float] = []
    candidate_p_values: list[float] = []
    scan_statistics: list[float] = []
    scan_p_values: list[float] = []
    retained_windows: list[tuple[int, int]] = []
    for start, end in scan_windows:
        scores = _scores_for_window(
            series=series,
            start=int(start),
            end=int(end),
            candidates=candidates,
            protocol=protocol,
        )
        p_values = {
            key: calibrator.candidate_p_value(nulls.candidate_nulls[key], score)
            for key, score in scores.items()
            if key in nulls.candidate_nulls
        }
        if not p_values:
            continue
        best_key = min(p_values, key=p_values.get)
        best_spec = _spec_by_key(candidates, best_key)
        scan_stat = scan_statistic_from_candidate_p_values(
            np.asarray(list(p_values.values()), dtype=np.float64)
        )
        retained_windows.append((int(start), int(end)))
        best_keys.append(best_key)
        best_families.append(best_spec.family if best_spec is not None else "unknown")
        best_witnesses.append(best_spec.witness if best_spec is not None else "none")
        best_projections.append(
            _format_projection(best_spec.projection) if best_spec is not None else ""
        )
        raw_scores.append(finite_float(scores.get(best_key, math.nan), default=math.nan))
        candidate_p_values.append(finite_float(p_values.get(best_key, math.nan), default=math.nan))
        scan_statistics.append(finite_float(scan_stat, default=math.nan))
        scan_p_values.append(
            finite_float(calibrator.scan_p_value(nulls.scan_null, scan_stat), default=math.nan)
        )
    return BlindScanScoreBlock(
        windows=np.asarray(retained_windows, dtype=np.int64).reshape((-1, 2)),
        best_keys=tuple(best_keys),
        best_families=tuple(best_families),
        best_witnesses=tuple(best_witnesses),
        best_projections=tuple(best_projections),
        raw_scores=np.asarray(raw_scores, dtype=np.float64),
        candidate_p_values=np.asarray(candidate_p_values, dtype=np.float64),
        scan_statistics=np.asarray(scan_statistics, dtype=np.float64),
        scan_p_values=np.asarray(scan_p_values, dtype=np.float64),
        window_count_total=int(scan_windows.shape[0]),
    )


def _cached_blind_scan_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    null_key: tuple[object, ...],
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    cache: CacheStore | None,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None = None,
) -> tuple[list[dict[str, object]], dict[float, int], int]:
    scan_length = window_length_bin(max(1, group.length), protocol)
    blind_window_count = len(list(_scan_windows(series.shape[0], scan_length, protocol, windows)))
    score_cache_key = (
        _instance_key(instance),
        int(scan_length),
        tuple(candidate.key for candidate in candidates),
        repr(null_key),
        int(series.shape[0]),
    )
    if cache is None:
        return _blind_scan_rows(
            instance=instance,
            group=group,
            series=series,
            candidates=candidates,
            nulls=nulls,
            protocol=protocol,
            windows=windows,
            calibrator=calibrator,
            blind_scan_cache=blind_scan_cache,
            score_cache_key=score_cache_key,
        )
    partition_id = event_uid(instance, group)
    fingerprint = cache.fingerprint(
        profile_name="blind_scan_events",
        partition_id=partition_id,
        extra={
            "null_key": repr(null_key),
            "candidate_keys": [candidate.key for candidate in candidates],
            "event_length": int(max(1, group.length)),
            "window_length_bin": int(scan_length),
            "window_count": int(blind_window_count),
        },
    )

    def compute() -> pd.DataFrame:
        rows, _, _ = _blind_scan_rows(
            instance=instance,
            group=group,
            series=series,
            candidates=candidates,
            nulls=nulls,
            protocol=protocol,
            windows=windows,
            calibrator=calibrator,
            blind_scan_cache=blind_scan_cache,
            score_cache_key=score_cache_key,
        )
        return pd.DataFrame(rows, columns=_blind_scan_columns())

    frame = cache.get_or_compute_table(
        profile_name="blind_scan_events",
        partition_id=partition_id,
        fingerprint=fingerprint,
        compute=compute,
    )
    frame = _normalize_blind_scan_frame(frame)
    return (
        frame.to_dict("records"),
        _false_alerts_by_alpha(frame, protocol),
        blind_window_count,
    )


def _scores_for_window(
    *,
    series: np.ndarray,
    start: int,
    end: int,
    candidates: Sequence[CandidateSpec],
    protocol: CapabilityProtocol,
) -> dict[str, float]:
    by_projection: dict[tuple[int, ...], dict[str, float]] = {}
    for candidate in candidates:
        if candidate.projection not in by_projection:
            by_projection[candidate.projection] = detection_window_scores(
                series=series,
                start=start,
                end=end,
                channels=candidate.projection,
                context_multiplier=protocol.context_window_multiplier,
                min_context_points=protocol.min_context_points,
            )
    result: dict[str, float] = {}
    for candidate in candidates:
        scores = by_projection.get(candidate.projection, {})
        if candidate.witness in scores:
            result[candidate.key] = finite_float(scores[candidate.witness])
    return result


def _candidate_specs(
    instance: InstanceRecord,
    group: EventGroup,
    protocol: CapabilityProtocol,
) -> tuple[CandidateSpec, ...]:
    channels = tuple(sorted(set(group.group_channels) | set(group.context_channels) | set(group.intervention_channels)))
    if not channels:
        channels = tuple(range(instance.channels))
    specs: list[CandidateSpec] = []
    for subset in channel_subsets(channels, protocol.max_projection_size):
        for witness in protocol.detection_witnesses:
            if witness_requires_projection_size(witness) <= len(subset):
                specs.append(CandidateSpec(witness=str(witness), projection=tuple(subset)))
    return tuple(specs)


def _calibration_null_instances(
    instances: Sequence[InstanceRecord],
    protocol: CapabilityProtocol,
) -> tuple[InstanceRecord, ...]:
    if protocol.calibration_split:
        preferred = tuple(instance for instance in instances if instance.split == protocol.calibration_split)
        if preferred:
            return preferred
    return tuple(instances)


def _spec_by_key(candidates: Sequence[CandidateSpec], key: str) -> CandidateSpec | None:
    for candidate in candidates:
        if candidate.key == key:
            return candidate
    return None


def _best_canonical_key(
    group: EventGroup,
    candidates: Sequence[CandidateSpec],
    candidate_p_values: Mapping[str, float],
) -> str | None:
    """Return the strongest calibrated canonical candidate for an event."""

    canonical: dict[str, float] = {}
    for candidate in candidates:
        if candidate.key not in candidate_p_values:
            continue
        if not _candidate_is_canonical(group, candidate):
            continue
        value = float(candidate_p_values[candidate.key])
        if math.isfinite(value):
            canonical[candidate.key] = value
    if not canonical:
        return None
    return min(canonical, key=canonical.get)


def _candidate_is_canonical(group: EventGroup, candidate: CandidateSpec) -> bool:
    """Return whether a candidate matches the event's canonical detection policy."""

    canonical = {
        "mean": ("mean_z",),
        "variance": ("variance_log_ratio", "local_energy_z"),
        "amplitude": ("variance_log_ratio", "local_energy_z"),
        "platform": ("mean_z", "local_energy_z"),
        "pattern": ("local_energy_z",),
        "frequency": ("local_energy_z",),
        "correlation-flip": ("correlation_shift", "covariance_shift"),
        "covariance-change": ("covariance_shift", "correlation_shift"),
        "lag-synchronization": ("correlation_shift", "covariance_shift"),
        "mode-correlation": ("correlation_shift", "covariance_shift"),
    }.get(group.anomaly_type, ())
    if candidate.witness not in canonical:
        return False
    if "dependence" in group.constraint_tag or "relation" in group.semantic_scope:
        return len(candidate.projection) >= 2
    return True


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
        "group_channels": _format_projection(group.group_channels),
    }


def _format_projection(channels: Sequence[int]) -> str:
    return "|".join(str(int(channel)) for channel in channels)


def _instance_key(instance: InstanceRecord) -> str:
    return f"{instance.variant_id}/{instance.split}/{instance.instance_id}"


def _overlaps(start: int, end: int, target_start: int, target_end: int) -> bool:
    return int(start) < int(target_end) and int(end) > int(target_start)


def _null_id(variant_id: str, event_length: int, key: tuple[object, ...]) -> str:
    payload = repr(key).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{variant_id}__len{int(event_length)}__{digest}"


def _read_array(instance: InstanceRecord, kind: str, arrays: ArrayStore | None) -> np.ndarray:
    if arrays is not None and kind == "clean":
        return arrays.get(instance, "clean")
    if arrays is not None and kind == "anomalous":
        return arrays.get(instance, "anomalous")
    if kind == "clean":
        return read_timeseries_csv(instance.clean_path)
    return read_timeseries_csv(instance.anomalous_path)


def _scan_windows(
    series_length: int,
    event_length: int,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> np.ndarray | list[tuple[int, int]]:
    if windows is None:
        return contiguous_windows(
            series_length,
            int(event_length),
            max_windows=protocol.max_scan_windows_per_length,
            stride_fraction=protocol.clean_window_stride_fraction,
        )
    return windows.get(
        WindowSpec(
            series_length=series_length,
            window_length=int(event_length),
            max_windows=protocol.max_scan_windows_per_length,
            stride_fraction=protocol.clean_window_stride_fraction,
        )
    )


def _frontier_columns() -> list[str]:
    return _common_columns() + [
        "scope",
        "family",
        "witness_or_model",
        "projection",
        "alpha",
        "raw_score",
        "candidate_level_p_value",
        "scan_statistic",
        "scan_level_p_value",
        "scan_threshold",
        "detected",
        "latency",
        "false_alert_count",
        "false_alert_rate",
        "candidate_null_count",
        "scan_null_count",
        "raw_candidate_count",
        "effective_candidate_count",
        "calibration_resolution_min_p",
        "calibration_status",
        "best_detection_witness",
        "best_canonical_witness",
        "best_canonical_family",
        "best_canonical_projection",
        "best_canonical_raw_score",
        "best_canonical_candidate_p_value",
        "best_canonical_scan_statistic",
        "best_canonical_detected",
        "candidate_count",
    ]


def _resolution_columns() -> list[str]:
    return [
        "event_id",
        "variant_id",
        "split",
        "instance_id",
        "anomaly_type",
        "alpha",
        "scope",
        "scan_null_count",
        "candidate_null_count",
        "raw_candidate_count",
        "effective_candidate_count",
        "calibration_resolution_min_p",
        "calibration_status",
    ]


def _blind_scan_columns() -> list[str]:
    return [
        "event_id",
        "variant_id",
        "split",
        "instance_id",
        "window_start",
        "window_end",
        "overlaps_oracle_event",
        "false_alert",
        "scope",
        "family",
        "witness_or_model",
        "projection",
        "alpha",
        "raw_score",
        "candidate_level_p_value",
        "scan_statistic",
        "scan_level_p_value",
        "scan_threshold",
        "detected",
        "candidate_null_count",
        "scan_null_count",
        "raw_candidate_count",
        "effective_candidate_count",
        "calibration_resolution_min_p",
        "calibration_status",
    ]


def _common_columns() -> list[str]:
    return [
        "event_id",
        "variant_id",
        "split",
        "instance_id",
        "base_oscillation",
        "anomaly_type",
        "constraint_tag",
        "semantic_scope",
        "start",
        "end",
        "length",
        "group_channels",
    ]


def _ordered_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> tuple[dict[str, object], ...]:
    ordered: list[dict[str, object]] = []
    for row in rows:
        active = {column: row.get(column) for column in columns}
        for key, value in row.items():
            if key not in active:
                active[str(key)] = value
        ordered.append(active)
    return tuple(ordered)


def _normalize_blind_scan_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_blind_scan_columns())
    active = frame.copy()
    for column in _blind_scan_columns():
        if column not in active.columns:
            active[column] = pd.Series(dtype=object)
    for column in ("overlaps_oracle_event", "false_alert", "detected"):
        active[column] = active[column].map(_as_bool)
    return active[_blind_scan_columns()]


def _false_alerts_by_alpha(
    frame: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> dict[float, int]:
    counts = {float(alpha): 0 for alpha in protocol.alpha_grid}
    if frame.empty:
        return counts
    active = _normalize_blind_scan_frame(frame)
    false_alerts = active[active["false_alert"].map(_as_bool)]
    if false_alerts.empty:
        return counts
    for alpha, count in false_alerts.groupby("alpha").size().items():
        counts[float(alpha)] = int(count)
    return counts


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
