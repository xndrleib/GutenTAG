"""Structural model-zoo capability frontier."""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..manifest import canonical_json_hash
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
from .dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid, instances_by_variant, read_timeseries_csv
from .models import CapabilityModel, default_model_registry
from .numerics import contiguous_windows, finite_float
from .partitions import PartitionSpec, partition_sequence, run_partitions
from .protocol import CapabilityProtocol, window_length_bin
from .windows import WindowLibrary, WindowSpec


@dataclass(frozen=True)
class ModelZooResult:
    """Structural model-zoo output tables."""

    frontier: pd.DataFrame
    event_scores: pd.DataFrame
    model_manifest: dict[str, object]


@dataclass(frozen=True)
class ModelNull:
    """Corrected clean nulls for one model/window length."""

    candidate_null: CandidateNull
    scan_null: ScanNull
    raw_candidate_count: int
    effective_candidate_count: float


@dataclass(frozen=True)
class ModelZooPartitionResult:
    """Model-zoo tables for one deterministic variant partition."""

    frontier: pd.DataFrame
    event_scores: pd.DataFrame
    manifest_rows: pd.DataFrame


def compute_model_zoo_frontier(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
    windows: WindowLibrary | None = None,
    models: Sequence[CapabilityModel] | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 1,
) -> ModelZooResult:
    """Compute model-zoo event and alpha frontiers."""

    active_models = tuple(models) if models is not None else None
    variant_instances = instances_by_variant(dataset.instances)
    variant_ids = tuple(variant_instances)
    if cache is None and int(n_jobs) <= 1:
        return _combine_partition_results(
            (
                _model_zoo_for_variants(
                    variant_ids=variant_ids,
                    variant_instances=variant_instances,
                    protocol=protocol,
                    arrays=arrays,
                    windows=windows,
                    active_models=active_models,
                ),
            )
        )
    partitions = partition_sequence(
        variant_ids,
        partition_size=max(1, int(partition_size)),
        prefix="model_zoo",
    )
    fingerprint_extra = {
        "model_fingerprint": _models_fingerprint(active_models),
        "partition_size": int(partition_size),
    }

    def worker(partition: PartitionSpec[str]) -> ModelZooPartitionResult:
        if cache is not None:
            return _cached_model_zoo_partition(
                cache=cache,
                partition_id=partition.partition_id,
                variant_ids=partition.items,
                variant_instances=variant_instances,
                protocol=protocol,
                arrays=arrays,
                windows=windows,
                active_models=active_models,
                fingerprint_extra=fingerprint_extra,
            )
        return _model_zoo_for_variants(
            variant_ids=partition.items,
            variant_instances=variant_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            active_models=active_models,
        )

    results = run_partitions(partitions, worker, n_jobs=n_jobs)
    return _combine_partition_results(tuple(result.value for result in results))


def _model_zoo_for_variants(
    *,
    variant_ids: Sequence[str],
    variant_instances: dict[str, list[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    active_models: Sequence[CapabilityModel] | None,
) -> ModelZooPartitionResult:
    """Compute model-zoo rows for a deterministic variant partition."""

    clean_cache: dict[str, np.ndarray] = {}
    anom_cache: dict[str, np.ndarray] = {}
    fitted_by_projection: dict[tuple[str, tuple[int, ...]], tuple[CapabilityModel, ...]] = {}
    null_cache: dict[tuple[object, ...], ModelNull] = {}
    frontier_rows: list[dict[str, object]] = []
    event_score_rows: list[dict[str, object]] = []
    manifest_rows: dict[str, dict[str, object]] = {}
    for variant_id in variant_ids:
        instances = variant_instances.get(variant_id, [])
        clean_instances = [_read_array(instance, "clean", arrays, clean_cache) for instance in instances]
        for projection in _variant_model_projections(instances):
            projected_clean = [_project_array(clean, projection) for clean in clean_instances]
            fitted = _fit_models(active_models, projected_clean)
            fitted_by_projection[(variant_id, projection)] = fitted
            projection_label = _projection_label(projection)
            for model in fitted:
                metadata = {
                    **model.metadata(),
                    "model_projection": projection_label,
                    "model_projection_size": len(projection),
                }
                manifest_key = f"{variant_id}:{model.model_id}:{projection_label}"
                manifest_rows[manifest_key] = {
                    **metadata,
                    "metadata_hash": canonical_json_hash(metadata),
                    "fit_variant_id": variant_id,
                    "fit_instance_count": len(clean_instances),
                }
    for variant_id in variant_ids:
        for instance in variant_instances.get(variant_id, []):
            _append_instance_model_rows(
                instance=instance,
                variant_instances=variant_instances,
                fitted_by_projection=fitted_by_projection,
                null_cache=null_cache,
                clean_cache=clean_cache,
                anom_cache=anom_cache,
                manifest_rows=manifest_rows,
                frontier_rows=frontier_rows,
                event_score_rows=event_score_rows,
                protocol=protocol,
                arrays=arrays,
                windows=windows,
            )
    return ModelZooPartitionResult(
        frontier=_sort_frontier(pd.DataFrame(frontier_rows)),
        event_scores=_sort_event_scores(pd.DataFrame(event_score_rows) if event_score_rows else pd.DataFrame()),
        manifest_rows=_sort_manifest(pd.DataFrame(manifest_rows.values()) if manifest_rows else pd.DataFrame()),
    )


def _append_instance_model_rows(
    *,
    instance: InstanceRecord,
    variant_instances: dict[str, list[InstanceRecord]],
    fitted_by_projection: dict[tuple[str, tuple[int, ...]], tuple[CapabilityModel, ...]],
    null_cache: dict[tuple[object, ...], ModelNull],
    clean_cache: dict[str, np.ndarray],
    anom_cache: dict[str, np.ndarray],
    manifest_rows: dict[str, dict[str, object]],
    frontier_rows: list[dict[str, object]],
    event_score_rows: list[dict[str, object]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
) -> None:
    anom = _read_array(instance, "anomalous", arrays, anom_cache)
    for group in instance.event_groups:
        projection = _model_projection(instance, group)
        projection_label = _projection_label(projection)
        projected_anom = _project_array(anom, projection)
        fitted = fitted_by_projection.get((instance.variant_id, projection), ())
        event_window = np.asarray([[group.start, group.end]], dtype=int)
        event_length = max(1, group.length)
        scan_length = window_length_bin(event_length, protocol)
        for model in fitted:
            null_key = (
                instance.variant_id,
                model.model_id,
                projection,
                scan_length,
                protocol.max_scan_windows_per_length,
                protocol.window_length_policy_mode,
                tuple(protocol.window_length_bins),
                protocol.clean_window_stride_fraction,
            )
            if null_key not in null_cache:
                null_cache[null_key] = _model_null(
                    model=model,
                    instances=variant_instances.get(instance.variant_id, [instance]),
                    projection=projection,
                    clean_cache=clean_cache,
                    arrays=arrays,
                    windows=windows,
                    event_length=scan_length,
                    protocol=protocol,
                )
            null = null_cache[null_key]
            raw_score = float(model.score_windows(projected_anom, event_window)[0])
            p_value = _model_candidate_p_value(null, raw_score)
            scan_stat = _model_scan_statistic(null, raw_score)
            scan_p_value = _model_scan_p_value(null, raw_score)
            intervention_score = raw_score
            context_score = _context_score(model, projected_anom, group, instance.length)
            metadata = manifest_rows.get(f"{instance.variant_id}:{model.model_id}:{projection_label}", model.metadata())
            metadata_hash = str(metadata.get("metadata_hash", canonical_json_hash(model.metadata())))
            event_score_rows.append(
                {
                    **_common(instance, group),
                    "model_id": model.model_id,
                    "model_family": model.family,
                    "model_projection": projection_label,
                    "model_projection_size": len(projection),
                    "fit_split": "all_clean",
                    "calibration_split": "all_clean",
                    "raw_score": finite_float(raw_score, default=math.nan),
                    "candidate_level_p_value": finite_float(p_value, default=math.nan),
                    "scan_statistic": finite_float(scan_stat, default=math.nan),
                    "scan_level_p_value": finite_float(scan_p_value, default=math.nan),
                    "candidate_null_count": int(null.candidate_null.scores.size),
                    "scan_null_count": int(null.scan_null.scan_statistics.size),
                    "raw_candidate_count": int(null.raw_candidate_count),
                    "effective_candidate_count": finite_float(null.effective_candidate_count, default=math.nan),
                    "intervention_channel_score": finite_float(intervention_score, default=math.nan),
                    "context_channel_score": finite_float(context_score, default=math.nan),
                    "model_metadata_hash": metadata_hash,
                }
            )
            for alpha in protocol.alpha_grid:
                threshold = EmpiricalCalibrator().scan_threshold(null.scan_null, float(alpha))
                detected = bool(scan_stat >= threshold) if math.isfinite(threshold) else False
                false_alert_count, false_alert_rate = _false_alert_rate(
                    model=model,
                    series=projected_anom,
                    group=group,
                    event_length=scan_length,
                    null=null,
                    threshold=threshold,
                    protocol=protocol,
                    windows=windows,
                )
                row = {
                    **_common(instance, group),
                    "model_id": model.model_id,
                    "model_family": model.family,
                    "model_projection": projection_label,
                    "model_projection_size": len(projection),
                    "fit_split": "all_clean",
                    "calibration_split": "all_clean",
                    "alpha": float(alpha),
                    "raw_score": finite_float(raw_score, default=math.nan),
                    "candidate_level_p_value": finite_float(p_value, default=math.nan),
                    "scan_statistic": finite_float(scan_stat, default=math.nan),
                    "scan_level_p_value": finite_float(scan_p_value, default=math.nan),
                    "scan_threshold": finite_float(threshold, default=math.nan),
                    "detected": detected,
                    "latency": 0,
                    "false_alert_count": int(false_alert_count),
                    "false_alert_rate": finite_float(false_alert_rate, default=math.nan),
                    "candidate_null_count": int(null.candidate_null.scores.size),
                    "scan_null_count": int(null.scan_null.scan_statistics.size),
                    "raw_candidate_count": int(null.raw_candidate_count),
                    "effective_candidate_count": finite_float(null.effective_candidate_count, default=math.nan),
                    "calibration_resolution_min_p": finite_float(
                        min_empirical_p(null.scan_null.scan_statistics.size),
                        default=math.nan,
                    ),
                    "calibration_status": calibration_status(
                        null.scan_null.scan_statistics.size,
                        float(alpha),
                        protocol.calibration_min_clean_scan_count_for_alpha,
                    ),
                    "intervention_channel_score": finite_float(intervention_score, default=math.nan),
                    "context_channel_score": finite_float(context_score, default=math.nan),
                    "model_metadata_hash": metadata_hash,
                }
                frontier_rows.append(row)


def _combine_partition_results(results: Sequence[ModelZooPartitionResult]) -> ModelZooResult:
    frontier_frames = [result.frontier for result in results if not result.frontier.empty]
    event_frames = [result.event_scores for result in results if not result.event_scores.empty]
    manifest_frames = [result.manifest_rows for result in results if not result.manifest_rows.empty]
    frontier = _sort_frontier(pd.concat(frontier_frames, ignore_index=True) if frontier_frames else pd.DataFrame())
    event_scores = _sort_event_scores(pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame())
    manifest_frame = _sort_manifest(pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame())
    if not manifest_frame.empty:
        subset = ["fit_variant_id", "model_id"]
        if "model_projection" in manifest_frame.columns:
            subset.append("model_projection")
        manifest_frame = manifest_frame.drop_duplicates(
            subset=subset,
            keep="first",
        )
    return ModelZooResult(
        frontier=frontier,
        event_scores=event_scores,
        model_manifest={
            "model_zoo_model_manifest_version": "synthgen.capability.model_zoo.v1",
            "models": manifest_frame.to_dict("records") if not manifest_frame.empty else [],
        },
    )


def _cached_model_zoo_partition(
    *,
    cache: CacheStore,
    partition_id: str,
    variant_ids: Sequence[str],
    variant_instances: dict[str, list[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    active_models: Sequence[CapabilityModel] | None,
    fingerprint_extra: dict[str, object],
) -> ModelZooPartitionResult:
    profiles = (
        "model_zoo_frontier",
        "model_zoo_event_scores",
        "model_zoo_manifest_rows",
    )
    fingerprints = {
        profile: cache.fingerprint(
            profile_name=profile,
            partition_id=partition_id,
            extra={
                **fingerprint_extra,
                "variant_ids": list(variant_ids),
            },
        )
        for profile in profiles
    }
    if cache.resume and all(
        cache.table_path(profile, partition_id).exists()
        and cache.is_valid(
            profile_name=profile,
            partition_id=partition_id,
            fingerprint=fingerprints[profile],
        )
        for profile in profiles
    ):
        return ModelZooPartitionResult(
            frontier=_read_cached_table(cache, "model_zoo_frontier", partition_id),
            event_scores=_read_cached_table(cache, "model_zoo_event_scores", partition_id),
            manifest_rows=_read_cached_table(cache, "model_zoo_manifest_rows", partition_id),
        )
    result = _model_zoo_for_variants(
        variant_ids=variant_ids,
        variant_instances=variant_instances,
        protocol=protocol,
        arrays=arrays,
        windows=windows,
        active_models=active_models,
    )
    cache.write_table_partition(
        profile_name="model_zoo_frontier",
        partition_id=partition_id,
        fingerprint=fingerprints["model_zoo_frontier"],
        frame=result.frontier,
    )
    cache.write_table_partition(
        profile_name="model_zoo_event_scores",
        partition_id=partition_id,
        fingerprint=fingerprints["model_zoo_event_scores"],
        frame=result.event_scores,
    )
    cache.write_table_partition(
        profile_name="model_zoo_manifest_rows",
        partition_id=partition_id,
        fingerprint=fingerprints["model_zoo_manifest_rows"],
        frame=result.manifest_rows,
    )
    return result


def _variant_model_projections(instances: Sequence[InstanceRecord]) -> tuple[tuple[int, ...], ...]:
    projections = {_model_projection(instance, group) for instance in instances for group in instance.event_groups}
    if not projections and instances:
        projections.add(tuple(range(instances[0].channels)))
    return tuple(sorted(projections))


def _model_projection(instance: InstanceRecord, group: EventGroup) -> tuple[int, ...]:
    if group.semantic_scope in {"relation", "regime_relation"}:
        projection = tuple(sorted(set(group.group_channels) | set(group.context_channels) | set(group.intervention_channels)))
        if len(projection) >= 2:
            return projection
    return tuple(range(instance.channels))


def _project_array(values: np.ndarray, projection: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not projection:
        return array
    if tuple(projection) == tuple(range(array.shape[1])):
        return array
    return array[:, projection]


def _projection_label(projection: tuple[int, ...]) -> str:
    return "|".join(str(channel) for channel in projection)


def _fit_models(
    models: Sequence[CapabilityModel] | None,
    clean_instances: Sequence[np.ndarray],
) -> tuple[CapabilityModel, ...]:
    fitted: list[CapabilityModel] = []
    model_instances = default_model_registry() if models is None else tuple(copy.deepcopy(model) for model in models)
    for model in model_instances:
        model.fit(clean_instances)
        fitted.append(model)
    return tuple(fitted)


def _model_null(
    *,
    model: CapabilityModel,
    instances: Sequence[InstanceRecord],
    projection: tuple[int, ...],
    clean_cache: dict[str, np.ndarray],
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    event_length: int,
    protocol: CapabilityProtocol,
) -> ModelNull:
    scores: list[float] = []
    for instance in instances:
        clean = _read_array(instance, "clean", arrays, clean_cache)
        projected_clean = _project_array(clean, projection)
        scan_windows = _scan_windows(clean.shape[0], event_length, protocol, windows)
        values = model.score_windows(projected_clean, np.asarray(scan_windows, dtype=int))
        scores.extend(float(value) for value in values if math.isfinite(float(value)))
    arr = np.asarray(scores, dtype=np.float64)
    candidate_null = CandidateNull(
        family=model.family,
        witness_or_model=model.model_id,
        projection_policy="model_window",
        window_length_bin=int(event_length),
        scores=arr,
        candidate_count=int(arr.size),
    )
    calibrator = EmpiricalCalibrator()
    scan_statistics = np.asarray(
        [
            scan_statistic_from_candidate_p_values(
                np.asarray([calibrator.candidate_p_value(candidate_null, score)], dtype=np.float64)
            )
            for score in arr
        ],
        dtype=np.float64,
    )
    scan_null = ScanNull(
        scope="model_zoo",
        family=model.family,
        aggregation="min_candidate_p",
        window_length_bin=int(event_length),
        scan_statistics=scan_statistics,
        raw_candidate_count=int(arr.size),
        effective_candidate_count=1.0 if arr.size else 0.0,
    )
    return ModelNull(
        candidate_null=candidate_null,
        scan_null=scan_null,
        raw_candidate_count=int(arr.size),
        effective_candidate_count=finite_float(scan_null.effective_candidate_count, default=math.nan),
    )


def _context_score(
    model: CapabilityModel,
    series: np.ndarray,
    group: EventGroup,
    length: int,
) -> float:
    width = max(1, group.length)
    start = max(0, group.start - width)
    end = min(int(length), group.end + width)
    if end <= start:
        return math.nan
    return float(model.score_windows(series, np.asarray([[start, end]], dtype=int))[0])


def _model_candidate_p_value(null: ModelNull, raw_score: float) -> float:
    return EmpiricalCalibrator().candidate_p_value(null.candidate_null, float(raw_score))


def _model_scan_statistic(null: ModelNull, raw_score: float) -> float:
    candidate_p = _model_candidate_p_value(null, raw_score)
    return scan_statistic_from_candidate_p_values(np.asarray([candidate_p], dtype=np.float64))


def _model_scan_p_value(null: ModelNull, raw_score: float) -> float:
    scan_statistic = _model_scan_statistic(null, raw_score)
    return EmpiricalCalibrator().scan_p_value(null.scan_null, scan_statistic)


def _false_alert_rate(
    *,
    model: CapabilityModel,
    series: np.ndarray,
    group: EventGroup,
    event_length: int,
    null: ModelNull,
    threshold: float,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> tuple[int, float]:
    if not math.isfinite(threshold):
        return 0, math.nan
    scan_windows = np.asarray(_scan_windows(series.shape[0], event_length, protocol, windows), dtype=int)
    if scan_windows.size == 0:
        return 0, math.nan
    scores = model.score_windows(series, scan_windows)
    false_alerts = 0
    eligible = 0
    for (start, end), score in zip(scan_windows, scores):
        if _overlaps(int(start), int(end), group.start, group.end):
            continue
        eligible += 1
        scan_stat = _model_scan_statistic(null, float(score))
        if math.isfinite(scan_stat) and scan_stat >= threshold:
            false_alerts += 1
    return false_alerts, false_alerts / max(float(eligible), 1.0)


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
        "length": group.length,
    }


def _read_array(
    instance: InstanceRecord,
    kind: str,
    arrays: ArrayStore | None,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    key = f"{kind}:{instance.variant_id}/{instance.split}/{instance.instance_id}"
    if key in cache:
        return cache[key]
    if arrays is not None:
        values = arrays.get(instance, kind)
    elif kind == "clean":
        values = read_timeseries_csv(instance.clean_path)
    else:
        values = read_timeseries_csv(instance.anomalous_path)
    cache[key] = np.asarray(values, dtype=np.float64)
    return cache[key]


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


def _overlaps(start: int, end: int, target_start: int, target_end: int) -> bool:
    return int(start) < int(target_end) and int(end) > int(target_start)


def _read_cached_table(
    cache: CacheStore,
    profile_name: str,
    partition_id: str,
) -> pd.DataFrame:
    return pd.read_csv(cache.table_path(profile_name, partition_id), compression="gzip")


def _sort_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    return _sort_table(
        frame,
        (
            "event_id",
            "model_id",
            "alpha",
        ),
    )


def _sort_event_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return _sort_table(
        frame,
        (
            "event_id",
            "model_id",
        ),
    )


def _sort_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    return _sort_table(
        frame,
        (
            "fit_variant_id",
            "model_id",
            "model_projection",
        ),
    )


def _sort_table(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    sort_columns = [column for column in columns if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _models_fingerprint(models: Sequence[CapabilityModel] | None) -> str:
    model_instances = default_model_registry() if models is None else tuple(copy.deepcopy(model) for model in models)
    payload = [
        {
            "model_id": model.model_id,
            "family": model.family,
            "metadata": model.metadata(),
        }
        for model in model_instances
    ]
    return canonical_json_hash({"models": payload})
