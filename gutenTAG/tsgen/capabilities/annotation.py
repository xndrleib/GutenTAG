"""Annotation-channel profile and alignment diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ..labels import annotation_channel_manifest, build_annotation_channels
from .cache import CacheStore
from .dataset import DatasetIndex, InstanceRecord, raw_events_for_instance
from .pandas_typing import as_series
from .partitions import PartitionSpec, partition_sequence, run_partitions
from .protocol import CapabilityProtocol

LABEL_TABLE_KEYS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class AnnotationProfileResult:
    """Annotation-channel output tables and manifest."""

    label_tables: dict[str, pd.DataFrame]
    alignment: pd.DataFrame
    robustness: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class AnnotationPartitionResult:
    """Annotation tables for one deterministic instance partition."""

    label_tables: dict[str, pd.DataFrame]
    alignment: pd.DataFrame


def compute_annotation_profiles(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    *,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 8,
    emit_label_tables: bool = True,
) -> AnnotationProfileResult:
    """Compute v12 annotation channels and alignment diagnostics."""

    del protocol
    if cache is None and int(n_jobs) <= 1:
        partition_result = _annotation_for_instances(
            dataset.instances,
            emit_label_tables=emit_label_tables,
        )
        return _combine_annotation_results(
            (partition_result,),
            dataset=dataset,
            emit_label_tables=emit_label_tables,
        )
    partitions = partition_sequence(
        dataset.instances,
        partition_size=max(1, int(partition_size)),
        prefix="annotation",
    )

    def worker(partition: PartitionSpec[InstanceRecord]) -> AnnotationPartitionResult:
        if cache is not None:
            return _cached_annotation_partition(
                cache=cache,
                partition_id=partition.partition_id,
                instances=partition.items,
                emit_label_tables=emit_label_tables,
            )
        return _annotation_for_instances(
            partition.items,
            emit_label_tables=emit_label_tables,
        )

    results = run_partitions(partitions, worker, n_jobs=n_jobs)
    return _combine_annotation_results(
        tuple(result.value for result in results),
        dataset=dataset,
        emit_label_tables=emit_label_tables,
    )


def _annotation_for_instances(
    instances: tuple[InstanceRecord, ...] | list[InstanceRecord],
    *,
    emit_label_tables: bool = True,
) -> AnnotationPartitionResult:
    """Compute annotation labels and alignment rows for an instance partition."""

    label_frames: dict[str, list[pd.DataFrame]] = {key: [] for key in LABEL_TABLE_KEYS}
    alignment_rows: list[dict[str, object]] = []
    for instance in instances:
        events = raw_events_for_instance(instance)
        channels = build_annotation_channels(
            length=instance.length,
            channels=instance.channels,
            events=events,
        )
        reference_arrays = {name: channel.values for name, channel in channels.items()}
        for name in LABEL_TABLE_KEYS:
            channel = channels[name]
            if emit_label_tables:
                label_frames[name].append(
                    _flatten_label_table(instance, channel.values, channel.columns)
                )
            if not events:
                continue
            reference = reference_arrays[channel.reference_channel]
            alignment_rows.append(
                _alignment_row(
                    instance=instance,
                    channel_name=name,
                    values=channel.values,
                    reference=reference,
                    event_count=len(events),
                )
            )
    label_tables = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in label_frames.items()
    }
    return AnnotationPartitionResult(
        label_tables=label_tables,
        alignment=pd.DataFrame(alignment_rows),
    )


def _combine_annotation_results(
    results: tuple[AnnotationPartitionResult, ...],
    *,
    dataset: DatasetIndex,
    emit_label_tables: bool = True,
) -> AnnotationProfileResult:
    label_tables = {
        name: _concat_frames(result.label_tables[name] for result in results)
        for name in LABEL_TABLE_KEYS
    }
    alignment = _concat_frames(result.alignment for result in results)
    manifest = annotation_channel_manifest()
    manifest["label_export"] = "full" if emit_label_tables else "diagnostics"
    manifest["label_tables_emitted"] = bool(emit_label_tables)
    manifest["table_paths"] = (
        {name: f"labels/{name}.csv" for name in LABEL_TABLE_KEYS}
        if emit_label_tables
        else {}
    )
    manifest["omitted_table_names"] = (
        [] if emit_label_tables else list(LABEL_TABLE_KEYS)
    )
    manifest["diagnostic_tables"] = {
        "annotation_alignment": "annotation_alignment.csv",
        "annotation_robustness": "annotation_robustness.csv",
    }
    manifest["instance_count"] = len(dataset.instances)
    manifest["event_group_count"] = int(
        sum(len(instance.event_groups) for instance in dataset.instances)
    )
    return AnnotationProfileResult(
        label_tables=label_tables,
        alignment=alignment,
        robustness=_robustness_summary(alignment),
        manifest=manifest,
    )


def _cached_annotation_partition(
    *,
    cache: CacheStore,
    partition_id: str,
    instances: tuple[InstanceRecord, ...],
    emit_label_tables: bool = True,
) -> AnnotationPartitionResult:
    profile_names = (
        (*LABEL_TABLE_KEYS, "annotation_alignment")
        if emit_label_tables
        else ("annotation_alignment",)
    )
    fingerprint_extra = {
        "instance_ids": [
            f"{instance.variant_id}/{instance.split}/{instance.instance_id}"
            for instance in instances
        ],
    }
    fingerprints = {
        profile_name: cache.fingerprint(
            profile_name=profile_name,
            partition_id=partition_id,
            extra=fingerprint_extra,
        )
        for profile_name in profile_names
    }
    if cache.resume and all(
        cache.table_path(profile_name, partition_id).exists()
        and cache.is_valid(
            profile_name=profile_name,
            partition_id=partition_id,
            fingerprint=fingerprints[profile_name],
        )
        for profile_name in profile_names
    ):
        return AnnotationPartitionResult(
            label_tables={
                key: (
                    _read_cached_table(cache, key, partition_id)
                    if emit_label_tables
                    else pd.DataFrame()
                )
                for key in LABEL_TABLE_KEYS
            },
            alignment=_read_cached_table(cache, "annotation_alignment", partition_id),
        )
    result = _annotation_for_instances(instances, emit_label_tables=emit_label_tables)
    if emit_label_tables:
        for key in LABEL_TABLE_KEYS:
            cache.write_table_partition(
                profile_name=key,
                partition_id=partition_id,
                fingerprint=fingerprints[key],
                frame=result.label_tables[key],
            )
    cache.write_table_partition(
        profile_name="annotation_alignment",
        partition_id=partition_id,
        fingerprint=fingerprints["annotation_alignment"],
        frame=result.alignment,
    )
    return result


def _flatten_label_table(
    instance: InstanceRecord,
    values: np.ndarray,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=np.int8)
    frame = pd.DataFrame(matrix, columns=pd.Index(list(columns)))
    frame.insert(0, "time_index", np.arange(instance.length, dtype=int))
    frame.insert(0, "instance_id", instance.instance_id)
    frame.insert(0, "split", instance.split)
    frame.insert(0, "variant_id", instance.variant_id)
    return frame


def _alignment_row(
    *,
    instance: InstanceRecord,
    channel_name: str,
    values: np.ndarray,
    reference: np.ndarray,
    event_count: int,
) -> dict[str, object]:
    labels = _flatten_binary(values)
    oracle = _flatten_binary(_broadcast_reference(reference, values))
    positives = int(labels.sum())
    oracle_positives = int(oracle.sum())
    intersection = int(np.logical_and(labels, oracle).sum())
    union = int(np.logical_or(labels, oracle).sum())
    false_positive = int(np.logical_and(labels, ~oracle).sum())
    false_negative = int(np.logical_and(~labels, oracle).sum())
    precision = _safe_ratio(intersection, positives)
    recall = _safe_ratio(intersection, oracle_positives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    jaccard = _safe_ratio(intersection, union)
    onset_delta = _onset_delta(values, reference)
    return {
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "annotation_channel": channel_name,
        "reference_channel": _reference_name(channel_name),
        "event_count": int(event_count),
        "label_positive_count": positives,
        "oracle_positive_count": oracle_positives,
        "intersection_count": intersection,
        "union_count": union,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "positive_rate": _safe_ratio(positives, labels.size),
        "oracle_positive_rate": _safe_ratio(oracle_positives, oracle.size),
        "onset_delta": onset_delta,
        "alignment_status": _alignment_status(precision, recall, jaccard),
    }


def _broadcast_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.int8)
    target = np.asarray(values, dtype=np.int8)
    if ref.shape == target.shape:
        return ref
    if ref.ndim == 2 and target.ndim == 2 and ref.shape[1] == 1:
        return np.repeat(ref, target.shape[1], axis=1)
    if ref.ndim == 2 and target.ndim == 2 and target.shape[1] == 1:
        return (ref.max(axis=1, keepdims=True) > 0).astype(np.int8)
    return (ref.reshape(ref.shape[0], -1).max(axis=1, keepdims=True) > 0).astype(
        np.int8
    )


def _flatten_binary(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int8).reshape(-1) > 0


def _reference_name(channel_name: str) -> str:
    if channel_name == "labels_oracle_intervention":
        return "labels_oracle_intervention"
    if channel_name == "labels_oracle_context":
        return "labels_oracle_context"
    return "labels_oracle_any"


def _onset_delta(values: np.ndarray, reference: np.ndarray) -> float:
    labels_any = _to_temporal_any(values)
    reference_any = _to_temporal_any(reference)
    label_starts = _run_starts(labels_any)
    oracle_starts = _run_starts(reference_any)
    if not label_starts or not oracle_starts:
        return float("nan")
    deltas: list[int] = []
    for oracle_start in oracle_starts:
        nearest = min(label_starts, key=lambda start: abs(start - oracle_start))
        deltas.append(int(nearest) - int(oracle_start))
    return float(np.mean(deltas)) if deltas else float("nan")


def _to_temporal_any(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.int8)
    if matrix.ndim == 1:
        return matrix > 0
    return matrix.max(axis=1) > 0


def _run_starts(values: np.ndarray) -> list[int]:
    mask = np.asarray(values, dtype=bool)
    if mask.size == 0:
        return []
    previous = np.concatenate([np.asarray([False]), mask[:-1]])
    return [int(index) for index in np.flatnonzero(mask & ~previous)]


def _robustness_summary(alignment: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = alignment.groupby(
        ["variant_id", "annotation_channel", "reference_channel"], dropna=False
    )
    for key, frame in grouped:
        variant_id, annotation_channel, reference_channel = cast(
            tuple[object, object, object], key
        )
        precision = as_series(frame["precision"]).astype(float)
        recall = as_series(frame["recall"]).astype(float)
        jaccard = as_series(frame["jaccard"]).astype(float)
        onset = as_series(frame["onset_delta"]).astype(float)
        rows.append(
            {
                "variant_id": variant_id,
                "annotation_channel": annotation_channel,
                "reference_channel": reference_channel,
                "instance_count": int(len(frame)),
                "median_precision": _median_finite(precision),
                "median_recall": _median_finite(recall),
                "median_jaccard": _median_finite(jaccard),
                "median_onset_delta": _median_finite(onset),
                "precision_iqr": _iqr(precision),
                "recall_iqr": _iqr(recall),
                "jaccard_iqr": _iqr(jaccard),
                "alignment_pass_rate": float(
                    (as_series(frame["alignment_status"]) == "aligned").mean()
                ),
                "robustness_status": _robustness_status(precision, recall, jaccard),
            }
        )
    return pd.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if abs(denom) <= 1e-12:
        return float("nan")
    return float(numerator) / denom


def _alignment_status(precision: float, recall: float, jaccard: float) -> str:
    if not all(math.isfinite(value) for value in (precision, recall, jaccard)):
        return "not_estimable"
    if precision >= 0.95 and recall >= 0.95 and jaccard >= 0.90:
        return "aligned"
    if recall < 0.50:
        return "low_recall_annotation"
    if precision < 0.50:
        return "noisy_annotation"
    return "partially_aligned"


def _robustness_status(
    precision: pd.Series, recall: pd.Series, jaccard: pd.Series
) -> str:
    med_precision = _median_finite(precision)
    med_recall = _median_finite(recall)
    med_jaccard = _median_finite(jaccard)
    if not all(
        math.isfinite(value) for value in (med_precision, med_recall, med_jaccard)
    ):
        return "not_estimable"
    if med_precision >= 0.95 and med_recall >= 0.95 and med_jaccard >= 0.90:
        return "oracle_equivalent"
    if med_recall >= 0.75 and med_precision >= 0.75:
        return "robust_annotation"
    if med_recall < 0.50:
        return "recall_limited_annotation"
    if med_precision < 0.50:
        return "precision_limited_annotation"
    return "weak_annotation"


def _median_finite(values: pd.Series) -> float:
    finite = _finite_array(values)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _iqr(values: pd.Series) -> float:
    finite = _finite_array(values)
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))


def _finite_array(values: pd.Series) -> np.ndarray:
    array = as_series(pd.to_numeric(values, errors="coerce")).to_numpy(
        dtype=np.dtype(np.float64)
    )
    return array[np.isfinite(array)]


def _concat_frames(frames: Any) -> pd.DataFrame:
    materialized = [frame for frame in frames if not frame.empty]
    return (
        pd.concat(materialized, ignore_index=True) if materialized else pd.DataFrame()
    )


def _read_cached_table(
    cache: CacheStore,
    profile_name: str,
    partition_id: str,
) -> pd.DataFrame:
    return pd.read_csv(cache.table_path(profile_name, partition_id), compression="gzip")
