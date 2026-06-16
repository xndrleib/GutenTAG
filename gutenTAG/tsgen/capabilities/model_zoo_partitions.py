"""Partition execution for structural model-zoo capability."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..manifest import canonical_json_hash
from .array_store import ArrayStore
from .calibration import EmpiricalCalibrator
from .dataset import EventGroup, InstanceRecord
from .models import CapabilityModel, default_model_registry
from .model_zoo_projection import (
    model_projection,
    project_array,
    projection_label_for,
    read_cached_array,
    variant_model_projections,
)
from .model_zoo_rows import model_event_score_row, model_frontier_row
from .model_zoo_scoring import (
    ModelNull,
    context_score,
    false_alert_rate,
    model_candidate_p_value,
    model_null,
    model_scan_p_value,
    model_scan_statistic,
)
from .model_zoo_tables import sort_event_scores, sort_frontier, sort_manifest
from .protocol import CapabilityProtocol, window_length_bin
from .windows import WindowLibrary


@dataclass(frozen=True)
class ModelEventScores:
    raw_score: float
    candidate_p_value: float
    scan_statistic: float
    scan_p_value: float
    intervention_score: float
    context_score: float
    metadata_hash: str


@dataclass(frozen=True)
class ModelZooPartitionResult:
    """Model-zoo tables for one deterministic variant partition."""

    frontier: pd.DataFrame
    event_scores: pd.DataFrame
    manifest_rows: pd.DataFrame


def model_zoo_for_variants(
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
    fitted_by_projection: dict[
        tuple[str, tuple[int, ...]], tuple[CapabilityModel, ...]
    ] = {}
    null_cache: dict[tuple[object, ...], ModelNull] = {}
    frontier_rows: list[dict[str, object]] = []
    event_score_rows: list[dict[str, object]] = []
    manifest_rows: dict[str, dict[str, object]] = {}
    for variant_id in variant_ids:
        instances = variant_instances.get(variant_id, [])
        clean_instances = [
            read_cached_array(instance, "clean", arrays, clean_cache)
            for instance in instances
        ]
        for projection in variant_model_projections(instances):
            projected_clean = [
                project_array(clean, projection) for clean in clean_instances
            ]
            fitted = _fit_models(active_models, projected_clean)
            fitted_by_projection[(variant_id, projection)] = fitted
            projection_label = projection_label_for(projection)
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
        frontier=sort_frontier(pd.DataFrame(frontier_rows)),
        event_scores=sort_event_scores(
            pd.DataFrame(event_score_rows) if event_score_rows else pd.DataFrame()
        ),
        manifest_rows=sort_manifest(
            pd.DataFrame(manifest_rows.values()) if manifest_rows else pd.DataFrame()
        ),
    )


def _append_instance_model_rows(
    *,
    instance: InstanceRecord,
    variant_instances: dict[str, list[InstanceRecord]],
    fitted_by_projection: dict[
        tuple[str, tuple[int, ...]], tuple[CapabilityModel, ...]
    ],
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
    anom = read_cached_array(instance, "anomalous", arrays, anom_cache)
    for group in instance.event_groups:
        _append_group_model_rows(
            instance=instance,
            group=group,
            anom=anom,
            variant_instances=variant_instances,
            fitted_by_projection=fitted_by_projection,
            null_cache=null_cache,
            clean_cache=clean_cache,
            manifest_rows=manifest_rows,
            frontier_rows=frontier_rows,
            event_score_rows=event_score_rows,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
        )


def _append_group_model_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    anom: np.ndarray,
    variant_instances: dict[str, list[InstanceRecord]],
    fitted_by_projection: dict[
        tuple[str, tuple[int, ...]], tuple[CapabilityModel, ...]
    ],
    null_cache: dict[tuple[object, ...], ModelNull],
    clean_cache: dict[str, np.ndarray],
    manifest_rows: dict[str, dict[str, object]],
    frontier_rows: list[dict[str, object]],
    event_score_rows: list[dict[str, object]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
) -> None:
    projection = model_projection(instance, group)
    projection_label = projection_label_for(projection)
    projected_anom = project_array(anom, projection)
    fitted = fitted_by_projection.get((instance.variant_id, projection), ())
    scan_length = window_length_bin(max(1, group.length), protocol)
    for model in fitted:
        null = _model_null_for_group(
            instance=instance,
            model=model,
            projection=projection,
            scan_length=scan_length,
            variant_instances=variant_instances,
            null_cache=null_cache,
            clean_cache=clean_cache,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
        )
        scores = _model_event_scores(
            instance=instance,
            group=group,
            model=model,
            projection_label=projection_label,
            projected_anom=projected_anom,
            null=null,
            manifest_rows=manifest_rows,
        )
        event_score_rows.append(
            _model_event_score_row(
                instance=instance,
                group=group,
                model=model,
                projection=projection,
                projection_label=projection_label,
                null=null,
                scores=scores,
            )
        )
        frontier_rows.extend(
            _model_frontier_rows(
                instance=instance,
                group=group,
                model=model,
                projection=projection,
                projection_label=projection_label,
                projected_anom=projected_anom,
                scan_length=scan_length,
                null=null,
                scores=scores,
                protocol=protocol,
                windows=windows,
            )
        )


def _model_null_for_group(
    *,
    instance: InstanceRecord,
    model: CapabilityModel,
    projection: tuple[int, ...],
    scan_length: int,
    variant_instances: dict[str, list[InstanceRecord]],
    null_cache: dict[tuple[object, ...], ModelNull],
    clean_cache: dict[str, np.ndarray],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
) -> ModelNull:
    null_key = _model_null_key(instance, model, projection, scan_length, protocol)
    if null_key not in null_cache:
        null_cache[null_key] = model_null(
            model=model,
            clean_instances=_projected_clean_instances(
                instance=instance,
                projection=projection,
                variant_instances=variant_instances,
                clean_cache=clean_cache,
                arrays=arrays,
            ),
            event_length=scan_length,
            protocol=protocol,
            windows=windows,
        )
    return null_cache[null_key]


def _model_null_key(
    instance: InstanceRecord,
    model: CapabilityModel,
    projection: tuple[int, ...],
    scan_length: int,
    protocol: CapabilityProtocol,
) -> tuple[object, ...]:
    return (
        instance.variant_id,
        model.model_id,
        projection,
        scan_length,
        protocol.max_scan_windows_per_length,
        protocol.window_length_policy_mode,
        tuple(protocol.window_length_bins),
        protocol.clean_window_stride_fraction,
    )


def _projected_clean_instances(
    *,
    instance: InstanceRecord,
    projection: tuple[int, ...],
    variant_instances: dict[str, list[InstanceRecord]],
    clean_cache: dict[str, np.ndarray],
    arrays: ArrayStore | None,
) -> list[np.ndarray]:
    return [
        project_array(
            read_cached_array(clean_instance, "clean", arrays, clean_cache),
            projection,
        )
        for clean_instance in variant_instances.get(instance.variant_id, [instance])
    ]


def _model_event_scores(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    model: CapabilityModel,
    projection_label: str,
    projected_anom: np.ndarray,
    null: ModelNull,
    manifest_rows: dict[str, dict[str, object]],
) -> ModelEventScores:
    event_window = np.asarray([[group.start, group.end]], dtype=int)
    raw_score = float(model.score_windows(projected_anom, event_window)[0])
    return ModelEventScores(
        raw_score=raw_score,
        candidate_p_value=model_candidate_p_value(null, raw_score),
        scan_statistic=model_scan_statistic(null, raw_score),
        scan_p_value=model_scan_p_value(null, raw_score),
        intervention_score=raw_score,
        context_score=context_score(model, projected_anom, group, instance.length),
        metadata_hash=_model_metadata_hash(
            instance=instance,
            model=model,
            projection_label=projection_label,
            manifest_rows=manifest_rows,
        ),
    )


def _model_metadata_hash(
    *,
    instance: InstanceRecord,
    model: CapabilityModel,
    projection_label: str,
    manifest_rows: dict[str, dict[str, object]],
) -> str:
    manifest_key = f"{instance.variant_id}:{model.model_id}:{projection_label}"
    metadata = manifest_rows.get(manifest_key, model.metadata())
    return str(metadata.get("metadata_hash", canonical_json_hash(model.metadata())))


def _model_event_score_row(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    model: CapabilityModel,
    projection: tuple[int, ...],
    projection_label: str,
    null: ModelNull,
    scores: ModelEventScores,
) -> dict[str, object]:
    return model_event_score_row(
        instance=instance,
        group=group,
        model=model,
        projection_label=projection_label,
        projection_size=len(projection),
        raw_score=scores.raw_score,
        candidate_p_value=scores.candidate_p_value,
        scan_statistic=scores.scan_statistic,
        scan_p_value=scores.scan_p_value,
        null=null,
        intervention_score=scores.intervention_score,
        context_score=scores.context_score,
        metadata_hash=scores.metadata_hash,
    )


def _model_frontier_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    model: CapabilityModel,
    projection: tuple[int, ...],
    projection_label: str,
    projected_anom: np.ndarray,
    scan_length: int,
    null: ModelNull,
    scores: ModelEventScores,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alpha in protocol.alpha_grid:
        rows.append(
            _model_frontier_row_for_alpha(
                instance=instance,
                group=group,
                model=model,
                projection=projection,
                projection_label=projection_label,
                projected_anom=projected_anom,
                scan_length=scan_length,
                alpha=float(alpha),
                null=null,
                scores=scores,
                protocol=protocol,
                windows=windows,
            )
        )
    return rows


def _model_frontier_row_for_alpha(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    model: CapabilityModel,
    projection: tuple[int, ...],
    projection_label: str,
    projected_anom: np.ndarray,
    scan_length: int,
    alpha: float,
    null: ModelNull,
    scores: ModelEventScores,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> dict[str, object]:
    threshold = EmpiricalCalibrator().scan_threshold(null.scan_null, alpha)
    detected = (
        bool(scores.scan_statistic >= threshold) if math.isfinite(threshold) else False
    )
    false_alert_count, rate = false_alert_rate(
        model=model,
        series=projected_anom,
        group=group,
        event_length=scan_length,
        null=null,
        threshold=threshold,
        protocol=protocol,
        windows=windows,
    )
    return model_frontier_row(
        instance=instance,
        group=group,
        model=model,
        projection_label=projection_label,
        projection_size=len(projection),
        alpha=alpha,
        raw_score=scores.raw_score,
        candidate_p_value=scores.candidate_p_value,
        scan_statistic=scores.scan_statistic,
        scan_p_value=scores.scan_p_value,
        scan_threshold=threshold,
        detected=detected,
        false_alert_count=false_alert_count,
        false_alert_rate=rate,
        null=null,
        protocol=protocol,
        intervention_score=scores.intervention_score,
        context_score=scores.context_score,
        metadata_hash=scores.metadata_hash,
    )


def _fit_models(
    models: Sequence[CapabilityModel] | None,
    clean_instances: Sequence[np.ndarray],
) -> tuple[CapabilityModel, ...]:
    fitted: list[CapabilityModel] = []
    model_instances = (
        default_model_registry()
        if models is None
        else tuple(copy.deepcopy(model) for model in models)
    )
    for model in model_instances:
        model.fit(clean_instances)
        fitted.append(model)
    return tuple(fitted)


__all__ = [
    "ModelEventScores",
    "ModelZooPartitionResult",
    "model_projection",
    "model_zoo_for_variants",
    "project_array",
    "projection_label_for",
    "variant_model_projections",
]
