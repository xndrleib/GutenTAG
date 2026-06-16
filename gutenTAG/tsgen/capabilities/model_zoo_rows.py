"""Row contracts for model-zoo capability tables."""

from __future__ import annotations

import math
from typing import Any

from .calibration import calibration_status, min_empirical_p
from .dataset import EventGroup, InstanceRecord, event_uid
from .models import CapabilityModel
from .numerics import finite_float
from .protocol import CapabilityProtocol


def model_event_score_row(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    model: CapabilityModel,
    projection_label: str,
    projection_size: int,
    raw_score: float,
    candidate_p_value: float,
    scan_statistic: float,
    scan_p_value: float,
    null: Any,
    intervention_score: float,
    context_score: float,
    metadata_hash: str,
) -> dict[str, object]:
    """Return one ``model_zoo_event_scores`` row."""

    return {
        **_common_model_event_fields(instance, group),
        "model_id": model.model_id,
        "model_family": model.family,
        "model_projection": projection_label,
        "model_projection_size": int(projection_size),
        "fit_split": "all_clean",
        "calibration_split": "all_clean",
        "raw_score": finite_float(raw_score, default=math.nan),
        "candidate_level_p_value": finite_float(candidate_p_value, default=math.nan),
        "scan_statistic": finite_float(scan_statistic, default=math.nan),
        "scan_level_p_value": finite_float(scan_p_value, default=math.nan),
        "candidate_null_count": int(null.candidate_null.scores.size),
        "scan_null_count": int(null.scan_null.scan_statistics.size),
        "raw_candidate_count": int(null.raw_candidate_count),
        "effective_candidate_count": finite_float(
            null.effective_candidate_count,
            default=math.nan,
        ),
        "intervention_channel_score": finite_float(
            intervention_score,
            default=math.nan,
        ),
        "context_channel_score": finite_float(context_score, default=math.nan),
        "model_metadata_hash": metadata_hash,
    }


def model_frontier_row(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    model: CapabilityModel,
    projection_label: str,
    projection_size: int,
    alpha: float,
    raw_score: float,
    candidate_p_value: float,
    scan_statistic: float,
    scan_p_value: float,
    scan_threshold: float,
    detected: bool,
    false_alert_count: int,
    false_alert_rate: float,
    null: Any,
    protocol: CapabilityProtocol,
    intervention_score: float,
    context_score: float,
    metadata_hash: str,
) -> dict[str, object]:
    """Return one ``model_zoo_frontier`` row."""

    scan_null_count = int(null.scan_null.scan_statistics.size)
    return {
        **_common_model_event_fields(instance, group),
        "model_id": model.model_id,
        "model_family": model.family,
        "model_projection": projection_label,
        "model_projection_size": int(projection_size),
        "fit_split": "all_clean",
        "calibration_split": "all_clean",
        "alpha": float(alpha),
        "raw_score": finite_float(raw_score, default=math.nan),
        "candidate_level_p_value": finite_float(candidate_p_value, default=math.nan),
        "scan_statistic": finite_float(scan_statistic, default=math.nan),
        "scan_level_p_value": finite_float(scan_p_value, default=math.nan),
        "scan_threshold": finite_float(scan_threshold, default=math.nan),
        "detected": bool(detected),
        "latency": 0,
        "false_alert_count": int(false_alert_count),
        "false_alert_rate": finite_float(false_alert_rate, default=math.nan),
        "candidate_null_count": int(null.candidate_null.scores.size),
        "scan_null_count": scan_null_count,
        "raw_candidate_count": int(null.raw_candidate_count),
        "effective_candidate_count": finite_float(
            null.effective_candidate_count,
            default=math.nan,
        ),
        "calibration_resolution_min_p": finite_float(
            min_empirical_p(scan_null_count),
            default=math.nan,
        ),
        "calibration_status": calibration_status(
            scan_null_count,
            float(alpha),
            protocol.calibration_min_clean_scan_count_for_alpha,
        ),
        "intervention_channel_score": finite_float(
            intervention_score,
            default=math.nan,
        ),
        "context_channel_score": finite_float(context_score, default=math.nan),
        "model_metadata_hash": metadata_hash,
    }


def _common_model_event_fields(
    instance: InstanceRecord,
    group: EventGroup,
) -> dict[str, object]:
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
