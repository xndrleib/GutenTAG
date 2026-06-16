"""Oracle-window row construction for corrected detectability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import (
    EmpiricalCalibrator,
    calibration_status,
    min_empirical_p,
    scan_statistic_from_candidate_p_values,
)
from .corrected_detectability_candidates import (
    CandidateSpec,
    best_canonical_key,
    format_projection,
    spec_by_key,
)
from .dataset import EventGroup
from .numerics import finite_float
from .protocol import CapabilityProtocol


@dataclass(frozen=True)
class OracleDetectionSummary:
    """Selected oracle-window detection statistics for one event."""

    best_key: str
    best_spec: CandidateSpec | None
    best_raw_score: float
    best_candidate_p: float
    canonical_key: str | None
    canonical_spec: CandidateSpec | None
    canonical_raw_score: float
    canonical_p: float
    canonical_scan_stat: float
    scan_stat: float
    scan_p: float
    candidate_count: int


def candidate_p_values_for_scores(
    *,
    event_scores: Mapping[str, float],
    nulls: Any,
    calibrator: EmpiricalCalibrator,
) -> dict[str, float]:
    """Return candidate-level empirical p-values for known candidate nulls."""

    return {
        key: calibrator.candidate_p_value(nulls.candidate_nulls[key], score)
        for key, score in event_scores.items()
        if key in nulls.candidate_nulls
    }


def summarize_oracle_detection(
    *,
    group: EventGroup,
    candidates: Sequence[CandidateSpec],
    event_scores: Mapping[str, float],
    candidate_p_values: Mapping[str, float],
    nulls: Any,
    calibrator: EmpiricalCalibrator,
) -> OracleDetectionSummary:
    """Select best and canonical candidates for one oracle event."""

    best_key = min(candidate_p_values, key=lambda key: candidate_p_values[key])
    best_spec = spec_by_key(candidates, best_key)
    canonical_key = best_canonical_key(group, candidates, candidate_p_values)
    best_raw_score = float(event_scores.get(best_key, math.nan))
    best_candidate_p = float(candidate_p_values[best_key])
    canonical_p = (
        float(candidate_p_values[canonical_key])
        if canonical_key is not None
        else math.nan
    )
    canonical_spec = (
        spec_by_key(candidates, canonical_key) if canonical_key is not None else None
    )
    canonical_raw_score = (
        float(event_scores.get(canonical_key, math.nan))
        if canonical_key is not None
        else math.nan
    )
    canonical_scan_stat = (
        scan_statistic_from_candidate_p_values(
            np.asarray([canonical_p], dtype=np.float64)
        )
        if math.isfinite(canonical_p)
        else math.nan
    )
    scan_stat = scan_statistic_from_candidate_p_values(
        np.asarray(list(candidate_p_values.values()), dtype=np.float64)
    )
    scan_p = calibrator.scan_p_value(nulls.scan_null, scan_stat)
    return OracleDetectionSummary(
        best_key=str(best_key),
        best_spec=best_spec,
        best_raw_score=best_raw_score,
        best_candidate_p=best_candidate_p,
        canonical_key=canonical_key,
        canonical_spec=canonical_spec,
        canonical_raw_score=canonical_raw_score,
        canonical_p=canonical_p,
        canonical_scan_stat=canonical_scan_stat,
        scan_stat=scan_stat,
        scan_p=scan_p,
        candidate_count=len(candidate_p_values),
    )


def oracle_frontier_row(
    *,
    common: Mapping[str, object],
    summary: OracleDetectionSummary,
    nulls: Any,
    alpha: float,
    threshold: float,
    false_alert_count: int,
    blind_window_count: int,
    protocol: CapabilityProtocol,
) -> dict[str, object]:
    """Build one corrected-detectability frontier row for an alpha level."""

    detected, canonical_detected = _oracle_detection_flags(summary, threshold)
    return {
        **common,
        **_oracle_best_candidate_fields(summary),
        "alpha": float(alpha),
        **_oracle_score_fields(summary, threshold),
        "detected": detected,
        **_oracle_false_alert_fields(false_alert_count, blind_window_count),
        **_oracle_null_fields(summary, nulls, alpha, protocol),
        "best_detection_witness": summary.best_key,
        "best_canonical_witness": summary.canonical_key or "",
        **_oracle_canonical_fields(summary, canonical_detected),
        "candidate_count": summary.candidate_count,
    }


def _oracle_detection_flags(
    summary: OracleDetectionSummary,
    threshold: float,
) -> tuple[bool, bool]:
    detected = (
        bool(summary.scan_stat >= threshold) if math.isfinite(threshold) else False
    )
    canonical_detected = (
        bool(summary.canonical_scan_stat >= threshold)
        if math.isfinite(summary.canonical_scan_stat) and math.isfinite(threshold)
        else False
    )
    return detected, canonical_detected


def _oracle_best_candidate_fields(summary: OracleDetectionSummary) -> dict[str, object]:
    return {
        "scope": "oracle_window",
        "family": (
            summary.best_spec.family if summary.best_spec is not None else "unknown"
        ),
        "witness_or_model": (
            summary.best_spec.witness if summary.best_spec is not None else "none"
        ),
        "projection": (
            format_projection(summary.best_spec.projection)
            if summary.best_spec is not None
            else ""
        ),
    }


def _oracle_score_fields(
    summary: OracleDetectionSummary,
    threshold: float,
) -> dict[str, object]:
    return {
        "raw_score": finite_float(summary.best_raw_score, default=math.nan),
        "candidate_level_p_value": finite_float(
            summary.best_candidate_p,
            default=math.nan,
        ),
        "scan_statistic": finite_float(summary.scan_stat, default=math.nan),
        "scan_level_p_value": finite_float(summary.scan_p, default=math.nan),
        "scan_threshold": finite_float(threshold, default=math.nan),
    }


def _oracle_false_alert_fields(
    false_alert_count: int,
    blind_window_count: int,
) -> dict[str, object]:
    return {
        "latency": 0,
        "false_alert_count": int(false_alert_count),
        "false_alert_rate": finite_float(
            int(false_alert_count) / max(float(blind_window_count), 1.0),
            default=math.nan,
        ),
    }


def _oracle_null_fields(
    summary: OracleDetectionSummary,
    nulls: Any,
    alpha: float,
    protocol: CapabilityProtocol,
) -> dict[str, object]:
    min_scan_p = min_empirical_p(len(nulls.scan_null.scan_statistics))
    return {
        "candidate_null_count": int(
            nulls.candidate_nulls[summary.best_key].scores.size
        ),
        "scan_null_count": int(nulls.scan_null.scan_statistics.size),
        "raw_candidate_count": int(nulls.scan_null.raw_candidate_count),
        "effective_candidate_count": finite_float(
            nulls.scan_null.effective_candidate_count,
            default=math.nan,
        ),
        "calibration_resolution_min_p": finite_float(min_scan_p, default=math.nan),
        "calibration_status": calibration_status(
            len(nulls.scan_null.scan_statistics),
            float(alpha),
            protocol.calibration_min_clean_scan_count_for_alpha,
        ),
    }


def _oracle_canonical_fields(
    summary: OracleDetectionSummary,
    canonical_detected: bool,
) -> dict[str, object]:
    return {
        "best_canonical_family": (
            summary.canonical_spec.family if summary.canonical_spec is not None else ""
        ),
        "best_canonical_projection": (
            format_projection(summary.canonical_spec.projection)
            if summary.canonical_spec is not None
            else ""
        ),
        "best_canonical_raw_score": finite_float(
            summary.canonical_raw_score,
            default=math.nan,
        ),
        "best_canonical_candidate_p_value": finite_float(
            summary.canonical_p,
            default=math.nan,
        ),
        "best_canonical_scan_statistic": finite_float(
            summary.canonical_scan_stat,
            default=math.nan,
        ),
        "best_canonical_detected": canonical_detected,
    }


def calibration_resolution_row(frontier_row: Mapping[str, object]) -> dict[str, object]:
    """Project a frontier row into the calibration-resolution table shape."""

    return {
        "event_id": frontier_row["event_id"],
        "variant_id": frontier_row["variant_id"],
        "split": frontier_row["split"],
        "instance_id": frontier_row["instance_id"],
        "anomaly_type": frontier_row["anomaly_type"],
        "alpha": frontier_row["alpha"],
        "scope": frontier_row["scope"],
        "scan_null_count": frontier_row["scan_null_count"],
        "candidate_null_count": frontier_row["candidate_null_count"],
        "raw_candidate_count": frontier_row["raw_candidate_count"],
        "effective_candidate_count": frontier_row["effective_candidate_count"],
        "calibration_resolution_min_p": frontier_row["calibration_resolution_min_p"],
        "calibration_status": frontier_row["calibration_status"],
    }


__all__ = [
    "OracleDetectionSummary",
    "calibration_resolution_row",
    "candidate_p_values_for_scores",
    "oracle_frontier_row",
    "summarize_oracle_detection",
]
