"""Detector-attribution row rules and field construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from ...contracts import ContractRegistry
from ..dataset import EventGroup, InstanceRecord
from ..numerics import finite_float

WITNESS_FAMILY: Mapping[str, str] = {
    "mean_z": "location.mean",
    "variance_log_ratio": "scale.variance",
    "local_energy_z": "local.energy",
    "correlation_shift": "dependence.correlation",
    "covariance_shift": "dependence.covariance",
}

CANONICAL_DETECTION_WITNESSES: Mapping[str, tuple[str, ...]] = {
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
}


@dataclass(frozen=True)
class AttributionContext:
    """Resolved detector-attribution evidence for one frontier row."""

    event_id: str
    instance: InstanceRecord
    group: EventGroup
    witness: str
    projection: str
    family: str
    forbidden: bool
    canonical: bool
    boundary_primary: bool
    shortcut_status: str
    detected: bool
    scan_p: float
    candidate_p: float
    canonical_detected: bool
    scan_null_count: int
    min_empirical_p: float
    evidence: float
    scope: str


def build_attribution_row_fields(
    context: AttributionContext,
    frontier_row: Mapping[str, object],
) -> dict[str, object]:
    """Build the full detector-attribution row payload."""

    row = _identity_fields(context)
    row.update(_detector_fields(context, frontier_row))
    row.update(_cause_fields(context, frontier_row))
    row.update(_calibration_fields(context, frontier_row))
    return row


def forbidden_shortcut(
    registry: ContractRegistry,
    anomaly_type: str,
    family: str,
) -> bool:
    """Return whether a witness family is a forbidden shortcut for an anomaly."""

    contract = registry.get_by_anomaly(anomaly_type)
    if contract is None:
        return False
    return family in set(contract.forbidden_shortcuts)


def min_empirical_p(scan_null_count: int) -> float:
    """Return the empirical p-value resolution implied by a scan null."""

    return 1.0 / (scan_null_count + 1.0) if scan_null_count >= 0 else math.nan


def frontier_witness_and_projection(row: Mapping[str, object]) -> tuple[str, str]:
    """Resolve witness and projection fields from legacy or corrected frontiers."""

    witness = str(row.get("witness_or_model", "") or "")
    projection = str(row.get("projection", "") or "")
    if witness:
        return witness, projection
    return _parse_witness(str(row.get("best_detection_witness", "")))


def field(
    row: Mapping[str, object],
    *names: str,
    default: object = math.nan,
) -> object:
    """Return the first present non-null value among candidate row fields."""

    for name in names:
        if name in row and _notna(row.get(name)):
            return row.get(name)
    return default


def projection_size(projection: str) -> int:
    """Return the number of channels encoded in a projection label."""

    if not projection:
        return 0
    return len([part for part in projection.split("|") if part != ""])


def is_canonical(group: EventGroup, witness: str, active_projection_size: int) -> bool:
    """Return whether a witness/projection is canonical for an event group."""

    canonical = CANONICAL_DETECTION_WITNESSES.get(group.anomaly_type, ())
    if witness not in canonical:
        return False
    if "dependence" in group.constraint_tag or "relation" in group.semantic_scope:
        return active_projection_size >= 2
    return True


def attribution_scope(
    *,
    canonical: bool,
    forbidden: bool,
    boundary_primary: bool,
    shortcut_status: str,
) -> str:
    """Classify a detector attribution row into an evidence scope."""

    if boundary_primary:
        return "boundary"
    if forbidden or shortcut_status == "shortcut_dominated":
        return "shortcut"
    if canonical:
        return "canonical"
    return "generic"


def primary_cause(
    *,
    detected: bool,
    canonical: bool,
    canonical_detected: bool,
    forbidden: bool,
    boundary_primary: bool,
    shortcut_status: str,
) -> str:
    """Classify the primary detector-cause label for an attribution row."""

    if not detected:
        return "observable_not_detected"
    if boundary_primary:
        return "boundary_artifact_suspected"
    if forbidden or shortcut_status == "shortcut_dominated":
        if canonical_detected:
            return "valid_with_shortcut"
        return "detected_wrong_reason"
    if canonical:
        return "valid_canonical_detection"
    if canonical_detected:
        if shortcut_status == "valid_with_shortcut":
            return "valid_with_shortcut"
        return "valid_canonical_detection"
    if shortcut_status == "valid_with_shortcut":
        return "valid_with_shortcut"
    return "valid_model_detection"


def as_bool(value: object) -> bool:
    """Coerce table values to bool using capability-table conventions."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if bool(pd.isna(value)):
        return False
    return bool(value)


def normalized_evidence(scan_p: float, minimum_empirical_p: float) -> float:
    """Convert scan p-value into bounded negative-log evidence."""

    if not math.isfinite(scan_p) or not math.isfinite(minimum_empirical_p):
        return math.nan
    return float(-math.log10(max(scan_p, minimum_empirical_p, 1e-300)))


def as_float(value: object) -> float:
    """Coerce a table value to a finite float or NaN."""

    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def as_int(value: object, default: int = 0) -> int:
    """Coerce a table value to int with a default for missing values."""

    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _notna(value: object) -> bool:
    return bool(pd.notna(value))


def _identity_fields(context: AttributionContext) -> dict[str, object]:
    return {
        "event_id": context.event_id,
        "variant_id": context.instance.variant_id,
        "split": context.instance.split,
        "instance_id": context.instance.instance_id,
        "anomaly_type": context.group.anomaly_type,
        "constraint_tag": context.group.constraint_tag,
        "semantic_scope": context.group.semantic_scope,
    }


def _detector_fields(
    context: AttributionContext,
    frontier_row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "alpha": finite_float(frontier_row.get("alpha", math.nan), default=math.nan),
        "scope": context.scope,
        "family": context.family,
        "witness_or_model": context.witness,
        "projection": context.projection,
        "raw_score": finite_float(
            field(frontier_row, "raw_score", "event_score"),
            default=math.nan,
        ),
        "candidate_level_p_value": finite_float(
            context.candidate_p,
            default=math.nan,
        ),
        "scan_level_p_value": finite_float(context.scan_p, default=math.nan),
        "normalized_evidence": finite_float(context.evidence, default=math.nan),
        "rank_within_event": 1,
        "is_canonical": context.canonical,
        "is_forbidden_shortcut": context.forbidden,
        "is_boundary": context.boundary_primary,
    }


def _cause_fields(
    context: AttributionContext,
    frontier_row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "primary_detection_cause": primary_cause(
            detected=context.detected,
            canonical=context.canonical,
            canonical_detected=context.canonical_detected,
            forbidden=context.forbidden,
            boundary_primary=context.boundary_primary,
            shortcut_status=context.shortcut_status,
        ),
        "detected": context.detected,
        "best_canonical_witness": str(
            field(frontier_row, "best_canonical_witness", default="") or ""
        ),
        "best_canonical_candidate_p_value": finite_float(
            field(frontier_row, "best_canonical_candidate_p_value"),
            default=math.nan,
        ),
        "best_canonical_detected": context.canonical_detected,
    }


def _calibration_fields(
    context: AttributionContext,
    frontier_row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "scan_threshold": finite_float(
            field(frontier_row, "scan_threshold"),
            default=math.nan,
        ),
        "candidate_null_count": as_int(
            field(
                frontier_row,
                "candidate_null_count",
                "raw_candidate_count",
                default=0,
            )
            or 0
        ),
        "scan_null_count": context.scan_null_count,
        "raw_candidate_count": as_int(
            field(frontier_row, "raw_candidate_count", default=0) or 0
        ),
        "effective_candidate_count": finite_float(
            field(
                frontier_row,
                "effective_candidate_count",
                "effective_candidate_count_proxy",
            ),
            default=math.nan,
        ),
        "calibration_resolution_min_p": finite_float(
            field(
                frontier_row,
                "calibration_resolution_min_p",
                default=context.min_empirical_p,
            ),
            default=math.nan,
        ),
    }


def _parse_witness(value: str) -> tuple[str, str]:
    if "@" not in value:
        return value or "none", ""
    witness, projection = value.split("@", maxsplit=1)
    return witness, projection


__all__ = [
    "AttributionContext",
    "WITNESS_FAMILY",
    "as_bool",
    "as_float",
    "attribution_scope",
    "build_attribution_row_fields",
    "field",
    "forbidden_shortcut",
    "frontier_witness_and_projection",
    "is_canonical",
    "min_empirical_p",
    "normalized_evidence",
    "primary_cause",
    "projection_size",
]
