"""Aggregate implementation-validity profile."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...contracts import ContractRegistry
from ..array_store import ArrayStore
from ..cache import CacheStore
from ..dataset import DatasetIndex
from ..pandas_typing import (
    as_frame,
    as_series,
    frame_groupby,
    numeric_column,
    sorted_frame,
)
from ..protocol import CapabilityProtocol
from .attribution import compute_detector_attribution
from .boundary import compute_boundary_audit
from .negative_controls import compute_negative_controls
from .realized_effects import compute_realized_effects
from .shortcut import compute_shortcut_audit
from .support import compute_support_integrity


@dataclass(frozen=True)
class ImplementationValidityResult:
    """Implementation-validity audit tables."""

    support_integrity: pd.DataFrame
    boundary_audit: pd.DataFrame
    shortcut_audit: pd.DataFrame
    realized_effects: pd.DataFrame
    detector_attribution: pd.DataFrame
    negative_controls: pd.DataFrame
    implementation_validity: pd.DataFrame


def compute_implementation_validity(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    observability: pd.DataFrame,
    event_summary: pd.DataFrame,
    detectability_frontier: pd.DataFrame,
    *,
    arrays: ArrayStore | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    registry: ContractRegistry | None = None,
) -> ImplementationValidityResult:
    """Compute implementation-validity audit tables."""

    del event_summary
    active_registry = registry or ContractRegistry.from_resource_defaults()
    support = compute_support_integrity(dataset, arrays=arrays)
    boundary = compute_boundary_audit(dataset, arrays=arrays, registry=active_registry)
    shortcut = compute_shortcut_audit(dataset, observability)
    realized = compute_realized_effects(dataset, arrays=arrays)
    attribution = compute_detector_attribution(
        dataset,
        detectability_frontier,
        boundary,
        shortcut,
        cache=cache,
        n_jobs=n_jobs,
        registry=active_registry,
    )
    negative_controls = compute_negative_controls(
        dataset,
        protocol,
        detectability_frontier,
        boundary,
        arrays=arrays,
        cache=cache,
        n_jobs=n_jobs,
    )
    validity = _implementation_validity(
        support,
        boundary,
        shortcut,
        realized,
        negative_controls,
        attribution,
    )
    return ImplementationValidityResult(
        support_integrity=support,
        boundary_audit=boundary,
        shortcut_audit=shortcut,
        realized_effects=realized,
        detector_attribution=attribution,
        negative_controls=negative_controls,
        implementation_validity=validity,
    )


def _implementation_validity(
    support: pd.DataFrame,
    boundary: pd.DataFrame,
    shortcut: pd.DataFrame,
    realized: pd.DataFrame,
    negative_controls: pd.DataFrame,
    detector_attribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    boundary_by_event = (
        boundary.set_index("event_id") if not boundary.empty else pd.DataFrame()
    )
    shortcut_by_event = (
        shortcut.set_index("event_id") if not shortcut.empty else pd.DataFrame()
    )
    realized_by_event = (
        realized.set_index("event_id") if not realized.empty else pd.DataFrame()
    )
    attribution_by_event = _attribution_by_event(detector_attribution)
    control_failures = _negative_control_failures(negative_controls)
    for _, support_row in support.iterrows():
        event_id = str(support_row["event_id"])
        boundary_row = _lookup(boundary_by_event, event_id)
        shortcut_row = _lookup(shortcut_by_event, event_id)
        realized_row = _lookup(realized_by_event, event_id)
        attribution_row = attribution_by_event.get(event_id, {})
        support_status = str(support_row.get("support_status", "unknown"))
        boundary_status = str(boundary_row.get("boundary_status", "unknown"))
        shortcut_status = str(shortcut_row.get("shortcut_status", "unknown"))
        attribution_primary_cause = str(
            attribution_row.get("primary_detection_cause", "unknown")
        )
        attribution_best_canonical_detected = _as_bool(
            attribution_row.get("best_canonical_detected", False)
        )
        negative_control_failed_count = int(control_failures.get(event_id, 0))
        status = _overall_status(
            support_status,
            boundary_status,
            shortcut_status,
            negative_control_failed_count,
            attribution_primary_cause,
            attribution_best_canonical_detected,
        )
        rows.append(
            {
                "event_id": event_id,
                "variant_id": support_row.get("variant_id"),
                "split": support_row.get("split"),
                "instance_id": support_row.get("instance_id"),
                "anomaly_type": support_row.get("anomaly_type"),
                "constraint_tag": support_row.get("constraint_tag"),
                "semantic_scope": support_row.get("semantic_scope"),
                "support_status": support_status,
                "boundary_status": boundary_status,
                "shortcut_status": shortcut_status,
                "detector_primary_detection_cause": attribution_primary_cause,
                "detector_best_canonical_detected": attribution_best_canonical_detected,
                "negative_control_failed_count": negative_control_failed_count,
                "implementation_validity_status": status,
                "inside_mass_share": support_row.get("inside_mass_share"),
                "far_field_mass_share": support_row.get("far_field_mass_share"),
                "boundary_energy_share": boundary_row.get("boundary_energy_share"),
                "canonical_vs_univariate_ratio": shortcut_row.get(
                    "canonical_vs_univariate_ratio"
                ),
                "realized_offset": realized_row.get("realized_offset"),
                "realized_log_var_ratio": realized_row.get("realized_log_var_ratio"),
                "realized_fisher_shift": realized_row.get("realized_fisher_shift"),
            }
        )
    return pd.DataFrame(rows)


def _attribution_by_event(
    detector_attribution: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    if detector_attribution is None or detector_attribution.empty:
        return {}
    if "event_id" not in detector_attribution.columns:
        return {}
    active = detector_attribution
    if "alpha" in active.columns:
        alpha_values = numeric_column(active, "alpha")
        if bool(alpha_values.notna().any()):
            min_alpha = float(alpha_values.min())
            active = as_frame(active[alpha_values.sub(min_alpha).abs() <= 1e-12])
    sort_columns = [
        column
        for column in ("event_id", "rank_within_event", "scan_level_p_value")
        if column in active.columns
    ]
    if sort_columns:
        active = sorted_frame(active, sort_columns)
    return {
        str(event_id): dict(group.iloc[0])
        for event_id, group in frame_groupby(active, "event_id", dropna=False)
    }


def _lookup(frame: pd.DataFrame, event_id: str) -> dict[str, object]:
    if frame.empty or event_id not in frame.index:
        return {}
    row = frame.loc[event_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return dict(as_series(row))


def _overall_status(
    support_status: str,
    boundary_status: str,
    shortcut_status: str,
    negative_control_failed_count: int,
    detector_primary_detection_cause: str = "unknown",
    detector_best_canonical_detected: bool = False,
) -> str:
    if support_status in {"support_mismatch", "insufficient_effect"}:
        return "insufficient_or_mismatched_effect"
    if support_status == "leaky":
        return "support_leakage_suspected"
    if boundary_status == "boundary_primary_detection_cause":
        return "boundary_artifact_suspected"
    if (
        shortcut_status == "shortcut_dominated"
        and not _has_confirmed_canonical_detection(
            detector_primary_detection_cause,
            detector_best_canonical_detected,
        )
    ):
        return "detected_wrong_reason"
    if negative_control_failed_count > 0:
        return "negative_control_failed"
    if support_status == "boundary_uncertain":
        return "needs_review"
    return "valid_candidate"


def _has_confirmed_canonical_detection(
    detector_primary_detection_cause: str,
    detector_best_canonical_detected: bool,
) -> bool:
    if detector_best_canonical_detected:
        return True
    return detector_primary_detection_cause in {
        "valid_canonical_detection",
        "valid_with_shortcut",
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if bool(pd.isna(value)):
        return False
    return bool(value)


def _negative_control_failures(negative_controls: pd.DataFrame) -> dict[str, int]:
    if negative_controls.empty:
        return {}
    failed = negative_controls[negative_controls["control_status"] == "control_failed"]
    if failed.empty:
        return {}
    return {
        str(event_id): int(count)
        for event_id, count in failed.groupby("event_id").size().items()
    }
