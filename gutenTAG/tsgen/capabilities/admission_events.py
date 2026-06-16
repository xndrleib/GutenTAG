"""Event-level admission evidence aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from .admission_common import (
    _alpha_rows,
    _bool_value,
    _index_by_event,
    _is_finite_number,
    _join_reasons,
    _manual_review_recommended,
    _min_alpha,
    _min_delta,
    _min_delta_rows,
)
from .protocol import CapabilityProtocol

if TYPE_CHECKING:
    from .admission import AdmissionPolicy


@dataclass(frozen=True)
class _EventAdmissionDecision:
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    gate_hits: tuple[str, ...]
    status: str
    impl_status: str
    canonical_observable: bool
    detected: bool
    detected_for_admission: bool
    detection_source: str
    calibration_status: str
    primary_cause: str
    repair_status: str
    model_detected: bool


def _event_admission(
    *,
    protocol: CapabilityProtocol,
    policy: "AdmissionPolicy",
    event_summary: pd.DataFrame,
    arity: pd.DataFrame,
    implementation_validity: pd.DataFrame,
    detector_attribution: pd.DataFrame,
    corrected_detectability: pd.DataFrame,
    repair_profile: pd.DataFrame,
    model_zoo_frontier: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    impl_by_event = _index_by_event(implementation_validity)
    repair_by_event = _index_by_event(repair_profile)
    arity_by_event = _min_delta_rows(arity, protocol)
    detect_by_event = _alpha_rows(corrected_detectability, protocol)
    attribution_by_event = _alpha_rows(detector_attribution, protocol)
    model_by_event = _model_detection_summary(model_zoo_frontier, protocol)
    rows: list[dict[str, object]] = []
    for _, summary_row in event_summary.iterrows():
        event_id = str(summary_row["event_id"])
        rows.append(
            _event_admission_row(
                summary_row=summary_row,
                protocol=protocol,
                policy_mode=policy.mode,
                impl=impl_by_event.get(event_id, {}),
                repair=repair_by_event.get(event_id, {}),
                arity_row=arity_by_event.get(event_id, {}),
                detect=detect_by_event.get(event_id, {}),
                attribution=attribution_by_event.get(event_id, {}),
                model=model_by_event.get(event_id, {}),
                metadata_row=metadata.get(event_id, {}),
            )
        )
    return pd.DataFrame(rows)


def _event_admission_row(
    *,
    summary_row: pd.Series,
    protocol: CapabilityProtocol,
    policy_mode: str,
    impl: Mapping[str, Any],
    repair: Mapping[str, Any],
    arity_row: Mapping[str, Any],
    detect: Mapping[str, Any],
    attribution: Mapping[str, Any],
    model: Mapping[str, Any],
    metadata_row: Mapping[str, Any],
) -> dict[str, object]:
    event_id = str(summary_row["event_id"])
    decision = _evaluate_event_evidence(
        impl=impl,
        repair=repair,
        arity_row=arity_row,
        detect=detect,
        attribution=attribution,
        model=model,
        metadata_row=metadata_row,
    )
    failures = list(decision.failures)
    warnings = list(decision.warnings)
    gate_hits = list(decision.gate_hits)
    return {
        "event_id": event_id,
        "variant_id": summary_row.get("variant_id"),
        "split": summary_row.get("split"),
        "instance_id": summary_row.get("instance_id"),
        "anomaly_type": summary_row.get("anomaly_type"),
        "constraint_tag": summary_row.get("constraint_tag"),
        "semantic_scope": summary_row.get("semantic_scope"),
        "admission_status": decision.status,
        "failure_reasons": _join_reasons(failures),
        "warning_reasons": _join_reasons(warnings),
        "provisional_gate_hits": _join_reasons(gate_hits),
        "manual_review_recommended": _manual_review_recommended(
            failures=failures,
            warnings=warnings,
            gate_hits=gate_hits,
            status=decision.status,
        ),
        "genotype_id": metadata_row.get("genotype_id"),
        "contract_id": metadata_row.get("contract_id"),
        "implementation_validity_status": decision.impl_status,
        "support_status": impl.get("support_status"),
        "boundary_status": impl.get("boundary_status"),
        "shortcut_status": impl.get("shortcut_status"),
        "detected_at_min_alpha": decision.detected,
        "detected_for_admission": decision.detected_for_admission,
        "admission_detection_source": decision.detection_source,
        "min_alpha": _min_alpha(protocol),
        "scan_level_p_value": detect.get("scan_level_p_value"),
        "calibration_status": decision.calibration_status,
        "primary_detection_cause": decision.primary_cause,
        "canonical_is_observable_at_min_delta": decision.canonical_observable,
        "min_delta": _min_delta(protocol),
        "canonical_observed_arity": arity_row.get("canonical_observed_arity"),
        "repair_status": decision.repair_status,
        "repair_gain": repair.get("repair_gain"),
        "model_zoo_detected_at_min_alpha": decision.model_detected,
        "model_zoo_detected_model_count": model.get("model_zoo_detected_model_count"),
        "policy_mode": policy_mode,
    }


def _evaluate_event_evidence(
    *,
    impl: Mapping[str, Any],
    repair: Mapping[str, Any],
    arity_row: Mapping[str, Any],
    detect: Mapping[str, Any],
    attribution: Mapping[str, Any],
    model: Mapping[str, Any],
    metadata_row: Mapping[str, Any],
) -> _EventAdmissionDecision:
    failures: list[str] = []
    warnings: list[str] = []
    gate_hits: list[str] = []
    _metadata_reasons(metadata_row, failures, warnings)
    impl_status = str(impl.get("implementation_validity_status", "missing"))
    _implementation_reasons(impl_status, failures, warnings, gate_hits)
    if not impl:
        failures.append("missing_implementation_validity_row")
    if not _has_finite_value(
        impl,
        ("realized_offset", "realized_log_var_ratio", "realized_fisher_shift"),
    ):
        warnings.append("missing_realized_effect_measurement")
    canonical_observable = _bool_value(
        arity_row.get("canonical_is_observable_at_delta"),
        default=False,
    )
    if not canonical_observable:
        warnings.append("canonical_not_observable_at_min_delta")
        gate_hits.append("min_canonical_observable_share")
    detected = _bool_value(detect.get("detected"), default=False)
    if not detect:
        warnings.append("missing_corrected_detectability_row")
    elif not detected:
        warnings.append("not_detected_at_min_alpha")
    model_detected = _bool_value(
        model.get("model_zoo_detected_at_min_alpha"),
        default=False,
    )
    detected_for_admission = detected or model_detected
    detection_source = _admission_detection_source(
        corrected_detected=detected,
        model_detected=model_detected,
    )
    if not detected and model_detected:
        warnings.append("model_zoo_only_detection")
    calibration_status = str(detect.get("calibration_status", "missing"))
    if calibration_status == "calibration_invalid_for_alpha":
        failures.append("calibration_invalid_for_alpha")
    elif calibration_status == "calibration_low_resolution":
        warnings.append("calibration_low_resolution")
    primary_cause = str(attribution.get("primary_detection_cause", "missing"))
    _attribution_reasons(primary_cause, warnings, gate_hits)
    repair_status = str(repair.get("repair_status", "missing"))
    if repair_status in {"repair_harmful", "repair_weak"}:
        warnings.append(repair_status)
        gate_hits.append("max_needs_repair_share")
    status = _event_status(
        failures=failures,
        warnings=warnings,
        impl_status=impl_status,
        detected=detected_for_admission,
        metadata_row=metadata_row,
    )
    return _EventAdmissionDecision(
        failures=tuple(failures),
        warnings=tuple(warnings),
        gate_hits=tuple(gate_hits),
        status=status,
        impl_status=impl_status,
        canonical_observable=canonical_observable,
        detected=detected,
        detected_for_admission=detected_for_admission,
        detection_source=detection_source,
        calibration_status=calibration_status,
        primary_cause=primary_cause,
        repair_status=repair_status,
        model_detected=model_detected,
    )


def _metadata_reasons(
    metadata_row: Mapping[str, Any],
    failures: list[str],
    warnings: list[str],
) -> None:
    if not metadata_row:
        warnings.append("missing_metadata_event_record")
        return
    if not metadata_row.get("genotype_id"):
        failures.append("missing_problem_genotype")
    if not metadata_row.get("contract_id"):
        failures.append("missing_contract_id")
    if metadata_row.get("legacy_provenance"):
        warnings.append("legacy_inferred_metadata")


def _implementation_reasons(
    impl_status: str,
    failures: list[str],
    warnings: list[str],
    gate_hits: list[str],
) -> None:
    if impl_status == "valid_candidate":
        return
    if impl_status == "insufficient_or_mismatched_effect":
        failures.append(impl_status)
        return
    if impl_status in {
        "support_leakage_suspected",
        "boundary_artifact_suspected",
        "detected_wrong_reason",
        "negative_control_failed",
    }:
        warnings.append(impl_status)
        gate_hits.append("max_needs_repair_share")
        if impl_status == "boundary_artifact_suspected":
            gate_hits.append("max_boundary_artifact_fail_share")
        return
    if impl_status == "needs_review":
        warnings.append(impl_status)
    elif impl_status != "missing":
        warnings.append(f"implementation_status:{impl_status}")


def _attribution_reasons(
    primary_cause: str,
    warnings: list[str],
    gate_hits: list[str],
) -> None:
    if primary_cause in {"valid_canonical_detection", "valid_model_detection"}:
        return
    if primary_cause == "valid_with_shortcut":
        warnings.append("shortcut_present_claim_downgraded")
        return
    if primary_cause == "boundary_artifact_suspected":
        warnings.append("boundary_artifact_primary_detection")
        gate_hits.append("max_boundary_artifact_fail_share")
    elif primary_cause == "detected_wrong_reason":
        warnings.append("detected_wrong_reason")
        gate_hits.append("max_needs_repair_share")
    elif primary_cause == "observable_not_detected":
        warnings.append("observable_not_detected")
    elif primary_cause == "missing":
        warnings.append("missing_detector_attribution")


def _event_status(
    *,
    failures: list[str],
    warnings: list[str],
    impl_status: str,
    detected: bool,
    metadata_row: Mapping[str, Any],
) -> str:
    missing_required_metadata = {
        "missing_problem_genotype",
        "missing_contract_id",
    }
    if any(reason in failures for reason in missing_required_metadata):
        return "reject"
    if failures:
        return "reject"
    if not metadata_row:
        return "legacy"
    if impl_status in {
        "support_leakage_suspected",
        "boundary_artifact_suspected",
        "detected_wrong_reason",
        "negative_control_failed",
        "needs_review",
    }:
        return "needs_repair"
    if not detected:
        return "debug"
    if warnings:
        return "release_with_warning"
    return "release"


def _admission_detection_source(
    *,
    corrected_detected: bool,
    model_detected: bool,
) -> str:
    if corrected_detected:
        return "corrected_detectability"
    if model_detected:
        return "model_zoo"
    return "none"


def _model_detection_summary(
    frame: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "alpha" not in frame.columns:
        return {}
    alpha = _min_alpha(protocol)
    active = frame[np.isclose(frame["alpha"].astype(float), alpha)]
    rows: dict[str, Mapping[str, Any]] = {}
    for event_id, group in active.groupby("event_id", dropna=False):
        rows[str(event_id)] = {
            "model_zoo_detected_at_min_alpha": bool(
                group["detected"].astype(bool).any()
            ),
            "model_zoo_detected_model_count": int(group["detected"].astype(bool).sum()),
        }
    return rows


def _has_finite_value(row: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key in row and _is_finite_number(row[key]):
            return True
    return False
