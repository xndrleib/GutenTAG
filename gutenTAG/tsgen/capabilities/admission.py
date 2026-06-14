"""Provisional release admission policy evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .dataset import DatasetIndex
from .protocol import CapabilityProtocol


ADMISSION_POLICY_VERSION = "synthgen.admission.v12.1"
_NUMERIC_GATE_NAMES = (
    "max_boundary_artifact_fail_share",
    "max_debug_event_share",
    "max_needs_repair_share",
    "min_canonical_observable_share",
    "min_oracle_annotation_pass_rate",
)


def _default_status_rules() -> dict[str, Any]:
    return {
        "all": {
            "require_problem_genotype": True,
            "require_contract_id": True,
            "require_realized_effect_row": True,
            "require_support_integrity_row": True,
        },
        "non_boundary_anomalies": {
            "boundary_primary_detection_allowed": False,
        },
        "relation_anomalies": {
            "require_canonical_relation_witness": True,
            "allow_shortcut_but_downgrade_claim": True,
        },
    }


def _default_numeric_gate_statuses() -> dict[str, str]:
    return {name: "provisional" for name in _NUMERIC_GATE_NAMES}


@dataclass(frozen=True)
class AdmissionPolicy:
    """Status-driven admission policy with provisional numeric gates."""

    version: str = ADMISSION_POLICY_VERSION
    mode: str = "provisional"
    numeric_gate_enforcement: str = "provisional"
    promote_after_full_runs: int = 3
    max_boundary_artifact_fail_share: float = 0.0
    max_debug_event_share: float = 0.05
    max_needs_repair_share: float = 0.05
    min_canonical_observable_share: float = 0.90
    min_oracle_annotation_pass_rate: float = 0.95
    status_rules: Mapping[str, Any] = field(default_factory=_default_status_rules)
    numeric_gate_statuses: Mapping[str, str] = field(default_factory=_default_numeric_gate_statuses)
    extra_numeric_gates: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy payload."""

        numeric_gates: dict[str, Any] = {
            "enforcement": self.numeric_gate_enforcement,
            "promote_after_full_runs": self.promote_after_full_runs,
        }
        for gate_name in _NUMERIC_GATE_NAMES:
            numeric_gates[gate_name] = {
                "value": getattr(self, gate_name),
                "status": str(self.numeric_gate_statuses.get(gate_name, "provisional")),
            }
        numeric_gates.update(dict(self.extra_numeric_gates))
        return {
            "admission_policy_version": self.version,
            "mode": self.mode,
            "status_rules": dict(self.status_rules),
            "numeric_gates": numeric_gates,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "AdmissionPolicy":
        """Construct an admission policy from a YAML-compatible mapping."""

        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Admission policy payload must contain a mapping")
        if "admission_policy" in payload:
            nested = payload["admission_policy"] or {}
            if not isinstance(nested, Mapping):
                raise ValueError("admission_policy must contain a mapping")
            payload = nested
        allowed = {
            "admission_policy_version",
            "version",
            "mode",
            "status_rules",
            "numeric_gates",
        }
        unknown = sorted(set(payload.keys()) - allowed)
        if unknown:
            raise ValueError(f"Unknown admission policy keys: {unknown}")
        numeric_gates = _mapping_or_empty(payload.get("numeric_gates"), "numeric_gates")
        gate_values: dict[str, float] = {}
        gate_statuses = _default_numeric_gate_statuses()
        extra_numeric_gates: dict[str, Any] = {}
        for gate_name in _NUMERIC_GATE_NAMES:
            if gate_name not in numeric_gates:
                continue
            gate_values[gate_name], gate_statuses[gate_name] = _parse_numeric_gate(
                gate_name,
                numeric_gates[gate_name],
            )
        for key, value in numeric_gates.items():
            if key in {*_NUMERIC_GATE_NAMES, "enforcement", "promote_after_full_runs"}:
                continue
            extra_numeric_gates[str(key)] = value
        status_rules = payload.get("status_rules")
        if status_rules is None:
            active_status_rules = _default_status_rules()
        elif isinstance(status_rules, Mapping):
            active_status_rules = dict(status_rules)
        else:
            raise ValueError("status_rules must contain a mapping")
        return cls(
            version=str(
                payload.get(
                    "admission_policy_version",
                    payload.get("version", ADMISSION_POLICY_VERSION),
                )
            ),
            mode=str(payload.get("mode", "provisional")),
            numeric_gate_enforcement=str(numeric_gates.get("enforcement", "provisional")),
            promote_after_full_runs=int(numeric_gates.get("promote_after_full_runs", 3)),
            status_rules=active_status_rules,
            numeric_gate_statuses=gate_statuses,
            extra_numeric_gates=extra_numeric_gates,
            **gate_values,
        )


def admission_policy_from_yaml(path: str | Path | None) -> AdmissionPolicy:
    """Load an admission policy from YAML if a path is supplied."""

    if path is None:
        return AdmissionPolicy()
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Admission policy YAML must contain a mapping")
    return AdmissionPolicy.from_mapping(payload)


def _mapping_or_empty(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must contain a mapping")
    return value


def _parse_numeric_gate(name: str, value: Any) -> tuple[float, str]:
    status = "provisional"
    raw_value = value
    if isinstance(value, Mapping):
        if "value" not in value:
            raise ValueError(f"numeric_gates.{name} must define a value")
        raw_value = value["value"]
        status = str(value.get("status", status))
    parsed = float(raw_value)
    if not math.isfinite(parsed):
        raise ValueError(f"numeric_gates.{name}.value must be finite")
    return parsed, status


@dataclass(frozen=True)
class AdmissionResult:
    """Admission output tables and JSON-compatible summaries."""

    events: pd.DataFrame
    variants: pd.DataFrame
    evaluation: dict[str, Any]
    release_summary: dict[str, Any]
    release_summary_markdown: str


def compute_admission_profiles(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    *,
    event_summary: pd.DataFrame,
    arity: pd.DataFrame,
    implementation_validity: pd.DataFrame,
    detector_attribution: pd.DataFrame,
    corrected_detectability: pd.DataFrame,
    repair_profile: pd.DataFrame | None = None,
    description_summary: pd.DataFrame | None = None,
    annotation_robustness: pd.DataFrame | None = None,
    model_zoo_frontier: pd.DataFrame | None = None,
    policy: AdmissionPolicy | None = None,
) -> AdmissionResult:
    """Compute provisional event and variant admission decisions."""

    active_policy = policy or AdmissionPolicy()
    metadata = dict(dataset.metadata_events) if dataset.metadata_events else _load_metadata_index(dataset.root)
    events = _event_admission(
        protocol=protocol,
        policy=active_policy,
        event_summary=event_summary,
        arity=arity,
        implementation_validity=implementation_validity,
        detector_attribution=detector_attribution,
        corrected_detectability=corrected_detectability,
        repair_profile=repair_profile if repair_profile is not None else pd.DataFrame(),
        model_zoo_frontier=model_zoo_frontier if model_zoo_frontier is not None else pd.DataFrame(),
        metadata=metadata,
    )
    variants = _variant_admission(
        events,
        policy=active_policy,
        description_summary=description_summary if description_summary is not None else pd.DataFrame(),
        annotation_robustness=annotation_robustness if annotation_robustness is not None else pd.DataFrame(),
    )
    evaluation = _policy_evaluation(events, variants, active_policy)
    release_summary = _release_summary(dataset, events, variants, evaluation)
    return AdmissionResult(
        events=events,
        variants=variants,
        evaluation=evaluation,
        release_summary=release_summary,
        release_summary_markdown=_release_summary_markdown(release_summary),
    )


def _event_admission(
    *,
    protocol: CapabilityProtocol,
    policy: AdmissionPolicy,
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
        impl = impl_by_event.get(event_id, {})
        repair = repair_by_event.get(event_id, {})
        arity_row = arity_by_event.get(event_id, {})
        detect = detect_by_event.get(event_id, {})
        attribution = attribution_by_event.get(event_id, {})
        model = model_by_event.get(event_id, {})
        metadata_row = metadata.get(event_id, {})
        failures: list[str] = []
        warnings: list[str] = []
        gate_hits: list[str] = []
        _metadata_reasons(metadata_row, failures, warnings)
        impl_status = str(impl.get("implementation_validity_status", "missing"))
        _implementation_reasons(impl_status, failures, warnings, gate_hits)
        if not impl:
            failures.append("missing_implementation_validity_row")
        if not _has_finite_value(impl, ("realized_offset", "realized_log_var_ratio", "realized_fisher_shift")):
            warnings.append("missing_realized_effect_measurement")
        canonical_observable = _bool_value(arity_row.get("canonical_is_observable_at_delta"), default=False)
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
        rows.append(
            {
                "event_id": event_id,
                "variant_id": summary_row.get("variant_id"),
                "split": summary_row.get("split"),
                "instance_id": summary_row.get("instance_id"),
                "anomaly_type": summary_row.get("anomaly_type"),
                "constraint_tag": summary_row.get("constraint_tag"),
                "semantic_scope": summary_row.get("semantic_scope"),
                "admission_status": status,
                "failure_reasons": _join_reasons(failures),
                "warning_reasons": _join_reasons(warnings),
                "provisional_gate_hits": _join_reasons(gate_hits),
                "manual_review_recommended": _manual_review_recommended(
                    failures=failures,
                    warnings=warnings,
                    gate_hits=gate_hits,
                    status=status,
                ),
                "genotype_id": metadata_row.get("genotype_id"),
                "contract_id": metadata_row.get("contract_id"),
                "implementation_validity_status": impl_status,
                "support_status": impl.get("support_status"),
                "boundary_status": impl.get("boundary_status"),
                "shortcut_status": impl.get("shortcut_status"),
                "detected_at_min_alpha": detected,
                "detected_for_admission": detected_for_admission,
                "admission_detection_source": detection_source,
                "min_alpha": _min_alpha(protocol),
                "scan_level_p_value": detect.get("scan_level_p_value"),
                "calibration_status": calibration_status,
                "primary_detection_cause": primary_cause,
                "canonical_is_observable_at_min_delta": canonical_observable,
                "min_delta": _min_delta(protocol),
                "canonical_observed_arity": arity_row.get("canonical_observed_arity"),
                "repair_status": repair_status,
                "repair_gain": repair.get("repair_gain"),
                "model_zoo_detected_at_min_alpha": model_detected,
                "model_zoo_detected_model_count": model.get("model_zoo_detected_model_count"),
                "policy_mode": policy.mode,
            }
        )
    return pd.DataFrame(rows)


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
    if any(reason in failures for reason in {"missing_problem_genotype", "missing_contract_id"}):
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


def _variant_admission(
    events: pd.DataFrame,
    *,
    policy: AdmissionPolicy,
    description_summary: pd.DataFrame,
    annotation_robustness: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    description_by_variant = _description_summary_by_variant(description_summary)
    annotation_by_variant = _annotation_summary_by_variant(annotation_robustness)
    rows: list[dict[str, object]] = []
    for variant_id, frame in events.groupby("variant_id", dropna=False):
        status_counts = frame["admission_status"].value_counts().to_dict()
        failures = _collect_reason_values(frame, "failure_reasons")
        warnings = _collect_reason_values(frame, "warning_reasons")
        gate_hits = _collect_reason_values(frame, "provisional_gate_hits")
        desc = description_by_variant.get(str(variant_id), {})
        annot = annotation_by_variant.get(str(variant_id), {})
        gate_hits.extend(_variant_gate_hits(frame, desc, annot, policy))
        status = _variant_status(frame, failures, warnings, gate_hits, policy)
        rows.append(
            {
                "variant_id": variant_id,
                "admission_status": status,
                "event_count": int(len(frame)),
                "release_event_count": int((frame["admission_status"] == "release").sum()),
                "release_with_warning_event_count": int((frame["admission_status"] == "release_with_warning").sum()),
                "debug_event_count": int((frame["admission_status"] == "debug").sum()),
                "legacy_event_count": int((frame["admission_status"] == "legacy").sum()),
                "needs_repair_event_count": int((frame["admission_status"] == "needs_repair").sum()),
                "reject_event_count": int((frame["admission_status"] == "reject").sum()),
                "release_event_share": _share(frame, {"release", "release_with_warning"}),
                "needs_repair_event_share": _share(frame, {"needs_repair"}),
                "reject_event_share": _share(frame, {"reject"}),
                "canonical_observable_share": _finite_mean_bool(frame["canonical_is_observable_at_min_delta"]),
                "detected_share_at_min_alpha": _finite_mean_bool(frame["detected_at_min_alpha"]),
                "median_repair_gain": desc.get("median_repair_gain"),
                "oracle_annotation_pass_rate": annot.get("oracle_annotation_pass_rate"),
                "failure_reasons": _join_reasons(failures),
                "warning_reasons": _join_reasons(warnings),
                "provisional_gate_hits": _join_reasons(gate_hits),
                "manual_review_recommended": _manual_review_recommended(
                    failures=failures,
                    warnings=warnings,
                    gate_hits=gate_hits,
                    status=status,
                ),
                "policy_mode": policy.mode,
            }
        )
    return pd.DataFrame(rows)


def _variant_status(
    frame: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    gate_hits: list[str],
    policy: AdmissionPolicy,
) -> str:
    statuses = set(frame["admission_status"].astype(str))
    if "reject" in statuses:
        return "reject"
    if statuses == {"legacy"}:
        return "legacy"
    if (
        "needs_repair" in statuses
        and _share(frame, {"needs_repair"}) > policy.max_needs_repair_share
    ):
        return "needs_repair"
    if (
        "debug" in statuses
        and _share(frame, {"debug"}) > policy.max_debug_event_share
    ):
        return "debug"
    if warnings or gate_hits or "release_with_warning" in statuses:
        return "release_with_warning"
    return "release"


def _manual_review_recommended(
    *,
    failures: list[str],
    warnings: list[str],
    gate_hits: list[str],
    status: str,
) -> bool:
    return bool(
        failures
        or gate_hits
        or status in {"debug", "needs_repair", "legacy", "reject"}
        or "model_zoo_only_detection" in warnings
    )


def _variant_gate_hits(
    frame: pd.DataFrame,
    description: Mapping[str, Any],
    annotation: Mapping[str, Any],
    policy: AdmissionPolicy,
) -> list[str]:
    hits: list[str] = []
    needs_repair_share = _share(frame, {"needs_repair"})
    if needs_repair_share > policy.max_needs_repair_share:
        hits.append("max_needs_repair_share")
    debug_share = _share(frame, {"debug"})
    if debug_share > policy.max_debug_event_share:
        hits.append("max_debug_event_share")
    boundary_hits = frame["provisional_gate_hits"].astype(str).str.contains("max_boundary_artifact_fail_share").mean()
    if float(boundary_hits) > policy.max_boundary_artifact_fail_share:
        hits.append("max_boundary_artifact_fail_share")
    canonical_share = _finite_mean_bool(frame["canonical_is_observable_at_min_delta"])
    if math.isfinite(canonical_share) and canonical_share < policy.min_canonical_observable_share:
        hits.append("min_canonical_observable_share")
    oracle_pass_rate = annotation.get("oracle_annotation_pass_rate")
    if _is_finite_number(oracle_pass_rate) and float(oracle_pass_rate) < policy.min_oracle_annotation_pass_rate:
        hits.append("min_oracle_annotation_pass_rate")
    repair_gain = description.get("median_repair_gain")
    if _is_finite_number(repair_gain) and float(repair_gain) < 0.20:
        hits.append("max_needs_repair_share")
    return hits


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


def _policy_evaluation(
    events: pd.DataFrame,
    variants: pd.DataFrame,
    policy: AdmissionPolicy,
) -> dict[str, Any]:
    event_counts = _value_counts(events, "admission_status")
    variant_counts = _value_counts(variants, "admission_status")
    gate_counts = _reason_counts(variants, "provisional_gate_hits")
    return {
        "admission_policy_version": policy.version,
        "mode": policy.mode,
        "numeric_gate_enforcement": policy.numeric_gate_enforcement,
        "policy": policy.to_dict(),
        "event_count": int(len(events)),
        "variant_count": int(len(variants)),
        "event_status_counts": event_counts,
        "variant_status_counts": variant_counts,
        "provisional_gate_hit_counts": gate_counts,
        "manual_review_event_count": int(events["manual_review_recommended"].sum()) if not events.empty else 0,
        "manual_review_variant_count": int(variants["manual_review_recommended"].sum()) if not variants.empty else 0,
        "outputs": {
            "admission_events": "admission_events.csv",
            "admission_variants": "admission_variants.csv",
            "admission_policy_evaluation": "admission_policy_evaluation.json",
            "release_summary": "release_summary.json",
            "release_summary_markdown": "release_summary.md",
        },
    }


def _release_summary(
    dataset: DatasetIndex,
    events: pd.DataFrame,
    variants: pd.DataFrame,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "release_summary_version": "synthgen.release_summary.v12.1",
        "dataset_root": str(dataset.root),
        "admission_policy_version": evaluation.get("admission_policy_version"),
        "policy_mode": evaluation.get("mode"),
        "event_count": int(len(events)),
        "variant_count": int(len(variants)),
        "event_status_counts": evaluation.get("event_status_counts", {}),
        "variant_status_counts": evaluation.get("variant_status_counts", {}),
        "provisional_gate_hit_counts": evaluation.get("provisional_gate_hit_counts", {}),
        "manual_review_event_count": evaluation.get("manual_review_event_count", 0),
        "manual_review_variant_count": evaluation.get("manual_review_variant_count", 0),
    }


def _release_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Synth-gen v12 release summary",
        "",
        f"Admission policy: `{summary.get('admission_policy_version')}`",
        f"Policy mode: `{summary.get('policy_mode')}`",
        "",
        "## Coverage",
        "",
        f"- Events: {summary.get('event_count', 0)}",
        f"- Variants: {summary.get('variant_count', 0)}",
        f"- Manual-review events: {summary.get('manual_review_event_count', 0)}",
        f"- Manual-review variants: {summary.get('manual_review_variant_count', 0)}",
        "",
        "## Event statuses",
        "",
        _dict_lines(summary.get("event_status_counts", {})),
        "",
        "## Variant statuses",
        "",
        _dict_lines(summary.get("variant_status_counts", {})),
        "",
        "## Provisional gate hits",
        "",
        _dict_lines(summary.get("provisional_gate_hit_counts", {})),
        "",
    ]
    return "\n".join(lines)


def _dict_lines(values: Mapping[str, Any]) -> str:
    if not values:
        return "No records."
    return "\n".join(f"- `{key}`: {value}" for key, value in sorted(values.items()))


def _load_metadata_index(dataset_root: Path) -> dict[str, Mapping[str, Any]]:
    path = Path(dataset_root) / "metadata" / "events.jsonl"
    if not path.exists():
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows[str(payload.get("event_id", ""))] = payload
    return rows


def _index_by_event(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "event_id" not in frame.columns:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for _, row in frame.iterrows():
        rows[str(row["event_id"])] = dict(row)
    return rows


def _min_delta_rows(arity: pd.DataFrame, protocol: CapabilityProtocol) -> dict[str, Mapping[str, Any]]:
    if arity.empty:
        return {}
    delta = _min_delta(protocol)
    frame = arity[np.isclose(arity["delta"].astype(float), delta)]
    return _index_by_event(frame)


def _alpha_rows(frame: pd.DataFrame, protocol: CapabilityProtocol) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "alpha" not in frame.columns:
        return {}
    alpha = _min_alpha(protocol)
    active = frame[np.isclose(frame["alpha"].astype(float), alpha)].copy()
    if "rank_within_event" in active.columns:
        active = active.sort_values(["event_id", "rank_within_event"])
    rows: dict[str, Mapping[str, Any]] = {}
    for event_id, group in active.groupby("event_id", dropna=False):
        rows[str(event_id)] = dict(group.iloc[0])
    return rows


def _model_detection_summary(frame: pd.DataFrame, protocol: CapabilityProtocol) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "alpha" not in frame.columns:
        return {}
    alpha = _min_alpha(protocol)
    active = frame[np.isclose(frame["alpha"].astype(float), alpha)]
    rows: dict[str, Mapping[str, Any]] = {}
    for event_id, group in active.groupby("event_id", dropna=False):
        rows[str(event_id)] = {
            "model_zoo_detected_at_min_alpha": bool(group["detected"].astype(bool).any()),
            "model_zoo_detected_model_count": int(group["detected"].astype(bool).sum()),
        }
    return rows


def _description_summary_by_variant(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "variant_id" not in frame.columns:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for variant_id, group in frame.groupby("variant_id", dropna=False):
        rows[str(variant_id)] = {
            "median_repair_gain": float(group["median_repair_gain"].median())
            if "median_repair_gain" in group
            else float("nan")
        }
    return rows


def _annotation_summary_by_variant(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "variant_id" not in frame.columns:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    oracle = frame[frame["annotation_channel"].astype(str).str.startswith("labels_oracle")]
    for variant_id, group in oracle.groupby("variant_id", dropna=False):
        rows[str(variant_id)] = {
            "oracle_annotation_pass_rate": float(group["alignment_pass_rate"].mean())
            if "alignment_pass_rate" in group
            else float("nan")
        }
    return rows


def _has_finite_value(row: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key in row and _is_finite_number(row[key]):
            return True
    return False


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if _is_finite_number(value):
        return bool(value)
    return default


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _min_alpha(protocol: CapabilityProtocol) -> float:
    return float(min(protocol.alpha_grid))


def _min_delta(protocol: CapabilityProtocol) -> float:
    return float(min(protocol.delta_grid))


def _join_reasons(reasons: list[str]) -> str:
    return "|".join(sorted(dict.fromkeys(reason for reason in reasons if reason)))


def _collect_reason_values(frame: pd.DataFrame, column: str) -> list[str]:
    values: list[str] = []
    if column not in frame.columns:
        return values
    for value in frame[column].fillna("").astype(str):
        values.extend(part for part in value.split("|") if part)
    return sorted(dict.fromkeys(values))


def _reason_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in _collect_reason_values(frame, column):
        counts[reason] = int(frame[column].fillna("").astype(str).str.contains(reason, regex=False).sum())
    return counts


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}


def _share(frame: pd.DataFrame, statuses: set[str]) -> float:
    if frame.empty:
        return float("nan")
    return float(frame["admission_status"].isin(statuses).mean())


def _finite_mean_bool(values: pd.Series) -> float:
    mapped = values.map(lambda value: _bool_value(value, default=False)).astype(float)
    if mapped.empty:
        return float("nan")
    return float(mapped.mean())
