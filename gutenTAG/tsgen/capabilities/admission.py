"""Provisional release admission policy evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .admission_common import (
    _collect_reason_values,
    _finite_mean_bool,
    _is_finite_number,
    _join_reasons,
    _manual_review_recommended,
    _reason_counts,
    _share,
    _value_counts,
)
from .admission_events import _event_admission
from .admission_policy import (
    AdmissionPolicy,
    admission_policy_from_yaml,
)
from .dataset import DatasetIndex
from .pandas_typing import column as frame_column
from .protocol import CapabilityProtocol

__all__ = [
    "AdmissionPolicy",
    "AdmissionResult",
    "admission_policy_from_yaml",
    "compute_admission_profiles",
]


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
    metadata = (
        dict(dataset.metadata_events)
        if dataset.metadata_events
        else _load_metadata_index(dataset.root)
    )
    events = _event_admission(
        protocol=protocol,
        policy=active_policy,
        event_summary=event_summary,
        arity=arity,
        implementation_validity=implementation_validity,
        detector_attribution=detector_attribution,
        corrected_detectability=corrected_detectability,
        repair_profile=repair_profile if repair_profile is not None else pd.DataFrame(),
        model_zoo_frontier=(
            model_zoo_frontier if model_zoo_frontier is not None else pd.DataFrame()
        ),
        metadata=metadata,
    )
    variants = _variant_admission(
        events,
        policy=active_policy,
        description_summary=(
            description_summary if description_summary is not None else pd.DataFrame()
        ),
        annotation_robustness=(
            annotation_robustness
            if annotation_robustness is not None
            else pd.DataFrame()
        ),
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
        admission_status = frame_column(frame, "admission_status")
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
                "release_event_count": int((admission_status == "release").sum()),
                "release_with_warning_event_count": int(
                    (admission_status == "release_with_warning").sum()
                ),
                "debug_event_count": int((admission_status == "debug").sum()),
                "legacy_event_count": int((admission_status == "legacy").sum()),
                "needs_repair_event_count": int(
                    (admission_status == "needs_repair").sum()
                ),
                "reject_event_count": int((admission_status == "reject").sum()),
                "release_event_share": _share(
                    frame, {"release", "release_with_warning"}
                ),
                "needs_repair_event_share": _share(frame, {"needs_repair"}),
                "reject_event_share": _share(frame, {"reject"}),
                "canonical_observable_share": _finite_mean_bool(
                    frame_column(frame, "canonical_is_observable_at_min_delta")
                ),
                "detected_share_at_min_alpha": _finite_mean_bool(
                    frame_column(frame, "detected_at_min_alpha")
                ),
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
    statuses = set(frame_column(frame, "admission_status").astype(str))
    if "reject" in statuses:
        return "reject"
    if statuses == {"legacy"}:
        return "legacy"
    if (
        "needs_repair" in statuses
        and _share(frame, {"needs_repair"}) > policy.max_needs_repair_share
    ):
        return "needs_repair"
    if "debug" in statuses and _share(frame, {"debug"}) > policy.max_debug_event_share:
        return "debug"
    if warnings or gate_hits or "release_with_warning" in statuses:
        return "release_with_warning"
    return "release"


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
    boundary_hits = (
        frame_column(frame, "provisional_gate_hits")
        .astype(str)
        .str.contains("max_boundary_artifact_fail_share")
        .mean()
    )
    if float(boundary_hits) > policy.max_boundary_artifact_fail_share:
        hits.append("max_boundary_artifact_fail_share")
    canonical_share = _finite_mean_bool(
        frame_column(frame, "canonical_is_observable_at_min_delta")
    )
    if (
        math.isfinite(canonical_share)
        and canonical_share < policy.min_canonical_observable_share
    ):
        hits.append("min_canonical_observable_share")
    oracle_pass_rate = _finite_number_value(
        annotation.get("oracle_annotation_pass_rate")
    )
    if (
        oracle_pass_rate is not None
        and float(oracle_pass_rate) < policy.min_oracle_annotation_pass_rate
    ):
        hits.append("min_oracle_annotation_pass_rate")
    repair_gain = _finite_number_value(description.get("median_repair_gain"))
    if repair_gain is not None and float(repair_gain) < 0.20:
        hits.append("max_needs_repair_share")
    return hits


def _finite_number_value(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


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
        "manual_review_event_count": (
            int(frame_column(events, "manual_review_recommended").sum())
            if not events.empty
            else 0
        ),
        "manual_review_variant_count": (
            int(frame_column(variants, "manual_review_recommended").sum())
            if not variants.empty
            else 0
        ),
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
        "provisional_gate_hit_counts": evaluation.get(
            "provisional_gate_hit_counts", {}
        ),
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


def _description_summary_by_variant(
    frame: pd.DataFrame,
) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "variant_id" not in frame.columns:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for variant_id, group in frame.groupby("variant_id", dropna=False):
        rows[str(variant_id)] = {
            "median_repair_gain": (
                float(group["median_repair_gain"].median())
                if "median_repair_gain" in group
                else float("nan")
            )
        }
    return rows


def _annotation_summary_by_variant(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "variant_id" not in frame.columns:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    oracle = frame[
        frame["annotation_channel"].astype(str).str.startswith("labels_oracle")
    ]
    for variant_id, group in oracle.groupby("variant_id", dropna=False):
        rows[str(variant_id)] = {
            "oracle_annotation_pass_rate": (
                float(group["alignment_pass_rate"].mean())
                if "alignment_pass_rate" in group
                else float("nan")
            )
        }
    return rows
