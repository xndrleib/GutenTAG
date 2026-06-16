"""Capability certificate and report construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd

from ..manifest import canonical_json_hash
from .dataset import DatasetIndex
from .pandas_typing import as_frame, as_series, column as frame_column, numeric_column
from .protocol import CapabilityProtocol

OUTPUT_FILENAMES: Mapping[str, str] = {
    "observability": "observability_profile.csv",
    "arity": "arity_profile.csv",
    "event_summary": "event_capability_summary.csv",
    "detectability_frontier": "detectability_frontier.csv",
    "detectability_summary": "detectability_summary.csv",
    "corrected_detectability_frontier": "corrected_detectability_frontier.csv",
    "oracle_window_diagnostic_frontier": "oracle_window_diagnostic_frontier.csv",
    "blind_scan_events": "blind_scan_events.csv",
    "calibration_resolution": "calibration_resolution.csv",
    "model_zoo_frontier": "model_zoo_frontier.csv",
    "model_zoo_event_scores": "model_zoo_event_scores.csv",
    "law_observability_profile": "law_observability_profile.csv",
    "law_observability_summary": "law_observability_summary.csv",
    "identifiability_embeddings": "identifiability_event_embeddings.csv",
    "identifiability_summary": "identifiability_summary.csv",
    "identifiability_pairwise": "identifiability_pairwise.csv",
    "diagnosis_confusion_matrix": "diagnosis_confusion_matrix.csv",
    "identifiability_quotient": "identifiability_quotient.csv",
    "description_profile": "description_profile.csv",
    "description_summary": "description_summary.csv",
    "description_stability": "description_stability.csv",
    "repair_profile": "repair_profile.csv",
    "support_integrity": "support_integrity.csv",
    "boundary_audit": "boundary_audit.csv",
    "shortcut_audit": "shortcut_audit.csv",
    "realized_effects": "realized_effects.csv",
    "detector_attribution": "detector_attribution.csv",
    "negative_controls": "negative_controls.csv",
    "implementation_validity": "implementation_validity.csv",
    "labels_oracle_any": "labels/labels_oracle_any.csv",
    "labels_oracle_intervention": "labels/labels_oracle_intervention.csv",
    "labels_oracle_context": "labels/labels_oracle_context.csv",
    "labels_event_only": "labels/labels_event_only.csv",
    "labels_delayed": "labels/labels_delayed.csv",
    "labels_weak_point": "labels/labels_weak_point.csv",
    "labels_visible_only": "labels/labels_visible_only.csv",
    "labels_noisy_boundary": "labels/labels_noisy_boundary.csv",
    "labels_censored": "labels/labels_censored.csv",
    "annotation_alignment": "annotation_alignment.csv",
    "annotation_robustness": "annotation_robustness.csv",
    "admission_events": "admission_events.csv",
    "admission_variants": "admission_variants.csv",
    "visual_audit_selection": "visual_audit_selection.csv",
}


def build_capability_certificate(
    *,
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    profile_names: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    run_manifest: Mapping[str, Any] | None = None,
    extra_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a compact machine-readable capability certificate."""

    protocol_dict = protocol.to_dict()
    event_summary = tables.get("event_summary", pd.DataFrame())
    summary = {
        "instance_count": len(dataset.instances),
        "event_group_count": (
            _dataset_event_count(dataset)
            if event_summary.empty
            else int(len(event_summary))
        ),
        "variant_count": (
            len({instance.variant_id for instance in dataset.instances})
            if event_summary.empty
            else int(frame_column(event_summary, "variant_id").nunique())
        ),
    }
    family_summary = family_capability_summary(tables)
    payload: dict[str, Any] = {
        "capability_certificate_version": protocol.protocol_version,
        "dataset_root": str(dataset.root),
        "analysis_output_dir": str(output_dir),
        "analysis_cache_dir": str(cache_dir) if cache_dir is not None else None,
        "profiles": list(profile_names or []),
        "protocol": protocol_dict,
        "protocol_hash": canonical_json_hash(protocol_dict),
        "run_manifest": dict(run_manifest or {}),
        "dataset_schema_version": (
            dataset.manifest.get("dataset_schema_version") if dataset.manifest else None
        ),
        "dataset_normalized_config_hash": (
            dataset.manifest.get("normalized_config_hash") if dataset.manifest else None
        ),
        "summary": summary,
        "family_summary": family_summary,
        "artifacts": {
            key: OUTPUT_FILENAMES[key]
            for key, frame in tables.items()
            if key in OUTPUT_FILENAMES and (not frame.empty or key in OUTPUT_FILENAMES)
        },
    }
    if extra_artifacts:
        payload["artifacts"].update(dict(extra_artifacts))
    return payload


def family_capability_summary(
    tables: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Build per-family summary rows for the capability certificate."""

    event_summary = tables.get("event_summary", pd.DataFrame())
    if event_summary.empty:
        return []
    rows: list[dict[str, Any]] = []
    detectability = tables.get("detectability_summary", pd.DataFrame())
    descriptions = tables.get("description_summary", pd.DataFrame())
    arity = tables.get("arity", pd.DataFrame())
    grouped = event_summary.groupby(
        ["variant_id", "anomaly_type", "constraint_tag", "semantic_scope"],
        dropna=False,
    )
    for key, frame in grouped:
        variant_id, anomaly_type, constraint_tag, semantic_scope = cast(
            tuple[object, object, object, object], key
        )
        record = _base_family_record(
            frame,
            variant_id=variant_id,
            anomaly_type=anomaly_type,
            constraint_tag=constraint_tag,
            semantic_scope=semantic_scope,
        )
        _add_arity_summary(record, arity, variant_id)
        _add_detectability_summary(record, detectability, variant_id)
        _add_description_summary(record, descriptions, variant_id)
        rows.append(record)
    return rows


def write_markdown_report(
    path: Path,
    certificate: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> None:
    """Write a compact Markdown report for a capability certificate."""

    lines: list[str] = []
    lines.append("# Synth-gen capability analysis report")
    lines.append("")
    lines.append(
        f"Certificate version: `{certificate.get('capability_certificate_version')}`"
    )
    lines.append(f"Protocol hash: `{certificate.get('protocol_hash')}`")
    lines.append("")
    summary = certificate.get("summary", {})
    lines.append("## Dataset coverage")
    lines.append("")
    lines.append(f"- Instances: {summary.get('instance_count', 0)}")
    lines.append(f"- Event groups: {summary.get('event_group_count', 0)}")
    lines.append(f"- Variants: {summary.get('variant_count', 0)}")
    lines.append("")
    lines.append("## Family capability summary")
    lines.append("")
    family_summary = certificate.get("family_summary", [])
    if family_summary:
        frame = pd.DataFrame(family_summary)
        preferred = [
            "variant_id",
            "event_count",
            "median_best_distance",
            "median_D_s1",
            "median_D_s2",
            "median_D_canonical_s1",
            "median_D_canonical_s2",
            "observable_share_at_min_delta",
            "median_observed_arity_at_min_delta",
            "canonical_observable_share_at_min_delta",
            "median_canonical_observed_arity_at_min_delta",
            "detected_rate_alpha_0.01",
            "median_witness_sufficiency",
            "median_repair_gain",
        ]
        columns = [column for column in preferred if column in frame.columns]
        lines.append(frame_to_markdown(as_frame(frame[columns])))
    else:
        lines.append("No event groups were available for analysis.")
    lines.append("")
    lines.append("## Identifiability summary")
    lines.append("")
    identifiability = tables.get("identifiability_summary", pd.DataFrame())
    if identifiability is not None and not identifiability.empty:
        lines.append(frame_to_markdown(identifiability))
    else:
        lines.append("Identifiability was not estimable for this run.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact pipe table without optional pandas dependencies."""

    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    rows = [
        [_format_markdown_cell(value) for value in record]
        for record in frame.to_numpy()
    ]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = (
        "| "
        + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns)))
        + " |"
    )
    separator = (
        "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    )
    body = [
        "| "
        + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns)))
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _dataset_event_count(dataset: DatasetIndex) -> int:
    return int(sum(len(instance.event_groups) for instance in dataset.instances))


def _base_family_record(
    frame: pd.DataFrame,
    *,
    variant_id: object,
    anomaly_type: object,
    constraint_tag: object,
    semantic_scope: object,
) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "anomaly_type": anomaly_type,
        "constraint_tag": constraint_tag,
        "semantic_scope": semantic_scope,
        "event_count": int(len(frame)),
        "median_best_distance": float(frame["best_distance"].median()),
        "median_D_s1": _median_optional(frame, "D_s1"),
        "median_D_s2": _median_optional(frame, "D_s2"),
        "median_D_canonical_s1": _median_optional(frame, "D_canonical_s1"),
        "median_D_canonical_s2": _median_optional(frame, "D_canonical_s2"),
        "median_witness_sufficiency_proxy": float(
            numeric_column(frame, "witness_sufficiency_proxy").median()
        ),
        "median_support_concentration_l2": float(
            numeric_column(frame, "support_concentration_l2_max").median()
        ),
    }


def _add_arity_summary(
    record: dict[str, Any],
    arity: pd.DataFrame,
    variant_id: object,
) -> None:
    if arity.empty:
        return
    delta = numeric_column(arity, "delta")
    arity_frame = as_frame(
        arity[
            (frame_column(arity, "variant_id") == variant_id) & (delta == delta.min())
        ]
    )
    if arity_frame.empty:
        return
    record["observable_share_at_min_delta"] = float(
        frame_column(arity_frame, "is_observable_at_delta").mean()
    )
    observed = numeric_column(arity_frame, "observed_arity").dropna()
    canonical_observed = (
        numeric_column(arity_frame, "canonical_observed_arity").dropna()
        if "canonical_observed_arity" in arity_frame
        else pd.Series(dtype=float)
    )
    record["median_observed_arity_at_min_delta"] = (
        float(observed.median()) if not observed.empty else float("nan")
    )
    record["canonical_observable_share_at_min_delta"] = (
        float(frame_column(arity_frame, "canonical_is_observable_at_delta").mean())
        if "canonical_is_observable_at_delta" in arity_frame
        else float("nan")
    )
    record["median_canonical_observed_arity_at_min_delta"] = (
        float(canonical_observed.median())
        if not canonical_observed.empty
        else float("nan")
    )


def _add_detectability_summary(
    record: dict[str, Any],
    detectability: pd.DataFrame,
    variant_id: object,
) -> None:
    if detectability.empty:
        return
    det_frame = as_frame(
        detectability[frame_column(detectability, "variant_id") == variant_id]
    )
    for alpha in (
        sorted(numeric_column(det_frame, "alpha").unique())
        if not det_frame.empty
        else []
    ):
        alpha_frame = as_frame(det_frame[numeric_column(det_frame, "alpha") == alpha])
        record[f"detected_rate_alpha_{float(alpha):g}"] = float(
            numeric_column(alpha_frame, "detected_rate").median()
        )


def _add_description_summary(
    record: dict[str, Any],
    descriptions: pd.DataFrame,
    variant_id: object,
) -> None:
    if descriptions.empty:
        return
    desc_frame = as_frame(
        descriptions[frame_column(descriptions, "variant_id") == variant_id]
    )
    if desc_frame.empty:
        return
    record["median_witness_sufficiency"] = float(
        numeric_column(desc_frame, "median_witness_sufficiency").median()
    )
    record["median_repair_gain"] = float(
        numeric_column(desc_frame, "median_repair_gain").median()
    )
    record["median_description_risk_proxy"] = float(
        numeric_column(desc_frame, "median_description_risk_proxy").median()
    )


def _median_optional(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    return float(as_series(frame.get(column, pd.Series(dtype=float))).median())


def _format_markdown_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = [
    "OUTPUT_FILENAMES",
    "build_capability_certificate",
    "family_capability_summary",
    "frame_to_markdown",
    "write_markdown_report",
]
