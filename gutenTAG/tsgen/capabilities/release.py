"""Release feedback and certificate builders for v12 capability runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..io import write_json
from .pandas_typing import row_mapping

FEEDBACK_VERSION = "synthgen.generator_feedback.v12.1"
RELEASE_CERTIFICATE_VERSION = "synthgen.release_certificate.v12.1"


def build_generator_feedback(analysis_dir: Path) -> dict[str, Any]:
    """Build generator feedback from admission and audit outputs."""

    root = Path(analysis_dir)
    variants = _read_csv(root / "admission_variants.csv")
    events = _read_csv(root / "admission_events.csv")
    visual = _read_csv(root / "visual_audit_selection.csv")
    reason_counts = _reason_counts(
        events, ("failure_reasons", "warning_reasons", "provisional_gate_hits")
    )
    recommendations = _variant_recommendations(variants, events, visual)
    return {
        "generator_feedback_version": FEEDBACK_VERSION,
        "analysis_dir": str(root),
        "variant_count": int(len(variants)),
        "event_count": int(len(events)),
        "variant_status_counts": _value_counts(variants, "admission_status"),
        "event_status_counts": _value_counts(events, "admission_status"),
        "top_reason_counts": dict(
            sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "recommendations": recommendations,
    }


def write_generator_feedback(analysis_dir: Path) -> dict[str, Any]:
    """Write generator feedback JSON and Markdown reports."""

    root = Path(analysis_dir)
    feedback = build_generator_feedback(root)
    write_json(root / "generator_feedback_report.json", feedback)
    (root / "generator_feedback_report.md").write_text(
        generator_feedback_markdown(feedback),
        encoding="utf-8",
    )
    return feedback


def generator_feedback_markdown(feedback: Mapping[str, Any]) -> str:
    """Render generator feedback as Markdown."""

    lines = [
        "# Generator Feedback Report",
        "",
        f"Version: `{feedback.get('generator_feedback_version')}`",
        "",
        "## Coverage",
        "",
        f"- Variants: {feedback.get('variant_count', 0)}",
        f"- Events: {feedback.get('event_count', 0)}",
        "",
        "## Variant Statuses",
        "",
        _dict_lines(feedback.get("variant_status_counts", {})),
        "",
        "## Event Statuses",
        "",
        _dict_lines(feedback.get("event_status_counts", {})),
        "",
        "## Top Reasons",
        "",
        _dict_lines(feedback.get("top_reason_counts", {})),
        "",
        "## Recommendations",
        "",
    ]
    recommendations = list(feedback.get("recommendations", []))
    if not recommendations:
        lines.append("No generator feedback recommendations were produced.")
    for item in recommendations:
        lines.append(f"### {item.get('variant_id')}")
        lines.append("")
        lines.append(f"- priority: `{item.get('priority')}`")
        lines.append(f"- admission status: `{item.get('admission_status')}`")
        lines.append(f"- intervention events: {item.get('intervention_event_count')}")
        reasons = item.get("reasons", [])
        if reasons:
            lines.append(f"- reasons: `{'|'.join(map(str, reasons))}`")
        actions = item.get("suggested_actions", [])
        for action in actions:
            lines.append(f"- action: {action}")
        visual_examples = item.get("visual_examples", [])
        for path in visual_examples:
            lines.append(f"- visual: `{path}`")
        lines.append("")
    return "\n".join(lines).replace("` ", "`") + "\n"


def build_release_certificate(
    *,
    dataset_root: Path,
    analysis_dir: Path,
    admission_policy_path: Path | None = None,
) -> dict[str, Any]:
    """Build a release certificate from generated dataset and analysis outputs."""

    dataset = Path(dataset_root)
    analysis = Path(analysis_dir)
    feedback_path = analysis / "generator_feedback_report.json"
    if not feedback_path.exists():
        write_generator_feedback(analysis)
    required = _required_artifacts(analysis)
    certificate = {
        "release_certificate_version": RELEASE_CERTIFICATE_VERSION,
        "dataset_root": str(dataset),
        "analysis_dir": str(analysis),
        "admission_policy_path": (
            str(admission_policy_path) if admission_policy_path is not None else None
        ),
        "dataset_manifest": _load_json(dataset / "dataset_manifest.json"),
        "capability_certificate": _load_json(analysis / "capability_certificate.json"),
        "admission_policy_evaluation": _load_json(
            analysis / "admission_policy_evaluation.json"
        ),
        "release_summary": _load_json(analysis / "release_summary.json"),
        "generator_feedback": _load_json(feedback_path),
        "visual_audit_manifest": _load_json(
            analysis / "visual_audit" / "visual_audit_manifest.json"
        ),
        "output_manifest": _load_json(analysis / "manifests" / "output_manifest.json"),
        "table_hashes": _load_json(analysis / "manifests" / "table_hashes.json"),
        "profile_run_manifest": _load_json(
            analysis / "manifests" / "profile_run_manifest.json"
        ),
        "required_artifacts": required,
        "artifact_hashes": _artifact_hashes(analysis, required),
    }
    certificate["certificate_status"] = (
        "complete" if all(item["exists"] for item in required) else "incomplete"
    )
    return certificate


def write_release_certificate(
    *,
    dataset_root: Path,
    analysis_dir: Path,
    admission_policy_path: Path | None = None,
) -> dict[str, Any]:
    """Write release certificate JSON and Markdown files."""

    analysis = Path(analysis_dir)
    certificate = build_release_certificate(
        dataset_root=Path(dataset_root),
        analysis_dir=analysis,
        admission_policy_path=admission_policy_path,
    )
    write_json(analysis / "release_certificate.json", certificate)
    (analysis / "release_certificate.md").write_text(
        release_certificate_markdown(certificate),
        encoding="utf-8",
    )
    return certificate


def release_certificate_markdown(certificate: Mapping[str, Any]) -> str:
    """Render a release certificate as Markdown."""

    release_summary = certificate.get("release_summary", {})
    admission = certificate.get("admission_policy_evaluation", {})
    lines = [
        "# Synth-gen v12 Release Certificate",
        "",
        f"Version: `{certificate.get('release_certificate_version')}`",
        f"Status: `{certificate.get('certificate_status')}`",
        "",
        "## Paths",
        "",
        f"- Dataset: `{certificate.get('dataset_root')}`",
        f"- Analysis: `{certificate.get('analysis_dir')}`",
        "",
        "## Admission",
        "",
        f"- Policy: `{admission.get('admission_policy_version')}`",
        f"- Mode: `{admission.get('mode')}`",
        f"- Events: {release_summary.get('event_count', 0)}",
        f"- Variants: {release_summary.get('variant_count', 0)}",
        f"- Manual-review events: {release_summary.get('manual_review_event_count', 0)}",
        f"- Manual-review variants: {release_summary.get('manual_review_variant_count', 0)}",
        "",
        "## Variant Statuses",
        "",
        _dict_lines(release_summary.get("variant_status_counts", {})),
        "",
        "## Required Artifacts",
        "",
    ]
    for item in certificate.get("required_artifacts", []):
        status = "present" if item.get("exists") else "missing"
        lines.append(f"- `{item.get('path')}`: {status}")
    lines.append("")
    lines.extend(["## Generator Feedback", ""])
    feedback = certificate.get("generator_feedback", {})
    recommendations = feedback.get("recommendations", [])
    if not recommendations:
        lines.append("No recommendations.")
    else:
        for item in recommendations[:10]:
            lines.append(
                f"- `{item.get('variant_id')}`: {item.get('priority')} / {item.get('admission_status')}"
            )
    lines.append("")
    return "\n".join(lines)


def _variant_recommendations(
    variants: pd.DataFrame,
    events: pd.DataFrame,
    visual: pd.DataFrame,
) -> list[dict[str, Any]]:
    if variants.empty:
        return []
    visual_by_variant = _visual_examples(visual)
    rows: list[dict[str, Any]] = []
    for _, row in variants.iterrows():
        variant_id = str(row.get("variant_id", ""))
        reasons = _row_reasons(row_mapping(row))
        status = str(row.get("admission_status", "unknown"))
        if status == "release" and not reasons:
            continue
        event_frame = (
            events[events["variant_id"].astype(str) == variant_id]
            if not events.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "variant_id": variant_id,
                "priority": _priority(status, reasons),
                "admission_status": status,
                "intervention_event_count": int(len(event_frame)),
                "release_event_share": _finite(row.get("release_event_share")),
                "needs_repair_event_share": _finite(
                    row.get("needs_repair_event_share")
                ),
                "reasons": reasons,
                "suggested_actions": _suggested_actions(reasons),
                "visual_examples": visual_by_variant.get(variant_id, [])[:3],
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            _priority_rank(str(item["priority"])),
            str(item["variant_id"]),
        ),
    )


def _suggested_actions(reasons: list[str]) -> list[str]:
    actions: list[str] = []
    if any("support_leakage" in reason or "support" in reason for reason in reasons):
        actions.append(
            "Review event support generation and label bounds; residual mass leaks outside declared support."
        )
    if any("boundary" in reason for reason in reasons):
        actions.append(
            "Smooth anomaly transitions or expand support to include boundary effects."
        )
    if any("wrong_reason" in reason or "shortcut" in reason for reason in reasons):
        actions.append(
            "Reduce marginal shortcuts and increase canonical relation witness strength."
        )
    if any(
        "not_detected" in reason or "observable_not_detected" in reason
        for reason in reasons
    ):
        actions.append(
            "Increase anomaly duration/strength or lower noise for this profile."
        )
    if any("negative_control" in reason for reason in reasons):
        actions.append(
            "Inspect negative-control triggers; detector evidence may not be specific to the intended event."
        )
    if any("calibration_low_resolution" in reason for reason in reasons):
        actions.append(
            "Increase clean calibration instances before treating low-alpha conclusions as stable."
        )
    if any(
        "repair" in reason or "max_needs_repair_share" in reason for reason in reasons
    ):
        actions.append(
            "Inspect generator parameters for this variant and rerun admission after repair."
        )
    if not actions:
        actions.append("Manual review recommended by provisional admission status.")
    return actions


def _row_reasons(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("failure_reasons", "warning_reasons", "provisional_gate_hits"):
        value = row.get(key, "")
        if bool(pd.isna(value)):
            continue
        values.extend(part for part in str(value).split("|") if part)
    return sorted(dict.fromkeys(values))


def _visual_examples(frame: pd.DataFrame) -> dict[str, list[str]]:
    if (
        frame.empty
        or "variant_id" not in frame.columns
        or "plot_path" not in frame.columns
    ):
        return {}
    rows: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        variant_id = str(row["variant_id"])
        path = str(row["plot_path"])
        if path:
            rows.setdefault(variant_id, []).append(f"visual_audit/{path}")
    return rows


def _priority(status: str, reasons: list[str]) -> str:
    if status == "reject":
        return "p0"
    if status == "needs_repair" or any(
        "boundary" in reason or "support" in reason for reason in reasons
    ):
        return "p1"
    if status == "debug" or reasons:
        return "p2"
    return "p3"


def _priority_rank(priority: str) -> int:
    return {"p0": 0, "p1": 1, "p2": 2, "p3": 3}.get(priority, 9)


def _reason_counts(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].fillna("").astype(str):
            for reason in value.split("|"):
                if reason:
                    counts[reason] = counts.get(reason, 0) + 1
    return counts


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


def _required_artifacts(analysis: Path) -> list[dict[str, Any]]:
    paths = (
        "capability_certificate.json",
        "admission_events.csv",
        "admission_variants.csv",
        "admission_policy_evaluation.json",
        "release_summary.json",
        "release_summary.md",
        "generator_feedback_report.json",
        "generator_feedback_report.md",
        "visual_audit/index.md",
        "visual_audit/index.html",
        "visual_audit/visual_audit_manifest.json",
        "visual_audit_selection.csv",
        "manifests/output_manifest.json",
        "manifests/table_hashes.json",
        "manifests/profile_run_manifest.json",
    )
    return [
        {
            "path": path,
            "exists": (analysis / path).exists(),
            "sha256": (
                _file_sha256(analysis / path) if (analysis / path).is_file() else None
            ),
        }
        for path in paths
    ]


def _artifact_hashes(
    analysis: Path, artifacts: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    return {
        str(item["path"]): str(item["sha256"])
        for item in artifacts
        if item.get("exists") and item.get("sha256")
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _dict_lines(values: Mapping[str, Any]) -> str:
    if not values:
        return "No records."
    return "\n".join(f"- `{key}`: {value}" for key, value in sorted(values.items()))
