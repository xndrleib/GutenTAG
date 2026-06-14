"""Visual audit selection and artifact generation."""

from __future__ import annotations

import re
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..array_store import ArrayStore
from ..dataset import DatasetIndex, InstanceRecord, read_timeseries_csv
from ..protocol import CapabilityProtocol
from .boundary import boundary_annotation
from .generic import plot_event_diagnostic
from .mode import is_mode_event
from .relation import (
    is_relation_event,
    plot_pca_residual_panel,
    plot_relation_scatter,
    plot_rolling_correlation_panel,
)


BUCKETS: tuple[str, ...] = (
    "strongest",
    "weakest",
    "borderline",
    "failed",
    "shortcut_dominated",
    "boundary_suspected",
)


@dataclass(frozen=True)
class VisualAuditResult:
    """Visual audit output artifacts."""

    selection: pd.DataFrame
    manifest: dict[str, Any]
    index_markdown: str
    index_html: str


def compute_visual_audit(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    *,
    output_dir: Path,
    event_summary: pd.DataFrame,
    admission_events: pd.DataFrame | None = None,
    implementation_validity: pd.DataFrame | None = None,
    detector_attribution: pd.DataFrame | None = None,
    boundary_audit: pd.DataFrame | None = None,
    shortcut_audit: pd.DataFrame | None = None,
    arrays: ArrayStore | None = None,
    max_events_per_bucket: int = 4,
) -> VisualAuditResult:
    """Select and plot representative events for visual audit."""

    audit_dir = Path(output_dir) / "visual_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_frame(
        event_summary=event_summary,
        admission_events=admission_events if admission_events is not None else pd.DataFrame(),
        implementation_validity=implementation_validity if implementation_validity is not None else pd.DataFrame(),
        boundary_audit=boundary_audit if boundary_audit is not None else pd.DataFrame(),
        shortcut_audit=shortcut_audit if shortcut_audit is not None else pd.DataFrame(),
    )
    selection = _select_bucket_rows(candidates, protocol, max_events_per_bucket=max_events_per_bucket)
    if not selection.empty:
        selection = _write_plots(
            dataset=dataset,
            audit_dir=audit_dir,
            selection=selection,
            detector_attribution=detector_attribution if detector_attribution is not None else pd.DataFrame(),
            arrays=arrays,
        )
    index_markdown = _build_index(selection)
    index_html = _build_html_index(selection)
    (audit_dir / "index.md").write_text(index_markdown, encoding="utf-8")
    (audit_dir / "index.html").write_text(index_html, encoding="utf-8")
    manifest = _manifest(selection)
    return VisualAuditResult(
        selection=selection,
        manifest=manifest,
        index_markdown=index_markdown,
        index_html=index_html,
    )


def _candidate_frame(
    *,
    event_summary: pd.DataFrame,
    admission_events: pd.DataFrame,
    implementation_validity: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    frame = event_summary.copy()
    for table in (admission_events, implementation_validity, boundary_audit, shortcut_audit):
        frame = _merge_event_table(frame, table)
    if "admission_status" not in frame.columns:
        frame["admission_status"] = "unknown"
    if "warning_reasons" not in frame.columns:
        frame["warning_reasons"] = ""
    if "failure_reasons" not in frame.columns:
        frame["failure_reasons"] = ""
    if "provisional_gate_hits" not in frame.columns:
        frame["provisional_gate_hits"] = ""
    return frame


def _merge_event_table(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "event_id" not in table.columns:
        return frame
    keep = [
        column
        for column in table.columns
        if column == "event_id" or column not in frame.columns
    ]
    if keep == ["event_id"]:
        return frame
    return frame.merge(table[keep], on="event_id", how="left")


def _select_bucket_rows(
    candidates: pd.DataFrame,
    protocol: CapabilityProtocol,
    *,
    max_events_per_bucket: int,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for bucket in BUCKETS:
        selected = _bucket_candidates(candidates, bucket, protocol).head(int(max_events_per_bucket))
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            payload = dict(row)
            payload["bucket"] = bucket
            payload["bucket_rank"] = rank
            payload["visual_priority_score"] = _priority_score(row, bucket, protocol)
            payload["relation_review"] = is_relation_event(payload)
            payload["mode_review"] = is_mode_event(payload)
            payload["boundary_annotation"] = boundary_annotation(payload)
            rows.append(payload)
    return pd.DataFrame(rows)


def _bucket_candidates(candidates: pd.DataFrame, bucket: str, protocol: CapabilityProtocol) -> pd.DataFrame:
    frame = candidates.copy()
    if bucket == "strongest":
        return frame.sort_values("best_canonical_distance", ascending=False)
    if bucket == "weakest":
        return frame.sort_values("best_canonical_distance", ascending=True)
    if bucket == "borderline":
        frame["_borderline"] = frame.apply(lambda row: _borderline_score(row, protocol), axis=1)
        return frame.sort_values("_borderline", ascending=True)
    if bucket == "failed":
        mask = frame["admission_status"].astype(str).isin({"reject", "needs_repair", "debug"})
        selected = frame[mask] if mask.any() else frame
        return selected.sort_values(["admission_status", "best_canonical_distance"], ascending=[True, True])
    if bucket == "shortcut_dominated":
        mask = (
            frame.get("shortcut_status", pd.Series("", index=frame.index)).astype(str).str.contains("shortcut", na=False)
            | frame.get("warning_reasons", pd.Series("", index=frame.index)).astype(str).str.contains("shortcut|wrong_reason", na=False)
        )
        selected = frame[mask] if mask.any() else frame.iloc[0:0]
        return selected.sort_values("best_canonical_distance", ascending=False)
    if bucket == "boundary_suspected":
        mask = (
            frame.get("boundary_status", pd.Series("", index=frame.index)).astype(str).str.contains("boundary_primary|suspected", na=False)
            | frame.get("warning_reasons", pd.Series("", index=frame.index)).astype(str).str.contains("boundary", na=False)
            | frame.get("provisional_gate_hits", pd.Series("", index=frame.index)).astype(str).str.contains("boundary", na=False)
        )
        selected = frame[mask] if mask.any() else frame.iloc[0:0]
        return selected.sort_values("best_canonical_distance", ascending=False)
    return frame


def _borderline_score(row: pd.Series, protocol: CapabilityProtocol) -> float:
    p_value = row.get("scan_level_p_value")
    try:
        if pd.notna(p_value):
            return abs(float(p_value) - float(min(protocol.alpha_grid)))
    except (TypeError, ValueError):
        pass
    status = str(row.get("admission_status", ""))
    if status == "release_with_warning":
        return 0.0
    if status == "debug":
        return 0.1
    return float(row.get("best_canonical_distance", 0.0))


def _priority_score(row: Mapping[str, object], bucket: str, protocol: CapabilityProtocol) -> float:
    if bucket == "borderline":
        return float(_borderline_score(pd.Series(row), protocol))
    try:
        return float(row.get("best_canonical_distance", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _write_plots(
    *,
    dataset: DatasetIndex,
    audit_dir: Path,
    selection: pd.DataFrame,
    detector_attribution: pd.DataFrame,
    arrays: ArrayStore | None,
) -> pd.DataFrame:
    instances = _instance_index(dataset)
    attribution_groups = detector_attribution.groupby("event_id") if not detector_attribution.empty else {}
    rows: list[dict[str, object]] = []
    for _, row in selection.iterrows():
        payload = dict(row)
        instance = instances.get((str(row["variant_id"]), str(row["split"]), str(row["instance_id"])))
        if instance is None:
            payload["plot_path"] = ""
            payload["plot_status"] = "missing_instance"
            rows.append(payload)
            continue
        clean = arrays.get(instance, "clean") if arrays is not None else read_timeseries_csv(instance.clean_path)
        anomalous = arrays.get(instance, "anomalous") if arrays is not None else read_timeseries_csv(instance.anomalous_path)
        event_id = str(row["event_id"])
        attribution = (
            attribution_groups.get_group(event_id)
            if not detector_attribution.empty and event_id in attribution_groups.groups
            else pd.DataFrame()
        )
        plot_path = Path(str(row["bucket"])) / f"{_safe_filename(event_id)}.png"
        plot_event_diagnostic(
            output_path=audit_dir / plot_path,
            clean=clean,
            anomalous=anomalous,
            event={**payload, "plot_path": plot_path.as_posix()},
            attribution=attribution,
        )
        payload["plot_path"] = plot_path.as_posix()
        payload["plot_status"] = "written"
        _write_relation_panels(
            audit_dir=audit_dir,
            payload=payload,
            clean=clean,
            anomalous=anomalous,
            event_id=event_id,
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def _write_relation_panels(
    *,
    audit_dir: Path,
    payload: dict[str, object],
    clean: np.ndarray,
    anomalous: np.ndarray,
    event_id: str,
) -> None:
    if not bool(payload.get("relation_review")) and not bool(payload.get("mode_review")):
        payload["relation_scatter_path"] = ""
        payload["rolling_correlation_path"] = ""
        payload["pca_residual_path"] = ""
        return
    bucket = str(payload.get("bucket", "visual"))
    stem = _safe_filename(event_id)
    panel_specs = (
        (
            "relation_scatter_path",
            Path(bucket) / f"{stem}__relation_scatter.png",
            plot_relation_scatter,
        ),
        (
            "rolling_correlation_path",
            Path(bucket) / f"{stem}__rolling_correlation.png",
            plot_rolling_correlation_panel,
        ),
        (
            "pca_residual_path",
            Path(bucket) / f"{stem}__pca_residual.png",
            plot_pca_residual_panel,
        ),
    )
    for column, relative_path, writer in panel_specs:
        written = writer(
            output_path=audit_dir / relative_path,
            clean=clean,
            anomalous=anomalous,
            event=payload,
        )
        payload[column] = relative_path.as_posix() if written else ""


def _instance_index(dataset: DatasetIndex) -> dict[tuple[str, str, str], InstanceRecord]:
    return {
        (instance.variant_id, instance.split, instance.instance_id): instance
        for instance in dataset.instances
    }


def _safe_filename(event_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", event_id).strip("_") or "event"


def _build_index(selection: pd.DataFrame) -> str:
    lines = ["# Visual Audit", ""]
    if selection.empty:
        lines.append("No visual audit events were selected.")
        return "\n".join(lines) + "\n"
    for bucket in BUCKETS:
        frame = selection[selection["bucket"] == bucket] if "bucket" in selection else pd.DataFrame()
        lines.extend([f"## {bucket}", ""])
        if frame.empty:
            lines.extend(["No events selected.", ""])
            continue
        for _, row in frame.iterrows():
            plot_path = str(row.get("plot_path", ""))
            status = str(row.get("admission_status", "unknown"))
            event_id = str(row.get("event_id", ""))
            warnings = str(row.get("warning_reasons", ""))
            lines.append(f"### {event_id}")
            lines.append("")
            lines.append(f"- status: `{status}`")
            if warnings:
                lines.append(f"- warnings: `{warnings}`")
            if plot_path:
                lines.append("")
                lines.append(f"![{event_id}]({plot_path})")
            for label, column in (
                ("relation scatter", "relation_scatter_path"),
                ("rolling correlation", "rolling_correlation_path"),
                ("PCA residual", "pca_residual_path"),
            ):
                extra_path = str(row.get(column, ""))
                if extra_path:
                    lines.append("")
                    lines.append(f"- {label}: [{extra_path}]({extra_path})")
            lines.append("")
    return "\n".join(lines)


def _build_html_index(selection: pd.DataFrame) -> str:
    if selection.empty:
        body = "<p>No visual audit events were selected.</p>"
    else:
        sections: list[str] = []
        for bucket in BUCKETS:
            frame = selection[selection["bucket"] == bucket] if "bucket" in selection else pd.DataFrame()
            cards = [_html_card(row) for _, row in frame.iterrows()]
            if not cards:
                cards = ["<p class=\"empty\">No events selected.</p>"]
            sections.append(
                "\n".join(
                    [
                        f"<section><h2>{escape(bucket)}</h2>",
                        "<div class=\"grid\">",
                        *cards,
                        "</div></section>",
                    ]
                )
            )
        body = "\n".join(sections)
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "  <title>Visual Audit</title>",
            "  <style>",
            _html_style(),
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <h1>Visual Audit</h1>",
            body,
            "  </main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _html_card(row: pd.Series) -> str:
    event_id = str(row.get("event_id", ""))
    status = str(row.get("admission_status", "unknown"))
    warnings = str(row.get("warning_reasons", ""))
    variant = str(row.get("variant_id", ""))
    plot_path = str(row.get("plot_path", ""))
    image = (
        f"<a href=\"{escape(plot_path)}\"><img src=\"{escape(plot_path)}\" alt=\"{escape(event_id)}\"></a>"
        if plot_path
        else "<div class=\"missing\">missing plot</div>"
    )
    panel_links = []
    for label, column in (
        ("scatter", "relation_scatter_path"),
        ("rolling corr", "rolling_correlation_path"),
        ("PCA residual", "pca_residual_path"),
    ):
        path = str(row.get(column, ""))
        if path:
            panel_links.append(f"<a href=\"{escape(path)}\">{escape(label)}</a>")
    links = " ".join(panel_links) if panel_links else "<span>standard panels only</span>"
    warning_html = f"<p class=\"reasons\">{escape(warnings)}</p>" if warnings else ""
    return "\n".join(
        [
            "<article class=\"card\">",
            image,
            f"<h3>{escape(event_id)}</h3>",
            f"<p><strong>{escape(status)}</strong> - {escape(variant)}</p>",
            warning_html,
            f"<nav>{links}</nav>",
            "</article>",
        ]
    )


def _html_style() -> str:
    return """
body {
  margin: 0;
  background: #f6f7f9;
  color: #1f2933;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
h1, h2, h3 {
  letter-spacing: 0;
}
section {
  margin: 28px 0;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}
.card {
  background: #ffffff;
  border: 1px solid #d8dee6;
  border-radius: 8px;
  padding: 12px;
  overflow: hidden;
}
.card img {
  width: 100%;
  aspect-ratio: 6 / 5;
  object-fit: contain;
  background: #ffffff;
  border: 1px solid #e5e7eb;
}
.card h3 {
  font-size: 13px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.card p {
  font-size: 13px;
}
.reasons {
  color: #5b6472;
  overflow-wrap: anywhere;
}
nav {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 13px;
}
a {
  color: #2457a6;
}
.missing, .empty {
  color: #697386;
}
""".strip()


def _manifest(selection: pd.DataFrame) -> dict[str, Any]:
    bucket_counts = (
        {str(key): int(value) for key, value in selection["bucket"].value_counts().sort_index().items()}
        if not selection.empty and "bucket" in selection
        else {}
    )
    return {
        "visual_audit_manifest_version": "synthgen.visual_audit.v12.2",
        "index_path": "visual_audit/index.md",
        "html_index_path": "visual_audit/index.html",
        "selection_table": "visual_audit_selection.csv",
        "bucket_counts": bucket_counts,
        "selected_event_count": int(len(selection)),
        "buckets": list(BUCKETS),
        "relation_scatter_count": _non_empty_count(selection, "relation_scatter_path"),
        "rolling_correlation_count": _non_empty_count(selection, "rolling_correlation_path"),
        "pca_residual_count": _non_empty_count(selection, "pca_residual_path"),
    }


def _non_empty_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna("").astype(str).ne("").sum())
