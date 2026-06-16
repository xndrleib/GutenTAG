"""Visual-audit index and manifest rendering."""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

BUCKETS: tuple[str, ...] = (
    "strongest",
    "weakest",
    "borderline",
    "failed",
    "shortcut_dominated",
    "boundary_suspected",
)


def build_markdown_index(selection: pd.DataFrame) -> str:
    """Render the visual-audit Markdown index."""

    lines = ["# Visual Audit", ""]
    if selection.empty:
        lines.append("No visual audit events were selected.")
        return "\n".join(lines) + "\n"
    for bucket in BUCKETS:
        frame = (
            selection[selection["bucket"] == bucket]
            if "bucket" in selection
            else pd.DataFrame()
        )
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


def build_html_index(selection: pd.DataFrame) -> str:
    """Render the visual-audit HTML gallery."""

    if selection.empty:
        body = "<p>No visual audit events were selected.</p>"
    else:
        sections: list[str] = []
        for bucket in BUCKETS:
            frame = (
                selection[selection["bucket"] == bucket]
                if "bucket" in selection
                else pd.DataFrame()
            )
            cards = [_html_card(row) for _, row in frame.iterrows()]
            if not cards:
                cards = ['<p class="empty">No events selected.</p>']
            sections.append(
                "\n".join(
                    [
                        f"<section><h2>{escape(bucket)}</h2>",
                        '<div class="grid">',
                        *cards,
                        "</div></section>",
                    ]
                )
            )
        body = "\n".join(sections)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
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


def build_manifest(selection: pd.DataFrame) -> dict[str, Any]:
    """Return the visual-audit manifest payload."""

    bucket_counts = (
        {
            str(key): int(value)
            for key, value in selection["bucket"].value_counts().sort_index().items()
        }
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
        "rolling_correlation_count": _non_empty_count(
            selection, "rolling_correlation_path"
        ),
        "pca_residual_count": _non_empty_count(selection, "pca_residual_path"),
    }


def _html_card(row: pd.Series) -> str:
    event_id = str(row.get("event_id", ""))
    status = str(row.get("admission_status", "unknown"))
    warnings = str(row.get("warning_reasons", ""))
    variant = str(row.get("variant_id", ""))
    plot_path = str(row.get("plot_path", ""))
    image = (
        f'<a href="{escape(plot_path)}"><img src="{escape(plot_path)}" alt="{escape(event_id)}"></a>'
        if plot_path
        else '<div class="missing">missing plot</div>'
    )
    panel_links = []
    for label, column in (
        ("scatter", "relation_scatter_path"),
        ("rolling corr", "rolling_correlation_path"),
        ("PCA residual", "pca_residual_path"),
    ):
        path = str(row.get(column, ""))
        if path:
            panel_links.append(f'<a href="{escape(path)}">{escape(label)}</a>')
    links = (
        " ".join(panel_links) if panel_links else "<span>standard panels only</span>"
    )
    warning_html = f'<p class="reasons">{escape(warnings)}</p>' if warnings else ""
    return "\n".join(
        [
            '<article class="card">',
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


def _non_empty_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna("").astype(str).ne("").sum())


__all__ = [
    "BUCKETS",
    "build_html_index",
    "build_manifest",
    "build_markdown_index",
]
