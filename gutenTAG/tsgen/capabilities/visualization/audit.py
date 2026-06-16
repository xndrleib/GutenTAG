"""Visual audit selection and artifact generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..array_store import ArrayStore
from ..dataset import DatasetIndex, InstanceRecord, read_timeseries_csv
from ..pandas_typing import (
    as_frame,
    column as frame_column,
    frame_groupby,
    row_mapping,
    sorted_frame,
)
from ..protocol import CapabilityProtocol
from .audit_index import (
    BUCKETS,
    build_html_index,
    build_manifest,
    build_markdown_index,
)
from .boundary import boundary_annotation
from .generic import plot_event_diagnostic
from .mode import is_mode_event
from .relation import (
    is_relation_event,
    plot_pca_residual_panel,
    plot_relation_scatter,
    plot_rolling_correlation_panel,
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
        admission_events=(
            admission_events if admission_events is not None else pd.DataFrame()
        ),
        implementation_validity=(
            implementation_validity
            if implementation_validity is not None
            else pd.DataFrame()
        ),
        boundary_audit=boundary_audit if boundary_audit is not None else pd.DataFrame(),
        shortcut_audit=shortcut_audit if shortcut_audit is not None else pd.DataFrame(),
    )
    selection = _select_bucket_rows(
        candidates, protocol, max_events_per_bucket=max_events_per_bucket
    )
    if not selection.empty:
        selection = _write_plots(
            dataset=dataset,
            audit_dir=audit_dir,
            selection=selection,
            detector_attribution=(
                detector_attribution
                if detector_attribution is not None
                else pd.DataFrame()
            ),
            arrays=arrays,
        )
    index_markdown = build_markdown_index(selection)
    index_html = build_html_index(selection)
    (audit_dir / "index.md").write_text(index_markdown, encoding="utf-8")
    (audit_dir / "index.html").write_text(index_html, encoding="utf-8")
    manifest = build_manifest(selection)
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
    for table in (
        admission_events,
        implementation_validity,
        boundary_audit,
        shortcut_audit,
    ):
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
        selected = _bucket_candidates(candidates, bucket, protocol).head(
            int(max_events_per_bucket)
        )
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            row_payload = row_mapping(row)
            payload = dict(row_payload)
            payload["bucket"] = bucket
            payload["bucket_rank"] = rank
            payload["visual_priority_score"] = _priority_score(
                row_payload, bucket, protocol
            )
            payload["relation_review"] = is_relation_event(payload)
            payload["mode_review"] = is_mode_event(payload)
            payload["boundary_annotation"] = boundary_annotation(payload)
            rows.append(payload)
    return pd.DataFrame(rows)


def _bucket_candidates(
    candidates: pd.DataFrame, bucket: str, protocol: CapabilityProtocol
) -> pd.DataFrame:
    frame = candidates.copy()
    if bucket == "strongest":
        return sorted_frame(frame, "best_canonical_distance").iloc[::-1]
    if bucket == "weakest":
        return sorted_frame(frame, "best_canonical_distance")
    if bucket == "borderline":
        frame["_borderline"] = frame.apply(
            lambda row: _borderline_score(row, protocol), axis=1
        )
        return sorted_frame(frame, "_borderline")
    if bucket == "failed":
        mask = _string_column(frame, "admission_status").isin(
            ["reject", "needs_repair", "debug"]
        )
        selected = as_frame(frame[mask]) if bool(mask.any()) else frame
        return as_frame(
            selected.sort_values(
                ["admission_status", "best_canonical_distance"], ascending=[True, True]
            )
        )
    if bucket == "shortcut_dominated":
        mask = _string_column(frame, "shortcut_status").str.contains(
            "shortcut", na=False
        ) | _string_column(frame, "warning_reasons").str.contains(
            "shortcut|wrong_reason", na=False
        )
        selected = as_frame(frame[mask]) if bool(mask.any()) else frame.iloc[0:0]
        return sorted_frame(selected, "best_canonical_distance").iloc[::-1]
    if bucket == "boundary_suspected":
        mask = (
            _string_column(frame, "boundary_status").str.contains(
                "boundary_primary|suspected", na=False
            )
            | _string_column(frame, "warning_reasons").str.contains(
                "boundary", na=False
            )
            | _string_column(frame, "provisional_gate_hits").str.contains(
                "boundary", na=False
            )
        )
        selected = as_frame(frame[mask]) if bool(mask.any()) else frame.iloc[0:0]
        return sorted_frame(selected, "best_canonical_distance").iloc[::-1]
    return frame


def _borderline_score(row: pd.Series, protocol: CapabilityProtocol) -> float:
    p_value = row.get("scan_level_p_value")
    try:
        if bool(pd.notna(p_value)):
            return abs(
                _float_value(p_value, default=math.nan)
                - float(min(protocol.alpha_grid))
            )
    except (TypeError, ValueError):
        pass
    status = str(row.get("admission_status", ""))
    if status == "release_with_warning":
        return 0.0
    if status == "debug":
        return 0.1
    return _float_value(row.get("best_canonical_distance", 0.0))


def _priority_score(
    row: Mapping[str, object], bucket: str, protocol: CapabilityProtocol
) -> float:
    if bucket == "borderline":
        return float(_borderline_score(pd.Series(row), protocol))
    try:
        return _float_value(row.get("best_canonical_distance", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _string_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame_column(frame, name).fillna("").astype(str)


def _write_plots(
    *,
    dataset: DatasetIndex,
    audit_dir: Path,
    selection: pd.DataFrame,
    detector_attribution: pd.DataFrame,
    arrays: ArrayStore | None,
) -> pd.DataFrame:
    instances = _instance_index(dataset)
    attribution_groups = (
        frame_groupby(detector_attribution, "event_id")
        if not detector_attribution.empty
        else None
    )
    rows: list[dict[str, object]] = []
    for _, row in selection.iterrows():
        payload = row_mapping(row)
        instance = instances.get(
            (
                str(payload["variant_id"]),
                str(payload["split"]),
                str(payload["instance_id"]),
            )
        )
        if instance is None:
            payload["plot_path"] = ""
            payload["plot_status"] = "missing_instance"
            rows.append(payload)
            continue
        clean = (
            arrays.get(instance, "clean")
            if arrays is not None
            else read_timeseries_csv(instance.clean_path)
        )
        anomalous = (
            arrays.get(instance, "anomalous")
            if arrays is not None
            else read_timeseries_csv(instance.anomalous_path)
        )
        event_id = str(payload["event_id"])
        attribution = (
            as_frame(attribution_groups.get_group(event_id))
            if attribution_groups is not None and event_id in attribution_groups.groups
            else pd.DataFrame()
        )
        plot_path = Path(str(payload["bucket"])) / f"{_safe_filename(event_id)}.png"
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
    if not bool(payload.get("relation_review")) and not bool(
        payload.get("mode_review")
    ):
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


def _instance_index(
    dataset: DatasetIndex,
) -> dict[tuple[str, str, str], InstanceRecord]:
    return {
        (instance.variant_id, instance.split, instance.instance_id): instance
        for instance in dataset.instances
    }


def _safe_filename(event_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", event_id).strip("_") or "event"


def _float_value(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, str, np.integer, np.floating)):
        try:
            return float(value)
        except ValueError:
            return default
    return default
