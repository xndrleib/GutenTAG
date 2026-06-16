"""Table contracts for corrected detectability evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import pandas as pd

from .protocol import CapabilityProtocol


def frontier_columns() -> list[str]:
    """Return the corrected-detectability frontier table columns."""

    return common_columns() + [
        "scope",
        "family",
        "witness_or_model",
        "projection",
        "alpha",
        "raw_score",
        "candidate_level_p_value",
        "scan_statistic",
        "scan_level_p_value",
        "scan_threshold",
        "detected",
        "latency",
        "false_alert_count",
        "false_alert_rate",
        "candidate_null_count",
        "scan_null_count",
        "raw_candidate_count",
        "effective_candidate_count",
        "calibration_resolution_min_p",
        "calibration_status",
        "best_detection_witness",
        "best_canonical_witness",
        "best_canonical_family",
        "best_canonical_projection",
        "best_canonical_raw_score",
        "best_canonical_candidate_p_value",
        "best_canonical_scan_statistic",
        "best_canonical_detected",
        "candidate_count",
    ]


def resolution_columns() -> list[str]:
    """Return the corrected-detectability calibration-resolution columns."""

    return [
        "event_id",
        "variant_id",
        "split",
        "instance_id",
        "anomaly_type",
        "alpha",
        "scope",
        "scan_null_count",
        "candidate_null_count",
        "raw_candidate_count",
        "effective_candidate_count",
        "calibration_resolution_min_p",
        "calibration_status",
    ]


def blind_scan_columns() -> list[str]:
    """Return the corrected-detectability blind-scan event table columns."""

    return [
        "event_id",
        "variant_id",
        "split",
        "instance_id",
        "window_start",
        "window_end",
        "overlaps_oracle_event",
        "false_alert",
        "scope",
        "family",
        "witness_or_model",
        "projection",
        "alpha",
        "raw_score",
        "candidate_level_p_value",
        "scan_statistic",
        "scan_level_p_value",
        "scan_threshold",
        "detected",
        "candidate_null_count",
        "scan_null_count",
        "raw_candidate_count",
        "effective_candidate_count",
        "calibration_resolution_min_p",
        "calibration_status",
    ]


def common_columns() -> list[str]:
    """Return columns shared by corrected-detectability oracle rows."""

    return [
        "event_id",
        "variant_id",
        "split",
        "instance_id",
        "base_oscillation",
        "anomaly_type",
        "constraint_tag",
        "semantic_scope",
        "start",
        "end",
        "length",
        "group_channels",
    ]


def ordered_rows(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Project cached rows onto a canonical column order while preserving extras."""

    ordered: list[dict[str, object]] = []
    for row in rows:
        active: dict[str, object] = {column: row.get(column) for column in columns}
        for key, value in row.items():
            if key not in active:
                active[str(key)] = value
        ordered.append(active)
    return tuple(ordered)


def normalize_blind_scan_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a blind-scan frame with canonical columns and boolean fields."""

    if frame.empty:
        return pd.DataFrame(columns=pd.Index(blind_scan_columns()))
    active = frame.copy()
    for column in blind_scan_columns():
        if column not in active.columns:
            active[column] = pd.Series(dtype=object)
    for column in ("overlaps_oracle_event", "false_alert", "detected"):
        active[column] = active[column].map(as_bool)
    return as_frame(active[blind_scan_columns()])


def false_alerts_by_alpha(
    frame: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> dict[float, int]:
    """Count blind-scan false alerts for each configured alpha level."""

    counts = {float(alpha): 0 for alpha in protocol.alpha_grid}
    if frame.empty:
        return counts
    active = normalize_blind_scan_frame(frame)
    false_alerts = active[active["false_alert"].map(as_bool)]
    if false_alerts.empty:
        return counts
    for alpha, count in false_alerts.groupby("alpha").size().items():
        counts[float(cast(Any, alpha))] = int(count)
    return counts


def as_bool(value: object) -> bool:
    """Normalize common serialized boolean values."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


__all__ = [
    "as_bool",
    "blind_scan_columns",
    "common_columns",
    "false_alerts_by_alpha",
    "frontier_columns",
    "normalize_blind_scan_frame",
    "ordered_rows",
    "resolution_columns",
]
from .pandas_typing import as_frame
