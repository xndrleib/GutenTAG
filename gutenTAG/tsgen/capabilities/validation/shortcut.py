"""Shortcut-dominance audit for capability observations."""

from __future__ import annotations

import math

import pandas as pd

from ..dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid
from ..numerics import finite_float


RELATION_SCOPES = {
    "relation",
    "regime_relation",
    "multichannel_relation",
    "collective",
}


def compute_shortcut_audit(
    dataset: DatasetIndex,
    observability: pd.DataFrame,
) -> pd.DataFrame:
    """Compare canonical relation evidence with univariate shortcut evidence."""

    rows: list[dict[str, object]] = []
    grouped = observability.groupby("event_id") if not observability.empty else {}
    for instance in dataset.instances:
        for group in instance.event_groups:
            uid = event_uid(instance, group)
            frame = grouped.get_group(uid) if not observability.empty and uid in grouped.groups else pd.DataFrame()
            rows.append(_shortcut_row(instance, group, frame))
    return pd.DataFrame(rows)


def _shortcut_row(
    instance: InstanceRecord,
    group: EventGroup,
    frame: pd.DataFrame,
) -> dict[str, object]:
    relation_like = _is_relation_like(group)
    best_canonical_relation = _max_score(
        frame,
        lambda row: bool(row.get("canonical_witness", False))
        and int(row.get("projection_size", 0)) >= 2,
    )
    best_univariate = _max_score(frame, lambda row: int(row.get("projection_size", 0)) == 1)
    best_boundary = math.nan
    ratio = best_canonical_relation / max(best_univariate, 1e-12) if math.isfinite(best_canonical_relation) else math.nan
    location_shortcut = _best_family(frame, {"mean_delta"})
    scale_shortcut = _best_family(frame, {"variance_delta"})
    energy_shortcut = _best_family(frame, {"energy_delta"})
    shortcut_dominated = bool(relation_like and math.isfinite(ratio) and ratio < 0.75 and best_univariate > 0.0)
    status = (
        "shortcut_dominated"
        if shortcut_dominated
        else "valid_with_shortcut"
        if relation_like and best_univariate > 0.0
        else "valid_no_shortcut"
    )
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "best_canonical_relation_witness": _best_witness(
            frame,
            lambda row: bool(row.get("canonical_witness", False))
            and int(row.get("projection_size", 0)) >= 2,
        ),
        "best_canonical_relation_score": finite_float(best_canonical_relation, default=math.nan),
        "best_univariate_witness": _best_witness(frame, lambda row: int(row.get("projection_size", 0)) == 1),
        "best_univariate_score": finite_float(best_univariate, default=math.nan),
        "best_boundary_witness": "",
        "best_boundary_score": finite_float(best_boundary, default=math.nan),
        "canonical_vs_univariate_ratio": finite_float(ratio, default=math.nan),
        "canonical_arity": 2 if relation_like else 1,
        "unrestricted_arity": 1 if best_univariate > 0.0 else math.nan,
        "location_shortcut": finite_float(location_shortcut, default=math.nan),
        "scale_shortcut": finite_float(scale_shortcut, default=math.nan),
        "energy_shortcut": finite_float(energy_shortcut, default=math.nan),
        "boundary_shortcut": finite_float(best_boundary, default=math.nan),
        "shortcut_status": status,
    }


def _is_relation_like(group: EventGroup) -> bool:
    if len(group.group_channels) >= 2 or len(group.context_channels) >= 2:
        return True
    scope = str(group.semantic_scope)
    return scope in RELATION_SCOPES or "relation" in scope or "dependence" in group.constraint_tag


def _max_score(frame: pd.DataFrame, predicate) -> float:
    if frame.empty:
        return math.nan
    values = [
        float(row["distance_value"])
        for _, row in frame.iterrows()
        if predicate(row) and pd.notna(row.get("distance_value"))
    ]
    return max(values) if values else math.nan


def _best_witness(frame: pd.DataFrame, predicate) -> str:
    if frame.empty:
        return ""
    best_value = -math.inf
    best_name = ""
    for _, row in frame.iterrows():
        if not predicate(row) or pd.isna(row.get("distance_value")):
            continue
        value = float(row["distance_value"])
        if value > best_value:
            best_value = value
            best_name = str(row.get("witness_family", ""))
    return best_name


def _best_family(frame: pd.DataFrame, witnesses: set[str]) -> float:
    return _max_score(frame, lambda row: str(row.get("witness_family", "")) in witnesses)
