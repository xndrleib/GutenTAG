"""Shared helpers for admission evidence aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .pandas_typing import (
    as_frame,
    as_series,
    column as frame_column,
    frame_groupby,
    numeric_column,
    sorted_frame,
)
from .protocol import CapabilityProtocol


def _index_by_event(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "event_id" not in frame.columns:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for _, row in frame.iterrows():
        rows[str(row["event_id"])] = dict(row)
    return rows


def _min_delta_rows(
    arity: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> dict[str, Mapping[str, Any]]:
    if arity.empty:
        return {}
    delta = _min_delta(protocol)
    frame = as_frame(arity[np.isclose(numeric_column(arity, "delta"), delta)])
    return _index_by_event(frame)


def _alpha_rows(
    frame: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> dict[str, Mapping[str, Any]]:
    if frame.empty or "alpha" not in frame.columns:
        return {}
    alpha = _min_alpha(protocol)
    active = as_frame(frame[np.isclose(numeric_column(frame, "alpha"), alpha)]).copy()
    if "rank_within_event" in active.columns:
        active = sorted_frame(active, ["event_id", "rank_within_event"])
    rows: dict[str, Mapping[str, Any]] = {}
    for event_id, group in frame_groupby(active, "event_id", dropna=False):
        rows[str(event_id)] = dict(group.iloc[0])
    return rows


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
    for value in as_series(frame[column]).fillna("").astype(str):
        values.extend(part for part in value.split("|") if part)
    return sorted(dict.fromkeys(values))


def _reason_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in _collect_reason_values(frame, column):
        counts[reason] = int(
            as_series(frame[column])
            .fillna("")
            .astype(str)
            .str.contains(reason, regex=False)
            .sum()
        )
    return counts


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in as_series(frame[column]).value_counts().sort_index().items()
    }


def _share(frame: pd.DataFrame, statuses: set[str]) -> float:
    if frame.empty:
        return float("nan")
    return float(frame_column(frame, "admission_status").isin(list(statuses)).mean())


def _finite_mean_bool(values: pd.Series) -> float:
    mapped = as_series(
        values.map(lambda value: _bool_value(value, default=False))
    ).astype(float)
    if mapped.empty:
        return float("nan")
    return float(mapped.mean())
