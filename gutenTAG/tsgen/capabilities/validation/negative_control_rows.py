"""Negative-control row construction and status semantics."""

from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd

from ..dataset import EventGroup, InstanceRecord, event_uid
from ..numerics import finite_float
from .negative_control_types import (
    NegativeControlEventContext,
    NegativeControlScores,
)


def frontier_negative_control_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    frontier: pd.DataFrame,
    context: NegativeControlEventContext,
    scores: NegativeControlScores,
) -> list[dict[str, object]]:
    """Build negative-control rows for every frontier alpha for one event."""

    rows: list[dict[str, object]] = []
    for _, frontier_row in frontier.iterrows():
        alpha = as_float(frontier_row.get("alpha", math.nan))
        threshold = as_float(frontier_row.get("scan_threshold", math.nan))
        rows.extend(
            _negative_control_rows_for_alpha(
                instance=instance,
                group=group,
                alpha=alpha,
                threshold=threshold,
                relation_like=context.relation_like,
                scores=scores,
            )
        )
    return rows


def as_float(value: object) -> float:
    """Coerce finite numeric-like values to float, otherwise ``nan``."""

    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _negative_control_rows_for_alpha(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    alpha: float,
    threshold: float,
    relation_like: bool,
    scores: NegativeControlScores,
) -> list[dict[str, object]]:
    return [
        _clean_vs_clean_row(instance, group, alpha, threshold, scores),
        _wrong_support_row(instance, group, alpha, threshold, scores),
        _wrong_witness_row(
            instance,
            group,
            alpha,
            threshold,
            relation_like,
            scores,
        ),
        _boundary_only_row(instance, group, alpha, scores),
    ]


def _clean_vs_clean_row(
    instance: InstanceRecord,
    group: EventGroup,
    alpha: float,
    threshold: float,
    scores: NegativeControlScores,
) -> dict[str, object]:
    return _control_row(
        instance,
        group,
        alpha=alpha,
        control_type="clean_vs_clean",
        control_score=scores.clean_score,
        reference_threshold=threshold,
        witness=scores.clean_witness,
        triggered=_triggered(scores.clean_score, threshold),
        applicability="applicable",
    )


def _wrong_support_row(
    instance: InstanceRecord,
    group: EventGroup,
    alpha: float,
    threshold: float,
    scores: NegativeControlScores,
) -> dict[str, object]:
    return _control_row(
        instance,
        group,
        alpha=alpha,
        control_type="wrong_support_control",
        control_score=scores.wrong_support_score,
        reference_threshold=threshold,
        witness=scores.wrong_support_witness,
        triggered=_triggered_against_baseline(
            scores.wrong_support_score,
            threshold,
            scores.clean_wrong_support_score,
        ),
        applicability="applicable",
        baseline_score=scores.clean_wrong_support_score,
    )


def _wrong_witness_row(
    instance: InstanceRecord,
    group: EventGroup,
    alpha: float,
    threshold: float,
    relation_like: bool,
    scores: NegativeControlScores,
) -> dict[str, object]:
    return _control_row(
        instance,
        group,
        alpha=alpha,
        control_type="wrong_witness_control",
        control_score=scores.marginal_score,
        reference_threshold=threshold,
        witness=scores.marginal_witness,
        triggered=_wrong_witness_triggered(relation_like, threshold, scores),
        applicability="applicable" if relation_like else "not_applicable",
        baseline_score=scores.clean_marginal_score,
    )


def _wrong_witness_triggered(
    relation_like: bool,
    threshold: float,
    scores: NegativeControlScores,
) -> bool:
    if not relation_like:
        return False
    return _triggered_against_baseline(
        scores.marginal_score,
        threshold,
        scores.clean_marginal_score,
    )


def _boundary_only_row(
    instance: InstanceRecord,
    group: EventGroup,
    alpha: float,
    scores: NegativeControlScores,
) -> dict[str, object]:
    return _control_row(
        instance,
        group,
        alpha=alpha,
        control_type="boundary_only_control",
        control_score=scores.boundary_ratio,
        reference_threshold=1.5,
        witness="boundary_to_canonical_ratio",
        triggered=scores.boundary_triggered,
        applicability="applicable",
    )


def _control_row(
    instance: InstanceRecord,
    group: EventGroup,
    *,
    alpha: float,
    control_type: str,
    control_score: float,
    reference_threshold: float,
    witness: str,
    triggered: bool,
    applicability: str,
    baseline_score: float = math.nan,
) -> dict[str, object]:
    status = (
        "not_applicable"
        if applicability == "not_applicable"
        else "control_failed" if triggered else "control_passed"
    )
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "alpha": finite_float(alpha, default=math.nan),
        "control_type": control_type,
        "control_score": finite_float(control_score, default=math.nan),
        "baseline_control_score": finite_float(baseline_score, default=math.nan),
        "reference_threshold": finite_float(reference_threshold, default=math.nan),
        "control_witness": witness,
        "control_triggered": bool(triggered),
        "expected_negative": True,
        "control_status": status,
        "applicability": applicability,
    }


def _triggered(score: float, threshold: float) -> bool:
    return math.isfinite(score) and math.isfinite(threshold) and score >= threshold


def _triggered_against_baseline(
    score: float, threshold: float, baseline: float
) -> bool:
    if not _triggered(score, threshold):
        return False
    if not math.isfinite(baseline):
        return True
    baseline_floor = max(
        float(threshold),
        float(baseline) * 1.25,
        float(baseline) + 0.25,
    )
    return float(score) >= baseline_floor


__all__ = ["as_float", "frontier_negative_control_rows"]
