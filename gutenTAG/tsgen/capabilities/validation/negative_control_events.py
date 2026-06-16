"""Event-level negative-control scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..dataset import EventGroup, InstanceRecord
from ..ontology import witness_requires_projection_size
from ..protocol import CapabilityProtocol
from ..witnesses import detection_window_scores
from .negative_control_rows import as_float, frontier_negative_control_rows
from .negative_control_support import negative_control_event_context
from .negative_control_types import (
    MARGINAL_WITNESSES,
    NegativeControlEventContext,
    NegativeControlScores,
)


def event_negative_controls(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    frontier: pd.DataFrame,
    boundary_row: dict[str, object],
    protocol: CapabilityProtocol,
) -> list[dict[str, object]]:
    """Return negative-control rows for one event group."""

    context = negative_control_event_context(
        instance=instance,
        group=group,
        protocol=protocol,
    )
    scores = _event_negative_control_scores(
        group=group,
        clean=clean,
        anomalous=anomalous,
        boundary_row=boundary_row,
        protocol=protocol,
        context=context,
    )
    return frontier_negative_control_rows(
        instance=instance,
        group=group,
        frontier=frontier,
        context=context,
        scores=scores,
    )


def _event_negative_control_scores(
    *,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    boundary_row: dict[str, object],
    protocol: CapabilityProtocol,
    context: NegativeControlEventContext,
) -> NegativeControlScores:
    clean_score, clean_witness = _best_score(
        series=clean,
        start=group.start,
        end=group.end,
        subsets=context.subsets,
        witnesses=context.clean_control_witnesses,
        protocol=protocol,
    )
    clean_wrong_support_score, wrong_support_score, wrong_support_witness = (
        _wrong_support_scores(
            clean=clean,
            anomalous=anomalous,
            protocol=protocol,
            context=context,
        )
    )
    clean_marginal_score, marginal_score, marginal_witness = _wrong_witness_scores(
        group=group,
        clean=clean,
        anomalous=anomalous,
        protocol=protocol,
        context=context,
    )
    boundary_ratio, boundary_triggered = _boundary_control_scores(boundary_row)
    return NegativeControlScores(
        clean_score=clean_score,
        clean_witness=clean_witness,
        clean_wrong_support_score=clean_wrong_support_score,
        wrong_support_score=wrong_support_score,
        wrong_support_witness=wrong_support_witness,
        clean_marginal_score=clean_marginal_score,
        marginal_score=marginal_score,
        marginal_witness=marginal_witness,
        boundary_ratio=boundary_ratio,
        boundary_triggered=boundary_triggered,
    )


def _wrong_support_scores(
    *,
    clean: np.ndarray,
    anomalous: np.ndarray,
    protocol: CapabilityProtocol,
    context: NegativeControlEventContext,
) -> tuple[float, float, str]:
    clean_score, _ = _best_score(
        series=clean,
        start=context.shifted_start,
        end=context.shifted_end,
        subsets=context.subsets,
        witnesses=context.clean_control_witnesses,
        protocol=protocol,
    )
    anomalous_score, witness = _best_score(
        series=anomalous,
        start=context.shifted_start,
        end=context.shifted_end,
        subsets=context.subsets,
        witnesses=context.clean_control_witnesses,
        protocol=protocol,
    )
    return clean_score, anomalous_score, witness


def _wrong_witness_scores(
    *,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    protocol: CapabilityProtocol,
    context: NegativeControlEventContext,
) -> tuple[float, float, str]:
    clean_score, _ = _best_score(
        series=clean,
        start=group.start,
        end=group.end,
        subsets=context.marginal_subsets,
        witnesses=MARGINAL_WITNESSES,
        protocol=protocol,
    )
    anomalous_score, witness = _best_score(
        series=anomalous,
        start=group.start,
        end=group.end,
        subsets=context.marginal_subsets,
        witnesses=MARGINAL_WITNESSES,
        protocol=protocol,
    )
    return clean_score, anomalous_score, witness


def _boundary_control_scores(boundary_row: dict[str, object]) -> tuple[float, bool]:
    boundary_ratio = as_float(boundary_row.get("boundary_to_canonical_ratio", math.nan))
    boundary_triggered = bool(
        boundary_row.get("boundary_primary_detection_cause", False)
    )
    return boundary_ratio, boundary_triggered


def _best_score(
    *,
    series: np.ndarray,
    start: int,
    end: int,
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
    protocol: CapabilityProtocol,
) -> tuple[float, str]:
    best_score = math.nan
    best_witness = "none"
    for subset in subsets:
        if not subset:
            continue
        scores = detection_window_scores(
            series=series,
            start=start,
            end=end,
            channels=subset,
            context_multiplier=protocol.context_window_multiplier,
            min_context_points=protocol.min_context_points,
        )
        for witness in witnesses:
            if witness not in scores or witness_requires_projection_size(witness) > len(
                subset
            ):
                continue
            score = float(scores[witness])
            if not math.isfinite(best_score) or score > best_score:
                best_score = score
                best_witness = (
                    f"{witness}@{'|'.join(str(channel) for channel in subset)}"
                )
    return best_score, best_witness


__all__ = ["event_negative_controls"]
