"""Null distribution objects and manifests for corrected detectability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .calibration import CandidateNull, ScanNull
from .corrected_detectability_candidates import CandidateSpec, format_projection
from .numerics import finite_float


@dataclass(frozen=True)
class CorrectedNullBundle:
    """Reusable nulls for one variant/window/candidate setup."""

    candidate_nulls: Mapping[str, CandidateNull]
    scan_null: ScanNull
    candidate_manifest: tuple[dict[str, object], ...]
    scan_manifest: dict[str, object]


def build_candidate_null(
    *,
    candidate: CandidateSpec,
    event_length: int,
    scores: np.ndarray,
) -> CandidateNull:
    """Build a candidate-level empirical null distribution."""

    active_scores = np.asarray(scores, dtype=np.float64)
    return CandidateNull(
        family=candidate.family,
        witness_or_model=candidate.witness,
        projection_policy=format_projection(candidate.projection),
        window_length_bin=int(event_length),
        scores=active_scores,
        candidate_count=int(active_scores.size),
    )


def candidate_null_manifest_row(
    *,
    null_id: str,
    candidate: CandidateSpec,
    null: CandidateNull,
) -> dict[str, object]:
    """Return one candidate-null manifest row."""

    scores = np.asarray(null.scores, dtype=np.float64)
    return {
        "candidate_null_id": f"{null_id}__{candidate.key}",
        "family": null.family,
        "witness_or_model": null.witness_or_model,
        "projection_policy": null.projection_policy,
        "window_length_bin": null.window_length_bin,
        "candidate_count": null.candidate_count,
        "score_min": (
            finite_float(np.min(scores), default=math.nan) if scores.size else math.nan
        ),
        "score_median": (
            finite_float(np.median(scores), default=math.nan)
            if scores.size
            else math.nan
        ),
        "score_max": (
            finite_float(np.max(scores), default=math.nan) if scores.size else math.nan
        ),
    }


def build_scan_null(
    *,
    event_length: int,
    scan_statistics: np.ndarray,
    raw_candidate_count: int,
) -> ScanNull:
    """Build the scan-level null distribution for one event length."""

    scan_arr = np.asarray(scan_statistics, dtype=np.float64)
    effective_count = float(raw_candidate_count) / max(float(scan_arr.size), 1.0)
    return ScanNull(
        scope="oracle_window",
        family="all",
        aggregation="min_candidate_p",
        window_length_bin=int(event_length),
        scan_statistics=scan_arr,
        raw_candidate_count=int(raw_candidate_count),
        effective_candidate_count=finite_float(effective_count, default=math.nan),
    )


def scan_null_manifest_row(
    *,
    null_id: str,
    null: ScanNull,
) -> dict[str, object]:
    """Return one scan-null manifest row."""

    scan_arr = np.asarray(null.scan_statistics, dtype=np.float64)
    return {
        "scan_null_id": null_id,
        "scope": null.scope,
        "family": null.family,
        "aggregation": null.aggregation,
        "window_length_bin": null.window_length_bin,
        "scan_null_count": int(null.scan_statistics.size),
        "raw_candidate_count": int(null.raw_candidate_count),
        "effective_candidate_count": finite_float(
            null.effective_candidate_count,
            default=math.nan,
        ),
        "scan_stat_min": (
            finite_float(np.min(scan_arr), default=math.nan)
            if scan_arr.size
            else math.nan
        ),
        "scan_stat_median": (
            finite_float(np.median(scan_arr), default=math.nan)
            if scan_arr.size
            else math.nan
        ),
        "scan_stat_max": (
            finite_float(np.max(scan_arr), default=math.nan)
            if scan_arr.size
            else math.nan
        ),
    }


__all__ = [
    "CorrectedNullBundle",
    "build_candidate_null",
    "build_scan_null",
    "candidate_null_manifest_row",
    "scan_null_manifest_row",
]
