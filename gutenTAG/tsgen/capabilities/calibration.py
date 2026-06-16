"""Empirical calibration primitives for corrected detectability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateNull:
    """Candidate-level empirical null distribution."""

    family: str
    witness_or_model: str
    projection_policy: str
    window_length_bin: int
    scores: np.ndarray
    candidate_count: int


@dataclass(frozen=True)
class ScanNull:
    """Scan-level null distribution over aggregated candidate p-values."""

    scope: str
    family: str
    aggregation: str
    window_length_bin: int
    scan_statistics: np.ndarray
    raw_candidate_count: int
    effective_candidate_count: float


class EmpiricalCalibrator:
    """Empirical p-value and threshold helper."""

    def candidate_p_value(self, null: CandidateNull, observed_score: float) -> float:
        values = _finite(null.scores)
        if values.size == 0:
            return float("nan")
        return float((1 + np.sum(values >= float(observed_score))) / (len(values) + 1))

    def scan_p_value(self, null: ScanNull, observed_scan_stat: float) -> float:
        values = _finite(null.scan_statistics)
        if values.size == 0:
            return float("nan")
        return float(
            (1 + np.sum(values >= float(observed_scan_stat))) / (len(values) + 1)
        )

    def scan_threshold(self, null: ScanNull, alpha: float) -> float:
        values = _finite(null.scan_statistics)
        if values.size == 0:
            return float("nan")
        return float(np.quantile(values, 1.0 - float(alpha), method="higher"))


def scan_statistic_from_candidate_p_values(candidate_p_values: np.ndarray) -> float:
    """Return ``-log(min p)`` with numerical guards."""

    values = _finite(candidate_p_values)
    if values.size == 0:
        return float("nan")
    min_p = max(float(np.min(values)), 1e-300)
    return float(-math.log(min_p))


def min_empirical_p(count: int) -> float:
    """Smallest empirical p-value resolvable with a null of ``count`` samples."""

    return float(1.0 / (int(count) + 1))


def calibration_status(
    null_count: int,
    alpha: float,
    minimum_counts: Sequence[tuple[float, int]] | None = None,
) -> str:
    """Classify calibration adequacy for an alpha level."""

    count = int(null_count)
    if count <= 0:
        return "calibration_invalid"
    if min_empirical_p(count) > float(alpha):
        return "calibration_invalid_for_alpha"
    minimum = _recommended_min_count(float(alpha), minimum_counts)
    if count < minimum:
        return "calibration_low_resolution"
    return "calibration_ok"


def _recommended_min_count(
    alpha: float,
    minimum_counts: Sequence[tuple[float, int]] | None = None,
) -> int:
    if minimum_counts is not None:
        for alpha_threshold, count in sorted(
            minimum_counts, key=lambda item: float(item[0])
        ):
            if alpha <= float(alpha_threshold):
                return int(count)
        return 25
    if alpha <= 0.01:
        return 300
    if alpha <= 0.05:
        return 100
    if alpha <= 0.10:
        return 50
    return 25


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]
