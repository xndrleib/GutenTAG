"""Blind-scan score blocks for corrected detectability."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .calibration import (
    EmpiricalCalibrator,
    scan_statistic_from_candidate_p_values,
)
from .corrected_detectability_candidates import (
    CandidateSpec,
    format_projection,
    spec_by_key,
)
from .corrected_detectability_nulls import CorrectedNullBundle
from .numerics import contiguous_windows, finite_float
from .protocol import CapabilityProtocol
from .windows import WindowLibrary, WindowSpec
from .witnesses import detection_window_scores


@dataclass(frozen=True)
class BlindScanScoreBlock:
    """Reusable blind-scan scores for one instance/candidate/null setup."""

    windows: np.ndarray
    best_keys: tuple[str, ...]
    best_families: tuple[str, ...]
    best_witnesses: tuple[str, ...]
    best_projections: tuple[str, ...]
    raw_scores: np.ndarray
    candidate_p_values: np.ndarray
    scan_statistics: np.ndarray
    scan_p_values: np.ndarray
    window_count_total: int


def blind_scan_score_block(
    *,
    series: np.ndarray,
    scan_length: int,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
) -> BlindScanScoreBlock:
    """Compute blind-scan statistics over all candidate windows."""

    candidate_windows = np.asarray(
        scan_windows(series.shape[0], scan_length, protocol, windows),
        dtype=np.int64,
    )
    best_keys: list[str] = []
    best_families: list[str] = []
    best_witnesses: list[str] = []
    best_projections: list[str] = []
    raw_scores: list[float] = []
    candidate_p_values: list[float] = []
    scan_statistics: list[float] = []
    scan_p_values: list[float] = []
    retained_windows: list[tuple[int, int]] = []
    for start, end in candidate_windows:
        scores = scores_for_window(
            series=series,
            start=int(start),
            end=int(end),
            candidates=candidates,
            protocol=protocol,
        )
        p_values = {
            key: calibrator.candidate_p_value(nulls.candidate_nulls[key], score)
            for key, score in scores.items()
            if key in nulls.candidate_nulls
        }
        if not p_values:
            continue
        best_key = min(p_values, key=lambda key: p_values[key])
        best_spec = spec_by_key(candidates, best_key)
        scan_stat = scan_statistic_from_candidate_p_values(
            np.asarray(list(p_values.values()), dtype=np.float64)
        )
        retained_windows.append((int(start), int(end)))
        best_keys.append(best_key)
        best_families.append(best_spec.family if best_spec is not None else "unknown")
        best_witnesses.append(best_spec.witness if best_spec is not None else "none")
        best_projections.append(
            format_projection(best_spec.projection) if best_spec is not None else ""
        )
        raw_scores.append(
            finite_float(scores.get(best_key, math.nan), default=math.nan)
        )
        candidate_p_values.append(
            finite_float(p_values.get(best_key, math.nan), default=math.nan)
        )
        scan_statistics.append(finite_float(scan_stat, default=math.nan))
        scan_p_values.append(
            finite_float(
                calibrator.scan_p_value(nulls.scan_null, scan_stat),
                default=math.nan,
            )
        )
    return BlindScanScoreBlock(
        windows=np.asarray(retained_windows, dtype=np.int64).reshape((-1, 2)),
        best_keys=tuple(best_keys),
        best_families=tuple(best_families),
        best_witnesses=tuple(best_witnesses),
        best_projections=tuple(best_projections),
        raw_scores=np.asarray(raw_scores, dtype=np.float64),
        candidate_p_values=np.asarray(candidate_p_values, dtype=np.float64),
        scan_statistics=np.asarray(scan_statistics, dtype=np.float64),
        scan_p_values=np.asarray(scan_p_values, dtype=np.float64),
        window_count_total=int(candidate_windows.shape[0]),
    )


def scores_for_window(
    *,
    series: np.ndarray,
    start: int,
    end: int,
    candidates: Sequence[CandidateSpec],
    protocol: CapabilityProtocol,
) -> dict[str, float]:
    """Score all corrected-detectability candidates on one window."""

    by_projection: dict[tuple[int, ...], dict[str, float]] = {}
    for candidate in candidates:
        if candidate.projection not in by_projection:
            by_projection[candidate.projection] = detection_window_scores(
                series=series,
                start=start,
                end=end,
                channels=candidate.projection,
                context_multiplier=protocol.context_window_multiplier,
                min_context_points=protocol.min_context_points,
            )
    result: dict[str, float] = {}
    for candidate in candidates:
        scores = by_projection.get(candidate.projection, {})
        if candidate.witness in scores:
            result[candidate.key] = finite_float(scores[candidate.witness])
    return result


def scan_windows(
    series_length: int,
    event_length: int,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> np.ndarray | list[tuple[int, int]]:
    """Return deterministic scan windows for corrected detectability."""

    if windows is None:
        return contiguous_windows(
            series_length,
            int(event_length),
            max_windows=protocol.max_scan_windows_per_length,
            stride_fraction=protocol.clean_window_stride_fraction,
        )
    return windows.get(
        WindowSpec(
            series_length=series_length,
            window_length=int(event_length),
            max_windows=protocol.max_scan_windows_per_length,
            stride_fraction=protocol.clean_window_stride_fraction,
        )
    )


__all__ = [
    "BlindScanScoreBlock",
    "blind_scan_score_block",
    "scan_windows",
    "scores_for_window",
]
