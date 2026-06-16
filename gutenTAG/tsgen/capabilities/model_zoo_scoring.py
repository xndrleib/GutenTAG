"""Model-zoo null calibration and scoring helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .calibration import (
    CandidateNull,
    EmpiricalCalibrator,
    ScanNull,
    scan_statistic_from_candidate_p_values,
)
from .dataset import EventGroup
from .models import CapabilityModel
from .numerics import contiguous_windows, finite_float
from .protocol import CapabilityProtocol
from .windows import WindowLibrary, WindowSpec


@dataclass(frozen=True)
class ModelNull:
    """Corrected clean nulls for one model/window length."""

    candidate_null: CandidateNull
    scan_null: ScanNull
    raw_candidate_count: int
    effective_candidate_count: float


def model_null(
    *,
    model: CapabilityModel,
    clean_instances: Sequence[np.ndarray],
    event_length: int,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> ModelNull:
    """Build clean candidate and scan nulls for one fitted model."""

    scores: list[float] = []
    for clean in clean_instances:
        scan_windows = scan_window_specs(
            clean.shape[0], event_length, protocol, windows
        )
        values = model.score_windows(clean, np.asarray(scan_windows, dtype=int))
        scores.extend(float(value) for value in values if math.isfinite(float(value)))
    arr = np.asarray(scores, dtype=np.float64)
    candidate_null = CandidateNull(
        family=model.family,
        witness_or_model=model.model_id,
        projection_policy="model_window",
        window_length_bin=int(event_length),
        scores=arr,
        candidate_count=int(arr.size),
    )
    calibrator = EmpiricalCalibrator()
    scan_statistics = np.asarray(
        [
            scan_statistic_from_candidate_p_values(
                np.asarray(
                    [calibrator.candidate_p_value(candidate_null, score)],
                    dtype=np.float64,
                )
            )
            for score in arr
        ],
        dtype=np.float64,
    )
    scan_null = ScanNull(
        scope="model_zoo",
        family=model.family,
        aggregation="min_candidate_p",
        window_length_bin=int(event_length),
        scan_statistics=scan_statistics,
        raw_candidate_count=int(arr.size),
        effective_candidate_count=1.0 if arr.size else 0.0,
    )
    return ModelNull(
        candidate_null=candidate_null,
        scan_null=scan_null,
        raw_candidate_count=int(arr.size),
        effective_candidate_count=finite_float(
            scan_null.effective_candidate_count,
            default=math.nan,
        ),
    )


def context_score(
    model: CapabilityModel,
    series: np.ndarray,
    group: EventGroup,
    length: int,
) -> float:
    """Score a context window surrounding an event group."""

    width = max(1, group.length)
    start = max(0, group.start - width)
    end = min(int(length), group.end + width)
    if end <= start:
        return math.nan
    return float(model.score_windows(series, np.asarray([[start, end]], dtype=int))[0])


def model_candidate_p_value(null: ModelNull, raw_score: float) -> float:
    """Return candidate p-value for one raw model score."""

    return EmpiricalCalibrator().candidate_p_value(
        null.candidate_null,
        float(raw_score),
    )


def model_scan_statistic(null: ModelNull, raw_score: float) -> float:
    """Return scan statistic for one raw model score."""

    candidate_p = model_candidate_p_value(null, raw_score)
    return scan_statistic_from_candidate_p_values(
        np.asarray([candidate_p], dtype=np.float64)
    )


def model_scan_p_value(null: ModelNull, raw_score: float) -> float:
    """Return scan p-value for one raw model score."""

    scan_statistic = model_scan_statistic(null, raw_score)
    return EmpiricalCalibrator().scan_p_value(null.scan_null, scan_statistic)


def false_alert_rate(
    *,
    model: CapabilityModel,
    series: np.ndarray,
    group: EventGroup,
    event_length: int,
    null: ModelNull,
    threshold: float,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> tuple[int, float]:
    """Return false-alert count/rate outside the target event window."""

    if not math.isfinite(threshold):
        return 0, math.nan
    scan_windows = np.asarray(
        scan_window_specs(series.shape[0], event_length, protocol, windows),
        dtype=int,
    )
    if scan_windows.size == 0:
        return 0, math.nan
    scores = model.score_windows(series, scan_windows)
    false_alerts = 0
    eligible = 0
    for (start, end), score in zip(scan_windows, scores):
        if overlaps(int(start), int(end), group.start, group.end):
            continue
        eligible += 1
        scan_stat = model_scan_statistic(null, float(score))
        if math.isfinite(scan_stat) and scan_stat >= threshold:
            false_alerts += 1
    return false_alerts, false_alerts / max(float(eligible), 1.0)


def scan_window_specs(
    series_length: int,
    event_length: int,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> np.ndarray | list[tuple[int, int]]:
    """Return scan windows from a cache or contiguous-window fallback."""

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


def overlaps(start: int, end: int, target_start: int, target_end: int) -> bool:
    """Return whether two half-open intervals overlap."""

    return int(start) < int(target_end) and int(end) > int(target_start)


__all__ = [
    "ModelNull",
    "context_score",
    "false_alert_rate",
    "model_candidate_p_value",
    "model_null",
    "model_scan_p_value",
    "model_scan_statistic",
    "overlaps",
    "scan_window_specs",
]
