"""Blind-scan scoring and row construction for corrected detectability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .cache import CacheStore
from .calibration import (
    EmpiricalCalibrator,
    calibration_status,
    min_empirical_p,
)
from .corrected_detectability_blind_scores import (
    BlindScanScoreBlock,
    blind_scan_score_block,
    scan_windows,
    scores_for_window,
)
from .corrected_detectability_candidates import CandidateSpec
from .corrected_detectability_nulls import CorrectedNullBundle
from .corrected_detectability_tables import (
    blind_scan_columns,
    false_alerts_by_alpha as count_false_alerts_by_alpha,
    normalize_blind_scan_frame,
)
from .dataset import EventGroup, InstanceRecord, event_uid
from .numerics import finite_float
from .protocol import CapabilityProtocol, window_length_bin
from .windows import WindowLibrary

__all__ = [
    "BlindScanScoreBlock",
    "cached_blind_scan_rows",
    "count_false_alerts_by_alpha",
    "scan_windows",
    "scores_for_window",
]


@dataclass(frozen=True)
class _BlindScanCacheContext:
    """Cache keys and scan geometry for one blind-scan event."""

    scan_length: int
    blind_window_count: int
    score_cache_key: tuple[object, ...]
    partition_id: str


def cached_blind_scan_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    null_key: tuple[object, ...],
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    cache: CacheStore | None,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None = None,
) -> tuple[list[dict[str, object]], dict[float, int], int]:
    """Return cached blind-scan rows and false-alert counts for one event."""

    context = _blind_scan_cache_context(
        instance=instance,
        group=group,
        series=series,
        candidates=candidates,
        null_key=null_key,
        protocol=protocol,
        windows=windows,
    )
    if cache is None:
        return _uncached_blind_scan_rows(
            instance=instance,
            group=group,
            series=series,
            candidates=candidates,
            nulls=nulls,
            protocol=protocol,
            windows=windows,
            calibrator=calibrator,
            blind_scan_cache=blind_scan_cache,
            score_cache_key=context.score_cache_key,
        )
    frame = _cached_blind_scan_frame(
        cache=cache,
        context=context,
        instance=instance,
        group=group,
        series=series,
        candidates=candidates,
        nulls=nulls,
        null_key=null_key,
        protocol=protocol,
        windows=windows,
        calibrator=calibrator,
        blind_scan_cache=blind_scan_cache,
    )
    frame = normalize_blind_scan_frame(frame)
    return (
        frame.to_dict("records"),
        count_false_alerts_by_alpha(frame, protocol),
        context.blind_window_count,
    )


def _blind_scan_cache_context(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    null_key: tuple[object, ...],
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> _BlindScanCacheContext:
    scan_length = window_length_bin(max(1, group.length), protocol)
    blind_window_count = len(
        list(scan_windows(series.shape[0], scan_length, protocol, windows))
    )
    return _BlindScanCacheContext(
        scan_length=int(scan_length),
        blind_window_count=int(blind_window_count),
        score_cache_key=(
            _instance_key(instance),
            int(scan_length),
            tuple(candidate.key for candidate in candidates),
            repr(null_key),
            int(series.shape[0]),
        ),
        partition_id=event_uid(instance, group),
    )


def _uncached_blind_scan_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None,
    score_cache_key: tuple[object, ...],
) -> tuple[list[dict[str, object]], dict[float, int], int]:
    return blind_scan_rows(
        instance=instance,
        group=group,
        series=series,
        candidates=candidates,
        nulls=nulls,
        protocol=protocol,
        windows=windows,
        calibrator=calibrator,
        blind_scan_cache=blind_scan_cache,
        score_cache_key=score_cache_key,
    )


def _cached_blind_scan_frame(
    *,
    cache: CacheStore,
    context: _BlindScanCacheContext,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    null_key: tuple[object, ...],
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None,
) -> pd.DataFrame:
    fingerprint = cache.fingerprint(
        profile_name="blind_scan_events",
        partition_id=context.partition_id,
        extra={
            "null_key": repr(null_key),
            "candidate_keys": [candidate.key for candidate in candidates],
            "event_length": int(max(1, group.length)),
            "window_length_bin": int(context.scan_length),
            "window_count": int(context.blind_window_count),
        },
    )

    def compute() -> pd.DataFrame:
        rows, _, _ = blind_scan_rows(
            instance=instance,
            group=group,
            series=series,
            candidates=candidates,
            nulls=nulls,
            protocol=protocol,
            windows=windows,
            calibrator=calibrator,
            blind_scan_cache=blind_scan_cache,
            score_cache_key=context.score_cache_key,
        )
        return pd.DataFrame(rows, columns=pd.Index(blind_scan_columns()))

    return cache.get_or_compute_table(
        profile_name="blind_scan_events",
        partition_id=context.partition_id,
        fingerprint=fingerprint,
        compute=compute,
    )


def blind_scan_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    series: np.ndarray,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None = None,
    score_cache_key: tuple[object, ...] | None = None,
) -> tuple[list[dict[str, object]], dict[float, int], int]:
    """Build detected blind-scan event rows for one oracle event."""

    scan_length = window_length_bin(max(1, group.length), protocol)
    block = cached_blind_scan_score_block(
        series=series,
        scan_length=scan_length,
        candidates=candidates,
        nulls=nulls,
        protocol=protocol,
        windows=windows,
        calibrator=calibrator,
        blind_scan_cache=blind_scan_cache,
        score_cache_key=score_cache_key,
    )
    rows: list[dict[str, object]] = []
    false_alerts_by_alpha = {float(alpha): 0 for alpha in protocol.alpha_grid}
    thresholds = {
        float(alpha): calibrator.scan_threshold(nulls.scan_null, float(alpha))
        for alpha in protocol.alpha_grid
    }
    for index, (start, end) in enumerate(block.windows):
        scan_stat = float(block.scan_statistics[index])
        if not math.isfinite(scan_stat):
            continue
        overlaps_event = _overlaps(int(start), int(end), group.start, group.end)
        for alpha in protocol.alpha_grid:
            alpha_value = float(alpha)
            threshold = thresholds[alpha_value]
            detected = (
                bool(scan_stat >= threshold) if math.isfinite(threshold) else False
            )
            if not detected:
                continue
            false_alert = not overlaps_event
            if false_alert:
                false_alerts_by_alpha[alpha_value] += 1
            rows.append(
                _blind_scan_row(
                    instance=instance,
                    group=group,
                    block=block,
                    nulls=nulls,
                    protocol=protocol,
                    index=index,
                    start=int(start),
                    end=int(end),
                    overlaps_event=overlaps_event,
                    false_alert=false_alert,
                    alpha_value=alpha_value,
                    scan_stat=scan_stat,
                    threshold=threshold,
                )
            )
    return rows, false_alerts_by_alpha, block.window_count_total


def cached_blind_scan_score_block(
    *,
    series: np.ndarray,
    scan_length: int,
    candidates: Sequence[CandidateSpec],
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
    calibrator: EmpiricalCalibrator,
    blind_scan_cache: dict[tuple[object, ...], BlindScanScoreBlock] | None,
    score_cache_key: tuple[object, ...] | None,
) -> BlindScanScoreBlock:
    """Return a memoized blind-scan score block."""

    key = score_cache_key or (
        int(series.shape[0]),
        int(scan_length),
        tuple(candidate.key for candidate in candidates),
        id(nulls),
    )
    if blind_scan_cache is not None and key in blind_scan_cache:
        return blind_scan_cache[key]
    block = blind_scan_score_block(
        series=series,
        scan_length=scan_length,
        candidates=candidates,
        nulls=nulls,
        protocol=protocol,
        windows=windows,
        calibrator=calibrator,
    )
    if blind_scan_cache is not None:
        blind_scan_cache[key] = block
    return block


def _blind_scan_row(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    block: BlindScanScoreBlock,
    nulls: CorrectedNullBundle,
    protocol: CapabilityProtocol,
    index: int,
    start: int,
    end: int,
    overlaps_event: bool,
    false_alert: bool,
    alpha_value: float,
    scan_stat: float,
    threshold: float,
) -> dict[str, object]:
    best_key = block.best_keys[index]
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "window_start": int(start),
        "window_end": int(end),
        "overlaps_oracle_event": overlaps_event,
        "false_alert": false_alert,
        "scope": "blind_scan",
        "family": block.best_families[index],
        "witness_or_model": block.best_witnesses[index],
        "projection": block.best_projections[index],
        "alpha": alpha_value,
        "raw_score": finite_float(block.raw_scores[index], default=math.nan),
        "candidate_level_p_value": finite_float(
            block.candidate_p_values[index],
            default=math.nan,
        ),
        "scan_statistic": finite_float(scan_stat, default=math.nan),
        "scan_level_p_value": finite_float(
            block.scan_p_values[index],
            default=math.nan,
        ),
        "scan_threshold": finite_float(threshold, default=math.nan),
        "detected": True,
        "candidate_null_count": int(nulls.candidate_nulls[best_key].scores.size),
        "scan_null_count": int(nulls.scan_null.scan_statistics.size),
        "raw_candidate_count": int(nulls.scan_null.raw_candidate_count),
        "effective_candidate_count": finite_float(
            nulls.scan_null.effective_candidate_count,
            default=math.nan,
        ),
        "calibration_resolution_min_p": finite_float(
            min_empirical_p(len(nulls.scan_null.scan_statistics)),
            default=math.nan,
        ),
        "calibration_status": calibration_status(
            len(nulls.scan_null.scan_statistics),
            alpha_value,
            protocol.calibration_min_clean_scan_count_for_alpha,
        ),
    }


def _instance_key(instance: InstanceRecord) -> str:
    return f"{instance.variant_id}/{instance.split}/{instance.instance_id}"


def _overlaps(start: int, end: int, target_start: int, target_end: int) -> bool:
    return int(start) < int(target_end) and int(end) > int(target_start)
