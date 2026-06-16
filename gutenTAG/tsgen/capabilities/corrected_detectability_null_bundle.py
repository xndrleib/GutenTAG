"""Clean null-bundle assembly for corrected detectability."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .array_store import ArrayStore
from .calibration import (
    CandidateNull,
    EmpiricalCalibrator,
    scan_statistic_from_candidate_p_values,
)
from .corrected_detectability_candidates import CandidateSpec
from .corrected_detectability_nulls import (
    CorrectedNullBundle,
    build_candidate_null,
    build_scan_null,
    candidate_null_manifest_row,
    scan_null_manifest_row,
)
from .dataset import InstanceRecord
from .protocol import CapabilityProtocol
from .scan_scores import CleanWindowScoreBlock, CleanWindowScoreCache
from .windows import WindowLibrary


def build_null_bundle(
    *,
    clean_instances: Sequence[InstanceRecord],
    clean_cache: dict[str, np.ndarray],
    score_cache: CleanWindowScoreCache,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    event_length: int,
    candidates: Sequence[CandidateSpec],
    protocol: CapabilityProtocol,
    null_id: str,
    calibrator: EmpiricalCalibrator,
) -> CorrectedNullBundle:
    """Build candidate and scan nulls for one corrected-detectability event key."""

    candidate_scores: dict[str, list[float]] = {
        candidate.key: [] for candidate in candidates
    }
    raw_candidate_count = 0
    blocks: dict[tuple[int, ...], CleanWindowScoreBlock] = {}
    for candidate in candidates:
        if candidate.projection not in blocks:
            blocks[candidate.projection] = score_cache.get(
                clean_instances=clean_instances,
                clean_cache=clean_cache,
                arrays=arrays,
                windows=windows,
                event_length=int(event_length),
                subset=candidate.projection,
                protocol=protocol,
            )
        block = blocks[candidate.projection]
        for values in block.scores_by_witness.get(candidate.witness, ()):
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            raw_candidate_count += int(finite.size)
            candidate_scores[candidate.key].extend(float(value) for value in finite)
    candidate_nulls: dict[str, CandidateNull] = {}
    candidate_manifest: list[dict[str, object]] = []
    for candidate in candidates:
        scores = np.asarray(candidate_scores[candidate.key], dtype=np.float64)
        null = build_candidate_null(
            candidate=candidate,
            event_length=int(event_length),
            scores=scores,
        )
        candidate_nulls[candidate.key] = null
        candidate_manifest.append(
            candidate_null_manifest_row(
                null_id=null_id,
                candidate=candidate,
                null=null,
            )
        )
    scan_stats: list[float] = []
    for scores in clean_scan_score_rows(candidates, blocks):
        p_values = []
        for candidate_key, score in scores.items():
            p_values.append(
                calibrator.candidate_p_value(candidate_nulls[candidate_key], score)
            )
        scan_stats.append(
            scan_statistic_from_candidate_p_values(
                np.asarray(p_values, dtype=np.float64)
            )
        )
    scan_arr = np.asarray(scan_stats, dtype=np.float64)
    scan_null = build_scan_null(
        event_length=int(event_length),
        scan_statistics=scan_arr,
        raw_candidate_count=int(raw_candidate_count),
    )
    scan_manifest = scan_null_manifest_row(null_id=null_id, null=scan_null)
    return CorrectedNullBundle(
        candidate_nulls=candidate_nulls,
        scan_null=scan_null,
        candidate_manifest=tuple(candidate_manifest),
        scan_manifest=scan_manifest,
    )


def clean_scan_score_rows(
    candidates: Sequence[CandidateSpec],
    blocks: Mapping[tuple[int, ...], CleanWindowScoreBlock],
) -> list[dict[str, float]]:
    """Return clean-window score rows aligned across candidate projections."""

    if not candidates:
        return []
    first = blocks.get(candidates[0].projection)
    if first is None:
        return []
    rows: list[dict[str, float]] = []
    for instance_index, window_count in enumerate(first.window_counts):
        for window_index in range(int(window_count)):
            scores: dict[str, float] = {}
            for candidate in candidates:
                block = blocks.get(candidate.projection)
                if block is None or instance_index >= len(block.window_counts):
                    continue
                values_by_instance = block.scores_by_witness.get(candidate.witness, ())
                if instance_index >= len(values_by_instance):
                    continue
                values = values_by_instance[instance_index]
                if window_index >= len(values):
                    continue
                score = float(values[window_index])
                if math.isfinite(score):
                    scores[candidate.key] = score
            if scores:
                rows.append(scores)
    return rows


__all__ = [
    "build_null_bundle",
    "clean_scan_score_rows",
]
