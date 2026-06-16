import numpy as np
import pytest

from gutenTAG.tsgen.capabilities.corrected_detectability_candidates import CandidateSpec
from gutenTAG.tsgen.capabilities.corrected_detectability_nulls import (
    build_candidate_null,
    build_scan_null,
    candidate_null_manifest_row,
    scan_null_manifest_row,
)


def test_candidate_null_and_manifest_row_are_stable() -> None:
    candidate = CandidateSpec("mean_z", (0, 2))
    scores = np.asarray([1.0, 3.0, 2.0], dtype=np.float64)

    null = build_candidate_null(candidate=candidate, event_length=12, scores=scores)
    row = candidate_null_manifest_row(
        null_id="variant__len12",
        candidate=candidate,
        null=null,
    )

    assert null.family == "location.mean"
    assert null.witness_or_model == "mean_z"
    assert null.projection_policy == "0|2"
    assert null.window_length_bin == 12
    assert null.candidate_count == 3
    assert row == {
        "candidate_null_id": "variant__len12__mean_z@0|2",
        "family": "location.mean",
        "witness_or_model": "mean_z",
        "projection_policy": "0|2",
        "window_length_bin": 12,
        "candidate_count": 3,
        "score_min": 1.0,
        "score_median": 2.0,
        "score_max": 3.0,
    }


def test_scan_null_and_manifest_row_are_stable() -> None:
    scan_statistics = np.asarray([0.5, 1.5, 2.5], dtype=np.float64)

    null = build_scan_null(
        event_length=16,
        scan_statistics=scan_statistics,
        raw_candidate_count=12,
    )
    row = scan_null_manifest_row(null_id="variant__len16", null=null)

    assert null.scope == "oracle_window"
    assert null.aggregation == "min_candidate_p"
    assert null.effective_candidate_count == pytest.approx(4.0)
    assert row == {
        "scan_null_id": "variant__len16",
        "scope": "oracle_window",
        "family": "all",
        "aggregation": "min_candidate_p",
        "window_length_bin": 16,
        "scan_null_count": 3,
        "raw_candidate_count": 12,
        "effective_candidate_count": 4.0,
        "scan_stat_min": 0.5,
        "scan_stat_median": 1.5,
        "scan_stat_max": 2.5,
    }


def test_empty_null_manifest_rows_use_nan_bounds() -> None:
    candidate = CandidateSpec("mean_z", (0,))
    candidate_null = build_candidate_null(
        candidate=candidate,
        event_length=8,
        scores=np.asarray([], dtype=np.float64),
    )
    scan_null = build_scan_null(
        event_length=8,
        scan_statistics=np.asarray([], dtype=np.float64),
        raw_candidate_count=0,
    )

    candidate_row = candidate_null_manifest_row(
        null_id="empty",
        candidate=candidate,
        null=candidate_null,
    )
    scan_row = scan_null_manifest_row(null_id="empty", null=scan_null)

    assert np.isnan(candidate_row["score_min"])
    assert np.isnan(candidate_row["score_median"])
    assert np.isnan(candidate_row["score_max"])
    assert np.isnan(scan_row["scan_stat_min"])
    assert np.isnan(scan_row["scan_stat_median"])
    assert np.isnan(scan_row["scan_stat_max"])
