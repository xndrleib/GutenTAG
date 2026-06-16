from types import SimpleNamespace

import numpy as np
import pytest

from gutenTAG.tsgen.capabilities.calibration import (
    CandidateNull,
    EmpiricalCalibrator,
    ScanNull,
)
from gutenTAG.tsgen.capabilities.corrected_detectability_candidates import CandidateSpec
from gutenTAG.tsgen.capabilities.corrected_detectability_oracle import (
    calibration_resolution_row,
    candidate_p_values_for_scores,
    oracle_frontier_row,
    summarize_oracle_detection,
)
from gutenTAG.tsgen.capabilities.dataset import EventGroup
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


def _group(anomaly_type: str = "mean") -> EventGroup:
    return EventGroup(
        group_id="event-0",
        start=10,
        end=20,
        source_start=10,
        source_end=20,
        anomaly_type=anomaly_type,
        constraint_tag="location.mean",
        repair_operator="none",
        semantic_scope="univariate",
        intervention_channels=(0,),
        context_channels=(),
        group_channels=(0,),
        primary_channels=(0,),
        event_scope="group",
        purity_hint="clean",
        raw_events=(),
    )


def _nulls() -> SimpleNamespace:
    return SimpleNamespace(
        candidate_nulls={
            "mean_z@0": CandidateNull(
                family="location.mean",
                witness_or_model="mean_z",
                projection_policy="0",
                window_length_bin=8,
                scores=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
                candidate_count=3,
            ),
            "local_energy_z@0": CandidateNull(
                family="local.energy",
                witness_or_model="local_energy_z",
                projection_policy="0",
                window_length_bin=8,
                scores=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                candidate_count=3,
            ),
        },
        scan_null=ScanNull(
            scope="oracle_window",
            family="all",
            aggregation="min_candidate_p",
            window_length_bin=8,
            scan_statistics=np.asarray([0.1, 1.0, 2.0, 4.0], dtype=np.float64),
            raw_candidate_count=12,
            effective_candidate_count=3.0,
        ),
    )


def test_oracle_detection_summary_separates_best_and_canonical_candidates() -> None:
    candidates = (
        CandidateSpec("mean_z", (0,)),
        CandidateSpec("local_energy_z", (0,)),
    )
    event_scores = {"mean_z@0": 0.5, "local_energy_z@0": 2.0}
    nulls = _nulls()
    calibrator = EmpiricalCalibrator()

    p_values = candidate_p_values_for_scores(
        event_scores=event_scores,
        nulls=nulls,
        calibrator=calibrator,
    )
    summary = summarize_oracle_detection(
        group=_group("mean"),
        candidates=candidates,
        event_scores=event_scores,
        candidate_p_values=p_values,
        nulls=nulls,
        calibrator=calibrator,
    )

    assert p_values["mean_z@0"] == pytest.approx(0.75)
    assert p_values["local_energy_z@0"] == pytest.approx(0.25)
    assert summary.best_key == "local_energy_z@0"
    assert summary.canonical_key == "mean_z@0"
    assert summary.best_raw_score == pytest.approx(2.0)
    assert summary.canonical_raw_score == pytest.approx(0.5)
    assert summary.candidate_count == 2


def test_oracle_frontier_row_and_resolution_projection_are_stable() -> None:
    candidates = (
        CandidateSpec("mean_z", (0,)),
        CandidateSpec("local_energy_z", (0,)),
    )
    event_scores = {"mean_z@0": 0.5, "local_energy_z@0": 2.0}
    nulls = _nulls()
    calibrator = EmpiricalCalibrator()
    p_values = candidate_p_values_for_scores(
        event_scores=event_scores,
        nulls=nulls,
        calibrator=calibrator,
    )
    summary = summarize_oracle_detection(
        group=_group("mean"),
        candidates=candidates,
        event_scores=event_scores,
        candidate_p_values=p_values,
        nulls=nulls,
        calibrator=calibrator,
    )

    row = oracle_frontier_row(
        common={
            "event_id": "event-1",
            "variant_id": "variant-a",
            "split": "train",
            "instance_id": "instance_000",
            "base_oscillation": "sine",
            "anomaly_type": "mean",
            "constraint_tag": "location.mean",
            "semantic_scope": "univariate",
            "start": 10,
            "end": 20,
            "length": 10,
            "group_channels": "0",
        },
        summary=summary,
        nulls=nulls,
        alpha=0.5,
        threshold=1.0,
        false_alert_count=2,
        blind_window_count=10,
        protocol=CapabilityProtocol(alpha_grid=(0.5,)),
    )
    resolution = calibration_resolution_row(row)

    assert row["family"] == "local.energy"
    assert row["witness_or_model"] == "local_energy_z"
    assert row["detected"] is True
    assert row["best_canonical_witness"] == "mean_z@0"
    assert row["best_canonical_detected"] is False
    assert row["false_alert_rate"] == pytest.approx(0.2)
    assert row["candidate_null_count"] == 3
    assert row["scan_null_count"] == 4
    assert resolution == {
        "event_id": "event-1",
        "variant_id": "variant-a",
        "split": "train",
        "instance_id": "instance_000",
        "anomaly_type": "mean",
        "alpha": 0.5,
        "scope": "oracle_window",
        "scan_null_count": 4,
        "candidate_null_count": 3,
        "raw_candidate_count": 12,
        "effective_candidate_count": 3.0,
        "calibration_resolution_min_p": pytest.approx(0.2),
        "calibration_status": "calibration_low_resolution",
    }
