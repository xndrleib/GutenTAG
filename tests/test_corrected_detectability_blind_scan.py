from pathlib import Path

import numpy as np

from gutenTAG.tsgen.capabilities.calibration import (
    CandidateNull,
    EmpiricalCalibrator,
    ScanNull,
)
from gutenTAG.tsgen.capabilities.corrected_detectability_blind_scan import (
    blind_scan_rows,
)
from gutenTAG.tsgen.capabilities.corrected_detectability_blind_scores import (
    blind_scan_score_block,
    scan_windows,
    scores_for_window,
)
from gutenTAG.tsgen.capabilities.corrected_detectability_candidates import CandidateSpec
from gutenTAG.tsgen.capabilities.corrected_detectability_nulls import (
    CorrectedNullBundle,
)
from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


def _protocol() -> CapabilityProtocol:
    return CapabilityProtocol(
        alpha_grid=(0.50,),
        detection_witnesses=("mean_z",),
        max_projection_size=1,
        max_scan_windows_per_length=5,
        clean_window_stride_fraction=0.50,
        calibration_min_clean_scan_count_for_alpha=((0.50, 2),),
    )


def _candidate() -> CandidateSpec:
    return CandidateSpec("mean_z", (0,))


def _nulls() -> CorrectedNullBundle:
    return CorrectedNullBundle(
        candidate_nulls={
            "mean_z@0": CandidateNull(
                family="location.mean",
                witness_or_model="mean_z",
                projection_policy="0",
                window_length_bin=8,
                scores=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                candidate_count=3,
            )
        },
        scan_null=ScanNull(
            scope="oracle_window",
            family="all",
            aggregation="min_candidate_p",
            window_length_bin=8,
            scan_statistics=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            raw_candidate_count=3,
            effective_candidate_count=1.0,
        ),
        candidate_manifest=(),
        scan_manifest={},
    )


def _instance() -> InstanceRecord:
    root = Path("/tmp/synth-gen-blind-scan-test")
    return InstanceRecord(
        dataset_root=root,
        variant_id="variant",
        split="train",
        instance_id="instance_000",
        instance_dir=root,
        clean_path=root / "clean.csv",
        anomalous_path=root / "anomalous.csv",
        events_path=root / "events.json",
        summary_path=root / "instance_summary.json",
        base_oscillation="sine",
        anomaly_type="mean",
        channels=1,
        length=40,
        event_groups=(),
    )


def _group() -> EventGroup:
    return EventGroup(
        group_id="event-0",
        start=10,
        end=18,
        source_start=10,
        source_end=18,
        anomaly_type="mean",
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


def _series() -> np.ndarray:
    series = np.zeros((40, 1), dtype=np.float64)
    series[10:18, 0] = 4.0
    return series


def test_scores_for_window_scores_candidate_keys() -> None:
    scores = scores_for_window(
        series=_series(),
        start=10,
        end=18,
        candidates=(_candidate(),),
        protocol=_protocol(),
    )

    assert set(scores) == {"mean_z@0"}
    assert scores["mean_z@0"] > 0.0


def test_blind_scan_score_block_records_best_candidate_metadata() -> None:
    block = blind_scan_score_block(
        series=_series(),
        scan_length=8,
        candidates=(_candidate(),),
        nulls=_nulls(),
        protocol=_protocol(),
        windows=None,
        calibrator=EmpiricalCalibrator(),
    )

    assert block.window_count_total == len(list(scan_windows(40, 8, _protocol(), None)))
    assert block.best_keys
    assert set(block.best_keys) == {"mean_z@0"}
    assert set(block.best_witnesses) == {"mean_z"}
    assert block.windows.shape[1] == 2


def test_blind_scan_rows_reports_detected_windows_and_false_alert_counts() -> None:
    rows, false_alerts, window_count = blind_scan_rows(
        instance=_instance(),
        group=_group(),
        series=_series(),
        candidates=(_candidate(),),
        nulls=_nulls(),
        protocol=_protocol(),
        windows=None,
        calibrator=EmpiricalCalibrator(),
    )

    assert window_count == len(list(scan_windows(40, 8, _protocol(), None)))
    assert rows
    assert all(row["detected"] is True for row in rows)
    assert any(row["overlaps_oracle_event"] for row in rows)
    assert false_alerts[0.50] == sum(1 for row in rows if row["false_alert"])
