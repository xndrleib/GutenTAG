from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gutenTAG.tsgen.capabilities.calibration import CandidateNull, ScanNull
from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord
from gutenTAG.tsgen.capabilities.model_zoo_rows import (
    model_event_score_row,
    model_frontier_row,
)
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


def _instance() -> InstanceRecord:
    root = Path("/tmp/synth-gen-model-zoo-rows-test")
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
        channels=2,
        length=64,
        event_groups=(),
    )


def _group() -> EventGroup:
    return EventGroup(
        group_id="0",
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


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="toy_model",
        family="structural.toy",
    )


def _null() -> SimpleNamespace:
    return SimpleNamespace(
        candidate_null=CandidateNull(
            family="structural.toy",
            witness_or_model="toy_model",
            projection_policy="model_window",
            window_length_bin=8,
            scores=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            candidate_count=3,
        ),
        scan_null=ScanNull(
            scope="model_zoo",
            family="structural.toy",
            aggregation="min_candidate_p",
            window_length_bin=8,
            scan_statistics=np.asarray([0.1, 0.5, 1.0], dtype=np.float64),
            raw_candidate_count=3,
            effective_candidate_count=1.0,
        ),
        raw_candidate_count=3,
        effective_candidate_count=1.0,
    )


def test_model_event_score_row_preserves_table_contract() -> None:
    row = model_event_score_row(
        instance=_instance(),
        group=_group(),
        model=_model(),
        projection_label="0|1",
        projection_size=2,
        raw_score=2.0,
        candidate_p_value=0.25,
        scan_statistic=1.25,
        scan_p_value=0.50,
        null=_null(),
        intervention_score=2.0,
        context_score=0.5,
        metadata_hash="abc123",
    )

    assert row["event_id"] == "variant/train/instance_000/g0"
    assert row["model_id"] == "toy_model"
    assert row["model_projection"] == "0|1"
    assert row["model_projection_size"] == 2
    assert row["fit_split"] == "all_clean"
    assert row["calibration_split"] == "all_clean"
    assert row["candidate_null_count"] == 3
    assert row["scan_null_count"] == 3
    assert row["model_metadata_hash"] == "abc123"


def test_model_frontier_row_preserves_calibration_fields() -> None:
    row = model_frontier_row(
        instance=_instance(),
        group=_group(),
        model=_model(),
        projection_label="0",
        projection_size=1,
        alpha=0.50,
        raw_score=2.0,
        candidate_p_value=0.25,
        scan_statistic=1.25,
        scan_p_value=0.50,
        scan_threshold=1.0,
        detected=True,
        false_alert_count=2,
        false_alert_rate=0.25,
        null=_null(),
        protocol=CapabilityProtocol(
            alpha_grid=(0.50,),
            calibration_min_clean_scan_count_for_alpha=((0.50, 2),),
        ),
        intervention_score=2.0,
        context_score=0.5,
        metadata_hash="abc123",
    )

    assert row["alpha"] == 0.50
    assert row["detected"] is True
    assert row["latency"] == 0
    assert row["false_alert_count"] == 2
    assert row["false_alert_rate"] == pytest.approx(0.25)
    assert row["calibration_resolution_min_p"] == pytest.approx(0.25)
    assert row["calibration_status"] == "calibration_ok"
