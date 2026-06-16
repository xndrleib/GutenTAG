from pathlib import Path

import numpy as np

from gutenTAG.tsgen.capabilities.corrected_detectability_candidates import CandidateSpec
from gutenTAG.tsgen.capabilities.corrected_detectability_partitions import (
    CorrectedPartitionResult,
    calibration_null_instances,
    clean_scan_score_rows,
    partition_from_payload,
    partition_to_payload,
)
from gutenTAG.tsgen.capabilities.dataset import InstanceRecord
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol
from gutenTAG.tsgen.capabilities.scan_scores import CleanWindowScoreBlock


def _instance(split: str) -> InstanceRecord:
    root = Path("/tmp/synth-gen-corrected-partitions-test")
    return InstanceRecord(
        dataset_root=root,
        variant_id="variant",
        split=split,
        instance_id=f"{split}_000",
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


def test_calibration_null_instances_prefers_configured_split() -> None:
    protocol = CapabilityProtocol(calibration_split="calibration")
    train = _instance("train")
    calibration = _instance("calibration")

    assert calibration_null_instances((train, calibration), protocol) == (calibration,)


def test_calibration_null_instances_falls_back_to_variant_instances() -> None:
    protocol = CapabilityProtocol(calibration_split="missing")
    train = _instance("train")

    assert calibration_null_instances((train,), protocol) == (train,)


def test_partition_payload_roundtrip_preserves_manifests_and_projects_rows() -> None:
    result = CorrectedPartitionResult(
        rows=({"event_id": "e1", "variant_id": "v1", "extra": "kept"},),
        blind_rows=({"event_id": "b1", "alpha": 0.5},),
        resolution_rows=({"event_id": "r1", "alpha": 0.5},),
        candidate_manifest_rows={"cand": {"candidate_null_id": "cand"}},
        scan_manifest_rows={"scan": {"scan_null_id": "scan"}},
    )

    restored = partition_from_payload(partition_to_payload(result))

    assert restored.rows[0]["event_id"] == "e1"
    assert restored.rows[0]["extra"] == "kept"
    assert restored.blind_rows == result.blind_rows
    assert restored.candidate_manifest_rows == result.candidate_manifest_rows
    assert restored.scan_manifest_rows == result.scan_manifest_rows


def test_clean_scan_score_rows_aligns_candidate_windows() -> None:
    candidates = (
        CandidateSpec("mean_z", (0,)),
        CandidateSpec("local_energy_z", (0,)),
    )
    block = CleanWindowScoreBlock(
        instance_keys=("instance",),
        window_counts=(3,),
        scores_by_witness={
            "mean_z": (np.asarray([1.0, np.nan, 3.0], dtype=np.float64),),
            "local_energy_z": (np.asarray([4.0, 5.0, 6.0], dtype=np.float64),),
        },
    )

    rows = clean_scan_score_rows(candidates, {(0,): block})

    assert rows == [
        {"mean_z@0": 1.0, "local_energy_z@0": 4.0},
        {"local_energy_z@0": 5.0},
        {"mean_z@0": 3.0, "local_energy_z@0": 6.0},
    ]
