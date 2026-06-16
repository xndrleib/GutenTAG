import pandas as pd

from gutenTAG.tsgen.capabilities.admission_events import (
    _evaluate_event_evidence,
    _model_detection_summary,
)
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


def test_event_decision_accepts_model_zoo_only_detection_with_warning() -> None:
    decision = _evaluate_event_evidence(
        impl={
            "implementation_validity_status": "valid_candidate",
            "realized_offset": 0.0,
        },
        repair={"repair_status": "repair_effective"},
        arity_row={"canonical_is_observable_at_delta": True},
        detect={"detected": False, "calibration_status": "calibration_ok"},
        attribution={"primary_detection_cause": "observable_not_detected"},
        model={"model_zoo_detected_at_min_alpha": True},
        metadata_row={
            "genotype_id": "genotype:variant",
            "contract_id": "synthgen.contract.pattern.v1",
        },
    )

    assert decision.status == "release_with_warning"
    assert decision.detected is False
    assert decision.detected_for_admission is True
    assert decision.detection_source == "model_zoo"
    assert "model_zoo_only_detection" in decision.warnings
    assert "not_detected_at_min_alpha" in decision.warnings


def test_model_detection_summary_uses_min_alpha_and_counts_models() -> None:
    protocol = CapabilityProtocol(alpha_grid=(0.01, 0.05))
    frame = pd.DataFrame(
        [
            {"event_id": "e1", "alpha": 0.01, "detected": False},
            {"event_id": "e1", "alpha": 0.01, "detected": True},
            {"event_id": "e1", "alpha": 0.05, "detected": True},
            {"event_id": "e2", "alpha": 0.05, "detected": True},
        ]
    )

    summary = _model_detection_summary(frame, protocol)

    assert summary["e1"]["model_zoo_detected_at_min_alpha"] is True
    assert summary["e1"]["model_zoo_detected_model_count"] == 1
    assert "e2" not in summary
