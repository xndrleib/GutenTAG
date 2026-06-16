import math
from pathlib import Path

from gutenTAG.tsgen.capabilities.validation.attribution_rules import (
    AttributionContext,
    attribution_scope,
    build_attribution_row_fields,
    frontier_witness_and_projection,
    min_empirical_p,
    normalized_evidence,
    primary_cause,
)

from tests.capability_execution_fixtures import _event_group, _support_instance_record


def test_frontier_witness_and_projection_supports_legacy_witness_field() -> None:
    witness, projection = frontier_witness_and_projection(
        {"best_detection_witness": "mean_z@0|1"}
    )

    assert witness == "mean_z"
    assert projection == "0|1"


def test_attribution_scope_and_primary_cause_prioritize_boundary_and_shortcut() -> None:
    assert (
        attribution_scope(
            canonical=True,
            forbidden=False,
            boundary_primary=True,
            shortcut_status="unknown",
        )
        == "boundary"
    )
    assert (
        primary_cause(
            detected=True,
            canonical=False,
            canonical_detected=False,
            forbidden=True,
            boundary_primary=False,
            shortcut_status="unknown",
        )
        == "detected_wrong_reason"
    )
    assert (
        primary_cause(
            detected=True,
            canonical=False,
            canonical_detected=True,
            forbidden=True,
            boundary_primary=False,
            shortcut_status="shortcut_dominated",
        )
        == "valid_with_shortcut"
    )


def test_build_attribution_row_fields_projects_identity_detector_and_calibration() -> (
    None
):
    root = Path("/tmp/synth-gen-attribution-rules-test")
    group = _event_group("0", 10, 20)
    instance = _support_instance_record(root, root, (group,))
    context = AttributionContext(
        event_id="event-0",
        instance=instance,
        group=group,
        witness="mean_z",
        projection="0",
        family="location.mean",
        forbidden=False,
        canonical=True,
        boundary_primary=False,
        shortcut_status="unknown",
        detected=True,
        scan_p=0.05,
        candidate_p=0.04,
        canonical_detected=True,
        scan_null_count=19,
        min_empirical_p=min_empirical_p(19),
        evidence=normalized_evidence(0.05, min_empirical_p(19)),
        scope="canonical",
    )

    row = build_attribution_row_fields(
        context,
        {
            "alpha": 0.10,
            "raw_score": 3.0,
            "scan_threshold": 2.5,
            "candidate_null_count": 30,
            "raw_candidate_count": 40,
            "effective_candidate_count": 2.0,
        },
    )

    assert row["event_id"] == "event-0"
    assert row["variant_id"] == "sine__mean__p00"
    assert row["scope"] == "canonical"
    assert row["primary_detection_cause"] == "valid_canonical_detection"
    assert row["scan_null_count"] == 19
    assert row["calibration_resolution_min_p"] == min_empirical_p(19)
    assert math.isclose(float(row["normalized_evidence"]), 1.3010299956639813)
