from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol
from gutenTAG.tsgen.capabilities.validation.negative_control_events import (
    event_negative_controls,
)
from gutenTAG.tsgen.capabilities.validation.negative_control_support import (
    is_relation_like,
    shifted_support,
)


def _instance(
    groups: tuple[EventGroup, ...],
    *,
    channels: int,
    length: int,
) -> InstanceRecord:
    root = Path("/tmp/synth-gen-negative-control-events-test")
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
        channels=channels,
        length=length,
        event_groups=groups,
    )


def _group(
    *,
    group_id: str = "g0",
    start: int = 10,
    end: int = 20,
    anomaly_type: str = "mean",
    constraint_tag: str = "location.mean",
    semantic_scope: str = "univariate",
    group_channels: tuple[int, ...] = (0,),
) -> EventGroup:
    return EventGroup(
        group_id=group_id,
        start=start,
        end=end,
        source_start=start,
        source_end=end,
        anomaly_type=anomaly_type,
        constraint_tag=constraint_tag,
        repair_operator="none",
        semantic_scope=semantic_scope,
        intervention_channels=group_channels,
        context_channels=(),
        group_channels=group_channels,
        primary_channels=group_channels,
        event_scope="unit_test",
        purity_hint="clean",
        raw_events=(),
    )


def _frontier() -> pd.DataFrame:
    return pd.DataFrame([{"alpha": 0.10, "scan_threshold": 1.0}])


def test_event_negative_controls_builds_boundary_and_applicability_rows() -> None:
    group = _group()
    instance = _instance((group,), channels=1, length=60)
    clean = np.zeros((60, 1), dtype=float)
    anomalous = clean.copy()

    rows = event_negative_controls(
        instance=instance,
        group=group,
        clean=clean,
        anomalous=anomalous,
        frontier=_frontier(),
        boundary_row={
            "boundary_to_canonical_ratio": 2.0,
            "boundary_primary_detection_cause": True,
        },
        protocol=CapabilityProtocol(),
    )

    by_type = {row["control_type"]: row for row in rows}
    assert set(by_type) == {
        "clean_vs_clean",
        "wrong_support_control",
        "wrong_witness_control",
        "boundary_only_control",
    }
    assert by_type["wrong_witness_control"]["applicability"] == "not_applicable"
    assert by_type["boundary_only_control"]["control_status"] == "control_failed"
    assert by_type["boundary_only_control"]["control_score"] == 2.0


def test_event_negative_controls_allows_relation_marginal_carrier() -> None:
    group = _group(
        anomaly_type="mode-correlation",
        constraint_tag="dependence.correlation",
        semantic_scope="relation",
        group_channels=(0, 1),
    )
    instance = _instance((group,), channels=2, length=80)
    clean = np.zeros((80, 2), dtype=float)
    carrier = np.tile([-1.0, 1.0], 5)
    clean[10:20, 0] = carrier
    clean[10:20, 1] = carrier
    anomalous = clean.copy()
    anomalous[10:20, 1] *= -1.0

    rows = event_negative_controls(
        instance=instance,
        group=group,
        clean=clean,
        anomalous=anomalous,
        frontier=_frontier(),
        boundary_row={},
        protocol=CapabilityProtocol(
            detection_witnesses=(
                "mean_z",
                "variance_log_ratio",
                "local_energy_z",
                "correlation_shift",
            )
        ),
    )

    wrong_witness = next(
        row for row in rows if row["control_type"] == "wrong_witness_control"
    )
    assert wrong_witness["applicability"] == "applicable"
    assert wrong_witness["control_status"] == "control_passed"
    assert wrong_witness["control_score"] == wrong_witness["baseline_control_score"]


def test_shifted_support_avoids_other_declared_events() -> None:
    current = _group(group_id="g0", start=10, end=20)
    other = _group(group_id="g1", start=20, end=30)
    instance = _instance((current, other), channels=1, length=80)

    start, end = shifted_support(
        instance,
        current,
        CapabilityProtocol(context_window_multiplier=0, min_context_points=0),
    )

    assert (start, end) == (0, 10)


def test_relation_like_detects_relation_metadata() -> None:
    group = _group(
        constraint_tag="dependence.correlation",
        semantic_scope="relation",
        group_channels=(0,),
    )

    assert is_relation_like(group) is True
