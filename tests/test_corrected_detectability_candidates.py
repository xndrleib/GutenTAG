from pathlib import Path

from gutenTAG.tsgen.capabilities.corrected_detectability_candidates import (
    CandidateSpec,
    best_canonical_key,
    candidate_is_canonical,
    candidate_specs,
    format_projection,
    spec_by_key,
)
from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol


def _instance(channels: int = 2) -> InstanceRecord:
    root = Path("/tmp/synth-gen-candidate-test")
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
        length=64,
        event_groups=(),
    )


def _group(
    anomaly_type: str = "mean",
    *,
    constraint_tag: str = "pointwise",
    semantic_scope: str = "univariate",
    group_channels: tuple[int, ...] = (0,),
    context_channels: tuple[int, ...] = (),
    intervention_channels: tuple[int, ...] = (),
) -> EventGroup:
    return EventGroup(
        group_id="event-0",
        start=10,
        end=20,
        source_start=10,
        source_end=20,
        anomaly_type=anomaly_type,
        constraint_tag=constraint_tag,
        repair_operator="none",
        semantic_scope=semantic_scope,
        intervention_channels=intervention_channels,
        context_channels=context_channels,
        group_channels=group_channels,
        primary_channels=group_channels,
        event_scope="group",
        purity_hint="clean",
        raw_events=(),
    )


def test_candidate_spec_key_and_family_are_stable() -> None:
    candidate = CandidateSpec(witness="correlation_shift", projection=(1, 3))

    assert candidate.key == "correlation_shift@1|3"
    assert candidate.family == "dependence.correlation"
    assert format_projection((2, 0)) == "2|0"


def test_candidate_specs_use_event_channels_and_projection_requirements() -> None:
    protocol = CapabilityProtocol(
        detection_witnesses=("mean_z", "correlation_shift"),
        max_projection_size=2,
    )
    group = _group(
        anomaly_type="mode-correlation",
        constraint_tag="dependence",
        semantic_scope="relation",
        group_channels=(0, 1),
    )

    keys = {
        candidate.key for candidate in candidate_specs(_instance(), group, protocol)
    }

    assert "mean_z@0" in keys
    assert "mean_z@1" in keys
    assert "mean_z@0|1" in keys
    assert "correlation_shift@0|1" in keys
    assert "correlation_shift@0" not in keys


def test_candidate_specs_fall_back_to_all_instance_channels() -> None:
    protocol = CapabilityProtocol(
        detection_witnesses=("mean_z",),
        max_projection_size=1,
    )
    group = _group(group_channels=(), context_channels=(), intervention_channels=())

    keys = {
        candidate.key
        for candidate in candidate_specs(_instance(channels=2), group, protocol)
    }

    assert keys == {"mean_z@0", "mean_z@1"}


def test_best_canonical_key_respects_anomaly_policy_before_score() -> None:
    group = _group(anomaly_type="mean")
    candidates = (
        CandidateSpec(witness="mean_z", projection=(0,)),
        CandidateSpec(witness="local_energy_z", projection=(0,)),
    )

    assert (
        best_canonical_key(
            group,
            candidates,
            {
                "mean_z@0": 0.20,
                "local_energy_z@0": 0.01,
            },
        )
        == "mean_z@0"
    )


def test_relation_canonical_candidates_require_pair_projection() -> None:
    group = _group(
        anomaly_type="mode-correlation",
        constraint_tag="dependence",
        semantic_scope="relation",
        group_channels=(0, 1),
    )

    assert not candidate_is_canonical(group, CandidateSpec("correlation_shift", (0,)))
    assert candidate_is_canonical(group, CandidateSpec("correlation_shift", (0, 1)))


def test_spec_by_key_returns_matching_candidate() -> None:
    candidates = (
        CandidateSpec(witness="mean_z", projection=(0,)),
        CandidateSpec(witness="variance_log_ratio", projection=(0,)),
    )

    assert spec_by_key(candidates, "variance_log_ratio@0") == candidates[1]
    assert spec_by_key(candidates, "missing@0") is None
