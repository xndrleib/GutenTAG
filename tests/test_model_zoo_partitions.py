from pathlib import Path

import numpy as np

from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord
from gutenTAG.tsgen.capabilities.model_zoo_projection import (
    model_projection,
    project_array,
    projection_label_for,
    variant_model_projections,
)


def _instance(
    groups: tuple[EventGroup, ...],
    *,
    channels: int = 3,
) -> InstanceRecord:
    root = Path("/tmp/synth-gen-model-zoo-partitions-test")
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
        length=40,
        event_groups=groups,
    )


def _group(
    *,
    semantic_scope: str = "univariate",
    group_channels: tuple[int, ...] = (0,),
    context_channels: tuple[int, ...] = (),
    intervention_channels: tuple[int, ...] = (),
) -> EventGroup:
    return EventGroup(
        group_id="0",
        start=10,
        end=18,
        source_start=10,
        source_end=18,
        anomaly_type="mean",
        constraint_tag="location.mean",
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


def test_model_projection_uses_relation_channels() -> None:
    group = _group(
        semantic_scope="relation",
        group_channels=(2,),
        context_channels=(0,),
        intervention_channels=(1,),
    )
    instance = _instance((group,))

    assert model_projection(instance, group) == (0, 1, 2)
    assert variant_model_projections((instance,)) == ((0, 1, 2),)


def test_model_projection_falls_back_to_all_channels_for_univariate() -> None:
    group = _group(group_channels=(1,))
    instance = _instance((group,), channels=3)

    assert model_projection(instance, group) == (0, 1, 2)


def test_project_array_and_projection_label_are_stable() -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)

    projected = project_array(values, (2, 0))

    assert projected.tolist() == values[:, (2, 0)].tolist()
    assert projection_label_for((2, 0)) == "2|0"
