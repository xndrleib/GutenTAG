"""Structural model-zoo capability frontier."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .array_store import ArrayStore
from .cache import CacheStore
from .dataset import DatasetIndex, instances_by_variant
from .models import CapabilityModel
from .model_zoo_cache import cached_model_zoo_partition, models_fingerprint
from .model_zoo_partitions import ModelZooPartitionResult, model_zoo_for_variants
from .model_zoo_tables import sort_event_scores, sort_frontier, sort_manifest
from .partitions import PartitionSpec, partition_sequence, run_partitions
from .protocol import CapabilityProtocol
from .windows import WindowLibrary


@dataclass(frozen=True)
class ModelZooResult:
    """Structural model-zoo output tables."""

    frontier: pd.DataFrame
    event_scores: pd.DataFrame
    model_manifest: dict[str, object]


def compute_model_zoo_frontier(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
    windows: WindowLibrary | None = None,
    models: Sequence[CapabilityModel] | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 1,
) -> ModelZooResult:
    """Compute model-zoo event and alpha frontiers."""

    active_models = tuple(models) if models is not None else None
    variant_instances = instances_by_variant(dataset.instances)
    variant_ids = tuple(variant_instances)
    if cache is None and int(n_jobs) <= 1:
        return _combine_partition_results(
            (
                model_zoo_for_variants(
                    variant_ids=variant_ids,
                    variant_instances=variant_instances,
                    protocol=protocol,
                    arrays=arrays,
                    windows=windows,
                    active_models=active_models,
                ),
            )
        )
    partitions = partition_sequence(
        variant_ids,
        partition_size=max(1, int(partition_size)),
        prefix="model_zoo",
    )
    fingerprint_extra = {
        "model_fingerprint": models_fingerprint(active_models),
        "partition_size": int(partition_size),
    }

    def worker(partition: PartitionSpec[str]) -> ModelZooPartitionResult:
        if cache is not None:
            return cached_model_zoo_partition(
                cache=cache,
                partition_id=partition.partition_id,
                variant_ids=partition.items,
                variant_instances=variant_instances,
                protocol=protocol,
                arrays=arrays,
                windows=windows,
                active_models=active_models,
                fingerprint_extra=fingerprint_extra,
            )
        return model_zoo_for_variants(
            variant_ids=partition.items,
            variant_instances=variant_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            active_models=active_models,
        )

    results = run_partitions(partitions, worker, n_jobs=n_jobs)
    return _combine_partition_results(tuple(result.value for result in results))


def _combine_partition_results(
    results: Sequence[ModelZooPartitionResult],
) -> ModelZooResult:
    frontier_frames = [
        result.frontier for result in results if not result.frontier.empty
    ]
    event_frames = [
        result.event_scores for result in results if not result.event_scores.empty
    ]
    manifest_frames = [
        result.manifest_rows for result in results if not result.manifest_rows.empty
    ]
    frontier = sort_frontier(
        pd.concat(frontier_frames, ignore_index=True)
        if frontier_frames
        else pd.DataFrame()
    )
    event_scores = sort_event_scores(
        pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    )
    manifest_frame = sort_manifest(
        pd.concat(manifest_frames, ignore_index=True)
        if manifest_frames
        else pd.DataFrame()
    )
    if not manifest_frame.empty:
        subset = ["fit_variant_id", "model_id"]
        if "model_projection" in manifest_frame.columns:
            subset.append("model_projection")
        manifest_frame = manifest_frame.drop_duplicates(
            subset=subset,
            keep="first",
        )
    return ModelZooResult(
        frontier=frontier,
        event_scores=event_scores,
        model_manifest={
            "model_zoo_model_manifest_version": "synthgen.capability.model_zoo.v1",
            "models": (
                manifest_frame.to_dict("records") if not manifest_frame.empty else []
            ),
        },
    )
