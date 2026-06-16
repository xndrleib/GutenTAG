"""Cache helpers for structural model-zoo partitions."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from ..manifest import canonical_json_hash
from .array_store import ArrayStore
from .cache import CacheStore
from .dataset import InstanceRecord
from .models import CapabilityModel, default_model_registry
from .model_zoo_partitions import ModelZooPartitionResult, model_zoo_for_variants
from .model_zoo_tables import read_cached_table
from .protocol import CapabilityProtocol
from .windows import WindowLibrary


def cached_model_zoo_partition(
    *,
    cache: CacheStore,
    partition_id: str,
    variant_ids: Sequence[str],
    variant_instances: dict[str, list[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    active_models: Sequence[CapabilityModel] | None,
    fingerprint_extra: dict[str, object],
) -> ModelZooPartitionResult:
    """Return a cached or freshly computed model-zoo partition."""

    profiles = (
        "model_zoo_frontier",
        "model_zoo_event_scores",
        "model_zoo_manifest_rows",
    )
    fingerprints = {
        profile: cache.fingerprint(
            profile_name=profile,
            partition_id=partition_id,
            extra={
                **fingerprint_extra,
                "variant_ids": list(variant_ids),
            },
        )
        for profile in profiles
    }
    if cache.resume and all(
        cache.table_path(profile, partition_id).exists()
        and cache.is_valid(
            profile_name=profile,
            partition_id=partition_id,
            fingerprint=fingerprints[profile],
        )
        for profile in profiles
    ):
        return ModelZooPartitionResult(
            frontier=read_cached_table(cache, "model_zoo_frontier", partition_id),
            event_scores=read_cached_table(
                cache,
                "model_zoo_event_scores",
                partition_id,
            ),
            manifest_rows=read_cached_table(
                cache,
                "model_zoo_manifest_rows",
                partition_id,
            ),
        )
    result = model_zoo_for_variants(
        variant_ids=variant_ids,
        variant_instances=variant_instances,
        protocol=protocol,
        arrays=arrays,
        windows=windows,
        active_models=active_models,
    )
    cache.write_table_partition(
        profile_name="model_zoo_frontier",
        partition_id=partition_id,
        fingerprint=fingerprints["model_zoo_frontier"],
        frame=result.frontier,
    )
    cache.write_table_partition(
        profile_name="model_zoo_event_scores",
        partition_id=partition_id,
        fingerprint=fingerprints["model_zoo_event_scores"],
        frame=result.event_scores,
    )
    cache.write_table_partition(
        profile_name="model_zoo_manifest_rows",
        partition_id=partition_id,
        fingerprint=fingerprints["model_zoo_manifest_rows"],
        frame=result.manifest_rows,
    )
    return result


def models_fingerprint(models: Sequence[CapabilityModel] | None) -> str:
    """Return a stable fingerprint for the active model registry."""

    model_instances = (
        default_model_registry()
        if models is None
        else tuple(copy.deepcopy(model) for model in models)
    )
    payload = [
        {
            "model_id": model.model_id,
            "family": model.family,
            "metadata": model.metadata(),
        }
        for model in model_instances
    ]
    return canonical_json_hash({"models": payload})


__all__ = ["cached_model_zoo_partition", "models_fingerprint"]
