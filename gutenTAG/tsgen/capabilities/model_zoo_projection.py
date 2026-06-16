"""Projection and cached array helpers for model-zoo capability."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .array_store import ArrayKind, ArrayStore
from .dataset import EventGroup, InstanceRecord, read_timeseries_csv


def variant_model_projections(
    instances: Sequence[InstanceRecord],
) -> tuple[tuple[int, ...], ...]:
    """Return model projections needed by all event groups in a variant."""

    projections = {
        model_projection(instance, group)
        for instance in instances
        for group in instance.event_groups
    }
    if not projections and instances:
        projections.add(tuple(range(instances[0].channels)))
    return tuple(sorted(projections))


def model_projection(instance: InstanceRecord, group: EventGroup) -> tuple[int, ...]:
    """Return the channel projection used to fit and score a model event."""

    if group.semantic_scope in {"relation", "regime_relation"}:
        projection = tuple(
            sorted(
                set(group.group_channels)
                | set(group.context_channels)
                | set(group.intervention_channels)
            )
        )
        if len(projection) >= 2:
            return projection
    return tuple(range(instance.channels))


def projection_label_for(projection: tuple[int, ...]) -> str:
    """Return a stable model projection label."""

    return "|".join(str(channel) for channel in projection)


def project_array(values: np.ndarray, projection: tuple[int, ...]) -> np.ndarray:
    """Project a series array onto selected model channels."""

    array = np.asarray(values, dtype=np.float64)
    if not projection:
        return array
    if tuple(projection) == tuple(range(array.shape[1])):
        return array
    return array[:, projection]


def read_cached_array(
    instance: InstanceRecord,
    kind: ArrayKind,
    arrays: ArrayStore | None,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Read a clean/anomalous instance array through an optional cache."""

    key = f"{kind}:{instance.variant_id}/{instance.split}/{instance.instance_id}"
    if key in cache:
        return cache[key]
    if arrays is not None:
        values = arrays.get(instance, kind)
    elif kind == "clean":
        values = read_timeseries_csv(instance.clean_path)
    else:
        values = read_timeseries_csv(instance.anomalous_path)
    cache[key] = np.asarray(values, dtype=np.float64)
    return cache[key]


__all__ = [
    "model_projection",
    "project_array",
    "projection_label_for",
    "read_cached_array",
    "variant_model_projections",
]
