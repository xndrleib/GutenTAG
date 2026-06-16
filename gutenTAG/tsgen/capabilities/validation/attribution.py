"""Detector attribution from calibrated detectability frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from ...contracts import ContractRegistry
from ..cache import CacheStore
from ..dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid
from ..hashes import table_content_hash
from ..partitions import PartitionSpec, partition_sequence, run_partitions
from .attribution_rules import (
    AttributionContext,
    WITNESS_FAMILY,
    as_bool,
    as_float,
    as_int,
    attribution_scope,
    build_attribution_row_fields,
    field,
    forbidden_shortcut,
    frontier_witness_and_projection,
    is_canonical,
    min_empirical_p,
    normalized_evidence,
    projection_size,
)


@dataclass(frozen=True)
class _AttributionLookups:
    groups: Mapping[str, tuple[InstanceRecord, EventGroup]]
    boundary_by_event: pd.DataFrame
    shortcut_by_event: pd.DataFrame


def compute_detector_attribution(
    dataset: DatasetIndex,
    detectability_frontier: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
    *,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 512,
    registry: ContractRegistry | None = None,
) -> pd.DataFrame:
    """Build detector-attribution rows from scan-calibrated frontier results."""

    if detectability_frontier.empty:
        return pd.DataFrame()
    active_registry = registry or ContractRegistry.from_resource_defaults()
    row_records: tuple[Mapping[str, object], ...] = tuple(
        {str(key): value for key, value in row.items()}
        for row in detectability_frontier.to_dict("records")
    )
    if cache is None and int(n_jobs) <= 1:
        frame = _attribution_for_rows(
            dataset=dataset,
            frontier_rows=row_records,
            boundary_audit=boundary_audit,
            shortcut_audit=shortcut_audit,
            registry=active_registry,
        )
        return _sort_attribution(frame)
    partitions: tuple[PartitionSpec[Mapping[str, object]], ...] = partition_sequence(
        row_records,
        partition_size=max(1, int(partition_size)),
        prefix="detector_attribution",
    )
    fingerprint_extra: dict[str, object] = {
        "detectability_frontier_hash": _input_table_hash(detectability_frontier),
        "boundary_audit_hash": _input_table_hash(boundary_audit),
        "shortcut_audit_hash": _input_table_hash(shortcut_audit),
        "partition_size": int(partition_size),
    }

    def worker(partition: PartitionSpec[Mapping[str, object]]) -> pd.DataFrame:
        return _attribution_for_rows(
            dataset=dataset,
            frontier_rows=partition.items,
            boundary_audit=boundary_audit,
            shortcut_audit=shortcut_audit,
            registry=active_registry,
        )

    results = run_partitions(
        partitions,
        worker,
        n_jobs=n_jobs,
        cache=cache,
        profile_name="detector_attribution",
        fingerprint_extra=fingerprint_extra,
        table_worker=worker if cache is not None else None,
    )
    frames = [result.value for result in results if not result.value.empty]
    return _sort_attribution(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )


def _attribution_for_rows(
    *,
    dataset: DatasetIndex,
    frontier_rows: Sequence[Mapping[str, object]],
    boundary_audit: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
    registry: ContractRegistry,
) -> pd.DataFrame:
    """Build detector-attribution rows for a deterministic frontier partition."""

    lookups = _attribution_lookups(
        dataset=dataset,
        boundary_audit=boundary_audit,
        shortcut_audit=shortcut_audit,
    )
    rows: list[dict[str, object]] = []
    for frontier_row in frontier_rows:
        row = _attribution_row_for_frontier(
            frontier_row=frontier_row,
            lookups=lookups,
            registry=registry,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def _attribution_lookups(
    *,
    dataset: DatasetIndex,
    boundary_audit: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
) -> _AttributionLookups:
    return _AttributionLookups(
        groups={
            event_uid(instance, group): (instance, group)
            for instance in dataset.instances
            for group in instance.event_groups
        },
        boundary_by_event=_audit_by_event(boundary_audit),
        shortcut_by_event=_audit_by_event(shortcut_audit),
    )


def _audit_by_event(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("event_id") if not frame.empty else pd.DataFrame()


def _attribution_row_for_frontier(
    *,
    frontier_row: Mapping[str, object],
    lookups: _AttributionLookups,
    registry: ContractRegistry,
) -> dict[str, object] | None:
    event_id = str(frontier_row["event_id"])
    instance_group = lookups.groups.get(event_id)
    if instance_group is None:
        return None
    context = _attribution_context(
        event_id=event_id,
        instance_group=instance_group,
        frontier_row=frontier_row,
        lookups=lookups,
        registry=registry,
    )
    return build_attribution_row_fields(context, frontier_row)


def _attribution_context(
    *,
    event_id: str,
    instance_group: tuple[InstanceRecord, EventGroup],
    frontier_row: Mapping[str, object],
    lookups: _AttributionLookups,
    registry: ContractRegistry,
) -> AttributionContext:
    instance, group = instance_group
    witness, projection = frontier_witness_and_projection(frontier_row)
    family = WITNESS_FAMILY.get(witness, "unknown")
    forbidden = forbidden_shortcut(registry, group.anomaly_type, family)
    canonical = is_canonical(group, witness, projection_size(projection))
    boundary_row = _lookup(lookups.boundary_by_event, event_id)
    shortcut_row = _lookup(lookups.shortcut_by_event, event_id)
    scan_p = as_float(field(frontier_row, "scan_level_p_value", "empirical_p_value"))
    scan_null_count = as_int(
        field(frontier_row, "scan_null_count", "null_scan_count", default=0)
    )
    min_p = min_empirical_p(scan_null_count)
    boundary_primary = bool(boundary_row.get("boundary_primary_detection_cause", False))
    shortcut_status = str(shortcut_row.get("shortcut_status", "unknown"))
    return AttributionContext(
        event_id=event_id,
        instance=instance,
        group=group,
        witness=witness,
        projection=projection,
        family=family,
        forbidden=forbidden,
        canonical=canonical,
        boundary_primary=boundary_primary,
        shortcut_status=shortcut_status,
        detected=bool(frontier_row.get("detected", False)),
        scan_p=scan_p,
        candidate_p=as_float(
            field(frontier_row, "candidate_level_p_value", "empirical_p_value")
        ),
        canonical_detected=as_bool(
            field(frontier_row, "best_canonical_detected", default=False)
        ),
        scan_null_count=scan_null_count,
        min_empirical_p=min_p,
        evidence=normalized_evidence(scan_p, min_p),
        scope=attribution_scope(
            canonical=canonical,
            forbidden=forbidden,
            boundary_primary=boundary_primary,
            shortcut_status=shortcut_status,
        ),
    )


def _sort_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["event_id", "alpha", "rank_within_event"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return frame


def _lookup(frame: pd.DataFrame, event_id: str) -> dict[str, object]:
    if frame.empty or event_id not in frame.index:
        return {}
    row = frame.loc[event_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return dict(row)


def _input_table_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return table_content_hash(frame)
    sort_columns = [
        column
        for column in ("event_id", "alpha", "witness_or_model", "projection", "scope")
        if column in frame.columns
    ]
    return table_content_hash(frame, sort_by=sort_columns or None)
