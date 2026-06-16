"""Analysis-level negative controls for implementation validity."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..array_store import ArrayStore
from ..cache import CacheStore
from ..dataset import DatasetIndex, InstanceRecord, event_uid, read_timeseries_csv
from ..hashes import table_content_hash
from ..pandas_typing import as_frame
from ..partitions import PartitionSpec, partition_sequence, run_partitions
from ..protocol import CapabilityProtocol
from .negative_control_events import event_negative_controls as _event_negative_controls


def compute_negative_controls(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    detectability_frontier: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    *,
    arrays: ArrayStore | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 8,
) -> pd.DataFrame:
    """Compute first-pass analysis-level negative controls."""

    if detectability_frontier.empty:
        return pd.DataFrame()
    if cache is None and int(n_jobs) <= 1:
        return _negative_controls_for_instances(
            instances=dataset.instances,
            protocol=protocol,
            detectability_frontier=detectability_frontier,
            boundary_audit=boundary_audit,
            arrays=arrays,
        )
    partitions = partition_sequence(
        dataset.instances,
        partition_size=max(1, int(partition_size)),
        prefix="negative_controls",
    )
    fingerprint_extra = {
        "detectability_frontier_hash": _input_table_hash(detectability_frontier),
        "boundary_audit_hash": _input_table_hash(boundary_audit),
        "partition_size": int(partition_size),
    }

    def worker(partition: PartitionSpec[InstanceRecord]) -> pd.DataFrame:
        return _negative_controls_for_instances(
            instances=partition.items,
            protocol=protocol,
            detectability_frontier=detectability_frontier,
            boundary_audit=boundary_audit,
            arrays=arrays,
        )

    results = run_partitions(
        partitions,
        worker,
        n_jobs=n_jobs,
        cache=cache,
        profile_name="negative_controls",
        fingerprint_extra=fingerprint_extra,
        table_worker=worker if cache is not None else None,
    )
    frames = [result.value for result in results if not result.value.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _negative_controls_for_instances(
    *,
    instances: Sequence[InstanceRecord],
    protocol: CapabilityProtocol,
    detectability_frontier: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    arrays: ArrayStore | None,
) -> pd.DataFrame:
    """Compute negative controls for a deterministic instance partition."""

    frontier_by_event = detectability_frontier.groupby("event_id")
    boundary_by_event = (
        boundary_audit.set_index("event_id")
        if not boundary_audit.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for instance in instances:
        clean = (
            arrays.get(instance, "clean")
            if arrays is not None
            else read_timeseries_csv(instance.clean_path)
        )
        anomalous = (
            arrays.get(instance, "anomalous")
            if arrays is not None
            else read_timeseries_csv(instance.anomalous_path)
        )
        for group in instance.event_groups:
            uid = event_uid(instance, group)
            if uid not in frontier_by_event.groups:
                continue
            frontier = as_frame(frontier_by_event.get_group(uid))
            rows.extend(
                _event_negative_controls(
                    instance=instance,
                    group=group,
                    clean=clean,
                    anomalous=anomalous,
                    frontier=frontier,
                    boundary_row=_lookup(boundary_by_event, uid),
                    protocol=protocol,
                )
            )
    return pd.DataFrame(rows)


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
        for column in (
            "event_id",
            "alpha",
            "control_type",
            "witness_or_model",
            "projection",
        )
        if column in frame.columns
    ]
    return table_content_hash(frame, sort_by=sort_columns or None)
