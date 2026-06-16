"""Corrected min-p detectability frontier."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .array_store import ArrayStore
from .cache import CacheStore
from .corrected_detectability_partitions import (
    cached_or_compute_partition,
    calibration_null_instances,
)
from .corrected_detectability_tables import (
    blind_scan_columns,
    frontier_columns,
    resolution_columns,
)
from .dataset import DatasetIndex, instances_by_variant
from .partitions import PartitionSpec, run_partitions
from .protocol import CapabilityProtocol
from .windows import WindowLibrary


@dataclass(frozen=True)
class CorrectedDetectabilityResult:
    """Corrected detectability tables and null manifests."""

    frontier: pd.DataFrame
    oracle_window_diagnostic: pd.DataFrame
    blind_scan_events: pd.DataFrame
    calibration_resolution: pd.DataFrame
    candidate_nulls_manifest: dict[str, object]
    scan_nulls_manifest: dict[str, object]


def compute_corrected_detectability_frontier(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
    windows: WindowLibrary | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
) -> CorrectedDetectabilityResult:
    """Compute corrected min-p detectability frontiers."""

    variant_instances = instances_by_variant(dataset.instances)
    null_instances = {
        variant_id: calibration_null_instances(instances, protocol)
        for variant_id, instances in variant_instances.items()
    }
    partitions = tuple(
        PartitionSpec(partition_id=str(variant_id), items=tuple(instances))
        for variant_id, instances in variant_instances.items()
    )
    partition_results = run_partitions(
        partitions,
        lambda partition: cached_or_compute_partition(
            partition=partition,
            null_instances=null_instances,
            protocol=protocol,
            arrays=arrays,
            windows=windows,
            cache=cache,
        ),
        n_jobs=n_jobs,
    )
    rows: list[dict[str, object]] = []
    blind_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    candidate_manifest_rows: dict[str, dict[str, object]] = {}
    scan_manifest_rows: dict[str, dict[str, object]] = {}
    for partition_result in partition_results:
        result = partition_result.value
        rows.extend(result.rows)
        blind_rows.extend(result.blind_rows)
        resolution_rows.extend(result.resolution_rows)
        candidate_manifest_rows.update(result.candidate_manifest_rows)
        scan_manifest_rows.update(result.scan_manifest_rows)
    frontier = pd.DataFrame(rows, columns=pd.Index(frontier_columns()))
    diagnostic = frontier.copy()
    blind_scan = pd.DataFrame(blind_rows, columns=pd.Index(blind_scan_columns()))
    resolution = pd.DataFrame(resolution_rows, columns=pd.Index(resolution_columns()))
    return CorrectedDetectabilityResult(
        frontier=frontier,
        oracle_window_diagnostic=diagnostic,
        blind_scan_events=blind_scan,
        calibration_resolution=resolution,
        candidate_nulls_manifest={
            "candidate_nulls_manifest_version": "synthgen.capability.candidate_nulls.v1",
            "candidate_nulls": list(candidate_manifest_rows.values()),
        },
        scan_nulls_manifest={
            "scan_nulls_manifest_version": "synthgen.capability.scan_nulls.v1",
            "scan_nulls": list(scan_manifest_rows.values()),
        },
    )
