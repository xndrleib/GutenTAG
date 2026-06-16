"""Deterministic partition execution helpers for capability profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Sequence, TypeVar, cast

import pandas as pd
from joblib import Parallel, delayed

from .cache import CacheFingerprint, CacheStore

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class PartitionSpec(Generic[T]):
    """A deterministic unit of profile work."""

    partition_id: str
    items: tuple[T, ...]


@dataclass(frozen=True)
class PartitionResult(Generic[R]):
    """Result for one completed partition."""

    partition_id: str
    value: R
    from_cache: bool = False


def partition_sequence(
    items: Sequence[T] | Iterable[T],
    *,
    partition_size: int,
    prefix: str = "p",
) -> tuple[PartitionSpec[T], ...]:
    """Split items into deterministic fixed-size partitions."""

    if partition_size <= 0:
        raise ValueError("partition_size must be positive")
    materialized = tuple(items)
    partitions: list[PartitionSpec[T]] = []
    for index in range(0, len(materialized), partition_size):
        partition_number = len(partitions)
        partitions.append(
            PartitionSpec(
                partition_id=f"{prefix}{partition_number:05d}",
                items=materialized[index : index + partition_size],
            )
        )
    return tuple(partitions)


def run_partitions(
    partitions: Sequence[PartitionSpec[T]],
    worker: Callable[[PartitionSpec[T]], R],
    *,
    n_jobs: int = 1,
    cache: CacheStore | None = None,
    profile_name: str | None = None,
    fingerprint_extra: dict[str, object] | None = None,
    table_worker: Callable[[PartitionSpec[T]], pd.DataFrame] | None = None,
) -> tuple[PartitionResult[R], ...]:
    """Run partitions serially or in parallel while preserving output order."""

    if cache is not None and table_worker is not None:
        if profile_name is None:
            raise ValueError("profile_name is required when using CacheStore")
        if int(n_jobs) <= 1:
            return cast(
                tuple[PartitionResult[R], ...],
                tuple(
                    _run_cached_partition(
                        cache=cache,
                        profile_name=profile_name,
                        partition=partition,
                        worker=table_worker,
                        fingerprint_extra=fingerprint_extra,
                    )
                    for partition in partitions
                ),
            )
        cached_results = cast(
            list[PartitionResult[pd.DataFrame]],
            Parallel(n_jobs=int(n_jobs))(
                delayed(_run_cached_partition)(
                    cache=cache,
                    profile_name=profile_name,
                    partition=partition,
                    worker=table_worker,
                    fingerprint_extra=fingerprint_extra,
                )
                for partition in partitions
            ),
        )
        return cast(
            tuple[PartitionResult[R], ...],
            tuple(
                sorted(
                    cached_results,
                    key=lambda item: _partition_index(partitions, item.partition_id),
                )
            ),
        )
    if int(n_jobs) <= 1:
        return tuple(
            PartitionResult(
                partition_id=partition.partition_id, value=worker(partition)
            )
            for partition in partitions
        )
    results = cast(
        list[PartitionResult[R]],
        Parallel(n_jobs=int(n_jobs))(
            delayed(_run_uncached_partition)(partition, worker)
            for partition in partitions
        ),
    )
    return tuple(
        sorted(
            results, key=lambda item: _partition_index(partitions, item.partition_id)
        )
    )


def _run_uncached_partition(
    partition: PartitionSpec[T],
    worker: Callable[[PartitionSpec[T]], R],
) -> PartitionResult[R]:
    return PartitionResult(partition_id=partition.partition_id, value=worker(partition))


def _run_cached_partition(
    *,
    cache: CacheStore,
    profile_name: str,
    partition: PartitionSpec[T],
    worker: Callable[[PartitionSpec[T]], pd.DataFrame],
    fingerprint_extra: dict[str, object] | None,
) -> PartitionResult[pd.DataFrame]:
    fingerprint = CacheFingerprint(
        dataset_hash=str(cache.run_fingerprint.get("dataset_hash", "")),
        protocol_hash=str(cache.run_fingerprint.get("protocol_hash", "")),
        code_hash=str(cache.run_fingerprint.get("code_hash", "")),
        profile_name=profile_name,
        partition_id=partition.partition_id,
        extra=dict(fingerprint_extra or {}),
    )
    from_cache = (
        cache.resume
        and cache.table_path(profile_name, partition.partition_id).exists()
        and cache.is_valid(
            profile_name=profile_name,
            partition_id=partition.partition_id,
            fingerprint=fingerprint,
        )
    )
    value = cache.get_or_compute_table(
        profile_name=profile_name,
        partition_id=partition.partition_id,
        fingerprint=fingerprint,
        compute=lambda: worker(partition),
    )
    return PartitionResult(
        partition_id=partition.partition_id, value=value, from_cache=from_cache
    )


def _partition_index(partitions: Sequence[PartitionSpec[T]], partition_id: str) -> int:
    for index, partition in enumerate(partitions):
        if partition.partition_id == partition_id:
            return index
    return len(partitions)
