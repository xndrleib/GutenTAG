"""Model-zoo table loading and deterministic ordering."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .cache import CacheStore


def read_cached_table(
    cache: CacheStore,
    profile_name: str,
    partition_id: str,
) -> pd.DataFrame:
    """Read a cached model-zoo table partition."""

    return pd.read_csv(cache.table_path(profile_name, partition_id), compression="gzip")


def sort_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort ``model_zoo_frontier`` rows deterministically."""

    return sort_table(
        frame,
        (
            "event_id",
            "model_id",
            "alpha",
        ),
    )


def sort_event_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort ``model_zoo_event_scores`` rows deterministically."""

    return sort_table(
        frame,
        (
            "event_id",
            "model_id",
        ),
    )


def sort_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort model-zoo manifest rows deterministically."""

    return sort_table(
        frame,
        (
            "fit_variant_id",
            "model_id",
            "model_projection",
        ),
    )


def sort_table(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Stable-sort a table by existing columns only."""

    if frame.empty:
        return frame.reset_index(drop=True)
    sort_columns = [column for column in columns if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


__all__ = [
    "read_cached_table",
    "sort_event_scores",
    "sort_frontier",
    "sort_manifest",
    "sort_table",
]
