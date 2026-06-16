"""Dataset and split summary statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..io import sanitize_json_value


def compute_split_statistics(
    split: str,
    instance_summaries: Sequence[Mapping[str, Any]],
    length: int,
    channels: int,
) -> dict[str, Any]:
    """Compute aggregate statistics for one generated split."""

    channel_totals, channel_stats = _per_channel_count_statistics(
        instance_summaries,
        channel_keys=[str(channel) for channel in range(channels)],
    )
    summary = {
        "split": split,
        "instances": len(instance_summaries),
        "length": length,
        "channels": channels,
    }
    summary.update(_density_statistics(instance_summaries))
    summary.update(_segment_count_statistics(instance_summaries))
    summary.update(_segment_length_statistics(instance_summaries))
    summary["per_channel_segment_counts_total"] = channel_totals
    summary["per_channel_segment_counts_stats"] = channel_stats
    return sanitize_json_value(summary)


def compute_dataset_statistics(
    instance_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics for a dataset or variant."""

    if len(instance_summaries) == 0:
        return _empty_dataset_statistics()
    channel_totals, channel_stats = _per_channel_count_statistics(
        instance_summaries,
        channel_keys=_observed_channel_keys(instance_summaries),
    )
    summary = {
        "instance_count": len(instance_summaries),
        "per_channel_segment_counts_total": channel_totals,
        "per_channel_segment_counts_stats": channel_stats,
    }
    summary.update(_density_statistics(instance_summaries))
    summary.update(_segment_count_statistics(instance_summaries))
    summary.update(_segment_length_statistics(instance_summaries))
    return sanitize_json_value(summary)


def _empty_dataset_statistics() -> dict[str, Any]:
    return {
        "instance_count": 0,
        "target_density_mean": None,
        "target_density_std": None,
        "target_density_min": None,
        "target_density_max": None,
        "achieved_density_mean": None,
        "achieved_density_std": None,
        "achieved_density_min": None,
        "achieved_density_max": None,
        "n_segments_mean": None,
        "n_segments_std": None,
        "n_segments_min": None,
        "n_segments_max": None,
        "segment_length_mean": None,
        "segment_length_std": None,
        "segment_length_median": None,
        "segment_length_min": None,
        "segment_length_max": None,
        "per_channel_segment_counts_total": {},
        "per_channel_segment_counts_stats": {},
    }


def _density_statistics(
    instance_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_densities = _summary_float_array(instance_summaries, "target_density")
    achieved_densities = _summary_float_array(instance_summaries, "achieved_density")
    return {
        **_float_stats("target_density", target_densities),
        **_float_stats("achieved_density", achieved_densities),
    }


def _segment_count_statistics(
    instance_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    n_segments = np.array(
        [int(item["n_segments"]) for item in instance_summaries],
        dtype=int,
    )
    return {
        "n_segments_mean": float(np.mean(n_segments)),
        "n_segments_std": float(np.std(n_segments)),
        "n_segments_min": int(np.min(n_segments)),
        "n_segments_max": int(np.max(n_segments)),
    }


def _segment_length_statistics(
    instance_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pooled_lengths = _collect_segment_lengths(instance_summaries)
    return {
        "segment_length_mean": float(np.mean(pooled_lengths)),
        "segment_length_std": float(np.std(pooled_lengths)),
        "segment_length_median": float(np.median(pooled_lengths)),
        "segment_length_min": int(np.min(pooled_lengths)),
        "segment_length_max": int(np.max(pooled_lengths)),
    }


def _summary_float_array(
    instance_summaries: Sequence[Mapping[str, Any]],
    key: str,
) -> np.ndarray:
    return np.array([float(item[key]) for item in instance_summaries], dtype=float)


def _float_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _per_channel_count_statistics(
    instance_summaries: Sequence[Mapping[str, Any]],
    channel_keys: Sequence[str],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    counts_by_channel = _per_channel_counts_by_instance(
        instance_summaries,
        channel_keys,
    )
    totals: dict[str, int] = {}
    stats: dict[str, dict[str, Any]] = {}
    for channel_key, counts in counts_by_channel.items():
        values = np.array(counts, dtype=int)
        totals[channel_key] = int(values.sum())
        stats[channel_key] = _count_stats(values)
    return totals, stats


def _per_channel_counts_by_instance(
    instance_summaries: Sequence[Mapping[str, Any]],
    channel_keys: Sequence[str],
) -> dict[str, list[int]]:
    counts_by_channel: dict[str, list[int]] = {
        str(channel_key): [] for channel_key in channel_keys
    }
    for item in instance_summaries:
        per_channel = item.get("per_channel_segment_counts", {})
        for channel_key in channel_keys:
            counts_by_channel[str(channel_key)].append(
                int(per_channel.get(str(channel_key), 0))
            )
    return counts_by_channel


def _observed_channel_keys(
    instance_summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        {
            str(channel)
            for item in instance_summaries
            for channel in item.get("per_channel_segment_counts", {}).keys()
        }
    )


def _count_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "total": int(values.sum()),
    }


def _collect_segment_lengths(
    instance_summaries: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    lengths: list[float] = []
    for item in instance_summaries:
        segment_lengths = item.get("segment_lengths", [])
        lengths.extend(float(length) for length in segment_lengths)
    if len(lengths) == 0:
        return np.array([0.0])
    return np.array(lengths, dtype=np.float64)
