"""Common segment-planning primitives."""

from __future__ import annotations

import numpy as np


def sample_segment_lengths(
    rng: np.random.Generator,
    total_points: int,
    n_segments: int,
    min_segment_length: int = 1,
) -> list[int]:
    """Sample positive segment lengths summing to a target size.

    Parameters
    ----------
    rng
        Random number generator controlling the length allocation.
    total_points
        Total number of points that must be covered by all sampled lengths.
    n_segments
        Number of segment lengths to sample.
    min_segment_length
        Preferred lower bound for each length. The effective lower bound is
        reduced when the requested total cannot satisfy it.

    Returns
    -------
    list[int]
        Segment lengths whose sum equals ``total_points``.
    """

    effective_min = max(1, min_segment_length)
    if effective_min * n_segments > total_points:
        effective_min = max(1, total_points // n_segments)

    raw = rng.random(n_segments)
    raw_sum = float(raw.sum())
    if raw_sum == 0:
        raw = np.ones(n_segments)
        raw_sum = float(raw.sum())
    lengths = np.floor(raw / raw_sum * total_points).astype(int)
    lengths = np.maximum(lengths, effective_min)

    diff = int(total_points - lengths.sum())
    if diff > 0:
        for idx in rng.permutation(n_segments):
            lengths[idx] += 1
            diff -= 1
            if diff == 0:
                break
        while diff > 0:
            idx = int(rng.integers(0, n_segments))
            lengths[idx] += 1
            diff -= 1
    elif diff < 0:
        while diff < 0:
            candidates = np.where(lengths > effective_min)[0]
            if len(candidates) == 0:
                break
            idx = int(rng.choice(candidates))
            lengths[idx] -= 1
            diff += 1

    lengths_list = [int(length) for length in lengths]
    assert sum(lengths_list) == total_points
    return lengths_list


def sample_bounded_integer_lengths(
    rng: np.random.Generator,
    total_points: int,
    n_segments: int,
    min_value: int,
    max_value: int,
) -> list[int]:
    """Sample bounded integer lengths with an exact sum.

    Parameters
    ----------
    rng
        Random number generator controlling residual allocation.
    total_points
        Required sum of returned values.
    n_segments
        Number of values to sample.
    min_value
        Inclusive lower bound for every value.
    max_value
        Inclusive upper bound for every value.

    Returns
    -------
    list[int]
        Integer values satisfying the requested bounds and total.
    """

    min_value = max(1, int(min_value))
    max_value = max(min_value, int(max_value))
    if n_segments <= 0:
        raise ValueError("n_segments must be > 0 for bounded integer sampling.")
    if n_segments * min_value > total_points:
        raise ValueError(
            "Cannot satisfy bounded integer sampling: "
            "n_segments * min_value > total_points "
            f"({n_segments}*{min_value}>{total_points})."
        )
    if n_segments * max_value < total_points:
        raise ValueError(
            "Cannot satisfy bounded integer sampling: "
            "n_segments * max_value < total_points "
            f"({n_segments}*{max_value}<{total_points})."
        )

    values = np.full(n_segments, min_value, dtype=int)
    capacities = np.full(n_segments, max_value - min_value, dtype=int)
    remaining = int(total_points - n_segments * min_value)

    while remaining > 0:
        candidates = np.where(capacities > 0)[0]
        if candidates.size == 0:
            raise ValueError(
                "Bounded integer sampler ran out of capacity before reaching "
                "target total."
            )
        idx = int(rng.choice(candidates))
        values[idx] += 1
        capacities[idx] -= 1
        remaining -= 1

    return [int(value) for value in values.tolist()]


def is_slot_available(
    start: int,
    end: int,
    channel: int,
    overlap_policy: str,
    occupied_global: np.ndarray,
    occupied_per_channel: np.ndarray,
) -> bool:
    """Return whether a candidate segment can occupy the requested slot."""

    if overlap_policy == "global":
        return int(occupied_global[start:end].sum()) == 0
    return int(occupied_per_channel[channel, start:end].sum()) == 0


def occupy_slot(
    start: int,
    end: int,
    channel: int,
    overlap_policy: str,
    occupied_global: np.ndarray,
    occupied_per_channel: np.ndarray,
) -> None:
    """Mark a segment slot as occupied according to overlap policy."""

    if overlap_policy == "global":
        occupied_global[start:end] = 1
    else:
        occupied_per_channel[channel, start:end] = 1
