"""Realized mode-boundary segment planning."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .common import sample_segment_lengths
from .mode_boundary_placement import place_mode_boundary_segments
from .mode_boundary_types import ModeBoundaryGeometry
from .types import SegmentPlan
from .uniform import sample_uniform_segments


def sample_mode_boundary_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    planner_cfg: Mapping[str, Any],
    anomaly_type: str,
    clean_values: np.ndarray | None,
    segment_count_range: Sequence[int],
    min_segment_length: int,
    logger: logging.Logger | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample segments aligned to realized mode-change boundaries.

    Parameters
    ----------
    rng, target_density, series_length, channels
        Randomness, target support density, and generated-series shape.
    max_placement_attempts, overlap_policy
        Random-placement retry budget and temporal overlap policy.
    planner_cfg, anomaly_type, clean_values
        Planner symmetry input, expected ``mode-correlation`` type, and clean data.
    segment_count_range, min_segment_length
        Bounds used to sample count and lengths.
    logger, segment_factory
        Optional fallback logger and segment-plan factory.

    Returns
    -------
    list[SegmentPlan]
        Boundary-aligned plans, or uniform fallback plans when changes are sparse.
    """

    del planner_cfg
    geometry = _mode_boundary_geometry(series_length, channels)
    clean_values = _validate_mode_boundary_request(
        anomaly_type,
        clean_values,
        geometry,
    )
    change_boundaries = _mode_change_boundaries(clean_values, geometry.length)
    if change_boundaries.shape[0] < 2:
        return _uniform_mode_boundary_fallback(
            rng=rng,
            target_density=target_density,
            geometry=geometry,
            max_placement_attempts=max_placement_attempts,
            overlap_policy=overlap_policy,
            segment_count_range=segment_count_range,
            min_segment_length=min_segment_length,
            logger=logger,
            segment_factory=segment_factory,
        )

    lengths = _sample_mode_boundary_lengths(
        rng,
        target_density=target_density,
        geometry=geometry,
        segment_count_range=segment_count_range,
        min_segment_length=int(min_segment_length),
    )
    segments = place_mode_boundary_segments(
        rng=rng,
        lengths=lengths,
        change_boundaries=change_boundaries,
        geometry=geometry,
        min_segment_length=int(min_segment_length),
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    )
    segments.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
    return segments


def _mode_boundary_geometry(series_length: int, channels: int) -> ModeBoundaryGeometry:
    return ModeBoundaryGeometry(length=int(series_length), n_channels=int(channels))


def _validate_mode_boundary_request(
    anomaly_type: str,
    clean_values: np.ndarray | None,
    geometry: ModeBoundaryGeometry,
) -> np.ndarray:
    if anomaly_type != "mode-correlation":
        raise ValueError(
            "mode_boundary_segments planner is only supported for "
            "anomaly_type='mode-correlation'."
        )
    if clean_values is None:
        raise ValueError("mode_boundary_segments planner requires clean_values.")
    if clean_values.shape != (geometry.length, geometry.n_channels):
        raise ValueError(
            "mode_boundary_segments expected clean_values shape "
            f"({geometry.length}, {geometry.n_channels}), got {clean_values.shape}."
        )
    return clean_values


def _mode_change_boundaries(clean_values: np.ndarray, length: int) -> np.ndarray:
    reference = np.asarray(clean_values[:, 0], dtype=np.float64)
    sign_trace = _stable_sign(reference)
    change_boundaries = (
        np.flatnonzero(sign_trace[1:] != sign_trace[:-1]).astype(int) + 1
    )
    return change_boundaries[
        (change_boundaries > 0) & (change_boundaries < int(length))
    ]


def _uniform_mode_boundary_fallback(
    *,
    rng: np.random.Generator,
    target_density: float,
    geometry: ModeBoundaryGeometry,
    max_placement_attempts: int,
    overlap_policy: str,
    segment_count_range: Sequence[int],
    min_segment_length: int,
    logger: logging.Logger | None,
    segment_factory: Callable[..., SegmentPlan],
) -> list[SegmentPlan]:
    if logger is not None:
        logger.warning(
            "mode_boundary_segments found too few mode changes; falling "
            "back to uniform placement."
        )
    return sample_uniform_segments(
        rng=rng,
        target_density=target_density,
        series_length=geometry.length,
        channels=geometry.n_channels,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        segment_count_range=segment_count_range,
        min_segment_length=min_segment_length,
        logger=logger,
        segment_factory=segment_factory,
    )


def _sample_mode_boundary_lengths(
    rng: np.random.Generator,
    *,
    target_density: float,
    geometry: ModeBoundaryGeometry,
    segment_count_range: Sequence[int],
    min_segment_length: int,
) -> list[int]:
    low, high = int(segment_count_range[0]), int(segment_count_range[1])
    if low > high:
        low, high = high, low
    n_segments = int(rng.integers(low, high + 1))
    target_points = int(round(float(target_density) * geometry.length))
    target_points = max(target_points, n_segments)
    target_points = min(target_points, geometry.length)

    max_segments_for_min_length = max(
        1,
        target_points // max(1, int(min_segment_length)),
    )
    if n_segments > max_segments_for_min_length:
        n_segments = max_segments_for_min_length
    return sample_segment_lengths(
        rng,
        target_points,
        n_segments,
        min_segment_length=int(min_segment_length),
    )


def _stable_sign(values: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(values, dtype=np.float64)).astype(np.int8)
    last = 1
    for idx, value in enumerate(signs):
        if value == 0:
            signs[idx] = last
        else:
            last = int(value)
    return signs
