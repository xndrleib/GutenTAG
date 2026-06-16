"""RMJ mode-grid segment planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .common import sample_bounded_integer_lengths
from .mode_grid_placement import place_mode_grid_segments
from .mode_grid_types import ModeGridGeometry, ModeGridOptions
from .types import SegmentPlan


def sample_mode_grid_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    planner_cfg: Mapping[str, Any],
    anomaly_policy: Mapping[str, Any],
    anomaly_type: str,
    base_period_size: int | None,
    density_range: Sequence[float],
    segment_count_range: Sequence[int],
    min_segment_length: int,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample mode-correlation supports on the RMJ block grid.

    Parameters
    ----------
    rng, target_density, series_length, channels
        Randomness, target support density, and generated-series shape.
    max_placement_attempts, overlap_policy
        Random-placement retry budget and temporal overlap policy.
    planner_cfg, anomaly_policy
        Mode-grid options such as block bounds and gap blocks.
    anomaly_type, base_period_size
        Expected ``mode-correlation`` type and RMJ block size.
    density_range, segment_count_range, min_segment_length
        Bounds used to derive block budget, count, and per-segment lengths.
    segment_factory
        Factory used to construct segment plans.

    Returns
    -------
    list[SegmentPlan]
        Sorted segment plan aligned to the RMJ block grid.
    """

    _validate_mode_grid_request(anomaly_type, base_period_size)
    assert base_period_size is not None
    geometry = _mode_grid_geometry(series_length, channels, int(base_period_size))
    n_segments = _sample_segment_count(rng, segment_count_range)
    target_blocks = _target_block_budget(target_density, density_range, geometry)
    options = _mode_grid_options(
        planner_cfg,
        anomaly_policy,
        min_segment_length=min_segment_length,
        target_blocks=target_blocks,
        geometry=geometry,
    )
    n_segments, target_blocks = _fit_segment_count_and_budget(
        n_segments,
        target_blocks,
        options,
    )
    block_lengths = sample_bounded_integer_lengths(
        rng,
        total_points=int(target_blocks),
        n_segments=int(n_segments),
        min_value=int(options.min_blocks),
        max_value=int(options.max_blocks),
    )

    segments = place_mode_grid_segments(
        rng=rng,
        block_lengths=block_lengths,
        geometry=geometry,
        options=options,
        max_placement_attempts=max_placement_attempts,
        overlap_policy=overlap_policy,
        segment_factory=segment_factory,
    )
    segments.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
    return segments


def _validate_mode_grid_request(
    anomaly_type: str,
    base_period_size: int | None,
) -> None:
    if anomaly_type != "mode-correlation":
        raise ValueError(
            "mode_grid_segments planner is only supported for "
            "anomaly_type='mode-correlation'."
        )
    if base_period_size is None or int(base_period_size) <= 0:
        raise ValueError(
            "mode_grid_segments requires a positive base_period_size from "
            "the base oscillator."
        )


def _mode_grid_geometry(
    series_length: int,
    channels: int,
    base_period_size: int,
) -> ModeGridGeometry:
    block_size = max(1, int(base_period_size))
    length = int(series_length)
    # Use complete observable blocks when the series length is not an exact
    # multiple of the RMJ grid. The final right-censored block is valid clean
    # data, but using it as an anomaly support makes event lengths look off-grid.
    n_blocks = max(1, int(length // block_size))
    return ModeGridGeometry(
        length=length,
        n_channels=int(channels),
        block_size=block_size,
        n_blocks=n_blocks,
    )


def _sample_segment_count(
    rng: np.random.Generator,
    segment_count_range: Sequence[int],
) -> int:
    low, high = int(segment_count_range[0]), int(segment_count_range[1])
    if low > high:
        low, high = high, low
    return int(rng.integers(low, high + 1))


def _target_block_budget(
    target_density: float,
    density_range: Sequence[float],
    geometry: ModeGridGeometry,
) -> int:
    target_points = int(round(float(target_density) * geometry.length))
    target_blocks = max(
        1,
        int(round(float(target_points) / float(geometry.block_size))),
    )
    min_target_blocks, max_target_blocks = _density_block_bounds(
        density_range,
        geometry,
    )
    return int(np.clip(target_blocks, min_target_blocks, max_target_blocks))


def _density_block_bounds(
    density_range: Sequence[float],
    geometry: ModeGridGeometry,
) -> tuple[int, int]:
    density_min = max(0.0, min(float(density_range[0]), float(density_range[1])))
    density_max = min(1.0, max(float(density_range[0]), float(density_range[1])))
    min_target_blocks = max(
        1,
        int(np.ceil((density_min * float(geometry.length)) / geometry.block_size)),
    )
    max_target_blocks = max(
        min_target_blocks,
        int(np.floor((density_max * float(geometry.length)) / geometry.block_size)),
    )
    min_target_blocks = min(min_target_blocks, geometry.n_blocks)
    max_target_blocks = min(max_target_blocks, geometry.n_blocks)
    if max_target_blocks < min_target_blocks:
        max_target_blocks = min_target_blocks
    return min_target_blocks, max_target_blocks


def _mode_grid_options(
    planner_cfg: Mapping[str, Any],
    anomaly_policy: Mapping[str, Any],
    *,
    min_segment_length: int,
    target_blocks: int,
    geometry: ModeGridGeometry,
) -> ModeGridOptions:
    raw_min_blocks = planner_cfg.get(
        "min_blocks",
        anomaly_policy.get(
            "min_blocks",
            int(np.ceil(float(min_segment_length) / float(geometry.block_size))),
        ),
    )
    min_blocks = max(1, int(raw_min_blocks))
    raw_max_blocks = planner_cfg.get(
        "max_blocks",
        anomaly_policy.get("max_blocks", target_blocks),
    )
    max_blocks = max(min_blocks, int(raw_max_blocks))
    max_blocks = min(max_blocks, geometry.n_blocks)
    min_gap_blocks = max(
        0,
        int(
            planner_cfg.get(
                "min_gap_blocks",
                anomaly_policy.get("min_gap_blocks", 0),
            )
        ),
    )
    return ModeGridOptions(
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        min_gap_blocks=min_gap_blocks,
        min_gap_points=int(min_gap_blocks * geometry.block_size),
    )


def _fit_segment_count_and_budget(
    n_segments: int,
    target_blocks: int,
    options: ModeGridOptions,
) -> tuple[int, int]:
    max_segments_for_min_blocks = max(1, target_blocks // max(1, options.min_blocks))
    if n_segments > max_segments_for_min_blocks:
        n_segments = max_segments_for_min_blocks
    if n_segments <= 0:
        raise ValueError("mode_grid_segments could not allocate any segments.")
    if n_segments * options.max_blocks < target_blocks:
        target_blocks = n_segments * options.max_blocks
    if n_segments * options.min_blocks > target_blocks:
        target_blocks = n_segments * options.min_blocks
    return n_segments, target_blocks
