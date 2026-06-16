"""Trend-parameter-aware segment planning."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..signal_energy import window_rms_from_prefix as _window_rms
from .types import SegmentPlan
from .trend_parameter_placement import place_trend_segments
from .trend_parameter_requirements import (
    build_trend_segment_requirements,
    fit_trend_requirements_to_budget,
    sample_segment_lengths_with_minima,
    trend_planner_settings,
)
from .trend_parameter_types import (
    ParameterRealizer,
    ParameterSanitizer,
    SeedDeriver,
)


@dataclass(frozen=True)
class _TrendPlannerRequest:
    rng: np.random.Generator
    target_density: float
    series_length: int
    channels: int
    max_placement_attempts: int
    overlap_policy: str
    planner_cfg: Mapping[str, Any]
    anomaly_parameter_template: Mapping[str, Any]
    parameter_seed: int
    clean_values: np.ndarray | None
    segment_count_range: Sequence[int]
    base_min_segment_length: int
    realize_parameters: ParameterRealizer
    sanitize_parameters: ParameterSanitizer
    derive_seed: SeedDeriver
    logger: logging.Logger | None
    segment_factory: Callable[..., SegmentPlan]


def sample_trend_parameter_aware_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    planner_cfg: Mapping[str, Any],
    anomaly_parameter_template: Mapping[str, Any],
    parameter_seed: int,
    clean_values: np.ndarray | None,
    segment_count_range: Sequence[int],
    base_min_segment_length: int,
    realize_parameters: ParameterRealizer,
    sanitize_parameters: ParameterSanitizer,
    derive_seed: SeedDeriver,
    logger: logging.Logger | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample trend segments with parameter-driven length constraints.

    Parameters
    ----------
    rng, target_density, series_length, channels
        Randomness, target support density, and generated-series shape.
    planner_cfg, anomaly_parameter_template, parameter_seed
        Trend constraints and deterministic parameter-realization inputs.
    remaining arguments
        Placement policy, parameter callbacks, optional clean values, and factory.

    Returns
    -------
    list[SegmentPlan]
        Sorted trend-aware segment plan.
    """
    return _sample_trend_parameter_aware_segments_from_request(
        _TrendPlannerRequest(**locals())
    )


def _sample_trend_parameter_aware_segments_from_request(
    request: _TrendPlannerRequest,
) -> list[SegmentPlan]:
    length = int(request.series_length)
    n_channels = int(request.channels)
    low, high = int(request.segment_count_range[0]), int(request.segment_count_range[1])
    if low > high:
        low, high = high, low
    n_segments = int(request.rng.integers(low, high + 1))
    if n_segments > length:
        raise ValueError(
            f"Requested n_segments={n_segments} exceeds series length={length}."
        )
    target_points = _target_points_for_density(
        request.target_density,
        length,
        n_segments,
    )
    settings = trend_planner_settings(request.planner_cfg)
    requirements = build_trend_segment_requirements(
        n_segments=n_segments,
        anomaly_parameter_template=request.anomaly_parameter_template,
        parameter_seed=request.parameter_seed,
        base_min_segment_length=request.base_min_segment_length,
        series_length=length,
        settings=settings,
        realize_parameters=request.realize_parameters,
        sanitize_parameters=request.sanitize_parameters,
        derive_seed=request.derive_seed,
    )
    fit_trend_requirements_to_budget(
        requirements=requirements,
        series_length=length,
        target_points=target_points,
        logger=request.logger,
    )

    target_points = min(target_points, length)
    lengths = sample_segment_lengths_with_minima(
        rng=request.rng,
        target_points=target_points,
        minimum_lengths=requirements.min_lengths,
        series_length=length,
    )
    segments = place_trend_segments(
        rng=request.rng,
        lengths=lengths,
        attrs=requirements.attrs,
        series_length=length,
        channels=n_channels,
        max_placement_attempts=request.max_placement_attempts,
        overlap_policy=request.overlap_policy,
        segment_factory=request.segment_factory,
    )

    if request.clean_values is not None:
        _annotate_clean_window_energy(
            segments=segments,
            clean_values=request.clean_values,
            series_length=length,
            channels=n_channels,
        )

    segments.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
    return segments


def _target_points_for_density(
    target_density: float,
    length: int,
    n_segments: int,
) -> int:
    target_points = int(round(float(target_density) * int(length)))
    target_points = max(target_points, int(n_segments))
    return min(target_points, int(length))


def _annotate_clean_window_energy(
    *,
    segments: Sequence[SegmentPlan],
    clean_values: np.ndarray,
    series_length: int,
    channels: int,
) -> None:
    if clean_values.shape != (int(series_length), int(channels)):
        raise ValueError(
            "trend_parameter_aware_segments expected clean_values shape "
            f"({int(series_length)}, {int(channels)}), got {clean_values.shape}."
        )
    sq_prefix_by_channel = [
        np.concatenate(
            [
                [0.0],
                np.cumsum(np.square(clean_values[:, channel]), dtype=np.float64),
            ]
        )
        for channel in range(int(channels))
    ]
    abs_by_channel = [
        np.abs(clean_values[:, channel]).astype(np.float64)
        for channel in range(int(channels))
    ]
    for segment in segments:
        channel = int(segment.channel)
        window_rms = _window_rms(
            sq_prefix_by_channel[channel],
            int(segment.start),
            int(segment.end),
        )
        window_peak = float(
            np.max(abs_by_channel[channel][int(segment.start) : int(segment.end)])
        )
        segment.attrs.update(
            {
                "window_rms": float(window_rms),
                "window_peak": float(window_peak),
            }
        )
