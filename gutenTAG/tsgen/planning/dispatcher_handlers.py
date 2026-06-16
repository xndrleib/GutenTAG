"""Concrete planner-dispatch handlers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ...generator.segment_groups import expand_segments_by_channel_policy
from ..config import parse_pair, parse_pair_int
from .dispatcher_context import (
    configured_density_range,
    configured_min_segment_length,
    configured_segment_count_range,
)
from .dispatcher_types import (
    PlannerDispatchContext,
    PlannerDispatchRequest,
    SegmentPlanSamplingConfig,
)
from .energy_aware import sample_energy_aware_segments
from .fixed_onset import sample_fixed_first_onset_segments
from .mode_boundary import sample_mode_boundary_segments
from .mode_grid import sample_mode_grid_segments
from .period_locked import sample_period_locked_frequency_segments
from .point_events import sample_point_event_segments
from .trend_parameter import sample_trend_parameter_aware_segments
from .types import SegmentPlan
from .uniform import sample_uniform_segments


def sample_raw_segments(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    """Dispatch to the concrete planner selected by context."""

    if context.planner_name == "point_events_from_density":
        return _dispatch_point_events(request, context)
    if context.planner_name == "fixed_first_onset_segments":
        return _dispatch_fixed_first_onset(request, context)
    if context.planner_name == "period_locked_frequency":
        return _dispatch_period_locked_frequency(request, context)
    if context.planner_name == "energy_aware_segments":
        return _dispatch_energy_aware(request, context)
    if context.planner_name == "trend_parameter_aware_segments":
        return _dispatch_trend_parameter_aware(request, context)
    if context.planner_name == "mode_boundary_segments":
        return _dispatch_mode_boundary(request, context)
    if context.planner_name == "mode_grid_segments":
        return _dispatch_mode_grid(request, context)
    return _dispatch_uniform(request, context)


def _dispatch_point_events(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    segment_count_range: tuple[int, int] | None = None
    if context.active_planner_cfg.get("segment_count_range") is not None:
        segment_count_range = parse_pair_int(
            context.active_planner_cfg["segment_count_range"],
            "segment_count_range",
        )
    density_range: tuple[float, float] | None = None
    if (
        segment_count_range is None
        and context.active_planner_cfg.get("density_range") is not None
    ):
        density_range = parse_pair(
            context.active_planner_cfg["density_range"],
            "density_range",
        )
    return sample_point_event_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        overlap_policy=context.overlap_policy,
        planner_cfg=context.active_planner_cfg,
        segment_count_range=segment_count_range,
        density_range=density_range,
        segment_factory=request.segment_factory,
    )


def _dispatch_fixed_first_onset(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    return sample_fixed_first_onset_segments(
        rng=request.rng,
        anomaly_type=request.anomaly_type,
        planner_cfg=context.active_planner_cfg,
        series_length=request.config.series_length,
        channels=request.config.channels,
        segment_factory=request.segment_factory,
        base_period_size=request.base_period_size,
    )


def _dispatch_period_locked_frequency(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    if request.anomaly_type != "frequency":
        raise ValueError(
            "period_locked_frequency planner is only supported for "
            "anomaly_type='frequency'."
        )
    return sample_period_locked_frequency_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        max_placement_attempts=request.config.max_placement_attempts,
        overlap_policy=context.overlap_policy,
        planner_cfg=context.active_planner_cfg,
        base_period_size=request.base_period_size,
        period_boundaries=request.period_boundaries,
        period_boundaries_by_channel=request.period_boundaries_by_channel,
        segment_count_range=configured_segment_count_range(request, context),
        logger=request.logger,
        segment_factory=request.segment_factory,
    )


def _dispatch_energy_aware(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    return sample_energy_aware_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        max_placement_attempts=request.config.max_placement_attempts,
        overlap_policy=context.overlap_policy,
        planner_cfg=context.active_planner_cfg,
        anomaly_type=request.anomaly_type,
        clean_values=request.clean_values,
        segment_count_range=configured_segment_count_range(request, context),
        min_segment_length=configured_min_segment_length(request, context),
        logger=request.logger,
        segment_factory=request.segment_factory,
    )


def _dispatch_trend_parameter_aware(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    if request.anomaly_type != "trend":
        raise ValueError(
            "trend_parameter_aware_segments planner is only supported for "
            "anomaly_type='trend'."
        )
    if request.anomaly_parameter_template is None or request.parameter_seed is None:
        raise ValueError(
            "trend_parameter_aware_segments planner requires "
            "anomaly_parameter_template and parameter_seed."
        )
    return sample_trend_parameter_aware_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        max_placement_attempts=request.config.max_placement_attempts,
        overlap_policy=context.overlap_policy,
        planner_cfg=context.active_planner_cfg,
        anomaly_parameter_template=request.anomaly_parameter_template,
        parameter_seed=int(request.parameter_seed),
        clean_values=request.clean_values,
        segment_count_range=configured_segment_count_range(request, context),
        base_min_segment_length=configured_min_segment_length(
            request,
            context,
            anomaly_type="trend",
        ),
        realize_parameters=request.realize_parameters,
        sanitize_parameters=request.sanitize_parameters,
        derive_seed=request.derive_seed,
        logger=request.logger,
        segment_factory=request.segment_factory,
    )


def _dispatch_mode_boundary(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    return sample_mode_boundary_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        max_placement_attempts=request.config.max_placement_attempts,
        overlap_policy=context.overlap_policy,
        planner_cfg=context.active_planner_cfg,
        anomaly_type=request.anomaly_type,
        clean_values=request.clean_values,
        segment_count_range=configured_segment_count_range(request, context),
        min_segment_length=configured_min_segment_length(request, context),
        logger=request.logger,
        segment_factory=request.segment_factory,
    )


def _dispatch_mode_grid(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    return sample_mode_grid_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        max_placement_attempts=request.config.max_placement_attempts,
        overlap_policy=context.overlap_policy,
        planner_cfg=context.active_planner_cfg,
        anomaly_policy=context.special_policy,
        anomaly_type=request.anomaly_type,
        base_period_size=request.base_period_size,
        density_range=configured_density_range(request, context),
        segment_count_range=configured_segment_count_range(request, context),
        min_segment_length=configured_min_segment_length(request, context),
        segment_factory=request.segment_factory,
    )


def _dispatch_uniform(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> list[SegmentPlan]:
    segment_count_range = configured_segment_count_range(request, context)
    if segment_count_range[0] > segment_count_range[1]:
        segment_count_range = (segment_count_range[1], segment_count_range[0])
    return sample_uniform_segments(
        rng=request.rng,
        target_density=request.target_density,
        series_length=request.config.series_length,
        channels=request.config.channels,
        max_placement_attempts=request.config.max_placement_attempts,
        overlap_policy=context.overlap_policy,
        segment_count_range=segment_count_range,
        min_segment_length=configured_min_segment_length(request, context),
        logger=request.logger,
        segment_factory=request.segment_factory,
    )


def apply_channel_policy(
    segments: Iterable[SegmentPlan],
    rng: np.random.Generator,
    config: SegmentPlanSamplingConfig,
    channel_policy: str,
    segment_factory: Callable[..., SegmentPlan],
) -> list[SegmentPlan]:
    """Expand raw segments according to effective channel policy."""

    return expand_segments_by_channel_policy(
        segments=segments,
        rng=rng,
        n_channels=config.channels,
        segment_factory=segment_factory,
        channel_policy=channel_policy,
    )


__all__ = [
    "apply_channel_policy",
    "sample_raw_segments",
]
