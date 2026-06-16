"""Segment-planning helpers for TS dataset generation."""

from .energy_aware import sample_energy_aware_segments
from .dispatcher import (
    SegmentPlanSamplingConfig,
    build_segment_plan_sampling_config,
    sample_segment_plan,
)
from .fixed_onset import (
    event_extra_from_segment_attrs,
    inject_fixed_onset_split_context,
    requested_pre_context_from_split,
    sample_fixed_first_onset_segments,
    temporal_union_density,
)
from .instance import InstanceSegmentPlanningResult, plan_instance_segments
from .mode_boundary import sample_mode_boundary_segments
from .mode_grid import sample_mode_grid_segments
from .period_geometry import compute_period_boundaries
from .period_locked import (
    sample_period_locked_frequency_segments,
)
from .policy import (
    resolve_minimum_segment_length,
    resolve_segment_planner,
    resolve_special_anomaly_policy,
)
from .point_events import sample_point_event_segments
from .trend_parameter import sample_trend_parameter_aware_segments
from .types import SegmentPlan
from .uniform import sample_uniform_segments

__all__ = [
    "SegmentPlan",
    "SegmentPlanSamplingConfig",
    "InstanceSegmentPlanningResult",
    "build_segment_plan_sampling_config",
    "compute_period_boundaries",
    "event_extra_from_segment_attrs",
    "inject_fixed_onset_split_context",
    "requested_pre_context_from_split",
    "resolve_minimum_segment_length",
    "resolve_segment_planner",
    "resolve_special_anomaly_policy",
    "sample_energy_aware_segments",
    "sample_fixed_first_onset_segments",
    "plan_instance_segments",
    "sample_mode_boundary_segments",
    "sample_mode_grid_segments",
    "sample_period_locked_frequency_segments",
    "sample_point_event_segments",
    "sample_segment_plan",
    "sample_trend_parameter_aware_segments",
    "sample_uniform_segments",
    "temporal_union_density",
]
