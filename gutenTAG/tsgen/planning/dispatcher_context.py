"""Planner-dispatch configuration and policy resolution."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from ..config import parse_pair, parse_pair_int
from .dispatcher_types import (
    PlannerDispatchContext,
    PlannerDispatchRequest,
    SegmentPlanRuntimeConfig,
    SegmentPlanSamplingConfig,
)
from .policy import (
    resolve_minimum_segment_length,
    resolve_segment_planner,
    resolve_special_anomaly_policy,
)


def build_segment_plan_sampling_config(
    runtime_config: SegmentPlanRuntimeConfig,
) -> SegmentPlanSamplingConfig:
    """Build planner dispatch config from generator runtime config.

    Parameters
    ----------
    runtime_config : SegmentPlanRuntimeConfig
        Object exposing the generator fields required by segment planning.

    Returns
    -------
    SegmentPlanSamplingConfig
        Planning-only runtime configuration.
    """
    return SegmentPlanSamplingConfig(
        series_length=int(runtime_config.length),
        channels=int(runtime_config.channels),
        max_placement_attempts=int(runtime_config.max_placement_attempts),
        overlap_policy=str(runtime_config.overlap_policy),
        channel_policy=str(runtime_config.channel_policy),
        density_range=runtime_config.density_range,
        segment_count_range=runtime_config.segment_count_range,
        min_segment_length_by_anomaly=runtime_config.min_segment_length_by_anomaly,
        segment_planner=runtime_config.segment_planner,
        special_anomaly_policies=runtime_config.special_anomaly_policies,
    )


def resolve_planner_dispatch_context(
    *,
    anomaly_type: str,
    config: SegmentPlanSamplingConfig,
    anomaly_policy: Mapping[str, Any] | None,
    planner_cfg: Mapping[str, Any] | None,
) -> PlannerDispatchContext:
    """Resolve planner config and effective policy overrides."""

    if planner_cfg is None:
        active_planner_cfg = resolve_segment_planner(
            anomaly_type=anomaly_type,
            segment_planner=config.segment_planner,
            special_anomaly_policies=config.special_anomaly_policies,
        )
    else:
        active_planner_cfg = copy.deepcopy(dict(planner_cfg))
    if anomaly_policy is None:
        special_policy = resolve_special_anomaly_policy(
            anomaly_type,
            config.special_anomaly_policies,
        )
    else:
        special_policy = copy.deepcopy(dict(anomaly_policy))
    planner_name = str(active_planner_cfg.get("planner", "uniform_segments")).lower()
    overlap_policy = str(
        special_policy.get(
            "overlap_policy",
            active_planner_cfg.get("overlap_policy", config.overlap_policy),
        )
    )
    channel_policy = str(special_policy.get("channel_policy", config.channel_policy))
    return PlannerDispatchContext(
        active_planner_cfg=active_planner_cfg,
        special_policy=special_policy,
        planner_name=planner_name,
        overlap_policy=overlap_policy,
        channel_policy=channel_policy,
    )


def configured_segment_count_range(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> tuple[int, int]:
    """Return effective segment-count range for a planner dispatch."""

    return parse_pair_int(
        context.special_policy.get(
            "segment_count_range",
            context.active_planner_cfg.get(
                "segment_count_range",
                request.config.segment_count_range,
            ),
        ),
        "segment_count_range",
    )


def configured_density_range(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
) -> tuple[float, float]:
    """Return effective density range for a planner dispatch."""

    return parse_pair(
        context.special_policy.get(
            "density_range",
            context.active_planner_cfg.get(
                "density_range", request.config.density_range
            ),
        ),
        "density_range",
    )


def configured_min_segment_length(
    request: PlannerDispatchRequest,
    context: PlannerDispatchContext,
    *,
    anomaly_type: str | None = None,
) -> int:
    """Return effective minimum segment length for a planner dispatch."""

    active_anomaly_type = anomaly_type or request.anomaly_type
    return int(
        context.special_policy.get(
            "min_segment_length",
            context.active_planner_cfg.get(
                "min_segment_length",
                minimum_segment_length(request.config, active_anomaly_type),
            ),
        )
    )


def minimum_segment_length(
    config: SegmentPlanSamplingConfig,
    anomaly_type: str,
) -> int:
    """Return default anomaly-specific minimum segment length."""

    return resolve_minimum_segment_length(
        anomaly_type,
        config.min_segment_length_by_anomaly,
    )


__all__ = [
    "build_segment_plan_sampling_config",
    "configured_density_range",
    "configured_min_segment_length",
    "configured_segment_count_range",
    "minimum_segment_length",
    "resolve_planner_dispatch_context",
]
