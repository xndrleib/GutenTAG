"""Planner dispatcher for TS dataset segment sampling."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from .dispatcher_context import (
    build_segment_plan_sampling_config,
    resolve_planner_dispatch_context,
)
from .dispatcher_handlers import apply_channel_policy, sample_raw_segments
from .dispatcher_types import (
    PlannerDispatchRequest,
    SegmentPlanRuntimeConfig,
    SegmentPlanSamplingConfig,
)
from .types import SegmentPlan

__all__ = [
    "SegmentPlanRuntimeConfig",
    "SegmentPlanSamplingConfig",
    "build_segment_plan_sampling_config",
    "sample_segment_plan",
]


def sample_segment_plan(
    *,
    rng: np.random.Generator,
    target_density: float,
    anomaly_type: str,
    config: SegmentPlanSamplingConfig,
    clean_values: np.ndarray | None = None,
    anomaly_policy: Mapping[str, Any] | None = None,
    planner_cfg: Mapping[str, Any] | None = None,
    base_period_size: int | None = None,
    period_boundaries: Iterable[int] | None = None,
    period_boundaries_by_channel: Mapping[int, Iterable[int] | None] | None = None,
    anomaly_parameter_template: Mapping[str, Any] | None = None,
    parameter_seed: int | None = None,
    realize_parameters: Callable[
        [Mapping[str, Any], np.random.Generator], dict[str, Any]
    ],
    sanitize_parameters: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    derive_seed: Callable[..., int],
    logger: logging.Logger | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample a channel-policy-expanded segment plan."""
    request = PlannerDispatchRequest(
        rng=rng,
        target_density=target_density,
        anomaly_type=anomaly_type,
        config=config,
        clean_values=clean_values,
        base_period_size=base_period_size,
        period_boundaries=period_boundaries,
        period_boundaries_by_channel=period_boundaries_by_channel,
        anomaly_parameter_template=anomaly_parameter_template,
        parameter_seed=parameter_seed,
        realize_parameters=realize_parameters,
        sanitize_parameters=sanitize_parameters,
        derive_seed=derive_seed,
        logger=logger,
        segment_factory=segment_factory,
    )
    context = resolve_planner_dispatch_context(
        anomaly_type=anomaly_type,
        config=config,
        anomaly_policy=anomaly_policy,
        planner_cfg=planner_cfg,
    )
    raw_segments = sample_raw_segments(request, context)
    return apply_channel_policy(
        raw_segments,
        rng,
        config,
        context.channel_policy,
        segment_factory,
    )
