"""Runtime instance-level segment planning."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..config import merge_dicts, parse_pair
from .dispatcher import SegmentPlanSamplingConfig, sample_segment_plan
from .fixed_onset import inject_fixed_onset_split_context, temporal_union_density
from .period_geometry import compute_period_boundaries
from .policy import (
    resolve_minimum_segment_length,
    resolve_segment_planner,
    resolve_special_anomaly_policy,
)
from .types import SegmentPlan


class BaseChannelState(Protocol):
    """Base-channel state required by period-aware planning."""

    def get_period_size(self) -> int | None:
        """Return the realized oscillator period size when available."""


@dataclass(frozen=True)
class InstanceSegmentPlanningResult:
    """Result of planning anomaly segments for one generated instance."""

    segment_plan: list[SegmentPlan]
    planner_cfg: dict[str, Any]
    special_policy: dict[str, Any]
    target_density: float
    active_density_range: tuple[float, float]
    active_density_tolerance: float
    active_overlap_policy: str
    active_channel_policy: str


@dataclass(frozen=True)
class _InstancePlanningRequest:
    rng: np.random.Generator
    anomaly_type: str
    split: str
    clean_values: np.ndarray
    channel_bos: Sequence[BaseChannelState]
    anomaly_parameter_template: Mapping[str, Any]
    parameter_seed: int
    variant_anomaly_policy: Mapping[str, Any] | None
    variant_segment_planner: Mapping[str, Any] | None
    config: SegmentPlanSamplingConfig
    default_density_tolerance: float
    realize_parameters: Callable[
        [Mapping[str, Any], np.random.Generator], dict[str, Any]
    ]
    sanitize_parameters: Callable[[str, Mapping[str, Any]], dict[str, Any]]
    derive_seed: Callable[..., int]
    logger: logging.Logger | None


@dataclass(frozen=True)
class _PeriodPlanningMetadata:
    base_period_size: int | None
    period_boundaries: np.ndarray | None
    period_boundaries_by_channel: dict[int, np.ndarray | None]


@dataclass(frozen=True)
class _InstancePlanningContext:
    special_policy: dict[str, Any]
    planner_cfg: dict[str, Any]
    planner_name: str
    target_density: float
    active_density_range: tuple[float, float]
    active_overlap_policy: str
    active_channel_policy: str
    period_metadata: _PeriodPlanningMetadata


def plan_instance_segments(
    *,
    rng: np.random.Generator,
    anomaly_type: str,
    split: str,
    clean_values: np.ndarray,
    channel_bos: Sequence[BaseChannelState],
    anomaly_parameter_template: Mapping[str, Any],
    parameter_seed: int,
    variant_anomaly_policy: Mapping[str, Any] | None,
    variant_segment_planner: Mapping[str, Any] | None,
    config: SegmentPlanSamplingConfig,
    default_density_tolerance: float,
    realize_parameters: Callable[
        [Mapping[str, Any], np.random.Generator], dict[str, Any]
    ],
    sanitize_parameters: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    derive_seed: Callable[..., int],
    logger: logging.Logger | None = None,
) -> InstanceSegmentPlanningResult:
    """Plan anomaly segments and expose summary metadata for one instance.

    Parameters
    ----------
    rng, anomaly_type, split, clean_values, channel_bos
        Runtime instance context and realized clean/base-channel state.
    variant_anomaly_policy, variant_segment_planner, config
        Effective variant and runtime planning configuration.
    remaining arguments
        Parameter callbacks, seed context, and optional logger.

    Returns
    -------
    InstanceSegmentPlanningResult
        Planned segments plus the effective planning metadata.
    """
    return _plan_instance_segments_from_request(_InstancePlanningRequest(**locals()))


def _plan_instance_segments_from_request(
    request: _InstancePlanningRequest,
) -> InstanceSegmentPlanningResult:
    context = _resolve_instance_planning_context(request)
    segment_plan = _sample_instance_segment_plan(request, context)
    target_density, active_density_range = _adjust_density_after_planning(
        segment_plan=segment_plan,
        planner_name=context.planner_name,
        planner_cfg=context.planner_cfg,
        series_length=request.config.series_length,
        target_density=context.target_density,
        active_density_range=context.active_density_range,
    )
    return InstanceSegmentPlanningResult(
        segment_plan=segment_plan,
        planner_cfg=context.planner_cfg,
        special_policy=context.special_policy,
        target_density=target_density,
        active_density_range=active_density_range,
        active_density_tolerance=_active_density_tolerance(
            context.special_policy,
            context.planner_cfg,
            request.default_density_tolerance,
        ),
        active_overlap_policy=context.active_overlap_policy,
        active_channel_policy=context.active_channel_policy,
    )


def _resolve_instance_planning_context(
    request: _InstancePlanningRequest,
) -> _InstancePlanningContext:
    special_policy = merge_dicts(
        resolve_special_anomaly_policy(
            request.anomaly_type,
            request.config.special_anomaly_policies,
        ),
        dict(request.variant_anomaly_policy or {}),
    )
    planner_cfg = _resolve_instance_planner_cfg(request)
    planner_name = str(planner_cfg.get("planner", "uniform_segments")).lower()
    active_density_range = _resolve_active_density_range(
        special_policy=special_policy,
        planner_cfg=planner_cfg,
        default_density_range=request.config.density_range,
    )
    target_density = float(
        request.rng.uniform(active_density_range[0], active_density_range[1])
    )
    period_metadata = _period_planning_metadata(request)
    _apply_trend_period_minimum(request, special_policy, period_metadata)
    active_overlap_policy = str(
        special_policy.get(
            "overlap_policy",
            planner_cfg.get("overlap_policy", request.config.overlap_policy),
        )
    )
    active_channel_policy = str(
        special_policy.get("channel_policy", request.config.channel_policy)
    )
    return _InstancePlanningContext(
        special_policy=special_policy,
        planner_cfg=planner_cfg,
        planner_name=planner_name,
        target_density=target_density,
        active_density_range=active_density_range,
        active_overlap_policy=active_overlap_policy,
        active_channel_policy=active_channel_policy,
        period_metadata=period_metadata,
    )


def _resolve_instance_planner_cfg(
    request: _InstancePlanningRequest,
) -> dict[str, Any]:
    if isinstance(request.variant_segment_planner, Mapping):
        planner_cfg = copy.deepcopy(dict(request.variant_segment_planner))
    else:
        planner_cfg = resolve_segment_planner(
            anomaly_type=request.anomaly_type,
            segment_planner=request.config.segment_planner,
            special_anomaly_policies=request.config.special_anomaly_policies,
        )
    return inject_fixed_onset_split_context(planner_cfg, request.split)


def _period_planning_metadata(
    request: _InstancePlanningRequest,
) -> _PeriodPlanningMetadata:
    return _PeriodPlanningMetadata(
        base_period_size=_resolve_base_period_size(request.channel_bos),
        period_boundaries=compute_period_boundaries(
            getattr(request.channel_bos[0], "frequency", None),
            series_length=request.config.series_length,
        ),
        period_boundaries_by_channel={
            channel: compute_period_boundaries(
                getattr(bo, "frequency", None),
                series_length=request.config.series_length,
            )
            for channel, bo in enumerate(request.channel_bos)
        },
    )


def _apply_trend_period_minimum(
    request: _InstancePlanningRequest,
    special_policy: dict[str, Any],
    period_metadata: _PeriodPlanningMetadata,
) -> None:
    base_period_size = period_metadata.base_period_size
    if (
        request.anomaly_type != "trend"
        or base_period_size is None
        or base_period_size <= 1
    ):
        return
    existing_min = int(
        special_policy.get(
            "min_segment_length",
            resolve_minimum_segment_length(
                "trend",
                request.config.min_segment_length_by_anomaly,
            ),
        )
    )
    special_policy["min_segment_length"] = int(max(existing_min, 2 * base_period_size))


def _sample_instance_segment_plan(
    request: _InstancePlanningRequest,
    context: _InstancePlanningContext,
) -> list[SegmentPlan]:
    return sample_segment_plan(
        rng=request.rng,
        target_density=context.target_density,
        anomaly_type=request.anomaly_type,
        config=request.config,
        clean_values=request.clean_values,
        anomaly_policy=context.special_policy,
        planner_cfg=context.planner_cfg,
        base_period_size=context.period_metadata.base_period_size,
        period_boundaries=context.period_metadata.period_boundaries,
        period_boundaries_by_channel=(
            context.period_metadata.period_boundaries_by_channel
        ),
        anomaly_parameter_template=request.anomaly_parameter_template,
        parameter_seed=int(request.parameter_seed),
        realize_parameters=request.realize_parameters,
        sanitize_parameters=request.sanitize_parameters,
        derive_seed=request.derive_seed,
        logger=request.logger,
    )


def _adjust_density_after_planning(
    *,
    segment_plan: Sequence[SegmentPlan],
    planner_name: str,
    planner_cfg: Mapping[str, Any],
    series_length: int,
    target_density: float,
    active_density_range: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    if (
        planner_name == "point_events_from_density"
        and planner_cfg.get("segment_count_range") is not None
    ):
        target_density = float(len(segment_plan) / int(series_length))
        active_density_range = (target_density, target_density)
    if planner_name == "fixed_first_onset_segments":
        target_density = temporal_union_density(segment_plan, series_length)
        active_density_range = (target_density, target_density)
    return target_density, active_density_range


def _active_density_tolerance(
    special_policy: Mapping[str, Any],
    planner_cfg: Mapping[str, Any],
    default_density_tolerance: float,
) -> float:
    return float(
        special_policy.get(
            "density_tolerance",
            planner_cfg.get("density_tolerance", default_density_tolerance),
        )
    )


def _resolve_active_density_range(
    *,
    special_policy: Mapping[str, Any],
    planner_cfg: Mapping[str, Any],
    default_density_range: Sequence[float],
) -> tuple[float, float]:
    active_density_range_raw = special_policy.get(
        "density_range",
        planner_cfg.get("density_range"),
    )
    if active_density_range_raw is not None:
        return parse_pair(active_density_range_raw, "density_range")
    return float(default_density_range[0]), float(default_density_range[1])


def _resolve_base_period_size(
    channel_bos: Sequence[BaseChannelState],
) -> int | None:
    base_period_size = channel_bos[0].get_period_size()
    if base_period_size is None:
        return None
    return int(base_period_size)
