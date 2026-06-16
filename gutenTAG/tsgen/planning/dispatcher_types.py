"""Shared planner-dispatch data contracts."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .types import SegmentPlan


@dataclass(frozen=True)
class SegmentPlanSamplingConfig:
    """Runtime configuration needed by segment-plan dispatch.

    Parameters
    ----------
    series_length : int
        Generated series length.
    channels : int
        Number of generated channels.
    max_placement_attempts : int
        Maximum random placement attempts.
    overlap_policy : str
        Default overlap policy.
    channel_policy : str
        Default channel policy.
    density_range : Sequence[float]
        Default anomaly density range.
    segment_count_range : Sequence[int]
        Default segment count range.
    min_segment_length_by_anomaly : Mapping[str, int]
        Per-anomaly minimum-length overrides.
    segment_planner : Mapping[str, Any]
        Configured planner mapping.
    special_anomaly_policies : Mapping[str, Any]
        Configured anomaly-specific policy mapping.
    """

    series_length: int
    channels: int
    max_placement_attempts: int
    overlap_policy: str
    channel_policy: str
    density_range: Sequence[float]
    segment_count_range: Sequence[int]
    min_segment_length_by_anomaly: Mapping[str, int]
    segment_planner: Mapping[str, Any]
    special_anomaly_policies: Mapping[str, Any]


class SegmentPlanRuntimeConfig(Protocol):
    """Runtime config fields required for segment-plan sampling."""

    @property
    def length(self) -> int: ...

    @property
    def channels(self) -> int: ...

    @property
    def max_placement_attempts(self) -> int: ...

    @property
    def overlap_policy(self) -> str: ...

    @property
    def channel_policy(self) -> str: ...

    @property
    def density_range(self) -> Sequence[float]: ...

    @property
    def segment_count_range(self) -> Sequence[int]: ...

    @property
    def min_segment_length_by_anomaly(self) -> Mapping[str, int]: ...

    @property
    def segment_planner(self) -> Mapping[str, Any]: ...

    @property
    def special_anomaly_policies(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PlannerDispatchContext:
    active_planner_cfg: dict[str, Any]
    special_policy: dict[str, Any]
    planner_name: str
    overlap_policy: str
    channel_policy: str


@dataclass(frozen=True)
class PlannerDispatchRequest:
    rng: np.random.Generator
    target_density: float
    anomaly_type: str
    config: SegmentPlanSamplingConfig
    clean_values: np.ndarray | None
    base_period_size: int | None
    period_boundaries: Iterable[int] | None
    period_boundaries_by_channel: Mapping[int, Iterable[int] | None] | None
    anomaly_parameter_template: Mapping[str, Any] | None
    parameter_seed: int | None
    realize_parameters: Callable[
        [Mapping[str, Any], np.random.Generator], dict[str, Any]
    ]
    sanitize_parameters: Callable[[str, Mapping[str, Any]], dict[str, Any]]
    derive_seed: Callable[..., int]
    logger: logging.Logger | None
    segment_factory: Callable[..., SegmentPlan]


__all__ = [
    "PlannerDispatchContext",
    "PlannerDispatchRequest",
    "SegmentPlanRuntimeConfig",
    "SegmentPlanSamplingConfig",
]
