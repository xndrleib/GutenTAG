"""Shared trend-parameter planning data contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ParameterRealizer = Callable[[Mapping[str, Any], np.random.Generator], dict[str, Any]]
ParameterSanitizer = Callable[[str, Mapping[str, Any]], dict[str, Any]]
SeedDeriver = Callable[..., int]


@dataclass(frozen=True)
class TrendPlannerSettings:
    sine_min_cycles: float
    random_walk_min_segment_length: int


@dataclass
class TrendSegmentRequirements:
    min_lengths: list[int]
    attrs: list[dict[str, Any]]


__all__ = [
    "ParameterRealizer",
    "ParameterSanitizer",
    "SeedDeriver",
    "TrendPlannerSettings",
    "TrendSegmentRequirements",
]
