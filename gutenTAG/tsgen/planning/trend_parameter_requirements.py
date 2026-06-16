"""Parameter-derived minimum-length policy for trend-aware planning."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...base_oscillations.utils.math_func_support import SAMPLING_F
from .trend_parameter_types import (
    ParameterRealizer,
    ParameterSanitizer,
    SeedDeriver,
    TrendPlannerSettings,
    TrendSegmentRequirements,
)


def trend_planner_settings(
    planner_cfg: Mapping[str, Any],
) -> TrendPlannerSettings:
    """Resolve trend-aware planner settings from raw planner config."""

    sine_min_cycles = float(planner_cfg.get("sine_min_cycles", 0.30))
    if sine_min_cycles <= 0.0:
        raise ValueError("trend_parameter_aware_segments.sine_min_cycles must be > 0")
    return TrendPlannerSettings(
        sine_min_cycles=sine_min_cycles,
        random_walk_min_segment_length=int(
            max(1, int(planner_cfg.get("random_walk_min_segment_length", 8)))
        ),
    )


def build_trend_segment_requirements(
    *,
    n_segments: int,
    anomaly_parameter_template: Mapping[str, Any],
    parameter_seed: int,
    base_min_segment_length: int,
    series_length: int,
    settings: TrendPlannerSettings,
    realize_parameters: ParameterRealizer,
    sanitize_parameters: ParameterSanitizer,
    derive_seed: SeedDeriver,
) -> TrendSegmentRequirements:
    """Realize trend params and derive per-segment minimum lengths."""

    requirements = TrendSegmentRequirements(min_lengths=[], attrs=[])
    for idx in range(int(n_segments)):
        params_rng = np.random.default_rng(
            derive_seed(parameter_seed, "trend-planner-params", str(idx))
        )
        trend_params = sanitize_parameters(
            "trend",
            realize_parameters(anomaly_parameter_template, params_rng),
        )
        attrs, min_segment_length = _trend_attrs_and_min_length(
            trend_params=trend_params,
            base_min_segment_length=base_min_segment_length,
            sine_min_cycles=settings.sine_min_cycles,
            random_walk_min_segment_length=settings.random_walk_min_segment_length,
            series_length=series_length,
        )
        requirements.min_lengths.append(min_segment_length)
        requirements.attrs.append(attrs)
    return requirements


def fit_trend_requirements_to_budget(
    *,
    requirements: TrendSegmentRequirements,
    series_length: int,
    target_points: int,
    logger: logging.Logger | None,
) -> None:
    """Mutate requirements so they fit series length and target budget."""

    dropped_segments = _drop_infeasible_minima(
        per_segment_min_lengths=requirements.min_lengths,
        per_segment_attrs=requirements.attrs,
        series_length=series_length,
        target_points=target_points,
    )
    _relax_single_minimum_if_needed(
        per_segment_min_lengths=requirements.min_lengths,
        per_segment_attrs=requirements.attrs,
        target_points=target_points,
        logger=logger,
    )
    _log_dropped_segments(
        dropped_segments=dropped_segments,
        remaining_count=len(requirements.min_lengths),
        target_points=target_points,
        logger=logger,
    )


def sample_segment_lengths_with_minima(
    *,
    rng: np.random.Generator,
    target_points: int,
    minimum_lengths: Sequence[int],
    series_length: int,
) -> list[int]:
    """Sample segment lengths subject to per-segment minimum lengths."""

    if len(minimum_lengths) == 0:
        return []
    minima = np.asarray([max(1, int(value)) for value in minimum_lengths], dtype=int)
    total_min = int(np.sum(minima))
    total_points = int(max(int(target_points), total_min))
    total_points = int(min(total_points, int(series_length)))
    if total_points < total_min:
        raise ValueError(
            "Requested target points are infeasible under per-segment minimum "
            f"lengths (target={total_points}, min_total={total_min})."
        )
    extra_points = int(total_points - total_min)
    if extra_points > 0:
        weights = rng.random(size=int(minima.shape[0])).astype(np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.full(minima.shape[0], 1.0 / float(minima.shape[0]))
        else:
            weights = weights / weight_sum
        extra = rng.multinomial(extra_points, weights)
        minima = minima + extra
    return [int(value) for value in minima.tolist()]


def _trend_attrs_and_min_length(
    *,
    trend_params: Mapping[str, Any],
    base_min_segment_length: int,
    sine_min_cycles: float,
    random_walk_min_segment_length: int,
    series_length: int,
) -> tuple[dict[str, Any], int]:
    attrs: dict[str, Any] = {"trend_params": copy.deepcopy(dict(trend_params))}
    min_segment_length = int(max(1, base_min_segment_length))
    oscillation = trend_params.get("oscillation")
    if isinstance(oscillation, Mapping):
        osc_kind = str(oscillation.get("kind", "")).lower()
        attrs["trend_oscillation_kind"] = osc_kind
        if osc_kind == "sine":
            frequency = float(oscillation.get("frequency", 0.0))
            if frequency > 0.0:
                min_from_cycles = int(
                    np.ceil((float(SAMPLING_F) * sine_min_cycles) / frequency)
                )
                min_segment_length = max(min_segment_length, min_from_cycles)
                attrs["trend_sine_frequency"] = float(frequency)
                attrs["trend_sine_min_cycles"] = float(sine_min_cycles)
        elif osc_kind == "random-walk":
            min_segment_length = max(
                min_segment_length,
                int(random_walk_min_segment_length),
            )
            attrs["trend_random_walk_min_segment_length"] = int(
                random_walk_min_segment_length
            )

    min_segment_length = int(np.clip(min_segment_length, 1, int(series_length)))
    attrs["trend_min_segment_length"] = int(min_segment_length)
    return attrs, min_segment_length


def _drop_infeasible_minima(
    *,
    per_segment_min_lengths: list[int],
    per_segment_attrs: list[dict[str, Any]],
    series_length: int,
    target_points: int,
) -> list[dict[str, Any]]:
    dropped_segments: list[dict[str, Any]] = []

    def drop_largest_minimum(reason: str) -> None:
        drop_idx = int(np.argmax(np.asarray(per_segment_min_lengths, dtype=int)))
        dropped = int(per_segment_min_lengths.pop(drop_idx))
        dropped_attrs = per_segment_attrs.pop(drop_idx)
        dropped_kind = str(dropped_attrs.get("trend_oscillation_kind", "unknown"))
        dropped_segments.append(
            {
                "reason": reason,
                "min_length": dropped,
                "kind": dropped_kind,
            }
        )

    while sum(per_segment_min_lengths) > int(series_length) and (
        len(per_segment_min_lengths) > 1
    ):
        drop_largest_minimum("series_length_infeasibility")

    while sum(per_segment_min_lengths) > int(target_points) and (
        len(per_segment_min_lengths) > 1
    ):
        drop_largest_minimum("target_density_infeasibility")

    return dropped_segments


def _relax_single_minimum_if_needed(
    *,
    per_segment_min_lengths: list[int],
    per_segment_attrs: list[dict[str, Any]],
    target_points: int,
    logger: logging.Logger | None,
) -> None:
    if not (
        len(per_segment_min_lengths) == 1
        and per_segment_min_lengths[0] > int(target_points)
    ):
        return
    original = int(per_segment_min_lengths[0])
    per_segment_min_lengths[0] = int(target_points)
    per_segment_attrs[0]["trend_min_segment_length_original"] = int(original)
    per_segment_attrs[0]["trend_min_segment_length_relaxed"] = int(target_points)
    if logger is not None:
        logger.warning(
            "Relaxing single trend minimum length to preserve target density "
            "(original=%s, relaxed=%s).",
            original,
            target_points,
        )


def _log_dropped_segments(
    *,
    dropped_segments: Sequence[Mapping[str, Any]],
    remaining_count: int,
    target_points: int,
    logger: logging.Logger | None,
) -> None:
    if len(dropped_segments) == 0 or logger is None:
        return
    reason_counts: dict[str, int] = {}
    for item in dropped_segments:
        key = str(item["reason"])
        reason_counts[key] = int(reason_counts.get(key, 0) + 1)
    max_dropped = int(max(int(item["min_length"]) for item in dropped_segments))
    logger.warning(
        "trend_parameter_aware_segments dropped segments to satisfy constraints "
        "(dropped=%s, remaining=%s, target_points=%s, max_dropped_min=%s, "
        "reasons=%s).",
        len(dropped_segments),
        remaining_count,
        target_points,
        max_dropped,
        reason_counts,
    )


__all__ = [
    "build_trend_segment_requirements",
    "fit_trend_requirements_to_budget",
    "sample_segment_lengths_with_minima",
    "trend_planner_settings",
]
