"""Adaptive effect-strength parameter policies."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .common import ordered_numeric_pair


class EffectSegment(Protocol):
    """Minimal segment contract for local effect-strength policies."""

    @property
    def attrs(self) -> Mapping[str, Any]: ...


def apply_amplitude_parameter_policy(
    *,
    segment_plan: Sequence[EffectSegment],
    segment_params: Sequence[Mapping[str, Any]],
    planner_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Adjust amplitude factors using local segment statistics.

    Parameters
    ----------
    segment_plan : Sequence[EffectSegment]
        Planned anomaly segments with optional local clean-window statistics.
    segment_params : Sequence[Mapping[str, Any]]
        Realized per-segment anomaly parameters.
    planner_cfg : Mapping[str, Any]
        Planner options for adaptive strength, bounds, and deadzones.

    Returns
    -------
    list[dict[str, Any]]
        Parameter dictionaries with amplitude policy applied.
    """
    adaptive_strength = bool(planner_cfg.get("adaptive_strength", False))
    min_effect_delta = max(0.0, float(planner_cfg.get("min_effect_delta", 0.0)))
    amplitude_factor_bounds = _optional_ordered_pair(
        planner_cfg,
        "amplitude_factor_bounds",
        "segment_planner.amplitude_factor_bounds",
    )
    deadzone = _optional_ordered_pair(
        planner_cfg,
        "amplitude_factor_deadzone",
        "segment_planner.amplitude_factor_deadzone",
    )

    resolved = _copy_segment_params(segment_params)
    for idx, segment in enumerate(segment_plan):
        if "amplitude_factor" not in resolved[idx]:
            continue
        factor = float(resolved[idx]["amplitude_factor"])
        if adaptive_strength and min_effect_delta > 0.0:
            local_scale = _local_amplitude_scale(segment.attrs)
            if local_scale > 1e-12:
                required_offset = min_effect_delta / local_scale
                current_offset = abs(factor - 1.0)
                if required_offset > current_offset:
                    direction = -1.0 if factor < 1.0 else 1.0
                    if current_offset <= 1e-12:
                        direction = -1.0 if (idx % 2 == 0) else 1.0
                    factor = 1.0 + direction * required_offset
            current_floor = float(resolved[idx].get("min_effect_delta", 0.0))
            resolved[idx]["min_effect_delta"] = float(
                max(current_floor, min_effect_delta)
            )
            resolved[idx].setdefault("center_mode", "linear")

        if deadzone is not None and deadzone[0] <= factor <= deadzone[1]:
            if factor >= 1.0:
                factor = deadzone[1]
            else:
                factor = deadzone[0]

        if amplitude_factor_bounds is not None:
            factor = float(
                np.clip(
                    factor,
                    amplitude_factor_bounds[0],
                    amplitude_factor_bounds[1],
                )
            )
        resolved[idx]["amplitude_factor"] = float(factor)
    return resolved


def apply_mean_parameter_policy(
    *,
    segment_params: Sequence[Mapping[str, Any]],
    planner_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Adjust mean offsets using configured deadzones and floors.

    Parameters
    ----------
    segment_params : Sequence[Mapping[str, Any]]
        Realized per-segment anomaly parameters.
    planner_cfg : Mapping[str, Any]
        Planner options for adaptive strength and offset deadzones.

    Returns
    -------
    list[dict[str, Any]]
        Parameter dictionaries with mean-offset policy applied.
    """
    adaptive_strength = bool(planner_cfg.get("adaptive_strength", False))
    min_effect_delta = max(0.0, float(planner_cfg.get("min_effect_delta", 0.0)))
    offset_deadzone = _optional_ordered_pair(
        planner_cfg, "offset_deadzone", "segment_planner.offset_deadzone"
    )

    resolved = _copy_segment_params(segment_params)
    for idx, params in enumerate(resolved):
        if "offset" not in params:
            continue
        offset = float(params["offset"])
        if (
            offset_deadzone is not None
            and offset_deadzone[0] <= offset <= offset_deadzone[1]
        ):
            offset = offset_deadzone[1] if offset >= 0.0 else offset_deadzone[0]
        if adaptive_strength and min_effect_delta > 0.0:
            if abs(offset) < min_effect_delta:
                direction = 1.0 if offset >= 0.0 else -1.0
                if abs(offset) <= 1e-12:
                    direction = -1.0 if (idx % 2 == 0) else 1.0
                offset = direction * min_effect_delta
            current_floor = float(params.get("min_effect_delta", 0.0))
            params["min_effect_delta"] = float(max(current_floor, min_effect_delta))
        params["offset"] = float(offset)
    return resolved


def apply_trend_parameter_policy(
    *,
    segment_plan: Sequence[EffectSegment],
    segment_params: Sequence[Mapping[str, Any]],
    planner_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach trend effect floors and local segment statistics.

    Parameters
    ----------
    segment_plan : Sequence[EffectSegment]
        Planned anomaly segments with optional local clean-window statistics.
    segment_params : Sequence[Mapping[str, Any]]
        Realized per-segment anomaly parameters.
    planner_cfg : Mapping[str, Any]
        Planner options for adaptive trend strength.

    Returns
    -------
    list[dict[str, Any]]
        Parameter dictionaries with trend policy applied.
    """
    adaptive_strength = bool(planner_cfg.get("adaptive_strength", False))
    min_effect_delta = max(0.0, float(planner_cfg.get("min_effect_delta", 0.0)))

    resolved = _copy_segment_params(segment_params)
    for idx, segment in enumerate(segment_plan):
        if adaptive_strength and min_effect_delta > 0.0:
            current_floor = float(resolved[idx].get("min_effect_delta", 0.0))
            resolved[idx]["min_effect_delta"] = float(
                max(current_floor, min_effect_delta)
            )
        if "window_rms" in segment.attrs:
            resolved[idx]["window_rms"] = float(segment.attrs["window_rms"])
        if "window_peak" in segment.attrs:
            resolved[idx]["window_peak"] = float(segment.attrs["window_peak"])
    return resolved


def _copy_segment_params(
    segment_params: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(params)) for params in segment_params]


def _local_amplitude_scale(attrs: Mapping[str, Any]) -> float:
    return float(
        attrs.get(
            "window_residual_scale",
            attrs.get(
                "window_robust_scale",
                attrs.get("window_std", attrs.get("window_rms", 0.0)),
            ),
        )
    )


def _optional_ordered_pair(
    planner_cfg: Mapping[str, Any],
    key: str,
    field_name: str,
) -> tuple[float, float] | None:
    raw_value = planner_cfg.get(key)
    if raw_value is None:
        return None
    return ordered_numeric_pair(raw_value, field_name)
