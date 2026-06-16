"""Anomaly parameter sanitization policy."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

TRANSITION_LENGTH_ANOMALIES = frozenset(
    {"amplitude", "trend", "mean", "platform", "variance", "pattern"}
)


def sanitize_anomaly_parameters(
    anomaly_type: str, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize anomaly parameters for generator compatibility."""

    resolved = copy.deepcopy(dict(parameters))
    _sanitize_transition_length(anomaly_type, resolved)
    if anomaly_type == "amplitude":
        _sanitize_amplitude_parameters(resolved)
    if anomaly_type == "mean":
        _sanitize_mean_parameters(resolved)
    if anomaly_type == "pattern":
        _sanitize_pattern_parameters(resolved)
    if anomaly_type == "trend":
        _sanitize_trend_parameters(resolved)
    if anomaly_type == "pattern-shift":
        _sanitize_pattern_shift_parameters(resolved)
    if anomaly_type == "platform":
        _clamp_nonnegative_float_param(resolved, "min_effect_delta")
    if anomaly_type == "variance":
        _clamp_nonnegative_float_param(resolved, "min_effect_delta")
    if anomaly_type == "extremum":
        _sanitize_extremum_parameters(resolved)
    return resolved


def _sanitize_transition_length(
    anomaly_type: str,
    params: dict[str, Any],
) -> None:
    if anomaly_type not in TRANSITION_LENGTH_ANOMALIES:
        return
    if "transition_length" in params:
        params["transition_length"] = max(0, int(params["transition_length"]))


def _sanitize_amplitude_parameters(params: dict[str, Any]) -> None:
    if "amplitude_factor" in params:
        params["amplitude_factor"] = float(params["amplitude_factor"])
    _lower_string_param(params, "center_mode")
    _clamp_nonnegative_float_param(params, "min_effect_delta")
    _clamp_nonnegative_float_param(params, "min_residual_scale")


def _sanitize_mean_parameters(params: dict[str, Any]) -> None:
    if "offset" in params:
        params["offset"] = float(params["offset"])
    _clamp_nonnegative_float_param(params, "min_effect_delta")


def _sanitize_pattern_parameters(params: dict[str, Any]) -> None:
    _clamp_nonnegative_float_param(params, "min_effect_delta")
    _clamp_nonnegative_float_param(params, "min_window_ptp")
    if "adaptive_blend" in params:
        params["adaptive_blend"] = bool(params["adaptive_blend"])
    _clamp_nonnegative_float_param(params, "blend_strength")


def _sanitize_trend_parameters(params: dict[str, Any]) -> None:
    _lower_string_param(params, "boundary_mode")
    _lower_string_param(params, "envelope_kind")
    _clamp_nonnegative_float_param(params, "min_effect_delta")


def _sanitize_pattern_shift_parameters(params: dict[str, Any]) -> None:
    transition_window = max(1, abs(int(params.get("transition_window", 10))))
    shift_by = int(params.get("shift_by", 0))
    shift_by = int(np.clip(shift_by, -transition_window, transition_window))
    if shift_by == 0 and transition_window > 0:
        shift_by = 1
    params["transition_window"] = transition_window
    params["shift_by"] = shift_by
    _lower_string_param(params, "crossfade_mode")
    _clamp_nonnegative_float_param(params, "min_effect_delta")


def _sanitize_extremum_parameters(params: dict[str, Any]) -> None:
    context_window = int(params.get("context_window", 10))
    params["context_window"] = max(1, abs(context_window))
    if "min" in params:
        params["min"] = bool(params["min"])
    if "local" in params:
        params["local"] = bool(params["local"])


def _lower_string_param(params: dict[str, Any], key: str) -> None:
    if key in params:
        params[key] = str(params[key]).lower()


def _clamp_nonnegative_float_param(params: dict[str, Any], key: str) -> None:
    if key in params:
        params[key] = max(0.0, float(params[key]))
