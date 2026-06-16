"""Transition parameter policy for anomaly segment parameters."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


class SegmentLike(Protocol):
    """Minimal segment contract required by transition policy."""

    @property
    def length(self) -> int: ...


SeedDeriver = Callable[..., int]
ParameterSanitizer = Callable[[str, Mapping[str, Any]], dict[str, Any]]

TRANSITION_LENGTH_ANOMALIES = frozenset(
    {
        "amplitude",
        "trend",
        "mean",
        "platform",
        "variance",
        "pattern",
        "covariance-change",
        "correlation-flip",
        "channel-rewiring",
        "lag-synchronization",
        "shared-factor-break",
    }
)
PATTERN_ADAPTIVE_BASE_FAMILIES = frozenset(
    {"discontinuous_periodic", "motif_rich", "mode_switching"}
)
TREND_TRANSITION_ENVELOPE_BASE_FAMILIES = frozenset(
    {"smooth_trend", "structural_periodic"}
)


@dataclass(frozen=True)
class _TransitionTemplateDefaults:
    transition_length: Any
    transition_window: Any
    template_transition_length: Any
    template_transition_window: Any


def transition_cap_for_anomaly(anomaly_type: str, segment_length: int) -> int:
    """Return the maximum transition span for an anomaly segment.

    Parameters
    ----------
    anomaly_type : str
        Canonical anomaly type.
    segment_length : int
        Segment length in time steps.

    Returns
    -------
    int
        Non-negative maximum transition span.
    """
    if anomaly_type == "trend":
        return max(0, int(segment_length))
    return max(0, int(segment_length // 2))


def sample_transition_span(
    *,
    anomaly_type: str,
    base_family: str,
    segment_length: int,
    rng: np.random.Generator,
) -> int:
    """Sample a transition span from anomaly and base-family policy."""

    cap = transition_cap_for_anomaly(anomaly_type, segment_length)
    if cap <= 0:
        return 0
    if anomaly_type == "extremum":
        return 0
    mode = _sample_transition_mode(
        anomaly_type=anomaly_type,
        base_family=base_family,
        rng=rng,
    )
    low_ratio, high_ratio = _transition_span_ratios(mode)
    sampled = int(
        round(float(rng.uniform(low_ratio, high_ratio)) * float(segment_length))
    )
    if mode != "abrupt":
        sampled = max(1, sampled)
    return int(max(0, min(cap, sampled)))


def _sample_transition_mode(
    *,
    anomaly_type: str,
    base_family: str,
    rng: np.random.Generator,
) -> str:
    mode = _sample_initial_transition_mode(anomaly_type, rng)
    return _adapt_transition_mode(
        mode=mode,
        anomaly_type=anomaly_type,
        base_family=base_family,
        rng=rng,
    )


def _sample_initial_transition_mode(
    anomaly_type: str,
    rng: np.random.Generator,
) -> str:
    if anomaly_type in {"mean", "platform", "variance"}:
        return str(rng.choice(["abrupt", "mixed", "smooth"], p=[0.25, 0.50, 0.25]))
    if anomaly_type in {
        "trend",
        "covariance-change",
        "correlation-flip",
        "shared-factor-break",
        "channel-rewiring",
    }:
        return str(rng.choice(["mixed", "smooth"], p=[0.30, 0.70]))
    if anomaly_type in {
        "pattern",
        "pattern-shift",
        "lag-synchronization",
        "frequency",
    }:
        return str(rng.choice(["abrupt", "mixed", "smooth"], p=[0.15, 0.55, 0.30]))
    return str(rng.choice(["abrupt", "mixed", "smooth"], p=[0.30, 0.50, 0.20]))


def _adapt_transition_mode(
    *,
    mode: str,
    anomaly_type: str,
    base_family: str,
    rng: np.random.Generator,
) -> str:
    if base_family in {"discontinuous_periodic", "mode_switching"} and mode == "smooth":
        mode = "mixed"
    if base_family == "smooth_trend" and anomaly_type in {
        "trend",
        "covariance-change",
        "correlation-flip",
        "shared-factor-break",
    }:
        mode = str(rng.choice(["mixed", "smooth"], p=[0.15, 0.85]))
    if base_family == "structural_periodic" and anomaly_type in {
        "covariance-change",
        "correlation-flip",
        "shared-factor-break",
    }:
        mode = str(rng.choice(["mixed", "smooth"], p=[0.65, 0.35]))
    if base_family == "discontinuous_periodic" and anomaly_type in {
        "pattern",
        "pattern-shift",
    }:
        mode = str(rng.choice(["mixed", "smooth"], p=[0.75, 0.25]))
    if base_family == "motif_rich" and anomaly_type == "pattern":
        mode = str(rng.choice(["mixed", "smooth"], p=[0.70, 0.30]))
    return mode


def _transition_span_ratios(mode: str) -> tuple[float, float]:
    if mode == "abrupt":
        return 0.0, 0.04
    if mode == "mixed":
        return 0.06, 0.16
    return 0.18, 0.32


def apply_transition_policy_to_segments(
    *,
    base_family: str,
    anomaly_type: str,
    anomaly_parameter_template: Mapping[str, Any],
    default_anomaly_overrides: Mapping[str, Mapping[str, Any]],
    segment_plan: Sequence[SegmentLike],
    segment_params: Sequence[Mapping[str, Any]],
    parameter_seed: int,
    derive_seed: SeedDeriver,
    sanitize_parameters: ParameterSanitizer,
) -> list[dict[str, Any]]:
    """Apply transition defaults to per-segment anomaly parameters."""

    if len(segment_params) == 0:
        return []
    defaults = _transition_template_defaults(
        anomaly_type=anomaly_type,
        anomaly_parameter_template=anomaly_parameter_template,
        default_anomaly_overrides=default_anomaly_overrides,
    )
    resolved: list[dict[str, Any]] = []
    for idx, segment in enumerate(segment_plan):
        rng = np.random.default_rng(
            derive_seed(parameter_seed, "transition-policy", str(idx))
        )
        params = _apply_segment_transition_policy(
            base_family=base_family,
            anomaly_type=anomaly_type,
            params=segment_params[idx],
            segment_length=max(1, int(segment.length)),
            defaults=defaults,
            rng=rng,
        )
        resolved.append(sanitize_parameters(anomaly_type, params))
    return resolved


def _transition_template_defaults(
    *,
    anomaly_type: str,
    anomaly_parameter_template: Mapping[str, Any],
    default_anomaly_overrides: Mapping[str, Mapping[str, Any]],
) -> _TransitionTemplateDefaults:
    default_overrides = default_anomaly_overrides.get(anomaly_type, {})
    return _TransitionTemplateDefaults(
        transition_length=default_overrides.get("transition_length"),
        transition_window=default_overrides.get("transition_window"),
        template_transition_length=anomaly_parameter_template.get("transition_length"),
        template_transition_window=anomaly_parameter_template.get("transition_window"),
    )


def _apply_segment_transition_policy(
    *,
    base_family: str,
    anomaly_type: str,
    params: Mapping[str, Any],
    segment_length: int,
    defaults: _TransitionTemplateDefaults,
    rng: np.random.Generator,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(params))
    _apply_transition_length_default(
        base_family=base_family,
        anomaly_type=anomaly_type,
        params=resolved,
        segment_length=segment_length,
        defaults=defaults,
        rng=rng,
    )
    _apply_pattern_shift_window_default(
        base_family=base_family,
        anomaly_type=anomaly_type,
        params=resolved,
        segment_length=segment_length,
        defaults=defaults,
        rng=rng,
    )
    _apply_pattern_defaults(
        base_family=base_family,
        anomaly_type=anomaly_type,
        params=resolved,
    )
    _apply_pattern_shift_crossfade_default(
        base_family=base_family,
        anomaly_type=anomaly_type,
        params=resolved,
        rng=rng,
    )
    _apply_trend_defaults(
        base_family=base_family,
        anomaly_type=anomaly_type,
        params=resolved,
    )
    return resolved


def _apply_transition_length_default(
    *,
    base_family: str,
    anomaly_type: str,
    params: dict[str, Any],
    segment_length: int,
    defaults: _TransitionTemplateDefaults,
    rng: np.random.Generator,
) -> None:
    if anomaly_type not in TRANSITION_LENGTH_ANOMALIES:
        return
    if not _uses_default_transition_length(params, defaults):
        return
    params["transition_length"] = int(
        sample_transition_span(
            anomaly_type=anomaly_type,
            base_family=base_family,
            segment_length=segment_length,
            rng=rng,
        )
    )


def _apply_pattern_shift_window_default(
    *,
    base_family: str,
    anomaly_type: str,
    params: dict[str, Any],
    segment_length: int,
    defaults: _TransitionTemplateDefaults,
    rng: np.random.Generator,
) -> None:
    if anomaly_type != "pattern-shift":
        return
    if not _uses_default_transition_window(params, defaults):
        return
    params["transition_window"] = int(
        max(
            1,
            sample_transition_span(
                anomaly_type=anomaly_type,
                base_family=base_family,
                segment_length=segment_length,
                rng=rng,
            ),
        )
    )


def _apply_pattern_defaults(
    *,
    base_family: str,
    anomaly_type: str,
    params: dict[str, Any],
) -> None:
    if anomaly_type != "pattern" or base_family not in PATTERN_ADAPTIVE_BASE_FAMILIES:
        return
    params.setdefault("adaptive_blend", True)
    params.setdefault("blend_strength", 1.0)


def _apply_pattern_shift_crossfade_default(
    *,
    base_family: str,
    anomaly_type: str,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    if anomaly_type != "pattern-shift" or "crossfade_mode" in params:
        return
    params["crossfade_mode"] = (
        "cosine"
        if base_family in PATTERN_ADAPTIVE_BASE_FAMILIES
        else ("linear" if rng.random() < 0.35 else "cosine")
    )


def _apply_trend_defaults(
    *,
    base_family: str,
    anomaly_type: str,
    params: dict[str, Any],
) -> None:
    if anomaly_type != "trend":
        return
    params.setdefault("boundary_mode", "inside_window_zero_endpoints")
    if "envelope_kind" not in params:
        params["envelope_kind"] = (
            "transition"
            if base_family in TREND_TRANSITION_ENVELOPE_BASE_FAMILIES
            else "sine2"
        )


def _uses_default_transition_length(
    params: Mapping[str, Any],
    defaults: _TransitionTemplateDefaults,
) -> bool:
    return (
        "transition_length" not in params
        or defaults.template_transition_length == defaults.transition_length
    )


def _uses_default_transition_window(
    params: Mapping[str, Any],
    defaults: _TransitionTemplateDefaults,
) -> bool:
    return (
        "transition_window" not in params
        or defaults.template_transition_window == defaults.transition_window
    )
