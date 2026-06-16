"""Per-segment anomaly parameter resolution."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .effects import (
    apply_amplitude_parameter_policy,
    apply_mean_parameter_policy,
    apply_trend_parameter_policy,
)
from .frequency import apply_period_locked_frequency_policy
from .sampling import realize_parameters
from .sanitization import sanitize_anomaly_parameters
from .transitions import apply_transition_policy_to_segments


class ParameterSegment(Protocol):
    """Minimal segment contract required by parameter resolution."""

    @property
    def length(self) -> int: ...

    @property
    def attrs(self) -> Mapping[str, Any]: ...


SeedDeriver = Callable[..., int]


def resolve_instance_anomaly_parameters(
    *,
    anomaly_parameter_policy: str,
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Resolve the instance-level anomaly-parameter snapshot.

    Parameters
    ----------
    anomaly_parameter_policy : str
        One of the generator's anomaly-parameter policies.
    anomaly_parameter_template : Mapping[str, Any]
        Parameter template for the anomaly type.
    fixed_anomaly_parameters : Mapping[str, Any] or None
        Variant-level realized parameters, when the variant policy fixed them.
    rng : numpy.random.Generator
        Random generator used for deterministic instance-level realization.

    Returns
    -------
    dict[str, Any] or None
        Instance-level anomaly parameters for summaries and per-segment
        resolution, or ``None`` when parameters are sampled per segment.
    """
    if anomaly_parameter_policy == "fixed_per_variant":
        if fixed_anomaly_parameters is None:
            return realize_parameters(anomaly_parameter_template, rng)
        return copy.deepcopy(dict(fixed_anomaly_parameters))

    if anomaly_parameter_policy == "random_per_instance":
        return realize_parameters(anomaly_parameter_template, rng)

    return None


def resolve_anomaly_parameters_for_segments(
    *,
    anomaly_parameter_policy: str,
    base_family: str,
    anomaly_type: str,
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Mapping[str, Any] | None,
    anomaly_parameters_instance: Mapping[str, Any] | None,
    segment_plan: Sequence[ParameterSegment],
    parameter_seed: int,
    planner_cfg: Mapping[str, Any] | None,
    default_anomaly_overrides: Mapping[str, Mapping[str, Any]],
    derive_seed: SeedDeriver,
) -> list[dict[str, Any]]:
    """Resolve sanitized anomaly parameters for every planned segment."""

    planner = dict(planner_cfg or {})
    segment_params = _realize_initial_segment_parameters(
        anomaly_parameter_policy=anomaly_parameter_policy,
        anomaly_type=anomaly_type,
        anomaly_parameter_template=anomaly_parameter_template,
        fixed_anomaly_parameters=fixed_anomaly_parameters,
        anomaly_parameters_instance=anomaly_parameters_instance,
        segment_plan=segment_plan,
        parameter_seed=parameter_seed,
        derive_seed=derive_seed,
    )
    segment_params = _apply_planner_parameter_policies(
        anomaly_type=anomaly_type,
        segment_plan=segment_plan,
        segment_params=segment_params,
        planner=planner,
        parameter_seed=parameter_seed,
        derive_seed=derive_seed,
    )
    return apply_transition_policy_to_segments(
        base_family=base_family,
        anomaly_type=anomaly_type,
        anomaly_parameter_template=anomaly_parameter_template,
        default_anomaly_overrides=default_anomaly_overrides,
        segment_plan=segment_plan,
        segment_params=segment_params,
        parameter_seed=parameter_seed,
        derive_seed=derive_seed,
        sanitize_parameters=sanitize_anomaly_parameters,
    )


def _apply_planner_parameter_policies(
    *,
    anomaly_type: str,
    segment_plan: Sequence[ParameterSegment],
    segment_params: Sequence[Mapping[str, Any]],
    planner: Mapping[str, Any],
    parameter_seed: int,
    derive_seed: SeedDeriver,
) -> list[dict[str, Any]]:
    planner_name = str(planner.get("planner", "uniform_segments")).lower()
    if anomaly_type == "trend" and planner_name == "trend_parameter_aware_segments":
        segment_params = _apply_planned_trend_parameters(
            segment_plan=segment_plan,
            segment_params=segment_params,
        )

    if anomaly_type == "frequency" and planner_name == "period_locked_frequency":
        segment_params = apply_period_locked_frequency_policy(
            segment_plan=segment_plan,
            segment_params=segment_params,
            planner_cfg=planner,
            parameter_seed=parameter_seed,
            derive_seed=derive_seed,
        )

    if anomaly_type == "amplitude":
        segment_params = apply_amplitude_parameter_policy(
            segment_plan=segment_plan,
            segment_params=segment_params,
            planner_cfg=planner,
        )

    if anomaly_type == "mean":
        segment_params = apply_mean_parameter_policy(
            segment_params=segment_params,
            planner_cfg=planner,
        )

    if anomaly_type == "trend":
        segment_params = apply_trend_parameter_policy(
            segment_plan=segment_plan,
            segment_params=segment_params,
            planner_cfg=planner,
        )
    return [copy.deepcopy(dict(params)) for params in segment_params]


def _realize_initial_segment_parameters(
    *,
    anomaly_parameter_policy: str,
    anomaly_type: str,
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Mapping[str, Any] | None,
    anomaly_parameters_instance: Mapping[str, Any] | None,
    segment_plan: Sequence[ParameterSegment],
    parameter_seed: int,
    derive_seed: SeedDeriver,
) -> list[dict[str, Any]]:
    if anomaly_parameter_policy == "fixed_per_variant":
        if fixed_anomaly_parameters is None:
            rng = np.random.default_rng(derive_seed(parameter_seed, "fixed-anomaly"))
            resolved = realize_parameters(anomaly_parameter_template, rng)
        else:
            resolved = copy.deepcopy(dict(fixed_anomaly_parameters))
        resolved = sanitize_anomaly_parameters(anomaly_type, resolved)
        return [copy.deepcopy(resolved) for _ in segment_plan]

    if anomaly_parameter_policy == "random_per_instance":
        if anomaly_parameters_instance is None:
            rng = np.random.default_rng(derive_seed(parameter_seed, "instance-anomaly"))
            resolved = realize_parameters(anomaly_parameter_template, rng)
        else:
            resolved = copy.deepcopy(dict(anomaly_parameters_instance))
        resolved = sanitize_anomaly_parameters(anomaly_type, resolved)
        return [copy.deepcopy(resolved) for _ in segment_plan]

    segment_params: list[dict[str, Any]] = []
    for idx, _segment in enumerate(segment_plan):
        rng = np.random.default_rng(derive_seed(parameter_seed, "segment", str(idx)))
        resolved = realize_parameters(anomaly_parameter_template, rng)
        segment_params.append(sanitize_anomaly_parameters(anomaly_type, resolved))
    return segment_params


def _apply_planned_trend_parameters(
    *,
    segment_plan: Sequence[ParameterSegment],
    segment_params: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    resolved = [copy.deepcopy(dict(params)) for params in segment_params]
    for idx, segment in enumerate(segment_plan):
        planned = segment.attrs.get("trend_params")
        if isinstance(planned, Mapping):
            resolved[idx] = sanitize_anomaly_parameters(
                "trend", copy.deepcopy(dict(planned))
            )
    return resolved
