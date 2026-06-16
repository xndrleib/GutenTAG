"""Config-bound runtime helpers for dataset generation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..anomalies import Anomaly
from ..generator.anomaly_application import AnomalyApplicationRuntime, apply_anomalies
from ..generator.base_channels import (
    compose_channel_noise_window,
    compose_channel_window_with_variations,
    replace_channel_noise_window,
    replace_channel_window_with_variations,
)
from .config import DEFAULT_ANOMALY_OVERRIDES, TSGeneratorConfig
from .io import sanitize_json_value
from .labels import normalize_subsequence_length, resolve_label_bounds_from_effect
from .parameters import (
    classify_base_family,
    resolve_anomaly_parameters_for_segments,
)
from .planning import SegmentPlan
from .seeding import derive_seed as _derive_seed


def resolve_runtime_anomaly_parameters_for_segments(
    *,
    config: TSGeneratorConfig,
    base_kind: str,
    anomaly_type: str,
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Mapping[str, Any] | None,
    anomaly_parameters_instance: Mapping[str, Any] | None,
    segment_plan: Sequence[SegmentPlan],
    parameter_seed: int,
    planner_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve per-segment anomaly parameters using generation config."""

    return resolve_anomaly_parameters_for_segments(
        anomaly_parameter_policy=config.anomaly_parameter_policy,
        base_family=classify_base_family(base_kind),
        anomaly_type=anomaly_type,
        anomaly_parameter_template=anomaly_parameter_template,
        fixed_anomaly_parameters=fixed_anomaly_parameters,
        anomaly_parameters_instance=anomaly_parameters_instance,
        segment_plan=list(segment_plan),
        parameter_seed=parameter_seed,
        planner_cfg=planner_cfg,
        default_anomaly_overrides=DEFAULT_ANOMALY_OVERRIDES,
        derive_seed=_derive_seed,
    )


def apply_runtime_anomalies(
    *,
    config: TSGeneratorConfig,
    anomaly_objects: Sequence[Anomaly],
    segment_plan: Sequence[SegmentPlan],
    base: np.ndarray,
    channel_bos: Sequence[Any],
    anomaly_seed: int,
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply runtime anomaly objects with config-bound callbacks."""

    return apply_anomalies(
        anomaly_objects=list(anomaly_objects),
        segment_plan=list(segment_plan),
        base=base,
        channel_bos=list(channel_bos),
        anomaly_seed=anomaly_seed,
        anomaly_type=anomaly_type,
        anomaly_parameters_per_segment=[
            dict(params) for params in anomaly_parameters_per_segment
        ],
        series_length=config.length,
        channels=config.channels,
        runtime=_anomaly_application_runtime(config),
    )


def _anomaly_application_runtime(
    config: TSGeneratorConfig,
) -> AnomalyApplicationRuntime:
    return AnomalyApplicationRuntime(
        compose_window=lambda *, base, bo, channel, start, end: (
            compose_channel_window_with_variations(
                base=base,
                bo=bo,
                channel=channel,
                start=start,
                end=end,
                series_length=config.length,
            )
        ),
        replace_window=lambda *, base, bo, channel, start, end, target_observed: (
            replace_channel_window_with_variations(
                base=base,
                bo=bo,
                channel=channel,
                start=start,
                end=end,
                target_observed=target_observed,
                series_length=config.length,
            )
        ),
        compose_noise=lambda *, bo, start, end: compose_channel_noise_window(
            bo=bo,
            start=start,
            end=end,
            series_length=config.length,
        ),
        replace_noise=lambda *, bo, start, end, target_noise: (
            replace_channel_noise_window(
                bo=bo,
                start=start,
                end=end,
                target_noise=target_noise,
                series_length=config.length,
            )
        ),
        resolve_label_bounds=lambda *, protocol_start, protocol_end, delta, anomaly_type: (
            resolve_label_bounds_from_effect(
                protocol_start=protocol_start,
                protocol_end=protocol_end,
                delta=delta,
                anomaly_type=anomaly_type,
                support_label_mode=config.support_label_mode,
                support_eps_mode=config.support_eps_mode,
                support_eps_value=config.support_eps_value,
                min_effective_label_length_non_extremum=(
                    config.min_effective_label_length_non_extremum
                ),
            )
        ),
        normalize_subsequence=lambda subsequence, expected_length: (
            normalize_subsequence_length(
                subsequence=subsequence,
                expected_length=expected_length,
                policy=config.length_normalization,
            )
        ),
        to_builtin=sanitize_json_value,
    )


__all__ = [
    "apply_runtime_anomalies",
    "resolve_runtime_anomaly_parameters_for_segments",
]
