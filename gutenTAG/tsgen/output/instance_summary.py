"""Per-instance summary helpers for TS dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import sanitize_json_value
from .instance_density import compute_density_metrics, validate_density
from .instance_events import SegmentLike, paired_event_stats


@dataclass(frozen=True)
class _CleanSummaryInput:
    instance_id: str
    split: str
    variant_id: str
    profile_id: str
    base_oscillation: str
    anomaly_type: str
    channels: int
    length_normalization: str
    support_label_mode: str
    support_eps_mode: str
    support_eps_value: float
    base_parameter_policy: str
    base_channel_parameter_policy: str
    anomaly_parameter_policy: str
    base_parameters: Mapping[str, Any]
    base_channel_parameters: Sequence[Mapping[str, Any]]
    base_parameters_per_channel: Sequence[Mapping[str, Any]]
    split_phase_shift: Mapping[str, Any]
    base_channel_correlation: Mapping[str, Any]
    seeds: Mapping[str, int]


@dataclass(frozen=True)
class _PairedSummaryInput:
    instance_id: str
    split: str
    variant_id: str
    profile_id: str
    base_oscillation: str
    anomaly_type: str
    target_density: float
    active_density_range: tuple[float, float]
    active_density_tolerance: float
    labels: np.ndarray
    events: Sequence[Mapping[str, Any]]
    segment_plan: Sequence[SegmentLike]
    channels: int
    channel_policy: str
    overlap_policy: str
    segment_planner: Mapping[str, Any]
    variant_anomaly_policy: Mapping[str, Any]
    length_normalization: str
    support_label_mode: str
    support_eps_mode: str
    support_eps_value: float
    base_parameter_policy: str
    base_channel_parameter_policy: str
    anomaly_parameter_policy: str
    base_parameters: Mapping[str, Any]
    base_channel_parameters: Sequence[Mapping[str, Any]]
    base_parameters_per_channel: Sequence[Mapping[str, Any]]
    split_phase_shift: Mapping[str, Any]
    base_channel_correlation: Mapping[str, Any]
    anomaly_parameters_instance: Mapping[str, Any] | None
    seeds: Mapping[str, int]


def build_clean_instance_summary(
    *,
    instance_id: str,
    split: str,
    variant_id: str,
    profile_id: str,
    base_oscillation: str,
    anomaly_type: str,
    channels: int,
    length_normalization: str,
    support_label_mode: str,
    support_eps_mode: str,
    support_eps_value: float,
    base_parameter_policy: str,
    base_channel_parameter_policy: str,
    anomaly_parameter_policy: str,
    base_parameters: Mapping[str, Any],
    base_channel_parameters: Sequence[Mapping[str, Any]],
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    split_phase_shift: Mapping[str, Any],
    base_channel_correlation: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Build a JSON-safe clean-only instance summary.

    Parameters
    ----------
    remaining arguments
        Stable identity, policy, parameter, and seed metadata.

    Returns
    -------
    dict[str, Any]
        JSON-safe clean-only instance summary.
    """
    return _build_clean_instance_summary_from_input(_CleanSummaryInput(**locals()))


def _build_clean_instance_summary_from_input(
    request: _CleanSummaryInput,
) -> dict[str, Any]:
    per_channel_counts = {str(channel): 0 for channel in range(int(request.channels))}
    return sanitize_json_value(
        {
            "instance_id": request.instance_id,
            "instance_role": "clean_only",
            "has_anomaly": False,
            "split": request.split,
            "variant_id": request.variant_id,
            "profile_id": request.profile_id,
            "base_oscillation": request.base_oscillation,
            "anomaly_type": request.anomaly_type,
            "target_density": 0.0,
            "achieved_density": 0.0,
            "achieved_density_labeled": 0.0,
            "achieved_density_source": 0.0,
            "density_validation_mode": "clean_only",
            "density_error": 0.0,
            "density_tolerance": 0.0,
            "n_segments": 0,
            "n_event_groups": 0,
            "segment_length_mean": 0.0,
            "segment_length_median": 0.0,
            "segment_length_std": 0.0,
            "segment_length_min": 0,
            "segment_length_max": 0,
            "segment_lengths": [],
            "source_segment_lengths": [],
            "effective_support_shrink_count": 0,
            "energy_fallback_count": 0,
            "per_channel_segment_counts": per_channel_counts,
            "channel_policy": "clean_only",
            "overlap_policy": "none",
            "segment_planner": {},
            "variant_anomaly_policy": {},
            "length_normalization": request.length_normalization,
            "support_label_mode": request.support_label_mode,
            "support_eps_mode": request.support_eps_mode,
            "support_eps_value": request.support_eps_value,
            "base_parameter_policy": request.base_parameter_policy,
            "base_channel_parameter_policy": request.base_channel_parameter_policy,
            "anomaly_parameter_policy": request.anomaly_parameter_policy,
            "base_parameters": request.base_parameters,
            "base_channel_parameters": request.base_channel_parameters,
            "base_parameters_per_channel": request.base_parameters_per_channel,
            "split_phase_shift": request.split_phase_shift,
            "base_channel_correlation": request.base_channel_correlation,
            "anomaly_parameters_instance": None,
            "seeds": dict(request.seeds),
        }
    )


def build_paired_instance_summary(
    *,
    instance_id: str,
    split: str,
    variant_id: str,
    profile_id: str,
    base_oscillation: str,
    anomaly_type: str,
    target_density: float,
    active_density_range: tuple[float, float],
    active_density_tolerance: float,
    labels: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    segment_plan: Sequence[SegmentLike],
    channels: int,
    channel_policy: str,
    overlap_policy: str,
    segment_planner: Mapping[str, Any],
    variant_anomaly_policy: Mapping[str, Any],
    length_normalization: str,
    support_label_mode: str,
    support_eps_mode: str,
    support_eps_value: float,
    base_parameter_policy: str,
    base_channel_parameter_policy: str,
    anomaly_parameter_policy: str,
    base_parameters: Mapping[str, Any],
    base_channel_parameters: Sequence[Mapping[str, Any]],
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    split_phase_shift: Mapping[str, Any],
    base_channel_correlation: Mapping[str, Any],
    anomaly_parameters_instance: Mapping[str, Any] | None,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Build and validate a JSON-safe paired instance summary.

    Parameters
    ----------
    labels, events, segment_plan
        Pointwise labels, emitted event records, and planned segments.
    active_density_range, active_density_tolerance
        Effective density bounds used for validation.
    remaining arguments
        Stable identity, policy, parameter, and seed metadata.

    Returns
    -------
    dict[str, Any]
        JSON-safe paired instance summary.

    Raises
    ------
    ValueError
        If density is outside the effective range/tolerance.
    """
    return _build_paired_instance_summary_from_input(_PairedSummaryInput(**locals()))


def _build_paired_instance_summary_from_input(
    request: _PairedSummaryInput,
) -> dict[str, Any]:
    density_metrics = compute_density_metrics(
        labels=request.labels,
        events=request.events,
        support_label_mode=request.support_label_mode,
    )
    validate_density(
        target_density=request.target_density,
        active_density_range=request.active_density_range,
        active_density_tolerance=request.active_density_tolerance,
        achieved_density_labeled=density_metrics["achieved_density_labeled"],
        achieved_density_source=density_metrics["achieved_density_source"],
        density_for_validation=density_metrics["density_for_validation"],
        density_validation_mode=density_metrics["density_validation_mode"],
    )

    event_stats = paired_event_stats(
        events=request.events,
        segment_plan=request.segment_plan,
        channels=request.channels,
    )
    density_for_validation = float(density_metrics["density_for_validation"])
    return sanitize_json_value(
        {
            **_paired_identity_payload(
                instance_id=request.instance_id,
                split=request.split,
                variant_id=request.variant_id,
                profile_id=request.profile_id,
                base_oscillation=request.base_oscillation,
                anomaly_type=request.anomaly_type,
                first_event=event_stats["first_event"],
            ),
            **_paired_density_payload(
                target_density=request.target_density,
                active_density_tolerance=request.active_density_tolerance,
                density_for_validation=density_for_validation,
                density_metrics=density_metrics,
            ),
            **_paired_segment_payload(request.events, event_stats),
            **_paired_policy_payload(
                channel_policy=request.channel_policy,
                overlap_policy=request.overlap_policy,
                segment_planner=request.segment_planner,
                variant_anomaly_policy=request.variant_anomaly_policy,
                length_normalization=request.length_normalization,
                support_label_mode=request.support_label_mode,
                support_eps_mode=request.support_eps_mode,
                support_eps_value=request.support_eps_value,
            ),
            **_paired_parameter_payload(
                base_parameter_policy=request.base_parameter_policy,
                base_channel_parameter_policy=request.base_channel_parameter_policy,
                anomaly_parameter_policy=request.anomaly_parameter_policy,
                base_parameters=request.base_parameters,
                base_channel_parameters=request.base_channel_parameters,
                base_parameters_per_channel=request.base_parameters_per_channel,
                split_phase_shift=request.split_phase_shift,
                base_channel_correlation=request.base_channel_correlation,
                anomaly_parameters_instance=request.anomaly_parameters_instance,
                seeds=request.seeds,
            ),
        }
    )


def _paired_identity_payload(
    *,
    instance_id: str,
    split: str,
    variant_id: str,
    profile_id: str,
    base_oscillation: str,
    anomaly_type: str,
    first_event: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "instance_role": "paired",
        "has_anomaly": True,
        "split": split,
        "variant_id": variant_id,
        "profile_id": profile_id,
        "base_oscillation": base_oscillation,
        "anomaly_type": anomaly_type,
        "requested_pre_context": first_event.get("requested_pre_context"),
        "onset_bucket": first_event.get("onset_bucket"),
        "actual_source_start": first_event.get("actual_source_start"),
        "actual_support_start": first_event.get("actual_support_start"),
        "alignment_strategy": first_event.get("alignment_strategy"),
        "alignment_error": first_event.get("alignment_error"),
    }


def _paired_density_payload(
    *,
    target_density: float,
    active_density_tolerance: float,
    density_for_validation: float,
    density_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_density": target_density,
        "achieved_density": density_for_validation,
        "achieved_density_labeled": density_metrics["achieved_density_labeled"],
        "achieved_density_source": density_metrics["achieved_density_source"],
        "density_validation_mode": density_metrics["density_validation_mode"],
        "density_error": abs(density_for_validation - float(target_density)),
        "density_tolerance": active_density_tolerance,
    }


def _paired_segment_payload(
    events: Sequence[Mapping[str, Any]],
    event_stats: Mapping[str, Any],
) -> dict[str, Any]:
    segment_lengths = event_stats["segment_lengths"]
    return {
        "n_segments": len(events),
        "n_event_groups": len(event_stats["unique_group_ids"]),
        "segment_length_mean": float(np.mean(segment_lengths)),
        "segment_length_median": float(np.median(segment_lengths)),
        "segment_length_std": float(np.std(segment_lengths)),
        "segment_length_min": int(np.min(segment_lengths)),
        "segment_length_max": int(np.max(segment_lengths)),
        "segment_lengths": segment_lengths,
        "source_segment_lengths": event_stats["source_segment_lengths"],
        "effective_support_shrink_count": event_stats["effective_support_shrink_count"],
        "energy_fallback_count": event_stats["energy_fallback_count"],
        "per_channel_segment_counts": event_stats["per_channel_counts"],
    }


def _paired_policy_payload(
    *,
    channel_policy: str,
    overlap_policy: str,
    segment_planner: Mapping[str, Any],
    variant_anomaly_policy: Mapping[str, Any],
    length_normalization: str,
    support_label_mode: str,
    support_eps_mode: str,
    support_eps_value: float,
) -> dict[str, Any]:
    return {
        "channel_policy": channel_policy,
        "overlap_policy": overlap_policy,
        "segment_planner": segment_planner,
        "variant_anomaly_policy": variant_anomaly_policy,
        "length_normalization": length_normalization,
        "support_label_mode": support_label_mode,
        "support_eps_mode": support_eps_mode,
        "support_eps_value": support_eps_value,
    }


def _paired_parameter_payload(
    *,
    base_parameter_policy: str,
    base_channel_parameter_policy: str,
    anomaly_parameter_policy: str,
    base_parameters: Mapping[str, Any],
    base_channel_parameters: Sequence[Mapping[str, Any]],
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    split_phase_shift: Mapping[str, Any],
    base_channel_correlation: Mapping[str, Any],
    anomaly_parameters_instance: Mapping[str, Any] | None,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "base_parameter_policy": base_parameter_policy,
        "base_channel_parameter_policy": base_channel_parameter_policy,
        "anomaly_parameter_policy": anomaly_parameter_policy,
        "base_parameters": base_parameters,
        "base_channel_parameters": base_channel_parameters,
        "base_parameters_per_channel": base_parameters_per_channel,
        "split_phase_shift": split_phase_shift,
        "base_channel_correlation": base_channel_correlation,
        "anomaly_parameters_instance": anomaly_parameters_instance,
        "seeds": dict(seeds),
    }
