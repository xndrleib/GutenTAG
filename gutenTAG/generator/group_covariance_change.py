"""Covariance-change group anomaly operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..anomalies.types import BaseAnomaly
from .group_context import resolve_group_context
from .group_event_application import apply_group_channel_effect
from .group_relation_policy import (
    effective_coupling_strength,
    effective_latent_transition_length,
    latent_shared_noise_attrs,
    prefer_observed_relation_rewrite,
)
from .multivariate_ops import (
    matched_coupling_window,
    residualized_matched_coupling_window,
    shared_noise_window_from_components,
)


@dataclass(frozen=True)
class _CovarianceChangeContext:
    source_start: int
    source_end: int
    group_id: int
    group_channels: list[int]
    segment_idx_by_channel: dict[int, int]
    anchor_channel: int
    target_channels: list[int]
    anchor_window: np.ndarray


@dataclass(frozen=True)
class _CovarianceTargetState:
    channel: int
    segment_idx: int
    params: dict[str, Any]
    before_window: np.ndarray
    before_noise: np.ndarray
    coupling_strength: float
    effective_strength: float
    transition_length: int


def apply_covariance_change_group(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> list[dict[str, Any]]:
    """Apply a covariance/coupling change to non-anchor group channels."""

    context = _resolve_covariance_change_context(
        group_indices=group_indices,
        segment_plan=segment_plan,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    if context is None:
        return []
    return [
        _apply_covariance_change_target(
            context=context,
            channel=channel,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
        for channel in context.target_channels
    ]


def _resolve_covariance_change_context(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> _CovarianceChangeContext | None:
    if len(group_indices) == 0:
        return None
    (
        _,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = resolve_group_context(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return None
    anchor_channel = int(group_channels[0])
    return _CovarianceChangeContext(
        source_start=int(source_start),
        source_end=int(source_end),
        group_id=int(group_id),
        group_channels=[int(channel) for channel in group_channels],
        segment_idx_by_channel={
            int(channel): int(segment_idx)
            for channel, segment_idx in segment_idx_by_channel.items()
        },
        anchor_channel=anchor_channel,
        target_channels=[int(channel) for channel in group_channels[1:]],
        anchor_window=runtime.compose_window(
            base=base,
            bo=channel_bos[anchor_channel],
            channel=anchor_channel,
            start=source_start,
            end=source_end,
        ),
    )


def _apply_covariance_change_target(
    *,
    context: _CovarianceChangeContext,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> dict[str, Any]:
    state = _covariance_target_state(
        context=context,
        channel=channel,
        base=base,
        channel_bos=channel_bos,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        runtime=runtime,
    )
    injection_level = _apply_covariance_rewrite(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return _covariance_change_event(
        context=context,
        state=state,
        injection_level=injection_level,
        base=base,
        channel_bos=channel_bos,
        labels=labels,
        used_positions=used_positions,
        anomaly_type=anomaly_type,
        runtime=runtime,
    )


def _covariance_target_state(
    *,
    context: _CovarianceChangeContext,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> _CovarianceTargetState:
    segment_idx = int(context.segment_idx_by_channel[int(channel)])
    params = anomaly_parameters_per_segment[segment_idx]
    coupling_strength = float(params.get("coupling_strength", -0.9))
    before_window = runtime.compose_window(
        base=base,
        bo=channel_bos[int(channel)],
        channel=int(channel),
        start=context.source_start,
        end=context.source_end,
    )
    before_noise = runtime.compose_noise(
        bo=channel_bos[int(channel)],
        start=context.source_start,
        end=context.source_end,
    )
    return _CovarianceTargetState(
        channel=int(channel),
        segment_idx=segment_idx,
        params=params,
        before_window=before_window,
        before_noise=before_noise,
        coupling_strength=coupling_strength,
        effective_strength=effective_coupling_strength(
            bo=channel_bos[int(channel)],
            coupling_strength=coupling_strength,
        ),
        transition_length=int(params.get("transition_length", 8)),
    )


def _apply_covariance_rewrite(
    *,
    context: _CovarianceChangeContext,
    state: _CovarianceTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> str:
    latent = latent_shared_noise_attrs(
        channel_bos[state.channel],
        context.source_start,
        context.source_end,
    )
    if latent is not None and not prefer_observed_relation_rewrite(
        channel_bos[state.channel]
    ):
        _rewrite_covariance_noise(
            latent=latent,
            context=context,
            state=state,
            channel_bos=channel_bos,
            runtime=runtime,
        )
        return "noise"
    _rewrite_covariance_observed_window(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return "observed_window"


def _rewrite_covariance_noise(
    *,
    latent: Any,
    context: _CovarianceChangeContext,
    state: _CovarianceTargetState,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    idio, shared, noise_mean, _, _ = latent
    candidate_noise = shared_noise_window_from_components(
        idiosyncratic_component=idio,
        shared_component=shared,
        noise_mean=noise_mean,
        target_shared_weight=float(state.effective_strength),
    )
    candidate_noise = BaseAnomaly.blend_with_reference(
        candidate_noise,
        state.before_noise,
        effective_latent_transition_length(
            channel_bos[state.channel],
            max(0, min(state.transition_length, 8)),
        ),
    )
    runtime.replace_noise(
        bo=channel_bos[state.channel],
        start=context.source_start,
        end=context.source_end,
        target_noise=candidate_noise,
    )


def _rewrite_covariance_observed_window(
    *,
    context: _CovarianceChangeContext,
    state: _CovarianceTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    candidate = _covariance_change_observed_candidate(
        channel_bo=channel_bos[state.channel],
        before_window=state.before_window,
        anchor_window=context.anchor_window,
        coupling_strength=state.effective_strength,
    )
    candidate = BaseAnomaly.blend_with_reference(
        candidate,
        state.before_window,
        state.transition_length,
    )
    runtime.replace_window(
        base=base,
        bo=channel_bos[state.channel],
        channel=state.channel,
        start=context.source_start,
        end=context.source_end,
        target_observed=candidate,
    )


def _covariance_change_event(
    *,
    context: _CovarianceChangeContext,
    state: _CovarianceTargetState,
    injection_level: str,
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    runtime: Any,
) -> dict[str, Any]:
    return apply_group_channel_effect(
        base=base,
        channel_bos=channel_bos,
        labels=labels,
        used_positions=used_positions,
        runtime=runtime,
        before_window=state.before_window,
        source_start=context.source_start,
        source_end=context.source_end,
        channel=state.channel,
        anomaly_type=anomaly_type,
        group_id=context.group_id,
        group_channels=context.group_channels,
        intervention_channels=context.target_channels,
        anomaly_object="shared_noise_coupling_change",
        channel_visible=False,
        purity_hint=(
            "operational_candidate"
            if injection_level == "noise"
            else "multivariate_preferred"
        ),
        params=state.params,
        extra={
            "anchor_channel": context.anchor_channel,
            "coupling_strength": float(state.coupling_strength),
            "effective_coupling_strength": float(state.effective_strength),
            "injection_level": injection_level,
        },
    )


def _covariance_change_observed_candidate(
    *,
    channel_bo: Any,
    before_window: np.ndarray,
    anchor_window: np.ndarray,
    coupling_strength: float,
) -> np.ndarray:
    if str(channel_bo.get_base_oscillation_kind()) in {"polynomial", "random-walk"}:
        return residualized_matched_coupling_window(
            reference=before_window,
            anchor=anchor_window,
            coupling_strength=coupling_strength,
        )
    return matched_coupling_window(
        reference=before_window,
        anchor=anchor_window,
        coupling_strength=coupling_strength,
    )
