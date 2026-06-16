"""Shared-factor-break group anomaly operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..anomalies.types import BaseAnomaly
from .group_context import resolve_group_context
from .group_event_application import apply_group_channel_effect
from .group_relation_policy import latent_shared_noise_attrs
from .multivariate_ops import (
    break_shared_factor_window,
    scaled_shared_noise_window_from_components,
)


@dataclass(frozen=True)
class _SharedFactorBreakContext:
    source_start: int
    source_end: int
    group_id: int
    group_channels: list[int]
    segment_idx_by_channel: dict[int, int]
    anchor_channel: int
    target_channels: list[int]
    anchor_window: np.ndarray


@dataclass(frozen=True)
class _SharedFactorTargetState:
    channel: int
    position: int
    segment_idx: int
    params: dict[str, Any]
    before_window: np.ndarray
    shared_factor_scale: float
    transition_length: int


def apply_shared_factor_break_group(
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
    """Break target-channel shared-factor alignment against the anchor channel."""

    context = _resolve_shared_factor_break_context(
        group_indices=group_indices,
        segment_plan=segment_plan,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    if context is None:
        return []
    return [
        _apply_shared_factor_target(
            context=context,
            position=position,
            channel=channel,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
        for position, channel in enumerate(context.target_channels)
    ]


def _resolve_shared_factor_break_context(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> _SharedFactorBreakContext | None:
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
    return _SharedFactorBreakContext(
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


def _apply_shared_factor_target(
    *,
    context: _SharedFactorBreakContext,
    position: int,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> dict[str, Any]:
    state = _shared_factor_target_state(
        context=context,
        position=position,
        channel=channel,
        base=base,
        channel_bos=channel_bos,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        runtime=runtime,
    )
    injection_level = _apply_shared_factor_rewrite(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return _shared_factor_break_event(
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


def _shared_factor_target_state(
    *,
    context: _SharedFactorBreakContext,
    position: int,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> _SharedFactorTargetState:
    segment_idx = int(context.segment_idx_by_channel[int(channel)])
    params = anomaly_parameters_per_segment[segment_idx]
    return _SharedFactorTargetState(
        channel=int(channel),
        position=int(position),
        segment_idx=segment_idx,
        params=params,
        before_window=runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=context.source_start,
            end=context.source_end,
        ),
        shared_factor_scale=float(params.get("shared_factor_scale", 0.0)),
        transition_length=int(params.get("transition_length", 8)),
    )


def _apply_shared_factor_rewrite(
    *,
    context: _SharedFactorBreakContext,
    state: _SharedFactorTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> str:
    latent = latent_shared_noise_attrs(
        channel_bos[state.channel],
        context.source_start,
        context.source_end,
    )
    if latent is not None:
        _rewrite_shared_factor_noise(
            latent=latent,
            context=context,
            state=state,
            channel_bos=channel_bos,
            runtime=runtime,
        )
        return "noise"
    _rewrite_shared_factor_observed_window(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return "observed_window"


def _rewrite_shared_factor_noise(
    *,
    latent: Any,
    context: _SharedFactorBreakContext,
    state: _SharedFactorTargetState,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    idio, shared, noise_mean, current_shared_weight, _ = latent
    before_noise = runtime.compose_noise(
        bo=channel_bos[state.channel],
        start=context.source_start,
        end=context.source_end,
    )
    candidate_noise = scaled_shared_noise_window_from_components(
        idiosyncratic_component=idio,
        shared_component=shared,
        noise_mean=noise_mean,
        current_shared_weight=current_shared_weight,
        shared_factor_scale=state.shared_factor_scale,
    )
    candidate_noise = BaseAnomaly.blend_with_reference(
        candidate_noise,
        before_noise,
        state.transition_length,
    )
    runtime.replace_noise(
        bo=channel_bos[state.channel],
        start=context.source_start,
        end=context.source_end,
        target_noise=candidate_noise,
    )


def _rewrite_shared_factor_observed_window(
    *,
    context: _SharedFactorBreakContext,
    state: _SharedFactorTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    candidate = break_shared_factor_window(
        reference=state.before_window,
        anchor=context.anchor_window,
        shared_factor_scale=state.shared_factor_scale,
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


def _shared_factor_break_event(
    *,
    context: _SharedFactorBreakContext,
    state: _SharedFactorTargetState,
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
        anomaly_object="shared_factor_break",
        channel_visible=False,
        purity_hint=(
            "operational_candidate"
            if injection_level == "noise"
            else "multivariate_preferred"
        ),
        params=state.params,
        extra={
            "anchor_channel": context.anchor_channel,
            "shared_factor_scale": float(state.shared_factor_scale),
            "target_alignment": float(state.shared_factor_scale),
            "surrogate_variant": state.position,
            "injection_level": injection_level,
        },
    )


__all__ = ["apply_shared_factor_break_group"]
