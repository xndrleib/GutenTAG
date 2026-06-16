"""Correlation-flip group anomaly operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..anomalies.types import BaseAnomaly
from .correlation_geometry import correlation_flip_window
from .group_context import resolve_group_context
from .group_event_application import apply_group_channel_effect
from .group_relation_policy import (
    effective_latent_transition_length,
    effective_relation_target,
    latent_shared_noise_attrs,
    prefer_observed_relation_rewrite,
)
from .multivariate_ops import (
    mixed_shared_noise_window_from_components,
    residualized_correlation_flip_window,
)


@dataclass(frozen=True)
class _CorrelationFlipContext:
    source_start: int
    source_end: int
    group_id: int
    group_channels: list[int]
    segment_idx_by_channel: dict[int, int]
    anchor_channel: int
    target_channels: list[int]
    anchor_window: np.ndarray


@dataclass(frozen=True)
class _CorrelationTargetState:
    channel: int
    position: int
    segment_idx: int
    params: dict[str, Any]
    before_window: np.ndarray
    before_noise: np.ndarray
    target_correlation: float | None
    effective_target_correlation: float | None
    transition_length: int


def apply_correlation_flip_group(
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
    """Apply a pairwise correlation flip to non-anchor group channels."""

    context = _resolve_correlation_flip_context(
        group_indices=group_indices,
        segment_plan=segment_plan,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    if context is None:
        return []
    return [
        _apply_correlation_flip_target(
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


def _resolve_correlation_flip_context(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> _CorrelationFlipContext | None:
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
    return _CorrelationFlipContext(
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


def _apply_correlation_flip_target(
    *,
    context: _CorrelationFlipContext,
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
    state = _correlation_target_state(
        context=context,
        position=position,
        channel=channel,
        base=base,
        channel_bos=channel_bos,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        runtime=runtime,
    )
    injection_level = _apply_correlation_rewrite(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return _correlation_flip_event(
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


def _correlation_target_state(
    *,
    context: _CorrelationFlipContext,
    position: int,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> _CorrelationTargetState:
    segment_idx = int(context.segment_idx_by_channel[int(channel)])
    params = anomaly_parameters_per_segment[segment_idx]
    target_correlation_raw = params.get("target_correlation", -0.85)
    target_correlation = (
        None if target_correlation_raw is None else float(target_correlation_raw)
    )
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
    return _CorrelationTargetState(
        channel=int(channel),
        position=int(position),
        segment_idx=segment_idx,
        params=params,
        before_window=before_window,
        before_noise=before_noise,
        target_correlation=target_correlation,
        effective_target_correlation=effective_relation_target(
            bo=channel_bos[int(channel)],
            target_correlation=target_correlation,
        ),
        transition_length=int(params.get("transition_length", 8)),
    )


def _apply_correlation_rewrite(
    *,
    context: _CorrelationFlipContext,
    state: _CorrelationTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> str:
    latent = latent_shared_noise_attrs(
        channel_bos[state.channel],
        context.source_start,
        context.source_end,
    )
    if _should_rewrite_latent_noise(
        latent=latent,
        effective_target_correlation=state.effective_target_correlation,
        channel_bo=channel_bos[state.channel],
    ):
        _rewrite_latent_noise(
            latent=latent,
            context=context,
            state=state,
            channel_bos=channel_bos,
            runtime=runtime,
        )
        return "noise"
    _rewrite_observed_window(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return "observed_window"


def _should_rewrite_latent_noise(
    *,
    latent: Any,
    effective_target_correlation: float | None,
    channel_bo: Any,
) -> bool:
    return (
        latent is not None
        and effective_target_correlation is not None
        and not prefer_observed_relation_rewrite(channel_bo)
    )


def _rewrite_latent_noise(
    *,
    latent: Any,
    context: _CorrelationFlipContext,
    state: _CorrelationTargetState,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    assert state.effective_target_correlation is not None
    idio, shared, noise_mean, current_shared_weight, _ = latent
    candidate_noise = mixed_shared_noise_window_from_components(
        idiosyncratic_component=idio,
        shared_component=shared,
        noise_mean=noise_mean,
        current_shared_weight=current_shared_weight,
        target_alignment=float(state.effective_target_correlation),
        surrogate_variant=state.position,
    )
    candidate_noise = BaseAnomaly.blend_with_reference(
        candidate_noise,
        state.before_noise,
        effective_latent_transition_length(
            channel_bos[state.channel],
            state.transition_length,
        ),
    )
    runtime.replace_noise(
        bo=channel_bos[state.channel],
        start=context.source_start,
        end=context.source_end,
        target_noise=candidate_noise,
    )


def _rewrite_observed_window(
    *,
    context: _CorrelationFlipContext,
    state: _CorrelationTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    candidate = _correlation_flip_observed_candidate(
        channel_bo=channel_bos[state.channel],
        before_window=state.before_window,
        anchor_window=context.anchor_window,
        target_correlation=state.effective_target_correlation,
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


def _correlation_flip_event(
    *,
    context: _CorrelationFlipContext,
    state: _CorrelationTargetState,
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
        anomaly_object="pair_correlation_flip",
        channel_visible=False,
        purity_hint=(
            "operational_candidate"
            if injection_level == "noise"
            else "multivariate_preferred"
        ),
        params=state.params,
        extra={
            "anchor_channel": context.anchor_channel,
            "target_correlation": state.target_correlation,
            "target_alignment": state.target_correlation,
            "effective_target_alignment": state.effective_target_correlation,
            "injection_level": injection_level,
        },
    )


def _correlation_flip_observed_candidate(
    *,
    channel_bo: Any,
    before_window: np.ndarray,
    anchor_window: np.ndarray,
    target_correlation: float | None,
) -> np.ndarray:
    if str(channel_bo.get_base_oscillation_kind()) in {"polynomial", "random-walk"}:
        return residualized_correlation_flip_window(
            reference=before_window,
            anchor=anchor_window,
            target_correlation=target_correlation,
        )
    return correlation_flip_window(
        reference=before_window,
        anchor=anchor_window,
        target_correlation=target_correlation,
    )
