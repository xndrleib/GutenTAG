"""Lag-synchronization group anomaly operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..anomalies.types import BaseAnomaly
from .group_context import resolve_group_context
from .group_event_application import apply_group_channel_effect
from .multivariate_ops import lag_shift_window


@dataclass(frozen=True)
class _LagSynchronizationContext:
    source_start: int
    source_end: int
    group_id: int
    group_channels: list[int]
    segment_idx_by_channel: dict[int, int]
    anchor_channel: int
    shifted_channels: list[int]


@dataclass(frozen=True)
class _LagSynchronizationTargetState:
    channel: int
    segment_idx: int
    params: dict[str, Any]
    before_window: np.ndarray
    target_observed: np.ndarray
    realized_lag: int


def apply_lag_synchronization_group(
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
    """Shift non-anchor channels by a realized lag and record per-channel events."""

    context = _resolve_lag_synchronization_context(
        group_indices=group_indices,
        segment_plan=segment_plan,
    )
    if context is None:
        return []
    return [
        _apply_lag_synchronization_target(
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
        for channel in context.shifted_channels
    ]


def _resolve_lag_synchronization_context(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
) -> _LagSynchronizationContext | None:
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
    return _LagSynchronizationContext(
        source_start=int(source_start),
        source_end=int(source_end),
        group_id=int(group_id),
        group_channels=[int(channel) for channel in group_channels],
        segment_idx_by_channel={
            int(channel): int(segment_idx)
            for channel, segment_idx in segment_idx_by_channel.items()
        },
        anchor_channel=anchor_channel,
        shifted_channels=[int(channel) for channel in group_channels[1:]],
    )


def _apply_lag_synchronization_target(
    *,
    context: _LagSynchronizationContext,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> dict[str, Any]:
    state = _lag_synchronization_target_state(
        context=context,
        channel=channel,
        base=base,
        channel_bos=channel_bos,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        runtime=runtime,
    )
    _replace_lag_synchronization_window(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    return _lag_synchronization_event(
        context=context,
        state=state,
        base=base,
        channel_bos=channel_bos,
        labels=labels,
        used_positions=used_positions,
        anomaly_type=anomaly_type,
        runtime=runtime,
    )


def _lag_synchronization_target_state(
    *,
    context: _LagSynchronizationContext,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> _LagSynchronizationTargetState:
    segment_idx = int(context.segment_idx_by_channel[int(channel)])
    params = anomaly_parameters_per_segment[segment_idx]
    full_series = runtime.compose_window(
        base=base,
        bo=channel_bos[int(channel)],
        channel=int(channel),
        start=0,
        end=base.shape[0],
    )
    before_window = np.array(
        full_series[context.source_start : context.source_end],
        copy=True,
    )
    shifted_window, realized_lag = lag_shift_window(
        series=full_series,
        start=context.source_start,
        end=context.source_end,
        lag_steps=int(params.get("lag_steps", 6)),
    )
    return _LagSynchronizationTargetState(
        channel=int(channel),
        segment_idx=segment_idx,
        params=params,
        before_window=before_window,
        target_observed=BaseAnomaly.blend_with_reference(
            shifted_window,
            before_window,
            int(params.get("transition_length", 8)),
        ),
        realized_lag=int(realized_lag),
    )


def _replace_lag_synchronization_window(
    *,
    context: _LagSynchronizationContext,
    state: _LagSynchronizationTargetState,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> None:
    runtime.replace_window(
        base=base,
        bo=channel_bos[state.channel],
        channel=state.channel,
        start=context.source_start,
        end=context.source_end,
        target_observed=state.target_observed,
    )


def _lag_synchronization_event(
    *,
    context: _LagSynchronizationContext,
    state: _LagSynchronizationTargetState,
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
        intervention_channels=context.shifted_channels,
        anomaly_object="lag_synchronization_shift",
        channel_visible=True,
        purity_hint="not_pure_local",
        params=state.params,
        extra={
            "anchor_channel": int(context.anchor_channel),
            "realized_lag_steps": int(state.realized_lag),
        },
    )


__all__ = ["apply_lag_synchronization_group"]
