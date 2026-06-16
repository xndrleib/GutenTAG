"""Mode-correlation group anomaly operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base_oscillations import RandomModeJump
from .group_context import resolve_group_context
from .group_event_application import apply_group_channel_effect


@dataclass(frozen=True)
class _ModeCorrelationContext:
    source_start: int
    source_end: int
    group_id: int
    group_channels: list[int]
    segment_idx_by_channel: dict[int, int]
    anchor_channel: int
    flipped_channels: list[int]
    segment_attrs: dict[str, Any]


def apply_mode_correlation_group(
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
    """Apply an RMJ mode-correlation sign flip to non-anchor channels."""

    context = _resolve_mode_correlation_context(
        group_indices=group_indices,
        segment_plan=segment_plan,
        channel_bos=channel_bos,
    )
    if context is None:
        return []
    before_windows = _compose_before_windows(
        context=context,
        base=base,
        channel_bos=channel_bos,
        runtime=runtime,
    )
    _flip_mode_correlation_channels(context=context, base=base)
    return [
        _mode_correlation_event(
            context=context,
            channel=channel,
            before_window=before_windows[int(channel)],
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
        for channel in context.flipped_channels
    ]


def _resolve_mode_correlation_context(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
    channel_bos: list[Any],
) -> _ModeCorrelationContext | None:
    if len(group_indices) == 0:
        return None
    (
        group_segments,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = resolve_group_context(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return None
    if not all(
        channel_bos[ch].get_base_oscillation_kind() == RandomModeJump.KIND
        for ch in group_channels
    ):
        return None
    anchor_channel = int(group_channels[0])
    return _ModeCorrelationContext(
        source_start=int(source_start),
        source_end=int(source_end),
        group_id=int(group_id),
        group_channels=[int(channel) for channel in group_channels],
        segment_idx_by_channel={
            int(channel): int(segment_idx)
            for channel, segment_idx in segment_idx_by_channel.items()
        },
        anchor_channel=anchor_channel,
        flipped_channels=[int(channel) for channel in group_channels[1:]],
        segment_attrs=dict(group_segments[0].attrs),
    )


def _compose_before_windows(
    *,
    context: _ModeCorrelationContext,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> dict[int, np.ndarray]:
    return {
        int(channel): runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=context.source_start,
            end=context.source_end,
        )
        for channel in context.flipped_channels
    }


def _flip_mode_correlation_channels(
    *,
    context: _ModeCorrelationContext,
    base: np.ndarray,
) -> None:
    for channel in context.flipped_channels:
        base[context.source_start : context.source_end, int(channel)] = (
            -1.0
            * np.asarray(
                base[context.source_start : context.source_end, int(channel)],
                dtype=np.float64,
            )
        )


def _mode_correlation_event(
    *,
    context: _ModeCorrelationContext,
    channel: int,
    before_window: np.ndarray,
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> dict[str, Any]:
    segment_idx = int(context.segment_idx_by_channel[int(channel)])
    return apply_group_channel_effect(
        base=base,
        channel_bos=channel_bos,
        labels=labels,
        used_positions=used_positions,
        runtime=runtime,
        before_window=before_window,
        source_start=context.source_start,
        source_end=context.source_end,
        channel=int(channel),
        anomaly_type=anomaly_type,
        group_id=context.group_id,
        group_channels=context.group_channels,
        intervention_channels=context.flipped_channels,
        anomaly_object="relation_sign_flip",
        channel_visible=False,
        purity_hint="relation_change",
        params=anomaly_parameters_per_segment[segment_idx],
        extra=_mode_correlation_event_extra(context),
    )


def _mode_correlation_event_extra(
    context: _ModeCorrelationContext,
) -> dict[str, Any]:
    segment_attrs = context.segment_attrs
    return {
        "anchor_channel": int(context.anchor_channel),
        "flipped_channels": [int(ch) for ch in context.flipped_channels],
        "mode_change_aligned": bool(segment_attrs.get("mode_change_aligned", False)),
        "mode_grid_aligned": bool(segment_attrs.get("mode_grid_aligned", False)),
        "support_independent_of_realized_mode_state": bool(
            segment_attrs.get(
                "support_independent_of_realized_mode_state",
                False,
            )
        ),
        "latent_mode_flip": True,
        **_mode_grid_metadata(segment_attrs),
    }


def _mode_grid_metadata(segment_attrs: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(segment_attrs[key])
        for key in (
            "mode_grid_block_size",
            "mode_grid_start_block",
            "mode_grid_end_block",
            "mode_grid_block_length",
            "mode_grid_min_gap_blocks",
        )
        if key in segment_attrs
    }


__all__ = ["apply_mode_correlation_group"]
