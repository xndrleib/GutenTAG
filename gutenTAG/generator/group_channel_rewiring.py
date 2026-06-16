"""Channel-rewiring group anomaly operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..anomalies.types import BaseAnomaly
from .group_context import resolve_group_context
from .group_event_application import apply_group_channel_effect
from .group_relation_policy import latent_shared_noise_attrs
from .multivariate_ops import rotated_pair_windows


@dataclass(frozen=True)
class _ChannelRewiringContext:
    source_start: int
    source_end: int
    group_id: int
    group_channels: list[int]
    segment_idx_by_channel: dict[int, int]
    first_channel: int
    second_channel: int
    intervention_channels: list[int]


@dataclass(frozen=True)
class _ChannelRewiringState:
    transition_length: int
    rotation_degrees: float
    before_by_channel: dict[int, np.ndarray]
    candidate_by_channel: dict[int, np.ndarray]
    injection_level: str


def apply_channel_rewiring_group(
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
    """Rotate a paired channel structure on observed windows or latent noise."""

    context = _resolve_channel_rewiring_context(
        group_indices=group_indices,
        segment_plan=segment_plan,
    )
    if context is None:
        return []
    state = _apply_channel_rewiring_pair(
        context=context,
        base=base,
        channel_bos=channel_bos,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        runtime=runtime,
    )
    return [
        _channel_rewiring_event(
            context=context,
            state=state,
            channel=channel,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
        for channel in context.intervention_channels
    ]


def _resolve_channel_rewiring_context(
    *,
    group_indices: list[int],
    segment_plan: list[Any],
) -> _ChannelRewiringContext | None:
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
    first_channel = int(group_channels[0])
    second_channel = int(group_channels[1])
    return _ChannelRewiringContext(
        source_start=int(source_start),
        source_end=int(source_end),
        group_id=int(group_id),
        group_channels=[int(channel) for channel in group_channels],
        segment_idx_by_channel={
            int(channel): int(segment_idx)
            for channel, segment_idx in segment_idx_by_channel.items()
        },
        first_channel=first_channel,
        second_channel=second_channel,
        intervention_channels=[first_channel, second_channel],
    )


def _apply_channel_rewiring_pair(
    *,
    context: _ChannelRewiringContext,
    base: np.ndarray,
    channel_bos: list[Any],
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> _ChannelRewiringState:
    first_params = anomaly_parameters_per_segment[
        int(context.segment_idx_by_channel[context.first_channel])
    ]
    transition_length = int(first_params.get("transition_length", 8))
    rotation_degrees = float(first_params.get("rotation_degrees", 25.0))
    before_by_channel = _compose_rewiring_windows(context, base, channel_bos, runtime)
    if _can_rewire_latent_noise(context, channel_bos):
        _replace_rewired_noise_pair(
            runtime=runtime,
            channel_bos=channel_bos,
            first_channel=context.first_channel,
            second_channel=context.second_channel,
            source_start=context.source_start,
            source_end=context.source_end,
            rotation_degrees=rotation_degrees,
            transition_length=transition_length,
        )
        return _ChannelRewiringState(
            transition_length=transition_length,
            rotation_degrees=rotation_degrees,
            before_by_channel=before_by_channel,
            candidate_by_channel={},
            injection_level="noise",
        )
    return _ChannelRewiringState(
        transition_length=transition_length,
        rotation_degrees=rotation_degrees,
        before_by_channel=before_by_channel,
        candidate_by_channel=_observed_rewiring_candidates(
            context,
            before_by_channel,
            rotation_degrees=rotation_degrees,
            transition_length=transition_length,
        ),
        injection_level="observed_window",
    )


def _compose_rewiring_windows(
    context: _ChannelRewiringContext,
    base: np.ndarray,
    channel_bos: list[Any],
    runtime: Any,
) -> dict[int, np.ndarray]:
    return {
        channel: runtime.compose_window(
            base=base,
            bo=channel_bos[channel],
            channel=channel,
            start=context.source_start,
            end=context.source_end,
        )
        for channel in context.intervention_channels
    }


def _can_rewire_latent_noise(
    context: _ChannelRewiringContext,
    channel_bos: list[Any],
) -> bool:
    latent_first = latent_shared_noise_attrs(
        channel_bos[context.first_channel],
        context.source_start,
        context.source_end,
    )
    latent_second = latent_shared_noise_attrs(
        channel_bos[context.second_channel],
        context.source_start,
        context.source_end,
    )
    return latent_first is not None and latent_second is not None


def _observed_rewiring_candidates(
    context: _ChannelRewiringContext,
    before_by_channel: dict[int, np.ndarray],
    *,
    rotation_degrees: float,
    transition_length: int,
) -> dict[int, np.ndarray]:
    before_first = before_by_channel[context.first_channel]
    before_second = before_by_channel[context.second_channel]
    rewired_first, rewired_second = rotated_pair_windows(
        before_first,
        before_second,
        rotation_degrees=rotation_degrees,
    )
    return {
        context.first_channel: BaseAnomaly.blend_with_reference(
            rewired_first,
            before_first,
            transition_length,
        ),
        context.second_channel: BaseAnomaly.blend_with_reference(
            rewired_second,
            before_second,
            transition_length,
        ),
    }


def _channel_rewiring_event(
    *,
    context: _ChannelRewiringContext,
    state: _ChannelRewiringState,
    channel: int,
    base: np.ndarray,
    channel_bos: list[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: list[dict[str, Any]],
    runtime: Any,
) -> dict[str, Any]:
    if state.injection_level == "observed_window":
        runtime.replace_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=context.source_start,
            end=context.source_end,
            target_observed=state.candidate_by_channel[int(channel)],
        )
    segment_idx = int(context.segment_idx_by_channel[int(channel)])
    return apply_group_channel_effect(
        base=base,
        channel_bos=channel_bos,
        labels=labels,
        used_positions=used_positions,
        runtime=runtime,
        before_window=state.before_by_channel[int(channel)],
        source_start=context.source_start,
        source_end=context.source_end,
        channel=int(channel),
        anomaly_type=anomaly_type,
        group_id=context.group_id,
        group_channels=context.group_channels,
        intervention_channels=context.intervention_channels,
        anomaly_object="paired_structure_rotation",
        channel_visible=(state.injection_level != "noise"),
        purity_hint=(
            "operational_candidate"
            if state.injection_level == "noise"
            else "not_pure_local"
        ),
        params=anomaly_parameters_per_segment[segment_idx],
        extra={
            "rewired_pair": context.intervention_channels,
            "rotation_degrees": float(state.rotation_degrees),
            "injection_level": state.injection_level,
        },
    )


def _replace_rewired_noise_pair(
    *,
    runtime: Any,
    channel_bos: list[Any],
    first_channel: int,
    second_channel: int,
    source_start: int,
    source_end: int,
    rotation_degrees: float,
    transition_length: int,
) -> None:
    before_noise_first = runtime.compose_noise(
        bo=channel_bos[int(first_channel)],
        start=source_start,
        end=source_end,
    )
    before_noise_second = runtime.compose_noise(
        bo=channel_bos[int(second_channel)],
        start=source_start,
        end=source_end,
    )
    rewired_noise_first, rewired_noise_second = rotated_pair_windows(
        before_noise_first,
        before_noise_second,
        rotation_degrees=rotation_degrees,
    )
    runtime.replace_noise(
        bo=channel_bos[int(first_channel)],
        start=source_start,
        end=source_end,
        target_noise=BaseAnomaly.blend_with_reference(
            rewired_noise_first,
            before_noise_first,
            transition_length,
        ),
    )
    runtime.replace_noise(
        bo=channel_bos[int(second_channel)],
        start=source_start,
        end=source_end,
        target_noise=BaseAnomaly.blend_with_reference(
            rewired_noise_second,
            before_noise_second,
            transition_length,
        ),
    )
