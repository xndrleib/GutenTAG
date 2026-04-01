from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np

from ..anomalies.types import BaseAnomaly
from ..base_oscillations import RandomModeJump
from .correlation_geometry import correlation_flip_window
from .event_metadata import build_event_record
from .multivariate_ops import (
    break_shared_factor_window,
    has_shared_noise_decomposition,
    lag_shift_window,
    matched_coupling_window,
    rotated_pair_windows,
    shared_noise_window_from_components,
)
from .segment_groups import collect_group_channels, group_source_bounds


@dataclass(frozen=True)
class GroupAnomalyRuntime:
    compose_window: Callable[..., np.ndarray]
    replace_window: Callable[..., None]
    compose_noise: Callable[..., np.ndarray]
    replace_noise: Callable[..., None]
    resolve_label_bounds: Callable[..., Tuple[int, int]]
    to_builtin: Callable[[Any], Any]


def _latent_shared_noise_attrs(
    bo: Any,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    if not has_shared_noise_decomposition(bo, start, end):
        return None
    idio = np.asarray(getattr(bo, "_idio_noise_component")[start:end], dtype=np.float64)
    shared = np.asarray(
        getattr(bo, "_shared_noise_component")[start:end], dtype=np.float64
    )
    mean = float(getattr(bo, "_noise_mean"))
    shared_weight = float(getattr(bo, "_shared_noise_weight"))
    return idio, shared, mean, shared_weight


def apply_group_anomaly(
    *,
    anomaly_type: str,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if anomaly_type == "mode-correlation":
        return _apply_mode_correlation_group(
            group_indices=group_indices,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
    if anomaly_type == "covariance-change":
        return _apply_covariance_change_group(
            group_indices=group_indices,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
    if anomaly_type == "correlation-flip":
        return _apply_correlation_flip_group(
            group_indices=group_indices,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
    if anomaly_type == "channel-rewiring":
        return _apply_channel_rewiring_group(
            group_indices=group_indices,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
    if anomaly_type == "lag-synchronization":
        return _apply_lag_synchronization_group(
            group_indices=group_indices,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
    if anomaly_type == "shared-factor-break":
        return _apply_shared_factor_break_group(
            group_indices=group_indices,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            labels=labels,
            used_positions=used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )
    raise ValueError(f"Unsupported group-level anomaly type: {anomaly_type}")


def _group_common(
    group_indices: List[int],
    segment_plan: List[Any],
) -> tuple[list[Any], int, int, int, list[int], dict[int, int]]:
    group_segments = [segment_plan[idx] for idx in group_indices]
    source_start, source_end = group_source_bounds(group_segments)
    group_id = int(group_segments[0].attrs.get("group_id", group_indices[0]))
    group_channels = collect_group_channels(group_segments)
    segment_idx_by_channel = {
        int(segment_plan[idx].channel): int(idx) for idx in group_indices
    }
    return (
        group_segments,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    )


def _apply_mode_correlation_group(
    *,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if len(group_indices) == 0:
        return []
    (
        group_segments,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = _group_common(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return []
    if not all(
        channel_bos[ch].get_base_oscillation_kind() == RandomModeJump.KIND
        for ch in group_channels
    ):
        return []
    anchor_channel = int(group_channels[0])
    flipped_channels = [int(ch) for ch in group_channels[1:]]
    before_windows = {
        int(channel): runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        for channel in flipped_channels
    }
    for channel in flipped_channels:
        runtime.replace_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
            target_observed=-1.0 * before_windows[int(channel)],
        )

    mode_change_aligned = bool(group_segments[0].attrs.get("mode_change_aligned", False))
    events: List[Dict[str, Any]] = []
    for channel in flipped_channels:
        segment_idx = int(segment_idx_by_channel[int(channel)])
        after_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        effective_delta = np.abs(after_window - before_windows[int(channel)])
        label_start, label_end = runtime.resolve_label_bounds(
            protocol_start=source_start,
            protocol_end=source_end,
            delta=effective_delta,
            anomaly_type=anomaly_type,
        )
        labels[label_start:label_end, int(channel)] = 1
        used_positions[int(channel)].append((source_start, source_end))
        events.append(
            build_event_record(
                start=int(label_start),
                end=int(label_end),
                channel=int(channel),
                anomaly_type=anomaly_type,
                group_id=int(group_id),
                group_channels=[int(ch) for ch in group_channels],
                affected_channels=[int(ch) for ch in flipped_channels],
                anomaly_object="relation_sign_flip",
                channel_visible=False,
                purity_hint="relation_change",
                params=runtime.to_builtin(anomaly_parameters_per_segment[int(segment_idx)]),
                source_start=int(source_start),
                source_end=int(source_end),
                extra={
                    "anchor_channel": int(anchor_channel),
                    "flipped_channels": [int(ch) for ch in flipped_channels],
                    "mode_change_aligned": bool(mode_change_aligned),
                },
            )
        )
    return events


def _apply_correlation_flip_group(
    *,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if len(group_indices) == 0:
        return []
    (
        _,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = _group_common(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return []
    anchor_channel = int(group_channels[0])
    target_channels = [int(ch) for ch in group_channels[1:]]
    anchor_window = runtime.compose_window(
        base=base,
        bo=channel_bos[int(anchor_channel)],
        channel=int(anchor_channel),
        start=source_start,
        end=source_end,
    )
    events: List[Dict[str, Any]] = []
    for channel in target_channels:
        segment_idx = int(segment_idx_by_channel[int(channel)])
        params = anomaly_parameters_per_segment[int(segment_idx)]
        target_correlation_raw = params.get("target_correlation", -0.85)
        target_correlation = (
            None if target_correlation_raw is None else float(target_correlation_raw)
        )
        transition_length = int(params.get("transition_length", 8))
        before_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        latent = _latent_shared_noise_attrs(channel_bos[int(channel)], source_start, source_end)
        injection_level = "observed_window"
        if latent is not None and target_correlation is not None:
            idio, shared, noise_mean, _ = latent
            candidate_noise = shared_noise_window_from_components(
                idiosyncratic_component=idio,
                shared_component=shared,
                noise_mean=noise_mean,
                target_shared_weight=float(target_correlation),
            )
            runtime.replace_noise(
                bo=channel_bos[int(channel)],
                start=source_start,
                end=source_end,
                target_noise=candidate_noise,
            )
            injection_level = "noise"
        else:
            candidate = correlation_flip_window(
                reference=before_window,
                anchor=anchor_window,
                target_correlation=target_correlation,
            )
            candidate = BaseAnomaly.blend_with_reference(
                candidate, before_window, transition_length
            )
            runtime.replace_window(
                base=base,
                bo=channel_bos[int(channel)],
                channel=int(channel),
                start=source_start,
                end=source_end,
                target_observed=candidate,
            )
        after_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        effective_delta = np.abs(after_window - before_window)
        label_start, label_end = runtime.resolve_label_bounds(
            protocol_start=source_start,
            protocol_end=source_end,
            delta=effective_delta,
            anomaly_type=anomaly_type,
        )
        labels[label_start:label_end, int(channel)] = 1
        used_positions[int(channel)].append((source_start, source_end))
        events.append(
            build_event_record(
                start=int(label_start),
                end=int(label_end),
                channel=int(channel),
                anomaly_type=anomaly_type,
                group_id=int(group_id),
                group_channels=[int(ch) for ch in group_channels],
                affected_channels=[int(ch) for ch in target_channels],
                anomaly_object="pair_correlation_flip",
                channel_visible=False,
                purity_hint=(
                    "operational_candidate"
                    if injection_level == "noise"
                    else "multivariate_preferred"
                ),
                params=runtime.to_builtin(params),
                source_start=int(source_start),
                source_end=int(source_end),
                extra={
                    "anchor_channel": int(anchor_channel),
                    "target_correlation": target_correlation,
                    "injection_level": injection_level,
                },
            )
        )
    return events


def _apply_covariance_change_group(
    *,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if len(group_indices) == 0:
        return []
    (
        _,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = _group_common(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return []
    anchor_channel = int(group_channels[0])
    target_channels = [int(ch) for ch in group_channels[1:]]
    anchor_window = runtime.compose_window(
        base=base,
        bo=channel_bos[int(anchor_channel)],
        channel=int(anchor_channel),
        start=source_start,
        end=source_end,
    )
    events: List[Dict[str, Any]] = []
    for channel in target_channels:
        segment_idx = int(segment_idx_by_channel[int(channel)])
        params = anomaly_parameters_per_segment[int(segment_idx)]
        coupling_strength = float(params.get("coupling_strength", -0.9))
        transition_length = int(params.get("transition_length", 8))
        before_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        latent = _latent_shared_noise_attrs(channel_bos[int(channel)], source_start, source_end)
        injection_level = "observed_window"
        if latent is not None:
            idio, shared, noise_mean, _ = latent
            candidate_noise = shared_noise_window_from_components(
                idiosyncratic_component=idio,
                shared_component=shared,
                noise_mean=noise_mean,
                target_shared_weight=float(coupling_strength),
            )
            runtime.replace_noise(
                bo=channel_bos[int(channel)],
                start=source_start,
                end=source_end,
                target_noise=candidate_noise,
            )
            injection_level = "noise"
        else:
            candidate = matched_coupling_window(
                reference=before_window,
                anchor=anchor_window,
                coupling_strength=coupling_strength,
            )
            candidate = BaseAnomaly.blend_with_reference(
                candidate, before_window, transition_length
            )
            runtime.replace_window(
                base=base,
                bo=channel_bos[int(channel)],
                channel=int(channel),
                start=source_start,
                end=source_end,
                target_observed=candidate,
            )
        after_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        effective_delta = np.abs(after_window - before_window)
        label_start, label_end = runtime.resolve_label_bounds(
            protocol_start=source_start,
            protocol_end=source_end,
            delta=effective_delta,
            anomaly_type=anomaly_type,
        )
        labels[label_start:label_end, int(channel)] = 1
        used_positions[int(channel)].append((source_start, source_end))
        events.append(
            build_event_record(
                start=int(label_start),
                end=int(label_end),
                channel=int(channel),
                anomaly_type=anomaly_type,
                group_id=int(group_id),
                group_channels=[int(ch) for ch in group_channels],
                affected_channels=[int(ch) for ch in target_channels],
                anomaly_object="shared_noise_coupling_change",
                channel_visible=False,
                purity_hint=(
                    "operational_candidate"
                    if injection_level == "noise"
                    else "multivariate_preferred"
                ),
                params=runtime.to_builtin(params),
                source_start=int(source_start),
                source_end=int(source_end),
                extra={
                    "anchor_channel": int(anchor_channel),
                    "coupling_strength": float(coupling_strength),
                    "injection_level": injection_level,
                },
            )
        )
    return events


def _apply_channel_rewiring_group(
    *,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if len(group_indices) == 0:
        return []
    (
        _,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = _group_common(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return []
    first_channel = int(group_channels[0])
    second_channel = int(group_channels[1])
    first_params = anomaly_parameters_per_segment[int(segment_idx_by_channel[first_channel])]
    transition_length = int(first_params.get("transition_length", 8))
    rotation_degrees = float(first_params.get("rotation_degrees", 25.0))
    before_first = runtime.compose_window(
        base=base,
        bo=channel_bos[int(first_channel)],
        channel=int(first_channel),
        start=source_start,
        end=source_end,
    )
    before_second = runtime.compose_window(
        base=base,
        bo=channel_bos[int(second_channel)],
        channel=int(second_channel),
        start=source_start,
        end=source_end,
    )
    rewired_first, rewired_second = rotated_pair_windows(
        before_first, before_second, rotation_degrees=rotation_degrees
    )
    candidate_by_channel = {
        int(first_channel): BaseAnomaly.blend_with_reference(
            rewired_first, before_first, transition_length
        ),
        int(second_channel): BaseAnomaly.blend_with_reference(
            rewired_second, before_second, transition_length
        ),
    }
    events: List[Dict[str, Any]] = []
    for channel, before_window in [
        (first_channel, before_first),
        (second_channel, before_second),
    ]:
        segment_idx = int(segment_idx_by_channel[int(channel)])
        runtime.replace_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
            target_observed=candidate_by_channel[int(channel)],
        )
        after_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        effective_delta = np.abs(after_window - before_window)
        label_start, label_end = runtime.resolve_label_bounds(
            protocol_start=source_start,
            protocol_end=source_end,
            delta=effective_delta,
            anomaly_type=anomaly_type,
        )
        labels[label_start:label_end, int(channel)] = 1
        used_positions[int(channel)].append((source_start, source_end))
        events.append(
            build_event_record(
                start=int(label_start),
                end=int(label_end),
                channel=int(channel),
                anomaly_type=anomaly_type,
                group_id=int(group_id),
                group_channels=[int(ch) for ch in group_channels],
                affected_channels=[int(first_channel), int(second_channel)],
                anomaly_object="paired_structure_rotation",
                channel_visible=True,
                purity_hint="not_pure_local",
                params=runtime.to_builtin(anomaly_parameters_per_segment[int(segment_idx)]),
                source_start=int(source_start),
                source_end=int(source_end),
                extra={
                    "rewired_pair": [int(first_channel), int(second_channel)],
                    "rotation_degrees": float(rotation_degrees),
                },
            )
        )
    return events


def _apply_lag_synchronization_group(
    *,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if len(group_indices) == 0:
        return []
    (
        _,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = _group_common(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return []
    anchor_channel = int(group_channels[0])
    shifted_channels = [int(ch) for ch in group_channels[1:]]
    events: List[Dict[str, Any]] = []
    for channel in shifted_channels:
        segment_idx = int(segment_idx_by_channel[int(channel)])
        params = anomaly_parameters_per_segment[int(segment_idx)]
        lag_steps = int(params.get("lag_steps", 6))
        transition_length = int(params.get("transition_length", 8))
        full_series = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=0,
            end=base.shape[0],
        )
        before_window = np.array(full_series[source_start:source_end], copy=True)
        shifted_window, realized_lag = lag_shift_window(
            series=full_series,
            start=source_start,
            end=source_end,
            lag_steps=lag_steps,
        )
        candidate = BaseAnomaly.blend_with_reference(
            shifted_window, before_window, transition_length
        )
        runtime.replace_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
            target_observed=candidate,
        )
        after_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        effective_delta = np.abs(after_window - before_window)
        label_start, label_end = runtime.resolve_label_bounds(
            protocol_start=source_start,
            protocol_end=source_end,
            delta=effective_delta,
            anomaly_type=anomaly_type,
        )
        labels[label_start:label_end, int(channel)] = 1
        used_positions[int(channel)].append((source_start, source_end))
        events.append(
            build_event_record(
                start=int(label_start),
                end=int(label_end),
                channel=int(channel),
                anomaly_type=anomaly_type,
                group_id=int(group_id),
                group_channels=[int(ch) for ch in group_channels],
                affected_channels=[int(ch) for ch in shifted_channels],
                anomaly_object="lag_synchronization_shift",
                channel_visible=True,
                purity_hint="not_pure_local",
                params=runtime.to_builtin(params),
                source_start=int(source_start),
                source_end=int(source_end),
                extra={
                    "anchor_channel": int(anchor_channel),
                    "realized_lag_steps": int(realized_lag),
                },
            )
        )
    return events


def _apply_shared_factor_break_group(
    *,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    if len(group_indices) == 0:
        return []
    (
        _,
        source_start,
        source_end,
        group_id,
        group_channels,
        segment_idx_by_channel,
    ) = _group_common(group_indices, segment_plan)
    if source_end <= source_start or len(group_channels) < 2:
        return []
    anchor_channel = int(group_channels[0])
    target_channels = [int(ch) for ch in group_channels[1:]]
    anchor_window = runtime.compose_window(
        base=base,
        bo=channel_bos[int(anchor_channel)],
        channel=int(anchor_channel),
        start=source_start,
        end=source_end,
    )
    events: List[Dict[str, Any]] = []
    for channel in target_channels:
        segment_idx = int(segment_idx_by_channel[int(channel)])
        params = anomaly_parameters_per_segment[int(segment_idx)]
        shared_factor_scale = float(params.get("shared_factor_scale", 0.0))
        transition_length = int(params.get("transition_length", 8))
        before_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        latent = _latent_shared_noise_attrs(channel_bos[int(channel)], source_start, source_end)
        injection_level = "observed_window"
        if latent is not None:
            idio, shared, noise_mean, current_shared_weight = latent
            target_shared_weight = float(
                np.clip(current_shared_weight * shared_factor_scale, -0.999, 0.999)
            )
            candidate_noise = shared_noise_window_from_components(
                idiosyncratic_component=idio,
                shared_component=shared,
                noise_mean=noise_mean,
                target_shared_weight=target_shared_weight,
            )
            runtime.replace_noise(
                bo=channel_bos[int(channel)],
                start=source_start,
                end=source_end,
                target_noise=candidate_noise,
            )
            injection_level = "noise"
        else:
            candidate = break_shared_factor_window(
                reference=before_window,
                anchor=anchor_window,
                shared_factor_scale=shared_factor_scale,
            )
            candidate = BaseAnomaly.blend_with_reference(
                candidate, before_window, transition_length
            )
            runtime.replace_window(
                base=base,
                bo=channel_bos[int(channel)],
                channel=int(channel),
                start=source_start,
                end=source_end,
                target_observed=candidate,
            )
        after_window = runtime.compose_window(
            base=base,
            bo=channel_bos[int(channel)],
            channel=int(channel),
            start=source_start,
            end=source_end,
        )
        effective_delta = np.abs(after_window - before_window)
        label_start, label_end = runtime.resolve_label_bounds(
            protocol_start=source_start,
            protocol_end=source_end,
            delta=effective_delta,
            anomaly_type=anomaly_type,
        )
        labels[label_start:label_end, int(channel)] = 1
        used_positions[int(channel)].append((source_start, source_end))
        events.append(
            build_event_record(
                start=int(label_start),
                end=int(label_end),
                channel=int(channel),
                anomaly_type=anomaly_type,
                group_id=int(group_id),
                group_channels=[int(ch) for ch in group_channels],
                affected_channels=[int(ch) for ch in target_channels],
                anomaly_object="shared_factor_break",
                channel_visible=False,
                purity_hint=(
                    "operational_candidate"
                    if injection_level == "noise"
                    else "multivariate_preferred"
                ),
                params=runtime.to_builtin(params),
                source_start=int(source_start),
                source_end=int(source_end),
                extra={
                    "anchor_channel": int(anchor_channel),
                    "shared_factor_scale": float(shared_factor_scale),
                    "injection_level": injection_level,
                },
            )
        )
    return events
