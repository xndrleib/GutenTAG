"""Apply runtime anomaly objects to generated time-series arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.random import SeedSequence

from ..tsgen.planning import event_extra_from_segment_attrs
from ..utils.types import GenerationContext
from .anomaly_objects import GROUP_LEVEL_ANOMALY_TYPES
from .event_metadata import build_event_record
from .group_anomalies import GroupAnomalyRuntime, apply_group_anomaly
from .segment_groups import group_indices_by_id


@dataclass(frozen=True)
class AnomalyApplicationRuntime:
    """Callbacks required by anomaly application.

    The generator facade owns variation-aware window composition and label
    policy configuration. This runtime keeps those dependencies explicit while
    moving the application workflow out of the facade.
    """

    compose_window: Callable[..., np.ndarray]
    replace_window: Callable[..., None]
    compose_noise: Callable[..., np.ndarray]
    replace_noise: Callable[..., None]
    resolve_label_bounds: Callable[..., tuple[int, int]]
    normalize_subsequence: Callable[[np.ndarray, int], np.ndarray]
    to_builtin: Callable[[Any], Any]


@dataclass
class _AnomalyApplicationState:
    labels: np.ndarray
    events: list[dict[str, Any]]
    ctx: GenerationContext
    used_positions: dict[int, list[tuple[int, int]]]
    processed_group_ids: set[int]
    segment_groups: dict[int, list[int]]


def apply_anomalies(
    *,
    anomaly_objects: Sequence[Any],
    segment_plan: Sequence[Any],
    base: np.ndarray,
    channel_bos: Sequence[Any],
    anomaly_seed: int,
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
    series_length: int,
    channels: int,
    runtime: AnomalyApplicationRuntime,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply runtime anomalies and return pointwise labels and sorted events."""

    state = _initial_application_state(
        anomaly_seed=anomaly_seed,
        segment_plan=segment_plan,
        series_length=series_length,
        channels=channels,
    )
    for segment_idx, anomaly in enumerate(anomaly_objects):
        _apply_segment_anomaly(
            segment_idx=segment_idx,
            anomaly=anomaly,
            state=state,
            segment_plan=segment_plan,
            base=base,
            channel_bos=channel_bos,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            runtime=runtime,
        )

    state.events.sort(
        key=lambda event: (event["start"], event["channel"], event["end"])
    )
    return state.labels, state.events


def _initial_application_state(
    *,
    anomaly_seed: int,
    segment_plan: Sequence[Any],
    series_length: int,
    channels: int,
) -> _AnomalyApplicationState:
    return _AnomalyApplicationState(
        labels=np.zeros((int(series_length), int(channels)), dtype=np.int8),
        events=[],
        ctx=GenerationContext(SeedSequence(anomaly_seed)),
        used_positions={channel: [] for channel in range(int(channels))},
        processed_group_ids=set(),
        segment_groups=group_indices_by_id(list(segment_plan)),
    )


def _apply_segment_anomaly(
    *,
    segment_idx: int,
    anomaly: Any,
    state: _AnomalyApplicationState,
    segment_plan: Sequence[Any],
    base: np.ndarray,
    channel_bos: Sequence[Any],
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
    runtime: AnomalyApplicationRuntime,
) -> None:
    segment_metadata = segment_plan[segment_idx]
    group_id, group_channels = _segment_group_metadata(segment_idx, segment_metadata)
    if _is_group_anomaly_segment(anomaly_type, group_channels):
        if group_id in state.processed_group_ids:
            return
        state.processed_group_ids.add(group_id)
        state.events.extend(
            _apply_group_anomaly(
                group_indices=state.segment_groups.get(group_id, [segment_idx]),
                segment_plan=segment_plan,
                base=base,
                channel_bos=channel_bos,
                labels=state.labels,
                used_positions=state.used_positions,
                anomaly_type=anomaly_type,
                anomaly_parameters_per_segment=anomaly_parameters_per_segment,
                runtime=runtime,
            )
        )
        return

    state.events.append(
        _apply_single_anomaly(
            segment_idx=segment_idx,
            anomaly=anomaly,
            segment_metadata=segment_metadata,
            base=base,
            channel_bos=channel_bos,
            labels=state.labels,
            used_positions=state.used_positions,
            anomaly_type=anomaly_type,
            anomaly_parameters_per_segment=anomaly_parameters_per_segment,
            ctx=state.ctx,
            runtime=runtime,
        )
    )


def _segment_group_metadata(
    segment_idx: int,
    segment_metadata: Any,
) -> tuple[int, list[int]]:
    group_id = int(segment_metadata.attrs.get("group_id", segment_idx))
    group_channels = [
        int(channel)
        for channel in segment_metadata.attrs.get(
            "group_channels", [int(segment_metadata.channel)]
        )
    ]
    return group_id, group_channels


def _is_group_anomaly_segment(
    anomaly_type: str,
    group_channels: Sequence[int],
) -> bool:
    return anomaly_type in GROUP_LEVEL_ANOMALY_TYPES and len(group_channels) > 1


def _apply_group_anomaly(
    *,
    group_indices: Sequence[int],
    segment_plan: Sequence[Any],
    base: np.ndarray,
    channel_bos: Sequence[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
    runtime: AnomalyApplicationRuntime,
) -> list[dict[str, Any]]:
    events = apply_group_anomaly(
        anomaly_type=anomaly_type,
        group_indices=list(group_indices),
        segment_plan=list(segment_plan),
        base=base,
        channel_bos=list(channel_bos),
        labels=labels,
        used_positions=used_positions,
        anomaly_parameters_per_segment=[
            dict(params) for params in anomaly_parameters_per_segment
        ],
        runtime=GroupAnomalyRuntime(
            compose_window=runtime.compose_window,
            replace_window=runtime.replace_window,
            compose_noise=runtime.compose_noise,
            replace_noise=runtime.replace_noise,
            resolve_label_bounds=runtime.resolve_label_bounds,
            to_builtin=runtime.to_builtin,
        ),
    )
    if len(group_indices) > 0:
        group_attrs = dict(segment_plan[int(group_indices[0])].attrs)
        for event in events:
            event.update(
                event_extra_from_segment_attrs(
                    group_attrs,
                    source_start=int(event.get("source_start", event["start"])),
                    support_start=int(event["start"]),
                    sanitize=runtime.to_builtin,
                )
            )
    return events


def _apply_single_anomaly(
    *,
    segment_idx: int,
    anomaly: Any,
    segment_metadata: Any,
    base: np.ndarray,
    channel_bos: Sequence[Any],
    labels: np.ndarray,
    used_positions: dict[int, list[tuple[int, int]]],
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
    ctx: GenerationContext,
    runtime: AnomalyApplicationRuntime,
) -> dict[str, Any]:
    channel = int(anomaly.channel)
    bo = channel_bos[channel]
    planned_start, planned_end = _planned_anomaly_bounds(anomaly)
    before_window = _compose_planned_window(
        base=base,
        bo=bo,
        channel=channel,
        planned_start=planned_start,
        planned_end=planned_end,
        runtime=runtime,
    )
    protocol = anomaly.generate(ctx.to_anomaly(bo, used_positions[channel]))
    effective_delta = _apply_single_protocol_effect(
        protocol=protocol,
        base=base,
        bo=bo,
        channel=channel,
        planned_start=planned_start,
        planned_end=planned_end,
        before_window=before_window,
        runtime=runtime,
    )

    label_start, label_end = runtime.resolve_label_bounds(
        protocol_start=int(protocol.start),
        protocol_end=int(protocol.end),
        delta=effective_delta,
        anomaly_type=anomaly_type,
    )
    labels[label_start:label_end, channel] = 1
    used_positions[channel].append((int(protocol.start), int(protocol.end)))
    return _build_single_event(
        segment_idx=segment_idx,
        segment_metadata=segment_metadata,
        channel=channel,
        anomaly_type=anomaly_type,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        label_start=int(label_start),
        label_end=int(label_end),
        protocol_start=int(protocol.start),
        protocol_end=int(protocol.end),
        runtime=runtime,
    )


def _planned_anomaly_bounds(anomaly: Any) -> tuple[int | None, int | None]:
    exact_position = getattr(anomaly, "exact_position", None)
    if exact_position is None:
        return None, None
    planned_start = int(exact_position)
    return planned_start, planned_start + int(anomaly.anomaly_length)


def _compose_planned_window(
    *,
    base: np.ndarray,
    bo: Any,
    channel: int,
    planned_start: int | None,
    planned_end: int | None,
    runtime: AnomalyApplicationRuntime,
) -> np.ndarray | None:
    if planned_start is None or planned_end is None:
        return None
    return runtime.compose_window(
        base=base,
        bo=bo,
        channel=channel,
        start=planned_start,
        end=planned_end,
    )


def _apply_single_protocol_effect(
    *,
    protocol: Any,
    base: np.ndarray,
    bo: Any,
    channel: int,
    planned_start: int | None,
    planned_end: int | None,
    before_window: np.ndarray | None,
    runtime: AnomalyApplicationRuntime,
) -> np.ndarray:
    expected_length = int(protocol.end - protocol.start)
    original_segment = np.array(
        base[protocol.start : protocol.end, channel],
        copy=True,
    )
    if protocol.subsequences:
        subsequence = _apply_protocol_subsequence(
            protocol=protocol,
            base=base,
            channel=channel,
            expected_length=expected_length,
            runtime=runtime,
        )
        return _single_effective_delta(
            protocol=protocol,
            base=base,
            bo=bo,
            channel=channel,
            planned_start=planned_start,
            planned_end=planned_end,
            before_window=before_window,
            expected_length=expected_length,
            original_segment=original_segment,
            subsequence=subsequence,
            runtime=runtime,
        )
    if _matches_planned_window(
        planned_start=planned_start,
        planned_end=planned_end,
        protocol_start=int(protocol.start),
        protocol_end=int(protocol.end),
        window=before_window,
        expected_length=expected_length,
    ):
        assert before_window is not None
        return _window_delta_after_protocol(
            protocol=protocol,
            base=base,
            bo=bo,
            channel=channel,
            before_window=before_window,
            runtime=runtime,
        )
    return np.zeros(expected_length, dtype=np.float64)


def _apply_protocol_subsequence(
    *,
    protocol: Any,
    base: np.ndarray,
    channel: int,
    expected_length: int,
    runtime: AnomalyApplicationRuntime,
) -> np.ndarray:
    subsequence = np.vstack(protocol.subsequences).sum(axis=0)
    subsequence = runtime.normalize_subsequence(subsequence, expected_length)
    base[protocol.start : protocol.end, channel] = subsequence
    return subsequence


def _single_effective_delta(
    *,
    protocol: Any,
    base: np.ndarray,
    bo: Any,
    channel: int,
    planned_start: int | None,
    planned_end: int | None,
    before_window: np.ndarray | None,
    expected_length: int,
    original_segment: np.ndarray,
    subsequence: np.ndarray,
    runtime: AnomalyApplicationRuntime,
) -> np.ndarray:
    if _matches_planned_window(
        planned_start=planned_start,
        planned_end=planned_end,
        protocol_start=int(protocol.start),
        protocol_end=int(protocol.end),
        window=before_window,
        expected_length=expected_length,
    ):
        assert before_window is not None
        return _window_delta_after_protocol(
            protocol=protocol,
            base=base,
            bo=bo,
            channel=channel,
            before_window=before_window,
            runtime=runtime,
        )
    if subsequence.shape[0] == original_segment.shape[0]:
        return np.abs(subsequence - original_segment)
    return np.zeros(expected_length, dtype=np.float64)


def _window_delta_after_protocol(
    *,
    protocol: Any,
    base: np.ndarray,
    bo: Any,
    channel: int,
    before_window: np.ndarray,
    runtime: AnomalyApplicationRuntime,
) -> np.ndarray:
    after_window = runtime.compose_window(
        base=base,
        bo=bo,
        channel=channel,
        start=protocol.start,
        end=protocol.end,
    )
    return np.abs(after_window - before_window)


def _matches_planned_window(
    *,
    planned_start: int | None,
    planned_end: int | None,
    protocol_start: int,
    protocol_end: int,
    window: np.ndarray | None,
    expected_length: int,
) -> bool:
    return (
        planned_start == protocol_start
        and planned_end == protocol_end
        and window is not None
        and window.shape[0] == expected_length
    )


def _build_single_event(
    *,
    segment_idx: int,
    segment_metadata: Any,
    channel: int,
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
    label_start: int,
    label_end: int,
    protocol_start: int,
    protocol_end: int,
    runtime: AnomalyApplicationRuntime,
) -> dict[str, Any]:
    attrs = dict(segment_metadata.attrs)
    return build_event_record(
        start=label_start,
        end=label_end,
        channel=channel,
        anomaly_type=anomaly_type,
        group_id=int(attrs.get("group_id", segment_idx)),
        group_channels=[
            int(group_channel)
            for group_channel in attrs.get("group_channels", [int(channel)])
        ],
        intervention_channels=[
            int(intervention_channel)
            for intervention_channel in attrs.get(
                "intervention_channels", [int(channel)]
            )
        ],
        anomaly_object=str(attrs.get("anomaly_object", anomaly_type)),
        channel_visible=bool(attrs.get("channel_visible", True)),
        purity_hint=str(attrs.get("purity_hint", "channel_visible")),
        params=runtime.to_builtin(anomaly_parameters_per_segment[segment_idx]),
        source_start=protocol_start,
        source_end=protocol_end,
        extra=event_extra_from_segment_attrs(
            attrs,
            source_start=protocol_start,
            support_start=label_start,
            sanitize=runtime.to_builtin,
        ),
    )
