"""Energy-aware segment planning."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..signal_energy import (
    peak_values_for_channel_length as _peak_values_for_channel_length,
    residual_stats_values as _residual_stats_values,
    rms_values_for_channel_length as _rms_values_for_channel_length,
    window_residual_scale,
    window_rms_from_prefix,
)
from .common import occupy_slot, sample_segment_lengths
from .energy_selection import (
    available_start_mask as _available_start_mask,
    sample_uniform_slot as _sample_uniform_slot,
    select_energy_candidate as _select_energy_candidate,
)
from .types import SegmentPlan

__all__ = [
    "sample_energy_aware_segments",
    "window_residual_scale",
]


@dataclass(frozen=True)
class _EnergyPlannerSettings:
    metric_mode: str
    rms_quantile: float
    peak_quantile: float
    weighted_sampling: bool
    fallback_mode: str
    use_residual_energy: bool
    amplitude_center_mode: str
    min_residual_scale: float


@dataclass
class _EnergyPlannerState:
    clean_values: np.ndarray
    length: int
    channels: int
    occupied_global: np.ndarray
    occupied_per_channel: np.ndarray
    sq_prefix_by_channel: list[np.ndarray]
    abs_by_channel: list[np.ndarray]
    rms_cache: dict[tuple[int, int], np.ndarray]
    peak_cache: dict[tuple[int, int], np.ndarray]
    residual_scale_cache: dict[tuple[int, int], np.ndarray]
    residual_peak_cache: dict[tuple[int, int], np.ndarray]


def sample_energy_aware_segments(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    planner_cfg: Mapping[str, Any],
    anomaly_type: str,
    clean_values: np.ndarray | None,
    segment_count_range: Sequence[int],
    min_segment_length: int,
    logger: logging.Logger | None = None,
    segment_factory: Callable[..., SegmentPlan] = SegmentPlan,
) -> list[SegmentPlan]:
    """Sample segments from high-energy clean windows."""
    length = int(series_length)
    n_channels = int(channels)
    active_clean_values = _validated_clean_values(
        clean_values=clean_values,
        length=length,
        channels=n_channels,
    )
    settings = _resolve_energy_planner_settings(planner_cfg, anomaly_type)
    lengths = _sample_energy_segment_lengths(
        rng=rng,
        target_density=target_density,
        series_length=length,
        segment_count_range=segment_count_range,
        min_segment_length=min_segment_length,
        logger=logger,
    )
    state = _build_energy_planner_state(
        clean_values=active_clean_values,
        length=length,
        channels=n_channels,
    )
    segments: list[SegmentPlan] = []

    for segment_length in lengths:
        selected, used_fallback = _select_energy_segment(
            rng=rng,
            segment_length=int(segment_length),
            overlap_policy=overlap_policy,
            max_placement_attempts=max_placement_attempts,
            state=state,
            settings=settings,
            segment_factory=segment_factory,
        )
        occupy_slot(
            selected.start,
            selected.end,
            selected.channel,
            overlap_policy,
            state.occupied_global,
            state.occupied_per_channel,
        )
        selected.attrs.update(
            _energy_segment_attrs(
                selected=selected,
                segment_length=int(segment_length),
                used_fallback=used_fallback,
                state=state,
                settings=settings,
            )
        )
        segments.append(selected)

    segments.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
    return segments


def _validated_clean_values(
    *,
    clean_values: np.ndarray | None,
    length: int,
    channels: int,
) -> np.ndarray:
    if clean_values is None:
        raise ValueError(
            "energy_aware_segments planner requires clean_values for "
            "window-energy checks."
        )
    if clean_values.shape != (length, channels):
        raise ValueError(
            "energy_aware_segments expected clean_values shape "
            f"({length}, {channels}), got {clean_values.shape}."
        )
    return np.asarray(clean_values, dtype=np.float64)


def _resolve_energy_planner_settings(
    planner_cfg: Mapping[str, Any],
    anomaly_type: str,
) -> _EnergyPlannerSettings:
    metric_mode = str(planner_cfg.get("energy_metric", "rms")).lower()
    if metric_mode not in ("rms", "peak", "rms_and_peak"):
        raise ValueError(
            "energy_aware_segments.energy_metric must be one of "
            "{'rms','peak','rms_and_peak'}"
        )
    fallback_mode = str(planner_cfg.get("fallback", "uniform_segments")).lower()
    if fallback_mode not in ("uniform_segments", "error"):
        raise ValueError(
            "energy_aware_segments.fallback must be one of "
            "{'uniform_segments','error'}"
        )
    return _EnergyPlannerSettings(
        metric_mode=metric_mode,
        rms_quantile=float(
            np.clip(float(planner_cfg.get("rms_quantile", 0.60)), 0.0, 1.0)
        ),
        peak_quantile=float(
            np.clip(float(planner_cfg.get("peak_quantile", 0.55)), 0.0, 1.0)
        ),
        weighted_sampling=bool(planner_cfg.get("weighted_sampling", True)),
        fallback_mode=fallback_mode,
        use_residual_energy=(
            anomaly_type == "amplitude"
            and max(0.0, float(planner_cfg.get("min_effect_delta", 0.0))) > 0.0
        ),
        amplitude_center_mode=str(planner_cfg.get("center_mode", "linear")),
        min_residual_scale=max(
            0.0,
            float(planner_cfg.get("min_residual_scale", 1e-6)),
        ),
    )


def _sample_energy_segment_lengths(
    *,
    rng: np.random.Generator,
    target_density: float,
    series_length: int,
    segment_count_range: Sequence[int],
    min_segment_length: int,
    logger: logging.Logger | None,
) -> list[int]:
    low, high = int(segment_count_range[0]), int(segment_count_range[1])
    if low > high:
        low, high = high, low
    n_segments = int(rng.integers(low, high + 1))
    if n_segments > int(series_length):
        raise ValueError(
            f"Requested n_segments={n_segments} exceeds "
            f"series length={int(series_length)}."
        )
    target_points = int(round(float(target_density) * int(series_length)))
    target_points = max(target_points, n_segments)
    target_points = min(target_points, int(series_length))
    min_length = int(min_segment_length)
    max_segments_for_min_length = max(1, target_points // max(1, min_length))
    if n_segments > max_segments_for_min_length:
        if logger is not None:
            logger.warning(
                "Reducing n_segments from %s to %s to satisfy "
                "min_segment_length=%s for target_points=%s "
                "(energy_aware_segments).",
                n_segments,
                max_segments_for_min_length,
                min_length,
                target_points,
            )
        n_segments = max_segments_for_min_length
    return sample_segment_lengths(
        rng,
        target_points,
        n_segments,
        min_segment_length=min_length,
    )


def _build_energy_planner_state(
    *,
    clean_values: np.ndarray,
    length: int,
    channels: int,
) -> _EnergyPlannerState:
    return _EnergyPlannerState(
        clean_values=clean_values,
        length=int(length),
        channels=int(channels),
        occupied_global=np.zeros(int(length), dtype=np.int8),
        occupied_per_channel=np.zeros((int(channels), int(length)), dtype=np.int8),
        sq_prefix_by_channel=[
            np.concatenate(
                [
                    [0.0],
                    np.cumsum(
                        np.square(clean_values[:, channel]),
                        dtype=np.float64,
                    ),
                ]
            )
            for channel in range(int(channels))
        ],
        abs_by_channel=[
            np.abs(clean_values[:, channel]).astype(np.float64)
            for channel in range(int(channels))
        ],
        rms_cache={},
        peak_cache={},
        residual_scale_cache={},
        residual_peak_cache={},
    )


def _select_energy_segment(
    *,
    rng: np.random.Generator,
    segment_length: int,
    overlap_policy: str,
    max_placement_attempts: int,
    state: _EnergyPlannerState,
    settings: _EnergyPlannerSettings,
    segment_factory: Callable[..., SegmentPlan],
) -> tuple[SegmentPlan, bool]:
    _validate_segment_length(segment_length, state.length)
    thresholds_rms, thresholds_peak = _populate_energy_thresholds(
        state=state,
        segment_length=segment_length,
        settings=settings,
    )
    selected = _select_energy_candidate(
        rng=rng,
        length=segment_length,
        channels=state.channels,
        metric_mode=settings.metric_mode,
        weighted_sampling=settings.weighted_sampling,
        min_residual_scale=settings.min_residual_scale,
        availability_by_channel=_availability_by_channel(
            state,
            segment_length,
            overlap_policy,
        ),
        rms_cache=state.rms_cache,
        peak_cache=state.peak_cache,
        residual_scale_cache=state.residual_scale_cache,
        thresholds_rms=thresholds_rms,
        thresholds_peak=thresholds_peak,
        segment_factory=segment_factory,
    )
    if selected is not None:
        return selected, False
    if settings.fallback_mode == "uniform_segments":
        selected = _sample_uniform_slot(
            rng=rng,
            length=segment_length,
            series_length=state.length,
            channels=state.channels,
            max_placement_attempts=max_placement_attempts,
            overlap_policy=overlap_policy,
            occupied_global=state.occupied_global,
            occupied_per_channel=state.occupied_per_channel,
            segment_factory=segment_factory,
        )
        if selected is not None:
            return selected, True
    raise ValueError(
        "Failed to place an energy-aware segment without overlap "
        f"(length={segment_length}, fallback={settings.fallback_mode})."
    )


def _validate_segment_length(segment_length: int, series_length: int) -> None:
    if int(series_length) - int(segment_length) + 1 <= 0:
        raise ValueError(
            f"Segment length {segment_length} is infeasible for "
            f"series length {series_length}."
        )


def _availability_by_channel(
    state: _EnergyPlannerState,
    segment_length: int,
    overlap_policy: str,
) -> dict[int, np.ndarray]:
    if overlap_policy == "global":
        return {
            channel: _available_start_mask(state.occupied_global, segment_length)
            for channel in range(state.channels)
        }
    return {
        channel: _available_start_mask(
            state.occupied_per_channel[channel, :],
            segment_length,
        )
        for channel in range(state.channels)
    }


def _populate_energy_thresholds(
    *,
    state: _EnergyPlannerState,
    segment_length: int,
    settings: _EnergyPlannerSettings,
) -> tuple[dict[int, float], dict[int, float]]:
    thresholds_rms: dict[int, float] = {}
    thresholds_peak: dict[int, float] = {}
    if settings.metric_mode in ("rms", "rms_and_peak"):
        for channel in range(state.channels):
            values = _rms_metric_values(state, channel, segment_length, settings)
            state.rms_cache[(channel, segment_length)] = values
            thresholds_rms[channel] = float(np.quantile(values, settings.rms_quantile))
    if settings.metric_mode in ("peak", "rms_and_peak"):
        for channel in range(state.channels):
            values = _peak_metric_values(state, channel, segment_length, settings)
            state.peak_cache[(channel, segment_length)] = values
            thresholds_peak[channel] = float(
                np.quantile(values, settings.peak_quantile)
            )
    return thresholds_rms, thresholds_peak


def _rms_metric_values(
    state: _EnergyPlannerState,
    channel: int,
    segment_length: int,
    settings: _EnergyPlannerSettings,
) -> np.ndarray:
    if settings.use_residual_energy:
        residual_scale, residual_peak = _residual_stats_values(
            state.clean_values[:, channel],
            length=segment_length,
            center_mode=settings.amplitude_center_mode,
        )
        state.residual_scale_cache[(channel, segment_length)] = residual_scale
        state.residual_peak_cache[(channel, segment_length)] = residual_peak
        return residual_scale
    return _rms_values_for_channel_length(
        sq_prefix=state.sq_prefix_by_channel[channel],
        length=segment_length,
    )


def _peak_metric_values(
    state: _EnergyPlannerState,
    channel: int,
    segment_length: int,
    settings: _EnergyPlannerSettings,
) -> np.ndarray:
    if settings.use_residual_energy:
        cache_key = (channel, segment_length)
        if cache_key not in state.residual_peak_cache:
            residual_scale, residual_peak = _residual_stats_values(
                state.clean_values[:, channel],
                length=segment_length,
                center_mode=settings.amplitude_center_mode,
            )
            state.residual_scale_cache[cache_key] = residual_scale
            state.residual_peak_cache[cache_key] = residual_peak
        return state.residual_peak_cache[cache_key]
    return _peak_values_for_channel_length(
        abs_values=state.abs_by_channel[channel],
        length=segment_length,
    )


def _energy_segment_attrs(
    *,
    selected: SegmentPlan,
    segment_length: int,
    used_fallback: bool,
    state: _EnergyPlannerState,
    settings: _EnergyPlannerSettings,
) -> dict[str, Any]:
    window_rms = window_rms_from_prefix(
        state.sq_prefix_by_channel[selected.channel],
        selected.start,
        selected.end,
    )
    window_peak = float(
        np.max(state.abs_by_channel[selected.channel][selected.start : selected.end])
    )
    attrs: dict[str, Any] = {
        "window_rms": float(window_rms),
        "window_peak": float(window_peak),
        "energy_metric": settings.metric_mode,
        "energy_fallback": bool(used_fallback),
    }
    if settings.use_residual_energy:
        cache_key = (selected.channel, int(segment_length))
        attrs.update(
            {
                "energy_reference": "amplitude_residual",
                "window_residual_scale": float(
                    state.residual_scale_cache[cache_key][selected.start]
                ),
                "window_residual_peak": float(
                    state.residual_peak_cache[cache_key][selected.start]
                ),
                "amplitude_center_mode": settings.amplitude_center_mode,
            }
        )
    return attrs
