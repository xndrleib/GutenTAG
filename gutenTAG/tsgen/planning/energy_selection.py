"""Candidate selection helpers for energy-aware segment planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from .common import is_slot_available
from .types import SegmentPlan


def select_energy_candidate(
    *,
    rng: np.random.Generator,
    length: int,
    channels: int,
    metric_mode: str,
    weighted_sampling: bool,
    min_residual_scale: float,
    availability_by_channel: Mapping[int, np.ndarray],
    rms_cache: Mapping[tuple[int, int], np.ndarray],
    peak_cache: Mapping[tuple[int, int], np.ndarray],
    residual_scale_cache: Mapping[tuple[int, int], np.ndarray],
    thresholds_rms: Mapping[int, float],
    thresholds_peak: Mapping[int, float],
    segment_factory: Callable[..., SegmentPlan],
) -> SegmentPlan | None:
    """Select one available high-energy candidate, or return ``None``."""

    candidate_channels: list[np.ndarray] = []
    candidate_starts: list[np.ndarray] = []
    candidate_weights: list[np.ndarray] = []
    for channel in range(channels):
        energy_mask = np.ones(availability_by_channel[channel].shape[0], dtype=bool)
        if metric_mode in ("rms", "rms_and_peak"):
            rms_values = rms_cache[(channel, int(length))]
            energy_mask &= rms_values >= thresholds_rms[channel]
        if metric_mode in ("peak", "rms_and_peak"):
            peak_values = peak_cache[(channel, int(length))]
            energy_mask &= peak_values >= thresholds_peak[channel]
        if (channel, int(length)) in residual_scale_cache:
            residual_values = residual_scale_cache[(channel, int(length))]
            energy_mask &= residual_values >= min_residual_scale
        mask = energy_mask & availability_by_channel[channel]
        starts = np.flatnonzero(mask)
        if starts.size == 0:
            continue
        candidate_channels.append(np.full(starts.shape[0], int(channel), dtype=int))
        candidate_starts.append(starts.astype(int))
        if weighted_sampling:
            candidate_weights.append(
                _candidate_weights(
                    starts=starts,
                    channel=channel,
                    length=length,
                    metric_mode=metric_mode,
                    rms_cache=rms_cache,
                    peak_cache=peak_cache,
                    residual_scale_cache=residual_scale_cache,
                    thresholds_rms=thresholds_rms,
                    thresholds_peak=thresholds_peak,
                    min_residual_scale=min_residual_scale,
                )
            )

    if len(candidate_starts) == 0:
        return None
    all_channels = np.concatenate(candidate_channels)
    all_starts = np.concatenate(candidate_starts)
    if weighted_sampling and len(candidate_weights) > 0:
        all_weights = np.concatenate(candidate_weights).astype(np.float64)
        weight_sum = float(np.sum(all_weights))
        if weight_sum > 0.0:
            probs = all_weights / weight_sum
            selected_idx = int(rng.choice(np.arange(all_starts.size), p=probs))
        else:
            selected_idx = int(rng.integers(0, all_starts.size))
    else:
        selected_idx = int(rng.integers(0, all_starts.size))
    selected_channel = int(all_channels[selected_idx])
    selected_start = int(all_starts[selected_idx])
    selected_end = int(selected_start + int(length))
    return segment_factory(
        start=selected_start,
        end=selected_end,
        length=int(length),
        channel=selected_channel,
    )


def available_start_mask(occupied: np.ndarray, length: int) -> np.ndarray:
    """Return starts whose fixed-length window has no occupied points."""

    if length <= 0:
        return np.zeros(0, dtype=bool)
    prefix = np.concatenate([[0], np.cumsum(occupied, dtype=np.int64)])
    return (prefix[length:] - prefix[:-length]) == 0


def sample_uniform_slot(
    *,
    rng: np.random.Generator,
    length: int,
    series_length: int,
    channels: int,
    max_placement_attempts: int,
    overlap_policy: str,
    occupied_global: np.ndarray,
    occupied_per_channel: np.ndarray,
    segment_factory: Callable[..., SegmentPlan],
) -> SegmentPlan | None:
    """Sample any available slot, with deterministic exhaustive fallback."""

    max_start = int(series_length) - int(length)
    if max_start < 0:
        return None
    for _ in range(int(max_placement_attempts)):
        channel = int(rng.integers(0, int(channels)))
        start = int(rng.integers(0, max_start + 1))
        end = start + int(length)
        if is_slot_available(
            start,
            end,
            channel,
            overlap_policy,
            occupied_global,
            occupied_per_channel,
        ):
            return segment_factory(
                start=start,
                end=end,
                length=int(length),
                channel=channel,
            )

    channel_order = rng.permutation(int(channels)).tolist()
    for channel in channel_order:
        for start in range(max_start + 1):
            end = start + int(length)
            if is_slot_available(
                start,
                end,
                int(channel),
                overlap_policy,
                occupied_global,
                occupied_per_channel,
            ):
                return segment_factory(
                    start=start,
                    end=end,
                    length=int(length),
                    channel=int(channel),
                )
    return None


def _candidate_weights(
    *,
    starts: np.ndarray,
    channel: int,
    length: int,
    metric_mode: str,
    rms_cache: Mapping[tuple[int, int], np.ndarray],
    peak_cache: Mapping[tuple[int, int], np.ndarray],
    residual_scale_cache: Mapping[tuple[int, int], np.ndarray],
    thresholds_rms: Mapping[int, float],
    thresholds_peak: Mapping[int, float],
    min_residual_scale: float,
) -> np.ndarray:
    weights = np.ones(starts.shape[0], dtype=np.float64)
    if metric_mode in ("rms", "rms_and_peak"):
        rms_part = np.maximum(
            rms_cache[(channel, int(length))][starts] - thresholds_rms[channel],
            0.0,
        )
        weights += rms_part
    if metric_mode in ("peak", "rms_and_peak"):
        peak_part = np.maximum(
            peak_cache[(channel, int(length))][starts] - thresholds_peak[channel],
            0.0,
        )
        weights += peak_part
    if (channel, int(length)) in residual_scale_cache:
        residual_part = np.maximum(
            residual_scale_cache[(channel, int(length))][starts] - min_residual_scale,
            0.0,
        )
        weights += residual_part
    return weights


__all__ = [
    "available_start_mask",
    "sample_uniform_slot",
    "select_energy_candidate",
]
