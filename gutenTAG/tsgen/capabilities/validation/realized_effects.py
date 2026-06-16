"""Realized-effect audit metrics for generated events."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..array_store import ArrayStore
from ..dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
    event_uid,
    read_timeseries_csv,
)
from ..numerics import finite_float, safe_corrcoef, safe_variance


def compute_realized_effects(
    dataset: DatasetIndex,
    *,
    arrays: ArrayStore | None = None,
) -> pd.DataFrame:
    """Compute family-aware realized-effect measurements from paired data."""

    rows: list[dict[str, object]] = []
    for instance in dataset.instances:
        clean = (
            arrays.get(instance, "clean")
            if arrays is not None
            else read_timeseries_csv(instance.clean_path)
        )
        anomalous = (
            arrays.get(instance, "anomalous")
            if arrays is not None
            else read_timeseries_csv(instance.anomalous_path)
        )
        for group in instance.event_groups:
            rows.append(_realized_row(instance, group, clean, anomalous))
    return pd.DataFrame(rows)


def _realized_row(
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
) -> dict[str, object]:
    channels = _channels(group, instance.channels)
    clean_seg = _segment(clean, group.start, group.end, channels)
    anom_seg = _segment(anomalous, group.start, group.end, channels)
    context_channels = _context_channels(group, instance.channels)
    context_idx = _context_indices(instance.length, group.start, group.end)
    row = _base_realized_row(instance, group)
    if clean_seg.size and anom_seg.size:
        row.update(_marginal_effect_metrics(clean_seg, anom_seg))
    if len(context_channels) >= 2:
        row.update(
            _relation_effect_metrics(
                clean=clean,
                anomalous=anomalous,
                group=group,
                context_channels=context_channels,
                context_idx=context_idx,
            )
        )
    return row


def _base_realized_row(
    instance: InstanceRecord,
    group: EventGroup,
) -> dict[str, object]:
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "realized_offset": math.nan,
        "standardized_offset": math.nan,
        "direction_match": None,
        "realized_log_var_ratio": math.nan,
        "realized_mad_ratio": math.nan,
        "mean_drift_side_effect": math.nan,
        "corr_context_clean": math.nan,
        "corr_event_clean": math.nan,
        "corr_event_anomalous": math.nan,
        "abs_corr_context_clean": math.nan,
        "abs_corr_event_clean": math.nan,
        "abs_corr_event_anomalous": math.nan,
        "realized_fisher_shift": math.nan,
        "marginal_mean_shift_target": math.nan,
        "marginal_var_shift_target": math.nan,
        "clean_mode_agreement": math.nan,
        "anomalous_mode_agreement": math.nan,
        "target_vs_ensemble_corr_context": math.nan,
        "target_vs_ensemble_corr_event": math.nan,
        "pca_residual_ratio": math.nan,
        "regression_residual_ratio": math.nan,
        "spectral_peak_clean": math.nan,
        "spectral_peak_anomalous": math.nan,
        "spectral_peak_shift": math.nan,
        "band_energy_shift": math.nan,
        "template_distance_clean": math.nan,
        "template_distance_anomalous": math.nan,
        "shape_residual_ratio": math.nan,
        "boundary_shape_residual_share": math.nan,
    }


def _marginal_effect_metrics(
    clean_seg: np.ndarray,
    anom_seg: np.ndarray,
) -> dict[str, object]:
    delta = anom_seg - clean_seg
    offset = float(np.mean(delta))
    clean_var = np.var(clean_seg, axis=0) + 1e-8
    anom_var = np.var(anom_seg, axis=0) + 1e-8
    spectral_clean = _spectral_peak(clean_seg)
    spectral_anom = _spectral_peak(anom_seg)
    return {
        "realized_offset": finite_float(offset),
        "standardized_offset": finite_float(
            offset / max(_clean_segment_scale(clean_seg), 1e-8)
        ),
        "realized_log_var_ratio": finite_float(
            float(np.mean(np.log(anom_var / clean_var)))
        ),
        "realized_mad_ratio": finite_float(_mad(anom_seg) / max(_mad(clean_seg), 1e-8)),
        "mean_drift_side_effect": finite_float(abs(offset)),
        "marginal_mean_shift_target": finite_float(
            _marginal_mean_shift(clean_seg, anom_seg)
        ),
        "marginal_var_shift_target": finite_float(
            float(np.mean(np.abs(anom_var - clean_var)))
        ),
        "shape_residual_ratio": finite_float(
            _shape_residual_ratio(clean_seg, anom_seg)
        ),
        "template_distance_clean": finite_float(_template_distance(clean_seg)),
        "template_distance_anomalous": finite_float(_template_distance(anom_seg)),
        "spectral_peak_clean": finite_float(spectral_clean, default=math.nan),
        "spectral_peak_anomalous": finite_float(spectral_anom, default=math.nan),
        "spectral_peak_shift": finite_float(
            spectral_anom - spectral_clean,
            default=math.nan,
        ),
        "band_energy_shift": finite_float(
            _band_energy(anom_seg) - _band_energy(clean_seg),
            default=math.nan,
        ),
    }


def _relation_effect_metrics(
    *,
    clean: np.ndarray,
    anomalous: np.ndarray,
    group: EventGroup,
    context_channels: tuple[int, ...],
    context_idx: np.ndarray,
) -> dict[str, object]:
    clean_event = _segment(clean, group.start, group.end, context_channels)
    anom_event = _segment(anomalous, group.start, group.end, context_channels)
    clean_context = _clean_relation_context(
        clean=clean,
        clean_event=clean_event,
        context_channels=context_channels,
        context_idx=context_idx,
    )
    mode_context = _target_vs_context_corr(clean_context, context_channels, group)
    mode_event = _target_vs_context_corr(anom_event, context_channels, group)
    return {
        "corr_context_clean": finite_float(
            _mean_signed_offdiag_corr(clean_context),
            default=math.nan,
        ),
        "corr_event_clean": finite_float(
            _mean_signed_offdiag_corr(clean_event),
            default=math.nan,
        ),
        "corr_event_anomalous": finite_float(
            _mean_signed_offdiag_corr(anom_event),
            default=math.nan,
        ),
        "abs_corr_context_clean": finite_float(
            _mean_abs_offdiag_corr(clean_context),
            default=math.nan,
        ),
        "abs_corr_event_clean": finite_float(
            _mean_abs_offdiag_corr(clean_event),
            default=math.nan,
        ),
        "abs_corr_event_anomalous": finite_float(
            _mean_abs_offdiag_corr(anom_event),
            default=math.nan,
        ),
        "realized_fisher_shift": finite_float(
            _mean_abs_offdiag_fisher_shift(clean_event, anom_event),
            default=math.nan,
        ),
        "target_vs_ensemble_corr_context": finite_float(
            mode_context,
            default=math.nan,
        ),
        "target_vs_ensemble_corr_event": finite_float(
            mode_event,
            default=math.nan,
        ),
        "pca_residual_ratio": finite_float(
            _pca_residual_ratio(anom_event, clean_event),
            default=math.nan,
        ),
        "regression_residual_ratio": finite_float(
            _regression_residual_ratio(anom_event, clean_event),
            default=math.nan,
        ),
        "clean_mode_agreement": finite_float(mode_context, default=math.nan),
        "anomalous_mode_agreement": finite_float(mode_event, default=math.nan),
    }


def _clean_segment_scale(clean_seg: np.ndarray) -> float:
    if clean_seg.shape[0] == 0:
        return 1.0
    return float(np.sqrt(np.mean(np.var(clean_seg, axis=0))))


def _marginal_mean_shift(clean_seg: np.ndarray, anom_seg: np.ndarray) -> float:
    clean_mean = np.mean(clean_seg, axis=0)
    anom_mean = np.mean(anom_seg, axis=0)
    return float(np.mean(np.abs(anom_mean - clean_mean)))


def _template_distance(values: np.ndarray) -> float:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.square(centered))))


def _clean_relation_context(
    *,
    clean: np.ndarray,
    clean_event: np.ndarray,
    context_channels: tuple[int, ...],
    context_idx: np.ndarray,
) -> np.ndarray:
    if not context_idx.size:
        return clean_event
    return clean[context_idx[:, None], list(context_channels)]


def _channels(group: EventGroup, channels: int) -> tuple[int, ...]:
    selected = sorted(set(group.intervention_channels) | set(group.primary_channels))
    if not selected:
        selected = sorted(set(group.group_channels) | set(group.context_channels))
    return tuple(channel for channel in selected if 0 <= int(channel) < int(channels))


def _context_channels(group: EventGroup, channels: int) -> tuple[int, ...]:
    selected = sorted(set(group.context_channels) | set(group.group_channels))
    return tuple(channel for channel in selected if 0 <= int(channel) < int(channels))


def _segment(
    series: np.ndarray, start: int, end: int, channels: tuple[int, ...]
) -> np.ndarray:
    if not channels:
        return np.empty((0, 0), dtype=np.float64)
    lo = max(0, min(int(start), series.shape[0]))
    hi = max(lo, min(int(end), series.shape[0]))
    return np.asarray(series, dtype=np.float64)[lo:hi, list(channels)]


def _context_indices(length: int, start: int, end: int) -> np.ndarray:
    width = max(1, int(end) - int(start))
    lo = max(0, int(start) - width)
    hi = min(int(length), int(end) + width)
    mask = np.ones(hi - lo, dtype=bool)
    inner_lo = max(0, int(start) - lo)
    inner_hi = max(inner_lo, min(hi - lo, int(end) - lo))
    mask[inner_lo:inner_hi] = False
    return np.arange(lo, hi, dtype=int)[mask]


def _mad(values: np.ndarray) -> float:
    center = np.median(values, axis=0, keepdims=True)
    return float(np.median(np.abs(values - center)))


def _mean_abs_offdiag_corr(values: np.ndarray) -> float:
    if values.shape[0] < 3 or values.shape[1] < 2:
        return math.nan
    corr = safe_corrcoef(values)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


def _mean_signed_offdiag_corr(values: np.ndarray) -> float:
    if values.shape[0] < 3 or values.shape[1] < 2:
        return math.nan
    corr = safe_corrcoef(values)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(corr[mask]))


def _mean_abs_offdiag_fisher_shift(
    clean_values: np.ndarray,
    anomalous_values: np.ndarray,
) -> float:
    if clean_values.shape[0] < 3 or clean_values.shape[1] < 2:
        return math.nan
    if anomalous_values.shape[0] < 3 or anomalous_values.shape[1] < 2:
        return math.nan
    clean_corr = safe_corrcoef(clean_values)
    anomalous_corr = safe_corrcoef(anomalous_values)
    mask = ~np.eye(clean_corr.shape[0], dtype=bool)
    clean_z = _fisher_array(clean_corr[mask])
    anomalous_z = _fisher_array(anomalous_corr[mask])
    return float(np.mean(np.abs(anomalous_z - clean_z)))


def _fisher_array(values: np.ndarray) -> np.ndarray:
    return np.arctanh(
        np.clip(np.asarray(values, dtype=np.float64), -0.999999, 0.999999)
    )


def _target_vs_context_corr(
    values: np.ndarray,
    context_channels: tuple[int, ...],
    group: EventGroup,
) -> float:
    if values.shape[0] < 3 or values.shape[1] < 2:
        return math.nan
    target_channels = tuple(
        dict.fromkeys(group.primary_channels or group.intervention_channels)
    )
    target_positions = [
        index
        for index, channel in enumerate(context_channels)
        if channel in target_channels
    ]
    if not target_positions:
        target_positions = [values.shape[1] - 1]
    context_positions = [
        index for index in range(values.shape[1]) if index not in set(target_positions)
    ]
    if not context_positions:
        return math.nan
    target = np.mean(values[:, target_positions], axis=1)
    ensemble = np.mean(values[:, context_positions], axis=1)
    if safe_variance(target) <= 1e-12 or safe_variance(ensemble) <= 1e-12:
        return math.nan
    corr = float(np.corrcoef(target, ensemble)[0, 1])
    return corr if math.isfinite(corr) else math.nan


def _pca_residual_ratio(anom_event: np.ndarray, clean_event: np.ndarray) -> float:
    return _rank1_residual(anom_event) / max(_rank1_residual(clean_event), 1e-8)


def _rank1_residual(values: np.ndarray) -> float:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return math.nan
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size <= 1:
        return 0.0
    return float(np.sqrt(np.sum(np.square(singular[1:])) / max(values.size, 1)))


def _regression_residual_ratio(
    anom_event: np.ndarray, clean_event: np.ndarray
) -> float:
    return _target_regression_residual(anom_event) / max(
        _target_regression_residual(clean_event),
        1e-8,
    )


def _target_regression_residual(values: np.ndarray) -> float:
    if values.shape[0] < 3 or values.shape[1] < 2:
        return math.nan
    x = values[:, :-1]
    y = values[:, -1]
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(np.sqrt(np.mean(np.square(residual))))


def _shape_residual_ratio(clean_seg: np.ndarray, anom_seg: np.ndarray) -> float:
    if clean_seg.size == 0 or anom_seg.size == 0:
        return math.nan
    raw = float(np.sqrt(np.mean(np.square(anom_seg - clean_seg))))
    if raw <= 1e-12:
        return 0.0
    centered_clean = clean_seg - np.mean(clean_seg, axis=0, keepdims=True)
    centered_anom = anom_seg - np.mean(anom_seg, axis=0, keepdims=True)
    residual = float(np.sqrt(np.mean(np.square(centered_anom - centered_clean))))
    return residual / max(raw, 1e-12)


def _spectral_peak(values: np.ndarray) -> float:
    if values.shape[0] < 4:
        return math.nan
    centered = values - np.mean(values, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0))
    if spectrum.shape[0] <= 1:
        return math.nan
    peak = int(np.argmax(np.mean(spectrum[1:], axis=1)) + 1)
    return float(peak)


def _band_energy(values: np.ndarray) -> float:
    if values.shape[0] < 4:
        return math.nan
    centered = values - np.mean(values, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0))
    return float(np.mean(np.square(spectrum[1:])))
