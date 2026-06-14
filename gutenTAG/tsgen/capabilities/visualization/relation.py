"""Relation-focused visual audit helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def is_relation_event(event: Mapping[str, object]) -> bool:
    """Return whether an event should receive relation-focused review."""

    scope = str(event.get("semantic_scope", ""))
    tag = str(event.get("constraint_tag", ""))
    return "relation" in scope or tag.startswith("dependence")


def plot_relation_scatter(
    *,
    output_path: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    event: Mapping[str, object],
) -> bool:
    """Write clean/anomalous relation scatter panels for the support window."""

    channels = _relation_channels(event, clean.shape[1])
    if len(channels) < 2:
        return False
    start, end = _event_bounds(event, clean.shape[0])
    if end <= start:
        return False
    x_channel, y_channel = channels[:2]
    clean_support = clean[start:end][:, [x_channel, y_channel]]
    anomalous_support = anomalous[start:end][:, [x_channel, y_channel]]
    left, right = _zoom_bounds(clean.shape[0], start, end)
    context_mask = np.ones(right - left, dtype=bool)
    context_mask[max(0, start - left):max(0, end - left)] = False
    clean_context = clean[left:right][context_mask][:, [x_channel, y_channel]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    if clean_context.size:
        axis.scatter(
            clean_context[:, 0],
            clean_context[:, 1],
            s=10,
            alpha=0.22,
            color="#9aa0a6",
            label="clean context",
        )
    axis.scatter(
        clean_support[:, 0],
        clean_support[:, 1],
        s=18,
        alpha=0.75,
        color="#2ca02c",
        label="clean support",
    )
    axis.scatter(
        anomalous_support[:, 0],
        anomalous_support[:, 1],
        s=18,
        alpha=0.75,
        color="#d62728",
        marker="x",
        label="anomalous support",
    )
    axis.set_xlabel(f"ch{x_channel}")
    axis.set_ylabel(f"ch{y_channel}")
    axis.set_title("relation scatter")
    axis.legend(loc="best", fontsize=8)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return True


def plot_rolling_correlation_panel(
    *,
    output_path: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    event: Mapping[str, object],
) -> bool:
    """Write rolling-correlation clean/anomalous comparison for a channel pair."""

    channels = _relation_channels(event, clean.shape[1])
    if len(channels) < 2:
        return False
    start, end = _event_bounds(event, clean.shape[0])
    if end <= start:
        return False
    left, right = _zoom_bounds(clean.shape[0], start, end)
    window = _rolling_window_size(start, end, right - left)
    x_channel, y_channel = channels[:2]
    clean_corr = _rolling_corr(clean[left:right, x_channel], clean[left:right, y_channel], window)
    anom_corr = _rolling_corr(anomalous[left:right, x_channel], anomalous[left:right, y_channel], window)
    time = np.arange(left, right)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.plot(time, clean_corr, color="#2ca02c", linewidth=1.4, label="clean")
    axis.plot(time, anom_corr, color="#d62728", linewidth=1.4, label="anomalous")
    axis.axvspan(start, end, color="#f4c542", alpha=0.18)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axis.set_ylim(-1.05, 1.05)
    axis.set_title(f"rolling correlation ch{x_channel}-ch{y_channel} (window={window})")
    axis.set_xlabel("time")
    axis.set_ylabel("corr")
    axis.legend(loc="best", fontsize=8)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return True


def plot_pca_residual_panel(
    *,
    output_path: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    event: Mapping[str, object],
) -> bool:
    """Write PCA reconstruction residual panel for collective/relation events."""

    channels = _relation_channels(event, clean.shape[1])
    if len(channels) < 2:
        return False
    start, end = _event_bounds(event, clean.shape[0])
    if end <= start:
        return False
    left, right = _zoom_bounds(clean.shape[0], start, end)
    projected_clean = clean[:, channels]
    projected_zoom_clean = clean[left:right][:, channels]
    projected_zoom_anom = anomalous[left:right][:, channels]
    context_mask = np.ones(clean.shape[0], dtype=bool)
    context_mask[start:end] = False
    fit_data = projected_clean[context_mask]
    if fit_data.shape[0] < len(channels) + 2:
        fit_data = projected_clean
    residual_clean = _pca_residual(projected_zoom_clean, fit_data)
    residual_anom = _pca_residual(projected_zoom_anom, fit_data)
    if residual_clean.size == 0 or residual_anom.size == 0:
        return False
    time = np.arange(left, right)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.plot(time, residual_clean, color="#2ca02c", linewidth=1.2, label="clean")
    axis.plot(time, residual_anom, color="#d62728", linewidth=1.2, label="anomalous")
    axis.axvspan(start, end, color="#f4c542", alpha=0.18)
    axis.set_title(f"PCA residual on channels {_format_channels(channels)}")
    axis.set_xlabel("time")
    axis.set_ylabel("reconstruction RMSE")
    axis.legend(loc="best", fontsize=8)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return True


def _pca_residual(values: np.ndarray, fit_data: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    reference = np.asarray(fit_data, dtype=np.float64)
    if matrix.ndim != 2 or reference.ndim != 2 or matrix.size == 0 or reference.size == 0:
        return np.asarray([], dtype=np.float64)
    center = np.mean(reference, axis=0, keepdims=True)
    centered = reference - center
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.asarray([], dtype=np.float64)
    rank = max(1, min(reference.shape[1] - 1, vt.shape[0]))
    basis = vt[:rank]
    active = matrix - center
    reconstructed = active @ basis.T @ basis + center
    return np.sqrt(np.mean(np.square(matrix - reconstructed), axis=1))


def _rolling_corr(x_values: np.ndarray, y_values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    half = max(1, int(window) // 2)
    for index in range(x.size):
        left = max(0, index - half)
        right = min(x.size, index + half + 1)
        x_window = x[left:right]
        y_window = y[left:right]
        if x_window.size < 3 or np.std(x_window) <= 1e-12 or np.std(y_window) <= 1e-12:
            result[index] = 0.0
            continue
        value = float(np.corrcoef(x_window, y_window)[0, 1])
        result[index] = value if np.isfinite(value) else 0.0
    return result


def _rolling_window_size(start: int, end: int, zoom_width: int) -> int:
    support_width = max(1, int(end) - int(start))
    return max(8, min(96, support_width // 2, max(8, int(zoom_width) // 8)))


def _event_bounds(event: Mapping[str, object], length: int) -> tuple[int, int]:
    start = _clip_int(event.get("start", event.get("support_start", 0)), 0, length)
    end = _clip_int(event.get("end", event.get("support_end", start + 1)), 0, length)
    return start, max(start, end)


def _zoom_bounds(length: int, start: int, end: int) -> tuple[int, int]:
    width = max(1, int(end) - int(start))
    margin = max(24, min(240, 4 * width))
    left = max(0, int(start) - margin)
    right = min(int(length), int(end) + margin)
    if right <= left:
        right = min(int(length), left + 1)
    return left, right


def _relation_channels(event: Mapping[str, object], channel_count: int) -> tuple[int, ...]:
    raw_channels = (
        _parse_channels(event.get("group_channels"))
        or _parse_channels(event.get("context_channels"))
        or _parse_channels(event.get("intervention_channels"))
    )
    channels = tuple(channel for channel in raw_channels if 0 <= channel < channel_count)
    if len(channels) >= 2:
        return channels
    return tuple(range(min(channel_count, 2)))


def _parse_channels(value: object) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace(",", "|").split("|")
    channels: list[int] = []
    for item in raw:
        try:
            channels.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(dict.fromkeys(channels)))


def _clip_int(value: object, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(low)
    return max(int(low), min(int(high), number))


def _format_channels(channels: tuple[int, ...]) -> str:
    return "|".join(str(channel) for channel in channels)
