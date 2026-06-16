"""Generic event diagnostic plots for visual audit."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..pandas_typing import (
    as_frame,
    as_series,
    numeric_column,
    row_mapping,
    sorted_frame,
)


def plot_event_diagnostic(
    *,
    output_path: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    event: Mapping[str, object],
    attribution: pd.DataFrame,
) -> None:
    """Write a compact visual diagnostic for one selected event."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = _int_event_field(event, "start", event.get("support_start", 0))
    end = _int_event_field(event, "end", event.get("support_end", start + 1))
    channels = _parse_channels(event.get("group_channels")) or _parse_channels(
        event.get("intervention_channels")
    )
    if not channels:
        channels = tuple(range(min(clean.shape[1], 3)))
    channels = tuple(channel for channel in channels if 0 <= channel < clean.shape[1])[
        :3
    ]
    if not channels:
        channels = (0,)
    left, right = _zoom_bounds(clean.shape[0], start, end)
    time = np.arange(left, right)
    clean_zoom = clean[left:right, list(channels)]
    anomalous_zoom = anomalous[left:right, list(channels)]
    residual_zoom = anomalous_zoom - clean_zoom

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    _plot_raw(axes[0], time, clean_zoom, anomalous_zoom, channels, start, end)
    _plot_residual(axes[1], time, residual_zoom, channels, start, end)
    _plot_residual_energy(axes[2], time, residual_zoom, start, end)
    _plot_attribution(axes[3], attribution)
    title = (
        f"{event.get('bucket', 'visual')} | {event.get('variant_id', '')} | "
        f"{event.get('anomaly_type', '')} | {event.get('admission_status', '')}"
    )
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def _plot_raw(
    axis: Axes,
    time: np.ndarray,
    clean: np.ndarray,
    anomalous: np.ndarray,
    channels: Sequence[int],
    start: int,
    end: int,
) -> None:
    for idx, channel in enumerate(channels):
        axis.plot(
            time, clean[:, idx], linewidth=1.0, alpha=0.75, label=f"clean ch{channel}"
        )
        axis.plot(
            time,
            anomalous[:, idx],
            linewidth=1.0,
            alpha=0.75,
            linestyle="--",
            label=f"anom ch{channel}",
        )
    _shade_support(axis, start, end)
    axis.set_title("raw clean vs anomalous / support zoom")
    axis.set_ylabel("value")
    axis.legend(loc="upper right", fontsize=7, ncol=2)


def _plot_residual(
    axis: Axes,
    time: np.ndarray,
    residual: np.ndarray,
    channels: Sequence[int],
    start: int,
    end: int,
) -> None:
    for idx, channel in enumerate(channels):
        axis.plot(time, residual[:, idx], linewidth=1.0, label=f"residual ch{channel}")
    _shade_support(axis, start, end)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("residual anomalous-clean")
    axis.set_ylabel("delta")
    axis.legend(loc="upper right", fontsize=7, ncol=3)


def _plot_residual_energy(
    axis: Axes,
    time: np.ndarray,
    residual: np.ndarray,
    start: int,
    end: int,
) -> None:
    energy = (
        np.sqrt(np.mean(np.square(residual), axis=1))
        if residual.size
        else np.zeros_like(time, dtype=float)
    )
    axis.plot(time, energy, color="#4c78a8", linewidth=1.5)
    axis.axvline(
        start, color="#d62728", linewidth=1.0, linestyle="--", label="support start"
    )
    axis.axvline(
        end, color="#d62728", linewidth=1.0, linestyle=":", label="support end"
    )
    _shade_support(axis, start, end)
    axis.set_title("boundary zoom / residual energy timeline")
    axis.set_ylabel("RMSE")
    axis.legend(loc="upper right", fontsize=7)


def _plot_attribution(axis: Axes, attribution: pd.DataFrame) -> None:
    if attribution.empty:
        axis.text(0.5, 0.5, "no detector attribution rows", ha="center", va="center")
        axis.set_axis_off()
        return
    frame = attribution.copy()
    if "alpha" in frame.columns and not numeric_column(frame, "alpha").dropna().empty:
        alpha_values = numeric_column(frame, "alpha")
        alpha = float(alpha_values.min())
        frame = as_frame(frame[np.isclose(alpha_values, alpha)])
    if "rank_within_event" in frame.columns:
        frame = sorted_frame(frame, "rank_within_event")
    frame = frame.head(8)
    labels = [
        f"{row.get('witness_or_model', row.get('family', 'score'))}\n{row.get('projection', '')}"
        for _, row in frame.iterrows()
    ]
    values = (
        as_series(frame.get("normalized_evidence", pd.Series([0.0] * len(frame))))
        .astype(float)
        .to_numpy()
    )
    colors = [_bar_color(row_mapping(row)) for _, row in frame.iterrows()]
    axis.bar(np.arange(len(frame)), values, color=colors)
    axis.set_xticks(np.arange(len(frame)))
    axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    axis.set_title("detector attribution bar")
    axis.set_ylabel("normalized evidence")


def _bar_color(row: Mapping[str, object]) -> str:
    if bool(row.get("is_boundary", False)):
        return "#d62728"
    if bool(row.get("is_forbidden_shortcut", False)):
        return "#ff7f0e"
    if bool(row.get("is_canonical", False)):
        return "#2ca02c"
    return "#7f7f7f"


def _shade_support(axis: Axes, start: int, end: int) -> None:
    axis.axvspan(start, end, color="#f4c542", alpha=0.18)


def _zoom_bounds(length: int, start: int, end: int) -> tuple[int, int]:
    width = max(1, int(end) - int(start))
    margin = max(16, min(200, 3 * width))
    left = max(0, int(start) - margin)
    right = min(int(length), int(end) + margin)
    if right <= left:
        right = min(int(length), left + 1)
    return left, right


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


def _int_event_field(event: Mapping[str, object], key: str, default: object) -> int:
    value = event.get(key, default)
    if isinstance(value, (int, float, str, np.integer, np.floating)):
        try:
            return int(value)
        except ValueError:
            return int(default) if isinstance(default, (int, float, str)) else 0
    return int(default) if isinstance(default, (int, float, str)) else 0
