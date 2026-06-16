"""Generator-side instance plot rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


@dataclass(frozen=True)
class InstancePlotConfig:
    """Configuration for generator-side instance plots."""

    length: int
    channels: int
    plot_dpi: int
    zoom_count: int
    zoom_margin: int
    zoom_margin_min: int
    zoom_margin_alpha: float
    zoom_fill_policy: str


@dataclass(frozen=True)
class _ChannelEventFlags:
    """Legend flags observed while annotating one plotted channel."""

    has_point_anomaly: bool
    has_selected_point: bool


def write_instance_plots(
    *,
    instance_dir: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    zoom_seed: int,
    config: InstancePlotConfig,
) -> None:
    """Write full-instance and zoom plots for one generated instance."""

    _plot_window(
        output_path=instance_dir / "plot_full.png",
        clean=clean,
        anomalous=anomalous,
        events=events,
        window_start=0,
        window_end=config.length,
        title="Full instance view",
        selected_event=None,
        config=config,
    )

    selected_events = _select_zoom_events(events, zoom_seed, config)
    for idx in range(config.zoom_count):
        output_path = instance_dir / f"zoom_{idx:02d}.png"
        event = selected_events[idx] if idx < len(selected_events) else None
        if event is None and config.zoom_fill_policy == "blank":
            _write_blank_zoom(output_path, idx, config)
            continue
        if event is None and len(events) > 0:
            event = events[idx % len(events)]
        if event is None:
            _write_blank_zoom(output_path, idx, config)
            continue
        start = int(event["start"])
        end = int(event["end"])
        margin = max(
            config.zoom_margin_min,
            config.zoom_margin,
            int(round(config.zoom_margin_alpha * (end - start))),
        )
        window_start = max(0, start - margin)
        window_end = min(config.length, end + margin)
        title = (
            f"Zoom {idx:02d} | type={event['anomaly_type']} | "
            f"channel={event['channel']} | ({start},{end})"
        )
        _plot_window(
            output_path=output_path,
            clean=clean,
            anomalous=anomalous,
            events=events,
            window_start=window_start,
            window_end=window_end,
            title=title,
            selected_event=event,
            config=config,
        )


def _select_zoom_events(
    events: Sequence[Mapping[str, Any]],
    zoom_seed: int,
    config: InstancePlotConfig,
) -> list[Mapping[str, Any]]:
    if len(events) == 0:
        return []
    rng = np.random.default_rng(zoom_seed)
    if len(events) <= config.zoom_count:
        selected = [events[i] for i in range(len(events))]
    else:
        indices = rng.choice(len(events), size=config.zoom_count, replace=False)
        selected = [events[int(i)] for i in indices.tolist()]
        selected = sorted(
            selected,
            key=lambda event: (int(event["start"]), int(event["channel"])),
        )
    if len(selected) < config.zoom_count and config.zoom_fill_policy == "repeat":
        repeated: list[Mapping[str, Any]] = []
        cursor = 0
        while len(selected) + len(repeated) < config.zoom_count:
            repeated.append(selected[cursor % len(selected)])
            cursor += 1
        selected = selected + repeated
    return selected[: config.zoom_count]


def _plot_window(
    *,
    output_path: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    window_start: int,
    window_end: int,
    title: str,
    selected_event: Mapping[str, Any] | None,
    config: InstancePlotConfig,
) -> None:
    fig, axes = _new_plot_axes(config)
    x_values = np.arange(window_start, window_end, dtype=int)
    selected_signature = _event_signature(selected_event)
    for channel in range(int(config.channels)):
        axis = axes[channel]
        _plot_channel_lines(
            axis, x_values, clean, anomalous, window_start, window_end, channel
        )
        flags = _annotate_channel_events(
            axis=axis,
            events=events,
            channel=channel,
            window_start=window_start,
            window_end=window_end,
            selected_signature=selected_signature,
        )
        axis.legend(
            handles=_legend_handles(
                flags.has_point_anomaly,
                flags.has_selected_point,
            )
        )
    _save_plot(fig, axes, title, output_path, config)


def _new_plot_axes(config: InstancePlotConfig) -> tuple[Any, list[Any]]:
    channels = int(config.channels)
    figure_height = max(2 * channels, 8)
    fig, axes = plt.subplots(channels, 1, figsize=(16, figure_height), sharex=True)
    if channels == 1:
        axes = [axes]
    return fig, list(axes)


def _event_signature(event: Mapping[str, Any] | None) -> tuple[int, int, int] | None:
    if event is None:
        return None
    return (
        int(event["start"]),
        int(event["end"]),
        int(event["channel"]),
    )


def _plot_channel_lines(
    axis: Any,
    x_values: np.ndarray,
    clean: np.ndarray,
    anomalous: np.ndarray,
    window_start: int,
    window_end: int,
    channel: int,
) -> None:
    axis.plot(
        x_values,
        clean[window_start:window_end, channel],
        linewidth=1.0,
        color="#7f7f7f",
        linestyle="--",
        alpha=0.95,
        label="clean",
        zorder=2,
    )
    axis.plot(
        x_values,
        anomalous[window_start:window_end, channel],
        linewidth=0.9,
        color="#1f77b4",
        alpha=0.95,
        label="anomalous",
        zorder=3,
    )
    axis.set_ylabel(f"ch {channel}")


def _annotate_channel_events(
    *,
    axis: Any,
    events: Sequence[Mapping[str, Any]],
    channel: int,
    window_start: int,
    window_end: int,
    selected_signature: tuple[int, int, int] | None,
) -> _ChannelEventFlags:
    has_point_anomaly = False
    has_selected_point = False
    for event in events:
        event_bounds = _visible_event_bounds(event, channel, window_start, window_end)
        if event_bounds is None:
            continue
        start, end, span_start, span_end = event_bounds
        signature = (start, end, channel)
        is_selected = selected_signature is not None and signature == selected_signature
        is_point_event = end - start <= 1
        if is_selected and is_point_event:
            has_selected_point = True
        elif is_point_event:
            has_point_anomaly = True
        _draw_event_marker(
            axis,
            start=start,
            span_start=span_start,
            span_end=span_end,
            selected=is_selected,
            point_event=is_point_event,
        )
    return _ChannelEventFlags(has_point_anomaly, has_selected_point)


def _visible_event_bounds(
    event: Mapping[str, Any],
    channel: int,
    window_start: int,
    window_end: int,
) -> tuple[int, int, int, int] | None:
    event_channel = int(event["channel"])
    if event_channel != channel:
        return None
    start = int(event["start"])
    end = int(event["end"])
    if end <= window_start or start >= window_end:
        return None
    return start, end, max(start, window_start), min(end, window_end)


def _draw_event_marker(
    axis: Any,
    *,
    start: int,
    span_start: int,
    span_end: int,
    selected: bool,
    point_event: bool,
) -> None:
    color = "#d62728" if selected else "#ff7f0e"
    if point_event:
        axis.axvline(
            x=start,
            color=color,
            alpha=0.85 if selected else 0.70,
            linestyle="--",
            linewidth=1.5 if selected else 1.2,
        )
        return
    axis.axvspan(
        span_start,
        span_end,
        color=color,
        alpha=0.30 if selected else 0.20,
        label="selected anomaly" if selected else "anomaly",
    )


def _save_plot(
    fig: Any,
    axes: Sequence[Any],
    title: str,
    output_path: Path,
    config: InstancePlotConfig,
) -> None:
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=config.plot_dpi,
        metadata={
            "Software": "GutenTAG ts-dataset-generator",
            "Date": "1970-01-01",
        },
    )
    plt.close(fig)


def _legend_handles(has_point_anomaly: bool, has_selected_point: bool) -> list[Any]:
    legend_handles: list[Any] = [
        Line2D(
            [0, 1],
            [0, 0],
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.95,
            label="clean",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#1f77b4",
            linewidth=0.9,
            alpha=0.95,
            label="anomalous",
        ),
        Patch(color="#ff7f0e", alpha=0.20, label="anomaly"),
        Patch(color="#d62728", alpha=0.30, label="selected anomaly"),
    ]
    if has_point_anomaly:
        legend_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#ff7f0e",
                alpha=0.70,
                linestyle="--",
                linewidth=1.2,
                label="point anomaly",
            )
        )
    if has_selected_point:
        legend_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#d62728",
                alpha=0.85,
                linestyle="--",
                linewidth=1.5,
                label="selected point anomaly",
            )
        )
    return legend_handles


def _write_blank_zoom(
    output_path: Path,
    zoom_index: int,
    config: InstancePlotConfig,
) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(8, 3))
    axis.text(0.5, 0.5, f"zoom_{zoom_index:02d}: no event", ha="center", va="center")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=config.plot_dpi,
        metadata={
            "Software": "GutenTAG ts-dataset-generator",
            "Date": "1970-01-01",
        },
    )
    plt.close(fig)
