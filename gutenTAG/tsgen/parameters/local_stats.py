"""Local clean-window statistics for parameter policies."""

from __future__ import annotations

from typing import Any, MutableMapping, Protocol, Sequence

import numpy as np

from ..signal_energy import window_residual_scale
from ..signal_ops import robust_scale


class SegmentWithWindow(Protocol):
    """Minimal segment contract required for local window statistics."""

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...

    @property
    def channel(self) -> int: ...

    @property
    def attrs(self) -> MutableMapping[str, Any]: ...


def annotate_segment_local_stats(
    segments: Sequence[SegmentWithWindow],
    clean_values: np.ndarray | None,
) -> None:
    """Attach deterministic clean-window statistics to segment attributes.

    Parameters
    ----------
    segments : Sequence[SegmentWithWindow]
        Planned segments to mutate in place.
    clean_values : numpy.ndarray or None
        Clean multichannel base values with shape ``(length, channels)``.
    """
    if clean_values is None or len(segments) == 0 or clean_values.ndim != 2:
        return
    n_rows, n_channels = clean_values.shape
    for segment in segments:
        channel = int(segment.channel)
        start = max(0, min(int(segment.start), n_rows))
        end = max(start, min(int(segment.end), n_rows))
        if channel < 0 or channel >= n_channels or end <= start:
            continue
        window = np.asarray(clean_values[start:end, channel], dtype=np.float64)
        if window.size == 0:
            continue
        median = float(np.median(window))
        mad = float(1.4826 * np.median(np.abs(window - median)))
        std = float(np.std(window))
        ptp = float(np.ptp(window))
        segment.attrs.update(
            {
                "window_rms": float(np.sqrt(np.mean(np.square(window)))),
                "window_peak": float(np.max(np.abs(window))),
                "window_std": std,
                "window_ptp": ptp,
                "window_median": median,
                "window_mad": mad,
                "window_robust_scale": robust_scale(window),
                "window_residual_scale": window_residual_scale(
                    window,
                    center_mode="linear",
                ),
            }
        )
