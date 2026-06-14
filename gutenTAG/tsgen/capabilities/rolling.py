"""Vectorized rolling/window statistics for capability profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RollingStats:
    """Precompute cumulative sums for fast interval statistics."""

    series: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.series, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("RollingStats expects a 2D array [time, channels]")
        self.series = values
        zero = np.zeros((1, values.shape[1]), dtype=np.float64)
        self.csum = np.vstack([zero, np.cumsum(values, axis=0)])
        self.csum2 = np.vstack([zero, np.cumsum(values * values, axis=0)])
        cross = values[:, :, None] * values[:, None, :]
        zero_cross = np.zeros((1, values.shape[1], values.shape[1]), dtype=np.float64)
        self.cross_csum = np.vstack([zero_cross, np.cumsum(cross, axis=0)])

    def mean(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
        channels: int | Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Return per-channel interval means."""

        starts_arr, ends_arr, lengths = self._intervals(starts, ends)
        channel_arr = self._channels(channels)
        sums = self.csum[ends_arr][:, channel_arr] - self.csum[starts_arr][:, channel_arr]
        result = sums / lengths[:, None]
        return _squeeze_channel_result(result, channels)

    def variance(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
        channels: int | Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Return per-channel population variances for intervals."""

        starts_arr, ends_arr, lengths = self._intervals(starts, ends)
        channel_arr = self._channels(channels)
        sums = self.csum[ends_arr][:, channel_arr] - self.csum[starts_arr][:, channel_arr]
        sums2 = self.csum2[ends_arr][:, channel_arr] - self.csum2[starts_arr][:, channel_arr]
        means = sums / lengths[:, None]
        variances = np.maximum((sums2 / lengths[:, None]) - means * means, 0.0)
        return _squeeze_channel_result(variances, channels)

    def covariance(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
        channel_i: int,
        channel_j: int,
    ) -> np.ndarray:
        """Return population covariance for channel pairs over intervals."""

        starts_arr, ends_arr, lengths = self._intervals(starts, ends)
        i = self._channel(channel_i)
        j = self._channel(channel_j)
        cross = self.cross_csum[ends_arr, i, j] - self.cross_csum[starts_arr, i, j]
        sum_i = self.csum[ends_arr, i] - self.csum[starts_arr, i]
        sum_j = self.csum[ends_arr, j] - self.csum[starts_arr, j]
        mean_i = sum_i / lengths
        mean_j = sum_j / lengths
        return (cross / lengths) - mean_i * mean_j

    def correlation(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
        channel_i: int,
        channel_j: int,
    ) -> np.ndarray:
        """Return Pearson correlation for channel pairs over intervals."""

        cov = self.covariance(starts, ends, channel_i, channel_j)
        var_i = self.variance(starts, ends, channel_i)
        var_j = self.variance(starts, ends, channel_j)
        denom = np.sqrt(np.maximum(var_i, 0.0) * np.maximum(var_j, 0.0))
        return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)

    def energy(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
        channels: int | Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Return sum of squared values over intervals and selected channels."""

        starts_arr, ends_arr, _ = self._intervals(starts, ends)
        channel_arr = self._channels(channels)
        sums2 = self.csum2[ends_arr][:, channel_arr] - self.csum2[starts_arr][:, channel_arr]
        return np.sum(sums2, axis=1)

    def _intervals(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts_arr = np.asarray(starts, dtype=np.int64)
        ends_arr = np.asarray(ends, dtype=np.int64)
        if starts_arr.ndim == 0:
            starts_arr = starts_arr.reshape(1)
        if ends_arr.ndim == 0:
            ends_arr = ends_arr.reshape(1)
        if starts_arr.shape != ends_arr.shape:
            raise ValueError("starts and ends must have the same shape")
        if np.any(starts_arr < 0) or np.any(ends_arr > len(self.series)) or np.any(ends_arr <= starts_arr):
            raise ValueError("Invalid interval bounds")
        return starts_arr, ends_arr, (ends_arr - starts_arr).astype(np.float64)

    def _channels(self, channels: int | Sequence[int] | np.ndarray) -> np.ndarray:
        if np.isscalar(channels):
            return np.asarray([self._channel(int(channels))], dtype=np.int64)
        channel_arr = np.asarray(channels, dtype=np.int64)
        if channel_arr.ndim != 1:
            raise ValueError("channels must be a scalar or a 1D sequence")
        for channel in channel_arr:
            self._channel(int(channel))
        return channel_arr

    def _channel(self, channel: int) -> int:
        if channel < 0 or channel >= self.series.shape[1]:
            raise ValueError(f"Channel out of bounds: {channel}")
        return int(channel)


def _squeeze_channel_result(
    result: np.ndarray,
    channels: int | Sequence[int] | np.ndarray,
) -> np.ndarray:
    if np.isscalar(channels):
        return result[:, 0]
    return result
