"""Base-channel variation and shared-noise helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .multivariate_ops import compose_observed_window, replace_observed_window


def shared_noise_weight(base_channel_correlation: Mapping[str, Any]) -> float:
    """Return clipped shared-noise weight from channel-correlation config.

    Parameters
    ----------
    base_channel_correlation : Mapping[str, Any]
        Channel-correlation configuration.

    Returns
    -------
    float
        Shared-noise weight clipped to ``[0, 1]``.
    """
    weight = float(dict(base_channel_correlation).get("shared_noise_weight", 0.0))
    return float(np.clip(weight, 0.0, 1.0))


def apply_shared_noise_correlation(
    *,
    channel_bos: Sequence[Any],
    seed: int,
    base_channel_correlation: Mapping[str, Any],
) -> None:
    """Mix channel noises with a deterministic shared-noise component.

    Parameters
    ----------
    channel_bos : Sequence[Any]
        Base-oscillation objects with optional ``noise`` arrays.
    seed : int
        Seed used to sample the shared component.
    base_channel_correlation : Mapping[str, Any]
        Channel-correlation configuration.

    Raises
    ------
    ValueError
        If non-empty channel noises have incompatible lengths.
    """
    shared_weight = shared_noise_weight(base_channel_correlation)
    if shared_weight <= 0.0 or len(channel_bos) <= 1:
        return
    noise_lengths = {
        int(np.asarray(bo.noise).shape[0]) for bo in channel_bos if bo.noise is not None
    }
    if len(noise_lengths) == 0:
        return
    if len(noise_lengths) > 1:
        raise ValueError(
            "Cannot apply shared-noise correlation to channel noises with "
            "different lengths."
        )
    noise_length = next(iter(noise_lengths))
    rng = np.random.default_rng(seed)
    shared_noise = rng.normal(0.0, 1.0, noise_length).astype(np.float64)
    shared_std = float(np.std(shared_noise))
    if shared_std > 0.0:
        shared_noise = (shared_noise - float(np.mean(shared_noise))) / shared_std
    else:
        shared_noise = np.zeros(noise_length, dtype=np.float64)
    residual_weight = float(np.sqrt(max(0.0, 1.0 - shared_weight**2)))
    for bo in channel_bos:
        if bo.noise is None:
            continue
        noise = np.asarray(bo.noise, dtype=np.float64)
        mean = float(np.mean(noise))
        centered = noise - mean
        channel_std = float(np.std(centered))
        if channel_std <= 1e-12:
            bo.noise = np.array(noise, copy=True)
            continue
        shared_component = shared_noise * channel_std
        mixed = residual_weight * centered + shared_weight * shared_component
        bo._noise_mean = float(mean)
        bo._idio_noise_component = centered.astype(np.float64)
        bo._shared_noise_component = shared_component.astype(np.float64)
        bo._shared_noise_weight = float(shared_weight)
        bo._residual_noise_weight = float(residual_weight)
        bo.noise = (mixed + mean).astype(np.float64)


def apply_variations(base: np.ndarray, channel_bos: Sequence[Any]) -> np.ndarray:
    """Compose observed values by adding fixed channel variations.

    Parameters
    ----------
    base : numpy.ndarray
        Base multichannel time-series.
    channel_bos : Sequence[Any]
        Base-oscillation objects with optional noise, trend, and offset fields.

    Returns
    -------
    numpy.ndarray
        Observed multichannel time-series.
    """
    result = np.array(base, dtype=np.float64, copy=True)
    for channel, bo in enumerate(channel_bos):
        if bo.noise is not None:
            result[:, channel] = result[:, channel] + bo.noise
        if bo.trend_series is not None:
            result[:, channel] = result[:, channel] + bo.trend_series
        if bo.offset is not None:
            result[:, channel] = result[:, channel] + bo.offset
    return result


def compose_channel_window_with_variations(
    *,
    base: np.ndarray,
    bo: Any,
    channel: int,
    start: int,
    end: int,
    series_length: int,
) -> np.ndarray:
    """Compose a clipped single-channel observed window.

    Parameters
    ----------
    base : numpy.ndarray
        Base multichannel time-series.
    bo : Any
        Base-oscillation object for ``channel``.
    channel : int
        Channel index.
    start : int
        Requested start index.
    end : int
        Requested end index.
    series_length : int
        Total series length used for clipping.

    Returns
    -------
    numpy.ndarray
        Composed observed window.
    """
    start_i, end_i = _clip_window(start, end, series_length)
    return compose_observed_window(
        base=base, bo=bo, channel=channel, start=start_i, end=end_i
    )


def replace_channel_window_with_variations(
    *,
    base: np.ndarray,
    bo: Any,
    channel: int,
    start: int,
    end: int,
    target_observed: np.ndarray,
    series_length: int,
) -> None:
    """Rewrite a clipped base window to match observed values.

    Parameters
    ----------
    base : numpy.ndarray
        Mutable base multichannel time-series.
    bo : Any
        Base-oscillation object for ``channel``.
    channel : int
        Channel index.
    start : int
        Requested start index.
    end : int
        Requested end index.
    target_observed : numpy.ndarray
        Desired observed values in the clipped window.
    series_length : int
        Total series length used for clipping.
    """
    start_i, end_i = _clip_window(start, end, series_length)
    replace_observed_window(
        base=base,
        bo=bo,
        channel=channel,
        start=start_i,
        end=end_i,
        target_observed=target_observed,
    )


def compose_channel_noise_window(
    *,
    bo: Any,
    start: int,
    end: int,
    series_length: int,
) -> np.ndarray:
    """Return a clipped channel-noise window.

    Parameters
    ----------
    bo : Any
        Base-oscillation object.
    start : int
        Requested start index.
    end : int
        Requested end index.
    series_length : int
        Total series length used for clipping.

    Returns
    -------
    numpy.ndarray
        Noise window, or zeros when the channel has no noise.
    """
    start_i, end_i = _clip_window(start, end, series_length)
    if getattr(bo, "noise", None) is None:
        return np.zeros(end_i - start_i, dtype=np.float64)
    return np.asarray(bo.noise[start_i:end_i], dtype=np.float64).copy()


def replace_channel_noise_window(
    *,
    bo: Any,
    start: int,
    end: int,
    target_noise: np.ndarray,
    series_length: int,
) -> None:
    """Rewrite a clipped channel-noise window.

    Parameters
    ----------
    bo : Any
        Mutable base-oscillation object.
    start : int
        Requested start index.
    end : int
        Requested end index.
    target_noise : numpy.ndarray
        Replacement noise values.
    series_length : int
        Total series length used for clipping.
    """
    start_i, end_i = _clip_window(start, end, series_length)
    if getattr(bo, "noise", None) is None:
        bo.noise = np.zeros(int(series_length), dtype=np.float64)
    bo.noise[start_i:end_i] = np.asarray(target_noise, dtype=np.float64)


def _clip_window(start: int, end: int, series_length: int) -> tuple[int, int]:
    start_i = max(0, min(int(start), int(series_length)))
    end_i = max(start_i, min(int(end), int(series_length)))
    return start_i, end_i
