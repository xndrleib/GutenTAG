"""Effective support helpers for pointwise anomaly labels."""

from __future__ import annotations

import numpy as np


def resolve_label_bounds_from_effect(
    *,
    protocol_start: int,
    protocol_end: int,
    delta: np.ndarray,
    anomaly_type: str,
    support_label_mode: str,
    support_eps_mode: str,
    support_eps_value: float,
    min_effective_label_length_non_extremum: int,
) -> tuple[int, int]:
    """Resolve label bounds from an observed anomaly effect signal.

    Parameters
    ----------
    protocol_start : int
        Start index reported by the anomaly protocol.
    protocol_end : int
        End index reported by the anomaly protocol.
    delta : numpy.ndarray
        Absolute effect signal over the protocol window.
    anomaly_type : str
        Canonical anomaly type.
    support_label_mode : str
        Label-support mode. ``strict_segment`` returns the full protocol span.
    support_eps_mode : str
        Epsilon interpretation, either ``relative`` or absolute.
    support_eps_value : float
        Threshold value for active effect support.
    min_effective_label_length_non_extremum : int
        Minimum support length for non-extremum anomalies.

    Returns
    -------
    tuple[int, int]
        Half-open label support bounds.
    """
    if support_label_mode == "strict_segment":
        return int(protocol_start), int(protocol_end)
    if delta.size == 0:
        return int(protocol_start), int(protocol_end)
    if support_eps_mode == "relative":
        peak_delta = float(np.max(delta))
        epsilon = float(support_eps_value) * peak_delta
    else:
        epsilon = float(support_eps_value)
    active = np.flatnonzero(delta > epsilon)
    if active.size == 0:
        return int(protocol_start), int(protocol_end)
    start = int(protocol_start + active[0])
    end = int(protocol_start + active[-1] + 1)
    if anomaly_type != "extremum":
        min_label_length = int(max(1, min_effective_label_length_non_extremum))
        if (end - start) < min_label_length:
            return expand_effective_support_to_min_length(
                protocol_start=protocol_start,
                protocol_end=protocol_end,
                delta=delta,
                min_label_length=min_label_length,
            )
    return start, end


def expand_effective_support_to_min_length(
    *,
    protocol_start: int,
    protocol_end: int,
    delta: np.ndarray,
    min_label_length: int,
) -> tuple[int, int]:
    """Expand an effect support window to a minimum length.

    Parameters
    ----------
    protocol_start : int
        Start index reported by the anomaly protocol.
    protocol_end : int
        End index reported by the anomaly protocol.
    delta : numpy.ndarray
        Absolute effect signal over the protocol window.
    min_label_length : int
        Minimum half-open support length.

    Returns
    -------
    tuple[int, int]
        Expanded half-open label support bounds.
    """
    source_length = max(0, int(protocol_end - protocol_start))
    if source_length <= 0:
        return int(protocol_start), int(protocol_end)
    if source_length <= 1 or min_label_length <= 1:
        end = min(int(protocol_start) + 1, int(protocol_end))
        return int(protocol_start), int(end)

    window = min(int(min_label_length), source_length)
    if window <= 1:
        end = min(int(protocol_start) + 1, int(protocol_end))
        return int(protocol_start), int(end)

    if delta.shape[0] != source_length:
        resized = np.interp(
            np.linspace(0.0, 1.0, source_length, endpoint=True),
            np.linspace(0.0, 1.0, max(1, delta.shape[0]), endpoint=True),
            delta if delta.shape[0] > 0 else np.zeros(1, dtype=np.float64),
        ).astype(np.float64)
    else:
        resized = delta

    scores = np.convolve(resized, np.ones(window, dtype=np.float64), mode="valid")
    best_start_local = int(np.argmax(scores)) if scores.size > 0 else 0
    start = int(protocol_start + best_start_local)
    end = int(start + window)
    return start, end


def normalize_subsequence_length(
    subsequence: np.ndarray,
    expected_length: int,
    policy: str,
) -> np.ndarray:
    """Normalize an anomaly subsequence to the expected protocol length.

    Parameters
    ----------
    subsequence : numpy.ndarray
        Raw anomaly subsequence values.
    expected_length : int
        Required output length.
    policy : str
        One of ``none``, ``crop``, ``pad``, or ``resample``.

    Returns
    -------
    numpy.ndarray
        Normalized subsequence as ``float64`` where new arrays are required.

    Raises
    ------
    ValueError
        If the selected policy cannot produce the requested length.
    """
    if expected_length <= 0:
        return np.array([], dtype=np.float64)
    if subsequence.shape[0] == expected_length:
        return subsequence
    if policy == "none":
        raise ValueError(
            f"Subsequence length mismatch ({subsequence.shape[0]} != {expected_length}) "
            "and length_normalization='none'."
        )
    if policy == "crop":
        if subsequence.shape[0] < expected_length:
            raise ValueError(
                f"Cannot crop subsequence of length {subsequence.shape[0]} "
                f"to larger expected length {expected_length}."
            )
        return subsequence[:expected_length]
    if policy == "pad":
        if subsequence.shape[0] > expected_length:
            return subsequence[:expected_length]
        if subsequence.shape[0] == 0:
            return np.zeros(expected_length, dtype=np.float64)
        return np.pad(
            subsequence,
            (0, expected_length - subsequence.shape[0]),
            mode="edge",
        )
    if subsequence.shape[0] == 0:
        return np.zeros(expected_length, dtype=np.float64)
    if expected_length == 1:
        return np.array([float(subsequence[0])], dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, subsequence.shape[0], endpoint=True)
    x_new = np.linspace(0.0, 1.0, expected_length, endpoint=True)
    return np.interp(x_new, x_old, subsequence).astype(np.float64)
