"""Dataset output file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..io import write_json
from ..labels import build_label_masks, write_label_masks


def check_labels_and_events_consistency(
    labels: np.ndarray,
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Validate that pointwise labels match event supports.

    Parameters
    ----------
    labels : numpy.ndarray
        Pointwise label matrix with shape ``(length, channels)``.
    events : Sequence[Mapping[str, Any]]
        Event records with ``start``, ``end``, and ``channel`` fields.

    Raises
    ------
    ValueError
        If reconstructed labels from events differ from ``labels``.
    """
    reconstructed = np.zeros_like(labels)
    for event in events:
        reconstructed[
            int(event["start"]) : int(event["end"]), int(event["channel"])
        ] = 1
    if not np.array_equal(labels, reconstructed):
        raise ValueError("labels_pointwise.csv is inconsistent with events.json")


def write_timeseries_csv(
    path: Path,
    values: np.ndarray,
    *,
    channels: int,
    csv_float_format: str,
) -> None:
    """Write multichannel time-series values to CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    values : numpy.ndarray
        Time-series matrix.
    channels : int
        Number of value columns.
    csv_float_format : str
        Float format passed to ``pandas.DataFrame.to_csv``.
    """
    columns = [f"value-{channel}" for channel in range(int(channels))]
    df = pd.DataFrame(values, columns=pd.Index(columns))
    df.to_csv(path, index=False, float_format=csv_float_format)


def write_labels_csv(path: Path, labels: np.ndarray, *, channels: int) -> None:
    """Write pointwise label matrix to CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    labels : numpy.ndarray
        Pointwise label matrix.
    channels : int
        Number of label columns.
    """
    columns = [f"label-{channel}" for channel in range(int(channels))]
    df = pd.DataFrame(labels.astype(np.int8), columns=pd.Index(columns))
    df.to_csv(path, index=False)


def write_instance_artifacts(
    *,
    instance_dir: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    labels: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    channels: int,
    csv_float_format: str,
) -> None:
    """Write the standard per-instance dataset artifact bundle.

    Parameters
    ----------
    instance_dir : pathlib.Path
        Instance output directory.
    clean : numpy.ndarray
        Clean time-series matrix.
    anomalous : numpy.ndarray
        Anomalous time-series matrix. For clean-only instances this may equal
        ``clean``.
    labels : numpy.ndarray
        Pointwise label matrix.
    events : Sequence[Mapping[str, Any]]
        Event records serialized to ``events.json``.
    channels : int
        Number of generated channels.
    csv_float_format : str
        Float formatting string for time-series CSV files.
    """
    check_labels_and_events_consistency(labels, events)
    write_timeseries_csv(
        instance_dir / "clean.csv",
        clean,
        channels=channels,
        csv_float_format=csv_float_format,
    )
    write_timeseries_csv(
        instance_dir / "anomalous.csv",
        anomalous,
        channels=channels,
        csv_float_format=csv_float_format,
    )
    write_labels_csv(instance_dir / "labels_pointwise.csv", labels, channels=channels)
    label_masks = build_label_masks(
        length=int(labels.shape[0]),
        channels=channels,
        events=events,
    )
    write_label_masks(instance_dir, label_masks)
    write_json(instance_dir / "events.json", events, sort_keys=True, indent=2)
