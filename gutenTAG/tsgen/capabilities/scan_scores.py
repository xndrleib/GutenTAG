"""Cached clean-window witness scores for detectability profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .array_store import ArrayStore
from .dataset import InstanceRecord, read_timeseries_csv
from .protocol import CapabilityProtocol
from .windows import WindowLibrary, WindowSpec
from .witnesses import detection_window_scores


@dataclass(frozen=True)
class CleanWindowScoreBlock:
    """Detection witness scores for one clean variant/window/subset block."""

    instance_keys: tuple[str, ...]
    window_counts: tuple[int, ...]
    scores_by_witness: Mapping[str, tuple[np.ndarray, ...]]


class CleanWindowScoreCache:
    """Memoize clean-window detection scores shared by null calibrators."""

    def __init__(self) -> None:
        self._blocks: dict[tuple[object, ...], CleanWindowScoreBlock] = {}

    def get(
        self,
        *,
        clean_instances: Sequence[InstanceRecord],
        clean_cache: dict[str, np.ndarray],
        arrays: ArrayStore | None,
        windows: WindowLibrary | None,
        event_length: int,
        subset: Sequence[int],
        protocol: CapabilityProtocol,
    ) -> CleanWindowScoreBlock:
        """Return cached scores for a clean instance set and channel subset."""

        instance_keys = tuple(_instance_key(instance) for instance in clean_instances)
        projection = tuple(int(channel) for channel in subset)
        key = (
            instance_keys,
            int(event_length),
            projection,
            protocol.max_scan_windows_per_length,
            protocol.clean_window_stride_fraction,
            protocol.context_window_multiplier,
            protocol.min_context_points,
            tuple(protocol.detection_witnesses),
        )
        if key not in self._blocks:
            self._blocks[key] = _compute_block(
                clean_instances=clean_instances,
                instance_keys=instance_keys,
                clean_cache=clean_cache,
                arrays=arrays,
                windows=windows,
                event_length=int(event_length),
                subset=projection,
                protocol=protocol,
            )
        return self._blocks[key]


def _compute_block(
    *,
    clean_instances: Sequence[InstanceRecord],
    instance_keys: tuple[str, ...],
    clean_cache: dict[str, np.ndarray],
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    event_length: int,
    subset: tuple[int, ...],
    protocol: CapabilityProtocol,
) -> CleanWindowScoreBlock:
    per_witness: dict[str, list[np.ndarray]] = {
        str(witness): [] for witness in protocol.detection_witnesses
    }
    window_counts: list[int] = []
    for instance in clean_instances:
        clean = _read_clean(instance, arrays, clean_cache)
        scan_windows = _scan_windows(
            clean.shape[0], int(event_length), protocol, windows
        )
        witness_values: dict[str, list[float]] = {
            str(witness): [] for witness in protocol.detection_witnesses
        }
        for start, end in scan_windows:
            scores = detection_window_scores(
                series=clean,
                start=int(start),
                end=int(end),
                channels=subset,
                context_multiplier=protocol.context_window_multiplier,
                min_context_points=protocol.min_context_points,
            )
            for witness in witness_values:
                value = scores.get(witness)
                witness_values[witness].append(
                    float(value) if value is not None else np.nan
                )
        window_counts.append(len(scan_windows))
        for witness, values in witness_values.items():
            per_witness[witness].append(np.asarray(values, dtype=np.float64))
    return CleanWindowScoreBlock(
        instance_keys=instance_keys,
        window_counts=tuple(window_counts),
        scores_by_witness={key: tuple(values) for key, values in per_witness.items()},
    )


def _read_clean(
    instance: InstanceRecord,
    arrays: ArrayStore | None,
    clean_cache: dict[str, np.ndarray],
) -> np.ndarray:
    key = _instance_key(instance)
    if key not in clean_cache:
        clean_cache[key] = (
            arrays.get(instance, "clean")
            if arrays is not None
            else read_timeseries_csv(instance.clean_path)
        )
    return clean_cache[key]


def _scan_windows(
    series_length: int,
    event_length: int,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> np.ndarray | list[tuple[int, int]]:
    if windows is None:
        from .numerics import contiguous_windows

        return contiguous_windows(
            series_length,
            int(event_length),
            max_windows=protocol.max_scan_windows_per_length,
            stride_fraction=protocol.clean_window_stride_fraction,
        )
    return windows.get(
        WindowSpec(
            series_length=series_length,
            window_length=int(event_length),
            max_windows=protocol.max_scan_windows_per_length,
            stride_fraction=protocol.clean_window_stride_fraction,
        )
    )


def _instance_key(instance: InstanceRecord) -> str:
    return f"{instance.variant_id}/{instance.split}/{instance.instance_id}"
