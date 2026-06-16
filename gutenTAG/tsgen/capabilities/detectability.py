"""Clean-calibrated, scan-corrected detectability frontiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd

from .array_store import ArrayStore
from .dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
    event_uid,
    instances_by_variant,
    read_timeseries_csv,
)
from .numerics import channel_subsets, contiguous_windows, finite_float
from .ontology import witness_requires_projection_size
from .protocol import CapabilityProtocol, window_length_bin
from .scan_scores import CleanWindowScoreCache
from .windows import WindowLibrary, WindowSpec
from .witnesses import detection_window_scores


@dataclass(frozen=True)
class DetectabilityResult:
    """Detectability frontier tables."""

    event_frontier: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class _DetectabilityState:
    """Mutable caches used while building the legacy detectability frontier."""

    null_instances: Mapping[str, tuple[InstanceRecord, ...]]
    clean_cache: dict[str, np.ndarray]
    anom_cache: dict[str, np.ndarray]
    null_cache: dict[tuple[object, ...], tuple[np.ndarray, int, float]]
    score_cache: CleanWindowScoreCache
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class _EventScanSpec:
    """Resolved projection and calibration inputs for one event group."""

    subsets: tuple[tuple[int, ...], ...]
    witnesses: tuple[str, ...]
    scan_length: int
    null_key: tuple[object, ...]


def compute_detectability_frontier(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
    windows: WindowLibrary | None = None,
) -> DetectabilityResult:
    """Compute event-level calibrated detectability at each alpha in the protocol."""

    state = _new_detectability_state(dataset, protocol)
    for instance in dataset.instances:
        _append_instance_frontier_rows(state, instance, protocol, arrays, windows)
    event_frontier = pd.DataFrame(state.rows)
    summary = _summarize_frontier(event_frontier)
    return DetectabilityResult(event_frontier=event_frontier, summary=summary)


def _new_detectability_state(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
) -> _DetectabilityState:
    variant_instances = instances_by_variant(dataset.instances)
    null_instances = {
        variant_id: _legacy_null_instances(instances, protocol)
        for variant_id, instances in variant_instances.items()
    }
    return _DetectabilityState(
        null_instances=null_instances,
        clean_cache={},
        anom_cache={},
        null_cache={},
        score_cache=CleanWindowScoreCache(),
        rows=[],
    )


def _append_instance_frontier_rows(
    state: _DetectabilityState,
    instance: InstanceRecord,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
) -> None:
    instance_key = _instance_key(instance)
    state.anom_cache[instance_key] = _read_array(instance, "anomalous", arrays)
    for group in instance.event_groups:
        spec = _event_scan_spec(instance, group, protocol)
        if spec is None:
            continue
        state.rows.extend(
            _event_frontier_rows(
                state,
                instance,
                group,
                spec,
                protocol,
                arrays,
                windows,
            )
        )


def _event_scan_spec(
    instance: InstanceRecord,
    group: EventGroup,
    protocol: CapabilityProtocol,
) -> _EventScanSpec | None:
    candidates = _candidate_channels(instance, group)
    subsets = tuple(channel_subsets(candidates, protocol.max_projection_size))
    witnesses = _admissible_detection_witnesses(subsets, protocol.detection_witnesses)
    if not subsets or not witnesses:
        return None
    scan_length = window_length_bin(max(1, group.length), protocol)
    return _EventScanSpec(
        subsets=subsets,
        witnesses=witnesses,
        scan_length=scan_length,
        null_key=_detectability_null_key(
            instance, scan_length, subsets, witnesses, protocol
        ),
    )


def _detectability_null_key(
    instance: InstanceRecord,
    scan_length: int,
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
    protocol: CapabilityProtocol,
) -> tuple[object, ...]:
    return (
        instance.variant_id,
        scan_length,
        tuple(subsets),
        tuple(witnesses),
        protocol.max_scan_windows_per_length,
        protocol.window_length_policy_mode,
        tuple(protocol.window_length_bins),
        protocol.legacy_detectability_max_clean_instances,
        protocol.calibration_split,
        protocol.clean_window_stride_fraction,
        protocol.context_window_multiplier,
        protocol.min_context_points,
    )


def _event_frontier_rows(
    state: _DetectabilityState,
    instance: InstanceRecord,
    group: EventGroup,
    spec: _EventScanSpec,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
) -> list[dict[str, object]]:
    null_scores, raw_candidate_count, effective_count = _cached_null_scan_scores(
        state,
        instance,
        spec,
        protocol,
        arrays,
        windows,
    )
    event_scores = _event_detection_scores(
        series=state.anom_cache[_instance_key(instance)],
        group=group,
        subsets=spec.subsets,
        witnesses=spec.witnesses,
        protocol=protocol,
    )
    return _alpha_frontier_rows(
        instance,
        group,
        event_scores,
        null_scores,
        raw_candidate_count,
        effective_count,
        protocol,
    )


def _cached_null_scan_scores(
    state: _DetectabilityState,
    instance: InstanceRecord,
    spec: _EventScanSpec,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
) -> tuple[np.ndarray, int, float]:
    if spec.null_key not in state.null_cache:
        state.null_cache[spec.null_key] = _null_scan_scores(
            clean_instances=state.null_instances.get(instance.variant_id, (instance,)),
            clean_cache=state.clean_cache,
            score_cache=state.score_cache,
            arrays=arrays,
            windows=windows,
            event_length=spec.scan_length,
            subsets=spec.subsets,
            witnesses=spec.witnesses,
            protocol=protocol,
        )
    return state.null_cache[spec.null_key]


def _alpha_frontier_rows(
    instance: InstanceRecord,
    group: EventGroup,
    event_scores: Mapping[str, float],
    null_scores: np.ndarray,
    raw_candidate_count: int,
    effective_count: float,
    protocol: CapabilityProtocol,
) -> list[dict[str, object]]:
    event_score = max(event_scores.values()) if event_scores else 0.0
    p_value = _empirical_p_value(null_scores, event_score)
    best_witness = (
        max(event_scores, key=lambda key: event_scores[key]) if event_scores else "none"
    )
    rows: list[dict[str, object]] = []
    for alpha in protocol.alpha_grid:
        threshold = _empirical_quantile(null_scores, 1.0 - float(alpha))
        rows.append(
            {
                **_common(instance, group),
                "alpha": finite_float(alpha, default=math.nan),
                "event_score": finite_float(event_score),
                "scan_threshold": finite_float(threshold),
                "empirical_p_value": finite_float(p_value),
                "detected": (
                    bool(event_score >= threshold) if null_scores.size else False
                ),
                "null_scan_count": int(null_scores.size),
                "raw_candidate_count": int(raw_candidate_count),
                "effective_candidate_count_proxy": finite_float(effective_count),
                "best_detection_witness": best_witness,
            }
        )
    return rows


def _null_scan_scores(
    *,
    clean_instances: Sequence[InstanceRecord],
    clean_cache: dict[str, np.ndarray],
    score_cache: CleanWindowScoreCache,
    arrays: ArrayStore | None,
    windows: WindowLibrary | None,
    event_length: int,
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
    protocol: CapabilityProtocol,
) -> tuple[np.ndarray, int, float]:
    local_scores: list[float] = []
    instance_best = np.zeros(len(clean_instances), dtype=np.float64)
    raw_candidate_count = 0
    for subset in subsets:
        block = score_cache.get(
            clean_instances=clean_instances,
            clean_cache=clean_cache,
            arrays=arrays,
            windows=windows,
            event_length=int(event_length),
            subset=subset,
            protocol=protocol,
        )
        for witness in witnesses:
            if witness_requires_projection_size(witness) > len(subset):
                continue
            for index, values in enumerate(block.scores_by_witness.get(witness, ())):
                finite = np.asarray(values, dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                if finite.size == 0:
                    continue
                raw_candidate_count += int(finite.size)
                local_scores.extend(float(value) for value in finite)
                instance_best[index] = max(
                    float(instance_best[index]), float(np.max(finite))
                )
    maxima_arr = np.asarray(instance_best, dtype=np.float64)
    local_arr = np.asarray(local_scores, dtype=np.float64)
    return (
        maxima_arr,
        raw_candidate_count,
        _effective_candidate_count(local_arr, maxima_arr),
    )


def _event_detection_scores(
    *,
    series: np.ndarray,
    group: EventGroup,
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
    protocol: CapabilityProtocol,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for subset in subsets:
        scores = detection_window_scores(
            series=series,
            start=group.start,
            end=group.end,
            channels=subset,
            context_multiplier=protocol.context_window_multiplier,
            min_context_points=protocol.min_context_points,
        )
        for witness in witnesses:
            if witness not in scores or witness_requires_projection_size(witness) > len(
                subset
            ):
                continue
            key = f"{witness}@{'|'.join(str(channel) for channel in subset)}"
            result[key] = finite_float(scores[witness])
    return result


def _effective_candidate_count(local_scores: np.ndarray, maxima: np.ndarray) -> float:
    """Estimate effective scan multiplicity from local and max null scores.

    The proxy solves the Sidak median equation using the empirical local CDF at
    the median clean scan maximum.  It is a diagnostic multiplicity measure, not
    a distribution-free guarantee.
    """

    local = local_scores[np.isfinite(local_scores)]
    max_values = maxima[np.isfinite(maxima)]
    if local.size < 2 or max_values.size == 0:
        return float("nan")
    median_max = float(np.median(max_values))
    local_cdf = float(np.mean(local <= median_max))
    local_cdf = min(max(local_cdf, 1e-9), 1.0 - 1e-9)
    estimate = math.log(0.5) / math.log(local_cdf)
    return max(1.0, min(float(local.size), float(estimate)))


def _empirical_p_value(null_scores: np.ndarray, observed: float) -> float:
    values = null_scores[np.isfinite(null_scores)]
    if values.size == 0:
        return float("nan")
    return float((1 + np.sum(values >= float(observed))) / (values.size + 1))


def _empirical_quantile(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    clipped = min(max(float(quantile), 0.0), 1.0)
    return float(np.quantile(finite, clipped, method="higher"))


def _candidate_channels(instance: InstanceRecord, group: EventGroup) -> tuple[int, ...]:
    channels = tuple(
        sorted(
            set(group.group_channels)
            | set(group.context_channels)
            | set(group.intervention_channels)
        )
    )
    return channels if channels else tuple(range(instance.channels))


def _admissible_detection_witnesses(
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
) -> tuple[str, ...]:
    max_size = max((len(subset) for subset in subsets), default=1)
    return tuple(
        witness
        for witness in witnesses
        if witness_requires_projection_size(witness) <= max_size
    )


def _legacy_null_instances(
    instances: Sequence[InstanceRecord],
    protocol: CapabilityProtocol,
) -> tuple[InstanceRecord, ...]:
    selected = _prefer_calibration_split(instances, protocol.calibration_split)
    limit = int(protocol.legacy_detectability_max_clean_instances)
    if limit > 0:
        selected = selected[:limit]
    return tuple(selected or instances)


def _prefer_calibration_split(
    instances: Sequence[InstanceRecord],
    calibration_split: str | None,
) -> tuple[InstanceRecord, ...]:
    if calibration_split:
        preferred = tuple(
            instance for instance in instances if instance.split == calibration_split
        )
        if preferred:
            return preferred
    return tuple(instances)


def _summarize_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    grouped = frontier.groupby(
        ["variant_id", "anomaly_type", "constraint_tag", "semantic_scope", "alpha"],
        dropna=False,
    )
    rows: list[dict[str, object]] = []
    for key, frame in grouped:
        variant_id, anomaly_type, constraint_tag, semantic_scope, alpha = cast(
            tuple[object, object, object, object, object], key
        )
        rows.append(
            {
                "variant_id": variant_id,
                "anomaly_type": anomaly_type,
                "constraint_tag": constraint_tag,
                "semantic_scope": semantic_scope,
                "alpha": finite_float(alpha, default=math.nan),
                "event_count": int(len(frame)),
                "detected_rate": float(frame["detected"].mean()),
                "median_event_score": float(frame["event_score"].median()),
                "median_scan_threshold": float(frame["scan_threshold"].median()),
                "median_p_value": float(frame["empirical_p_value"].median()),
                "median_effective_candidate_count_proxy": float(
                    frame["effective_candidate_count_proxy"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _common(instance: InstanceRecord, group: EventGroup) -> dict[str, object]:
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "base_oscillation": instance.base_oscillation,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "start": group.start,
        "end": group.end,
        "length": group.length,
        "group_channels": "|".join(str(channel) for channel in group.group_channels),
    }


def _instance_key(instance: InstanceRecord) -> str:
    return f"{instance.variant_id}/{instance.split}/{instance.instance_id}"


def _read_array(
    instance: InstanceRecord, kind: str, arrays: ArrayStore | None
) -> np.ndarray:
    if arrays is not None and kind == "clean":
        return arrays.get(instance, "clean")
    if arrays is not None and kind == "anomalous":
        return arrays.get(instance, "anomalous")
    if kind == "clean":
        return read_timeseries_csv(instance.clean_path)
    return read_timeseries_csv(instance.anomalous_path)


def _scan_windows(
    series_length: int,
    event_length: int,
    protocol: CapabilityProtocol,
    windows: WindowLibrary | None,
) -> np.ndarray | list[tuple[int, int]]:
    if windows is None:
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
