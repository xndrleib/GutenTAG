"""Observed separability and quantitative arity profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
    event_uid,
    read_timeseries_csv,
)
from .array_store import ArrayStore
from .numerics import channel_subsets, finite_float
from .ontology import (
    available_witnesses_for_projection,
    canonical_witnesses_for_anomaly,
)
from .pandas_typing import row_mapping
from .protocol import CapabilityProtocol
from .witnesses import paired_witness_scores


@dataclass(frozen=True)
class ObservabilityResult:
    """Observed separability tables."""

    observability: pd.DataFrame
    arity: pd.DataFrame
    event_summary: pd.DataFrame


def compute_observability_profiles(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
) -> ObservabilityResult:
    """Compute per-event witness distances and arity curves."""

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for instance in dataset.instances:
        clean = (
            arrays.get(instance, "clean")
            if arrays is not None
            else read_timeseries_csv(instance.clean_path)
        )
        anomalous = (
            arrays.get(instance, "anomalous")
            if arrays is not None
            else read_timeseries_csv(instance.anomalous_path)
        )
        for group in instance.event_groups:
            event_rows = _event_observability_rows(
                instance=instance,
                group=group,
                clean=clean,
                anomalous=anomalous,
                protocol=protocol,
            )
            rows.extend(event_rows)
            summary_rows.append(
                _event_summary_row(instance, group, event_rows, protocol)
            )
    observability = pd.DataFrame(rows)
    event_summary = pd.DataFrame(summary_rows)
    arity = _build_arity_profile(event_summary, protocol)
    return ObservabilityResult(
        observability=observability, arity=arity, event_summary=event_summary
    )


def _event_observability_rows(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    protocol: CapabilityProtocol,
) -> list[dict[str, object]]:
    uid = event_uid(instance, group)
    candidate_channels = tuple(
        sorted(
            set(group.group_channels)
            | set(group.context_channels)
            | set(group.intervention_channels)
        )
    )
    if not candidate_channels:
        candidate_channels = tuple(range(instance.channels))
    rows: list[dict[str, object]] = []
    for subset in channel_subsets(candidate_channels, protocol.max_projection_size):
        scores = paired_witness_scores(
            clean=clean,
            anomalous=anomalous,
            start=group.start,
            end=group.end,
            channels=subset,
            length=instance.length,
            context_multiplier=protocol.context_window_multiplier,
            min_context_points=protocol.min_context_points,
        )
        admissible = set(
            available_witnesses_for_projection(
                len(subset),
                candidate_witnesses=protocol.witness_families,
            )
        )
        for witness, value in scores.items():
            if witness == "support_concentration_l2" or witness not in admissible:
                continue
            rows.append(
                {
                    **_event_common(instance, group, uid),
                    "projection_size": len(subset),
                    "channel_subset": _format_subset(subset),
                    "witness_family": witness,
                    "distance_value": finite_float(value),
                    "support_concentration_l2": finite_float(
                        scores.get("support_concentration_l2", float("nan")),
                        default=float("nan"),
                    ),
                    "canonical_witness": witness
                    in canonical_witnesses_for_anomaly(group.anomaly_type),
                }
            )
    return rows


def _event_summary_row(
    instance: InstanceRecord,
    group: EventGroup,
    rows: Sequence[Mapping[str, object]],
    protocol: CapabilityProtocol,
) -> dict[str, object]:
    common = _event_common(instance, group, event_uid(instance, group))
    values_by_size: dict[int, float] = {}
    canonical_values_by_size: dict[int, float] = {}
    support_concentration_values: list[float] = []
    for row in rows:
        size = _int_mapping(row, "projection_size")
        value = _float_mapping(row, "distance_value")
        values_by_size[size] = max(values_by_size.get(size, 0.0), value)
        if bool(row.get("canonical_witness", False)):
            canonical_values_by_size[size] = max(
                canonical_values_by_size.get(size, 0.0), value
            )
        support_value = row.get("support_concentration_l2")
        if support_value is not None:
            support_value_float = _float_value(support_value, default=float("nan"))
            if math.isfinite(support_value_float):
                support_concentration_values.append(support_value_float)
    cumulative: dict[int, float] = {}
    best = 0.0
    for size in range(1, int(protocol.max_projection_size) + 1):
        best = max(best, values_by_size.get(size, 0.0))
        cumulative[size] = best
    canonical_cumulative: dict[int, float] = {}
    canonical_best = 0.0
    for size in range(1, int(protocol.max_projection_size) + 1):
        canonical_best = max(canonical_best, canonical_values_by_size.get(size, 0.0))
        canonical_cumulative[size] = canonical_best
    result = {
        **common,
        "best_distance": finite_float(best),
        "best_canonical_distance": finite_float(canonical_best),
        "witness_sufficiency_proxy": finite_float(canonical_best / max(best, 1e-12)),
        "support_concentration_l2_max": finite_float(
            (
                max(support_concentration_values)
                if support_concentration_values
                else float("nan")
            ),
            default=float("nan"),
        ),
    }
    for size in range(1, int(protocol.max_projection_size) + 1):
        result[f"D_s{size}"] = finite_float(cumulative.get(size, 0.0))
        result[f"D_canonical_s{size}"] = finite_float(
            canonical_cumulative.get(size, 0.0)
        )
    return result


def _build_arity_profile(
    event_summary: pd.DataFrame, protocol: CapabilityProtocol
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if event_summary.empty:
        return pd.DataFrame(rows)
    for _, record_series in event_summary.iterrows():
        record = row_mapping(record_series)
        d_values = {
            size: _float_mapping(record, f"D_s{size}")
            for size in range(1, protocol.max_projection_size + 1)
        }
        canonical_d_values = {
            size: _float_mapping(record, f"D_canonical_s{size}")
            for size in range(1, protocol.max_projection_size + 1)
        }
        for delta in protocol.delta_grid:
            observed_arity: int | None = None
            canonical_observed_arity: int | None = None
            for size in range(1, protocol.max_projection_size + 1):
                if observed_arity is None and d_values[size] > float(delta):
                    observed_arity = size
                if canonical_observed_arity is None and canonical_d_values[
                    size
                ] > float(delta):
                    canonical_observed_arity = size
            rows.append(
                {
                    "event_id": record["event_id"],
                    "variant_id": record["variant_id"],
                    "split": record["split"],
                    "instance_id": record["instance_id"],
                    "anomaly_type": record["anomaly_type"],
                    "constraint_tag": record["constraint_tag"],
                    "semantic_scope": record["semantic_scope"],
                    "delta": float(delta),
                    "observed_arity": (
                        observed_arity if observed_arity is not None else math.nan
                    ),
                    "is_observable_at_delta": observed_arity is not None,
                    "canonical_observed_arity": (
                        canonical_observed_arity
                        if canonical_observed_arity is not None
                        else math.nan
                    ),
                    "canonical_is_observable_at_delta": canonical_observed_arity
                    is not None,
                    "semantic_arity": _semantic_arity(record),
                    "D_at_max_projection": d_values[protocol.max_projection_size],
                    "D_canonical_at_max_projection": canonical_d_values[
                        protocol.max_projection_size
                    ],
                    "arity_gap_s2_minus_s1": finite_float(
                        d_values.get(2, d_values.get(1, 0.0)) - d_values.get(1, 0.0)
                    ),
                    "canonical_arity_gap_s2_minus_s1": finite_float(
                        canonical_d_values.get(2, canonical_d_values.get(1, 0.0))
                        - canonical_d_values.get(1, 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _semantic_arity(record: Mapping[str, object]) -> int:
    channels = str(record.get("group_channels", ""))
    if not channels:
        return 0
    return len([part for part in channels.split("|") if part != ""])


def _event_common(
    instance: InstanceRecord, group: EventGroup, uid: str
) -> dict[str, object]:
    return {
        "event_id": uid,
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "base_oscillation": instance.base_oscillation,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "event_scope": group.event_scope,
        "purity_hint": group.purity_hint,
        "start": group.start,
        "end": group.end,
        "length": group.length,
        "source_start": group.source_start,
        "source_end": group.source_end,
        "source_length": group.source_length,
        "intervention_channels": _format_subset(group.intervention_channels),
        "context_channels": _format_subset(group.context_channels),
        "group_channels": _format_subset(group.group_channels),
        "primary_channels": _format_subset(group.primary_channels),
    }


def _format_subset(channels: Sequence[int]) -> str:
    return "|".join(str(int(channel)) for channel in channels)


def _int_mapping(row: Mapping[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if isinstance(value, (int, float, str, np.integer, np.floating)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _float_mapping(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    return _float_value(row.get(key, default), default=default)


def _float_value(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float, str, np.integer, np.floating)):
        try:
            resolved = float(value)
        except ValueError:
            return default
    else:
        return default
    if not math.isfinite(resolved):
        return default
    return resolved
