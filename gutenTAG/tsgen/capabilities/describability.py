"""Witness- and repair-based describability profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid, read_timeseries_csv
from .array_store import ArrayStore
from .numerics import finite_float, log2_comb, segment
from .ontology import canonical_witnesses_for_anomaly, repair_operator_for_anomaly
from .protocol import CapabilityProtocol
from .repair import repair_segment


@dataclass(frozen=True)
class DescribabilityResult:
    """Description-quality tables."""

    description_profile: pd.DataFrame
    summary: pd.DataFrame
    description_stability: pd.DataFrame


def compute_describability_profiles(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    observability: pd.DataFrame,
    event_summary: pd.DataFrame,
    arrays: ArrayStore | None = None,
) -> DescribabilityResult:
    """Compute witness sufficiency, repair gain, and descriptor complexity."""

    obs_by_event = observability.groupby("event_id") if not observability.empty else {}
    summary_by_event = event_summary.set_index("event_id") if not event_summary.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for instance in dataset.instances:
        clean = arrays.get(instance, "clean") if arrays is not None else read_timeseries_csv(instance.clean_path)
        anomalous = arrays.get(instance, "anomalous") if arrays is not None else read_timeseries_csv(instance.anomalous_path)
        for group in instance.event_groups:
            uid = event_uid(instance, group)
            frame = obs_by_event.get_group(uid) if not observability.empty and uid in obs_by_event.groups else pd.DataFrame()
            event_summary_row = summary_by_event.loc[uid] if not summary_by_event.empty and uid in summary_by_event.index else None
            rows.append(
                _description_row(
                    instance=instance,
                    group=group,
                    clean=clean,
                    anomalous=anomalous,
                    observability_frame=frame,
                    event_summary_row=event_summary_row,
                )
            )
    profile = pd.DataFrame(rows)
    return DescribabilityResult(
        description_profile=profile,
        summary=_summarize(profile),
        description_stability=_description_stability(profile),
    )


def _description_row(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    observability_frame: pd.DataFrame,
    event_summary_row: pd.Series | None,
) -> dict[str, object]:
    canonical = canonical_witnesses_for_anomaly(group.anomaly_type)
    if observability_frame.empty:
        best_distance = 0.0
        best_canonical_distance = 0.0
        canonical_witness_best = "none"
    else:
        best_idx = observability_frame["distance_value"].astype(float).idxmax()
        best_distance = float(observability_frame.loc[best_idx, "distance_value"])
        canonical_frame = observability_frame[observability_frame["witness_family"].isin(canonical)]
        if canonical_frame.empty:
            best_canonical_distance = 0.0
            canonical_witness_best = "none"
        else:
            canonical_idx = canonical_frame["distance_value"].astype(float).idxmax()
            best_canonical_distance = float(canonical_frame.loc[canonical_idx, "distance_value"])
            canonical_witness_best = str(canonical_frame.loc[canonical_idx, "witness_family"])
    witness_sufficiency = best_canonical_distance / max(best_distance, 1e-12)
    repair_gain = _repair_gain(clean, anomalous, group)
    support_concentration = float(event_summary_row.get("support_concentration_l2_max", float("nan"))) if event_summary_row is not None else float("nan")
    description_length_bits = _description_length_bits(instance, group, canonical)
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "base_oscillation": instance.base_oscillation,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "repair_operator": repair_operator_for_anomaly(group.anomaly_type),
        "canonical_witnesses": "|".join(canonical),
        "best_canonical_witness": canonical_witness_best,
        "best_distance": finite_float(best_distance),
        "best_canonical_distance": finite_float(best_canonical_distance),
        "witness_sufficiency": finite_float(witness_sufficiency),
        "repair_gain": finite_float(repair_gain),
        "support_concentration_l2": finite_float(support_concentration, default=float("nan")),
        "description_length_bits": finite_float(description_length_bits),
        "description_risk_proxy": finite_float((1.0 - min(1.0, witness_sufficiency)) + (1.0 - max(0.0, repair_gain))),
    }


def _repair_gain(clean: np.ndarray, anomalous: np.ndarray, group: EventGroup) -> float:
    channels = tuple(sorted(set(group.group_channels) | set(group.context_channels) | set(group.intervention_channels)))
    if not channels:
        channels = tuple(range(clean.shape[1]))
    clean_seg = segment(clean, group.start, group.end, channels)
    anom_seg = segment(anomalous, group.start, group.end, channels)
    if clean_seg.size == 0 or anom_seg.size == 0:
        return 0.0
    raw = float(np.sqrt(np.mean(np.square(anom_seg - clean_seg))))
    if raw <= 1e-12:
        return 1.0
    repaired = repair_segment(
        clean_seg,
        anom_seg,
        constraint_tag=group.constraint_tag,
        repair_operator=group.repair_operator,
    )
    residual = float(np.sqrt(np.mean(np.square(repaired - clean_seg))))
    return float(max(-1.0, min(1.0, 1.0 - residual / raw)))


def _description_length_bits(instance: InstanceRecord, group: EventGroup, canonical_witnesses: Sequence[str]) -> float:
    temporal_bits = 2.0 * math.log2(max(instance.length, 2))
    arity = max(1, len(set(group.group_channels) | set(group.context_channels) | set(group.intervention_channels)))
    channel_bits = log2_comb(instance.channels, min(arity, instance.channels))
    witness_bits = math.log2(max(2, len(canonical_witnesses)))
    constraint_bits = math.log2(16.0)
    return float(temporal_bits + channel_bits + witness_bits + constraint_bits)


def _summarize(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    grouped = profile.groupby(["variant_id", "anomaly_type", "constraint_tag", "semantic_scope"], dropna=False)
    rows: list[dict[str, object]] = []
    for key, frame in grouped:
        variant_id, anomaly_type, constraint_tag, semantic_scope = key
        rows.append(
            {
                "variant_id": variant_id,
                "anomaly_type": anomaly_type,
                "constraint_tag": constraint_tag,
                "semantic_scope": semantic_scope,
                "event_count": int(len(frame)),
                "median_witness_sufficiency": float(frame["witness_sufficiency"].median()),
                "median_repair_gain": float(frame["repair_gain"].median()),
                "median_description_length_bits": float(frame["description_length_bits"].median()),
                "median_description_risk_proxy": float(frame["description_risk_proxy"].median()),
            }
        )
    return pd.DataFrame(rows)


def _description_stability(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    grouped = profile.groupby(
        ["variant_id", "anomaly_type", "constraint_tag", "semantic_scope", "repair_operator"],
        dropna=False,
    )
    rows: list[dict[str, object]] = []
    for key, frame in grouped:
        variant_id, anomaly_type, constraint_tag, semantic_scope, repair_operator = key
        witness_values = frame["witness_sufficiency"].astype(float)
        repair_values = frame["repair_gain"].astype(float)
        risk_values = frame["description_risk_proxy"].astype(float)
        rows.append(
            {
                "variant_id": variant_id,
                "anomaly_type": anomaly_type,
                "constraint_tag": constraint_tag,
                "semantic_scope": semantic_scope,
                "repair_operator": repair_operator,
                "event_count": int(len(frame)),
                "witness_sufficiency_iqr": _iqr(witness_values),
                "repair_gain_iqr": _iqr(repair_values),
                "description_risk_iqr": _iqr(risk_values),
                "witness_sufficiency_cv": _coefficient_of_variation(witness_values),
                "repair_gain_cv": _coefficient_of_variation(repair_values),
                "description_stability_status": _stability_status(witness_values, repair_values),
            }
        )
    return pd.DataFrame(rows)


def _iqr(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    return float(finite.quantile(0.75) - finite.quantile(0.25))


def _coefficient_of_variation(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    mean = float(finite.mean())
    if abs(mean) <= 1e-12:
        return 0.0 if float(finite.std(ddof=0)) <= 1e-12 else float("inf")
    return float(finite.std(ddof=0) / abs(mean))


def _stability_status(witness_values: pd.Series, repair_values: pd.Series) -> str:
    witness_iqr = _iqr(witness_values)
    repair_iqr = _iqr(repair_values)
    if not math.isfinite(witness_iqr) or not math.isfinite(repair_iqr):
        return "not_estimable"
    if witness_iqr <= 0.10 and repair_iqr <= 0.10:
        return "stable_description"
    if witness_iqr <= 0.25 and repair_iqr <= 0.25:
        return "moderately_stable_description"
    return "unstable_description"
