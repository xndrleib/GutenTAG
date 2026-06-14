"""Analysis-level negative controls for implementation validity."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..array_store import ArrayStore
from ..cache import CacheStore
from ..dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid, read_timeseries_csv
from ..hashes import table_content_hash
from ..numerics import channel_subsets, finite_float
from ..ontology import DETECTION_WITNESSES, witness_requires_projection_size
from ..partitions import PartitionSpec, partition_sequence, run_partitions
from ..protocol import CapabilityProtocol
from ..witnesses import detection_window_scores


MARGINAL_WITNESSES = ("mean_z", "variance_log_ratio", "local_energy_z")


def compute_negative_controls(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    detectability_frontier: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    *,
    arrays: ArrayStore | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 8,
) -> pd.DataFrame:
    """Compute first-pass analysis-level negative controls."""

    if detectability_frontier.empty:
        return pd.DataFrame()
    if cache is None and int(n_jobs) <= 1:
        return _negative_controls_for_instances(
            instances=dataset.instances,
            protocol=protocol,
            detectability_frontier=detectability_frontier,
            boundary_audit=boundary_audit,
            arrays=arrays,
        )
    partitions = partition_sequence(
        dataset.instances,
        partition_size=max(1, int(partition_size)),
        prefix="negative_controls",
    )
    fingerprint_extra = {
        "detectability_frontier_hash": _input_table_hash(detectability_frontier),
        "boundary_audit_hash": _input_table_hash(boundary_audit),
        "partition_size": int(partition_size),
    }

    def worker(partition: PartitionSpec[InstanceRecord]) -> pd.DataFrame:
        return _negative_controls_for_instances(
            instances=partition.items,
            protocol=protocol,
            detectability_frontier=detectability_frontier,
            boundary_audit=boundary_audit,
            arrays=arrays,
        )

    results = run_partitions(
        partitions,
        worker,
        n_jobs=n_jobs,
        cache=cache,
        profile_name="negative_controls",
        fingerprint_extra=fingerprint_extra,
        table_worker=worker if cache is not None else None,
    )
    frames = [result.value for result in results if not result.value.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _negative_controls_for_instances(
    *,
    instances: Sequence[InstanceRecord],
    protocol: CapabilityProtocol,
    detectability_frontier: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    arrays: ArrayStore | None,
) -> pd.DataFrame:
    """Compute negative controls for a deterministic instance partition."""

    frontier_by_event = detectability_frontier.groupby("event_id")
    boundary_by_event = boundary_audit.set_index("event_id") if not boundary_audit.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for instance in instances:
        clean = arrays.get(instance, "clean") if arrays is not None else read_timeseries_csv(instance.clean_path)
        anomalous = arrays.get(instance, "anomalous") if arrays is not None else read_timeseries_csv(instance.anomalous_path)
        for group in instance.event_groups:
            uid = event_uid(instance, group)
            if uid not in frontier_by_event.groups:
                continue
            frontier = frontier_by_event.get_group(uid)
            rows.extend(
                _event_negative_controls(
                    instance=instance,
                    group=group,
                    clean=clean,
                    anomalous=anomalous,
                    frontier=frontier,
                    boundary_row=_lookup(boundary_by_event, uid),
                    protocol=protocol,
                )
            )
    return pd.DataFrame(rows)


def _event_negative_controls(
    *,
    instance: InstanceRecord,
    group: EventGroup,
    clean: np.ndarray,
    anomalous: np.ndarray,
    frontier: pd.DataFrame,
    boundary_row: dict[str, object],
    protocol: CapabilityProtocol,
) -> list[dict[str, object]]:
    candidates = _candidate_channels(instance, group)
    subsets = channel_subsets(candidates, protocol.max_projection_size)
    witnesses = _admissible_witnesses(subsets, protocol.detection_witnesses)
    relation_like = _is_relation_like(group)
    clean_control_witnesses = _relation_witnesses(witnesses) if relation_like else witnesses
    clean_score, clean_witness = _best_score(
        series=clean,
        start=group.start,
        end=group.end,
        subsets=subsets,
        witnesses=clean_control_witnesses,
        protocol=protocol,
    )
    shifted_start, shifted_end = _shifted_support(instance, group, protocol)
    wrong_support_witnesses = clean_control_witnesses if relation_like else witnesses
    clean_wrong_support_score, _ = _best_score(
        series=clean,
        start=shifted_start,
        end=shifted_end,
        subsets=subsets,
        witnesses=wrong_support_witnesses,
        protocol=protocol,
    )
    wrong_support_score, wrong_support_witness = _best_score(
        series=anomalous,
        start=shifted_start,
        end=shifted_end,
        subsets=subsets,
        witnesses=wrong_support_witnesses,
        protocol=protocol,
    )
    marginal_subsets = tuple(subset for subset in subsets if len(subset) == 1)
    clean_marginal_score, _ = _best_score(
        series=clean,
        start=group.start,
        end=group.end,
        subsets=marginal_subsets,
        witnesses=MARGINAL_WITNESSES,
        protocol=protocol,
    )
    marginal_score, marginal_witness = _best_score(
        series=anomalous,
        start=group.start,
        end=group.end,
        subsets=marginal_subsets,
        witnesses=MARGINAL_WITNESSES,
        protocol=protocol,
    )
    boundary_ratio = _as_float(boundary_row.get("boundary_to_canonical_ratio", math.nan))
    boundary_triggered = bool(boundary_row.get("boundary_primary_detection_cause", False))
    rows: list[dict[str, object]] = []
    for _, frontier_row in frontier.iterrows():
        alpha = _as_float(frontier_row.get("alpha", math.nan))
        threshold = _as_float(frontier_row.get("scan_threshold", math.nan))
        rows.append(
            _control_row(
                instance,
                group,
                alpha=alpha,
                control_type="clean_vs_clean",
                control_score=clean_score,
                reference_threshold=threshold,
                witness=clean_witness,
                triggered=_triggered(clean_score, threshold),
                applicability="applicable",
            )
        )
        rows.append(
            _control_row(
                instance,
                group,
                alpha=alpha,
                control_type="wrong_support_control",
                control_score=wrong_support_score,
                reference_threshold=threshold,
                witness=wrong_support_witness,
                triggered=_triggered_against_baseline(
                    wrong_support_score,
                    threshold,
                    clean_wrong_support_score,
                ),
                applicability="applicable",
                baseline_score=clean_wrong_support_score,
            )
        )
        rows.append(
            _control_row(
                instance,
                group,
                alpha=alpha,
                control_type="wrong_witness_control",
                control_score=marginal_score,
                reference_threshold=threshold,
                witness=marginal_witness,
                triggered=_triggered_against_baseline(
                    marginal_score,
                    threshold,
                    clean_marginal_score,
                )
                if relation_like
                else False,
                applicability="applicable" if relation_like else "not_applicable",
                baseline_score=clean_marginal_score,
            )
        )
        rows.append(
            _control_row(
                instance,
                group,
                alpha=alpha,
                control_type="boundary_only_control",
                control_score=boundary_ratio,
                reference_threshold=1.5,
                witness="boundary_to_canonical_ratio",
                triggered=boundary_triggered,
                applicability="applicable",
            )
        )
    return rows


def _control_row(
    instance: InstanceRecord,
    group: EventGroup,
    *,
    alpha: float,
    control_type: str,
    control_score: float,
    reference_threshold: float,
    witness: str,
    triggered: bool,
    applicability: str,
    baseline_score: float = math.nan,
) -> dict[str, object]:
    status = (
        "not_applicable"
        if applicability == "not_applicable"
        else "control_failed"
        if triggered
        else "control_passed"
    )
    return {
        "event_id": event_uid(instance, group),
        "variant_id": instance.variant_id,
        "split": instance.split,
        "instance_id": instance.instance_id,
        "anomaly_type": group.anomaly_type,
        "constraint_tag": group.constraint_tag,
        "semantic_scope": group.semantic_scope,
        "alpha": finite_float(alpha, default=math.nan),
        "control_type": control_type,
        "control_score": finite_float(control_score, default=math.nan),
        "baseline_control_score": finite_float(baseline_score, default=math.nan),
        "reference_threshold": finite_float(reference_threshold, default=math.nan),
        "control_witness": witness,
        "control_triggered": bool(triggered),
        "expected_negative": True,
        "control_status": status,
        "applicability": applicability,
    }


def _best_score(
    *,
    series: np.ndarray,
    start: int,
    end: int,
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
    protocol: CapabilityProtocol,
) -> tuple[float, str]:
    best_score = math.nan
    best_witness = "none"
    for subset in subsets:
        if not subset:
            continue
        scores = detection_window_scores(
            series=series,
            start=start,
            end=end,
            channels=subset,
            context_multiplier=protocol.context_window_multiplier,
            min_context_points=protocol.min_context_points,
        )
        for witness in witnesses:
            if witness not in scores or witness_requires_projection_size(witness) > len(subset):
                continue
            score = float(scores[witness])
            if not math.isfinite(best_score) or score > best_score:
                best_score = score
                best_witness = f"{witness}@{'|'.join(str(channel) for channel in subset)}"
    return best_score, best_witness


def _candidate_channels(instance: InstanceRecord, group: EventGroup) -> tuple[int, ...]:
    channels = tuple(sorted(set(group.group_channels) | set(group.context_channels) | set(group.intervention_channels)))
    return channels if channels else tuple(range(instance.channels))


def _admissible_witnesses(
    subsets: Sequence[tuple[int, ...]],
    witnesses: Sequence[str],
) -> tuple[str, ...]:
    max_size = max((len(subset) for subset in subsets), default=1)
    return tuple(witness for witness in witnesses if witness_requires_projection_size(witness) <= max_size)


def _relation_witnesses(witnesses: Sequence[str]) -> tuple[str, ...]:
    relation_witnesses = tuple(
        witness for witness in witnesses if witness in {"correlation_shift", "covariance_shift"}
    )
    return relation_witnesses if relation_witnesses else tuple(witnesses)


def _shifted_support(
    instance: InstanceRecord,
    group: EventGroup,
    protocol: CapabilityProtocol,
) -> tuple[int, int]:
    length = int(instance.length)
    width = max(1, group.length)
    protected = tuple(_declared_support_bounds(other, length) for other in instance.event_groups)
    for start in _shifted_support_candidates(group, width, length):
        end = start + width
        context_start, context_end = _context_bounds(start, end, length, protocol)
        if not _overlaps_any(start, end, protected) and not _overlaps_any(context_start, context_end, protected):
            return start, end
    if int(group.end) + width <= length:
        return int(group.end), int(group.end) + width
    if int(group.start) - width >= 0:
        return int(group.start) - width, int(group.start)
    return 0, min(length, width)


def _shifted_support_candidates(group: EventGroup, width: int, length: int) -> tuple[int, ...]:
    starts: list[int] = []
    for multiplier in range(1, max(2, length // max(1, width)) + 2):
        starts.extend(
            (
                int(group.end) + (multiplier - 1) * width,
                int(group.start) - multiplier * width,
            )
        )
    starts.extend(range(0, max(1, length - width + 1), max(1, width)))
    deduped: list[int] = []
    seen: set[int] = set()
    for start in starts:
        clipped = max(0, min(int(start), max(0, length - width)))
        if clipped in seen:
            continue
        seen.add(clipped)
        deduped.append(clipped)
    return tuple(deduped)


def _declared_support_bounds(group: EventGroup, length: int) -> tuple[int, int]:
    start = group.source_start if group.source_end > group.source_start else group.start
    end = group.source_end if group.source_end > group.source_start else group.end
    lo = max(0, min(int(start), int(length)))
    hi = max(lo, min(int(end), int(length)))
    return lo, hi


def _context_bounds(
    start: int,
    end: int,
    length: int,
    protocol: CapabilityProtocol,
) -> tuple[int, int]:
    width = max(1, int(end) - int(start))
    radius = max(int(protocol.min_context_points), int(protocol.context_window_multiplier) * width)
    return max(0, int(start) - radius), min(int(length), int(end) + radius)


def _overlaps_any(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> bool:
    for other_start, other_end in intervals:
        if int(start) < int(other_end) and int(end) > int(other_start):
            return True
    return False


def _is_relation_like(group: EventGroup) -> bool:
    return (
        len(group.group_channels) >= 2
        or len(group.context_channels) >= 2
        or "relation" in str(group.semantic_scope)
        or "dependence" in str(group.constraint_tag)
    )


def _triggered(score: float, threshold: float) -> bool:
    return math.isfinite(score) and math.isfinite(threshold) and score >= threshold


def _triggered_against_baseline(score: float, threshold: float, baseline: float) -> bool:
    if not _triggered(score, threshold):
        return False
    if not math.isfinite(baseline):
        return True
    baseline_floor = max(float(threshold), float(baseline) * 1.25, float(baseline) + 0.25)
    return float(score) >= baseline_floor


def _lookup(frame: pd.DataFrame, event_id: str) -> dict[str, object]:
    if frame.empty or event_id not in frame.index:
        return {}
    row = frame.loc[event_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return dict(row)


def _as_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _input_table_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return table_content_hash(frame)
    sort_columns = [
        column
        for column in ("event_id", "alpha", "control_type", "witness_or_model", "projection")
        if column in frame.columns
    ]
    return table_content_hash(frame, sort_by=sort_columns or None)
