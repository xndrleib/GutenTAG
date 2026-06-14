"""Detector attribution from calibrated detectability frontiers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd

from ...contracts import ContractRegistry
from ..cache import CacheStore
from ..dataset import DatasetIndex, EventGroup, InstanceRecord, event_uid
from ..hashes import table_content_hash
from ..numerics import finite_float
from ..partitions import PartitionSpec, partition_sequence, run_partitions


WITNESS_FAMILY: Mapping[str, str] = {
    "mean_z": "location.mean",
    "variance_log_ratio": "scale.variance",
    "local_energy_z": "local.energy",
    "correlation_shift": "dependence.correlation",
    "covariance_shift": "dependence.covariance",
}

CANONICAL_DETECTION_WITNESSES: Mapping[str, tuple[str, ...]] = {
    "mean": ("mean_z",),
    "variance": ("variance_log_ratio", "local_energy_z"),
    "amplitude": ("variance_log_ratio", "local_energy_z"),
    "platform": ("mean_z", "local_energy_z"),
    "pattern": ("local_energy_z",),
    "frequency": ("local_energy_z",),
    "correlation-flip": ("correlation_shift", "covariance_shift"),
    "covariance-change": ("covariance_shift", "correlation_shift"),
    "lag-synchronization": ("correlation_shift", "covariance_shift"),
    "mode-correlation": ("correlation_shift", "covariance_shift"),
}


def compute_detector_attribution(
    dataset: DatasetIndex,
    detectability_frontier: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
    *,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 512,
    registry: ContractRegistry | None = None,
) -> pd.DataFrame:
    """Build detector-attribution rows from scan-calibrated frontier results."""

    if detectability_frontier.empty:
        return pd.DataFrame()
    active_registry = registry or ContractRegistry.from_resource_defaults()
    row_records = tuple(detectability_frontier.to_dict("records"))
    if cache is None and int(n_jobs) <= 1:
        frame = _attribution_for_rows(
            dataset=dataset,
            frontier_rows=row_records,
            boundary_audit=boundary_audit,
            shortcut_audit=shortcut_audit,
            registry=active_registry,
        )
        return _sort_attribution(frame)
    partitions = partition_sequence(
        row_records,
        partition_size=max(1, int(partition_size)),
        prefix="detector_attribution",
    )
    fingerprint_extra = {
        "detectability_frontier_hash": _input_table_hash(detectability_frontier),
        "boundary_audit_hash": _input_table_hash(boundary_audit),
        "shortcut_audit_hash": _input_table_hash(shortcut_audit),
        "partition_size": int(partition_size),
    }

    def worker(partition: PartitionSpec[Mapping[str, object]]) -> pd.DataFrame:
        return _attribution_for_rows(
            dataset=dataset,
            frontier_rows=partition.items,
            boundary_audit=boundary_audit,
            shortcut_audit=shortcut_audit,
            registry=active_registry,
        )

    results = run_partitions(
        partitions,
        worker,
        n_jobs=n_jobs,
        cache=cache,
        profile_name="detector_attribution",
        fingerprint_extra=fingerprint_extra,
        table_worker=worker if cache is not None else None,
    )
    frames = [result.value for result in results if not result.value.empty]
    return _sort_attribution(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def _attribution_for_rows(
    *,
    dataset: DatasetIndex,
    frontier_rows: Sequence[Mapping[str, object]],
    boundary_audit: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
    registry: ContractRegistry,
) -> pd.DataFrame:
    """Build detector-attribution rows for a deterministic frontier partition."""

    groups = {
        event_uid(instance, group): (instance, group)
        for instance in dataset.instances
        for group in instance.event_groups
    }
    boundary_by_event = boundary_audit.set_index("event_id") if not boundary_audit.empty else pd.DataFrame()
    shortcut_by_event = shortcut_audit.set_index("event_id") if not shortcut_audit.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for frontier_row in frontier_rows:
        event_id = str(frontier_row["event_id"])
        instance_group = groups.get(event_id)
        if instance_group is None:
            continue
        instance, group = instance_group
        witness, projection = _frontier_witness_and_projection(frontier_row)
        family = WITNESS_FAMILY.get(witness, "unknown")
        contract = registry.get_by_anomaly(group.anomaly_type)
        forbidden = (
            family in set(contract.forbidden_shortcuts)
            if contract is not None
            else False
        )
        projection_size = _projection_size(projection)
        canonical = _is_canonical(group, witness, projection_size)
        boundary_row = _lookup(boundary_by_event, event_id)
        shortcut_row = _lookup(shortcut_by_event, event_id)
        boundary_primary = bool(boundary_row.get("boundary_primary_detection_cause", False))
        shortcut_status = str(shortcut_row.get("shortcut_status", "unknown"))
        detected = bool(frontier_row.get("detected", False))
        scan_p = _as_float(_field(frontier_row, "scan_level_p_value", "empirical_p_value"))
        candidate_p = _as_float(_field(frontier_row, "candidate_level_p_value", "empirical_p_value"))
        canonical_detected = _as_bool(_field(frontier_row, "best_canonical_detected", default=False))
        scan_null_count = int(_field(frontier_row, "scan_null_count", "null_scan_count", default=0) or 0)
        min_empirical_p = 1.0 / (scan_null_count + 1.0) if scan_null_count >= 0 else math.nan
        evidence = _normalized_evidence(scan_p, min_empirical_p)
        scope = _scope(
            canonical=canonical,
            forbidden=forbidden,
            boundary_primary=boundary_primary,
            shortcut_status=shortcut_status,
        )
        rows.append(
            {
                "event_id": event_id,
                "variant_id": instance.variant_id,
                "split": instance.split,
                "instance_id": instance.instance_id,
                "anomaly_type": group.anomaly_type,
                "constraint_tag": group.constraint_tag,
                "semantic_scope": group.semantic_scope,
                "alpha": finite_float(frontier_row.get("alpha", math.nan), default=math.nan),
                "scope": scope,
                "family": family,
                "witness_or_model": witness,
                "projection": projection,
                "raw_score": finite_float(_field(frontier_row, "raw_score", "event_score"), default=math.nan),
                "candidate_level_p_value": finite_float(candidate_p, default=math.nan),
                "scan_level_p_value": finite_float(scan_p, default=math.nan),
                "normalized_evidence": finite_float(evidence, default=math.nan),
                "rank_within_event": 1,
                "is_canonical": canonical,
                "is_forbidden_shortcut": forbidden,
                "is_boundary": boundary_primary,
                "primary_detection_cause": _primary_cause(
                    detected=detected,
                    canonical=canonical,
                    canonical_detected=canonical_detected,
                    forbidden=forbidden,
                    boundary_primary=boundary_primary,
                    shortcut_status=shortcut_status,
                ),
                "detected": detected,
                "best_canonical_witness": str(_field(frontier_row, "best_canonical_witness", default="") or ""),
                "best_canonical_candidate_p_value": finite_float(
                    _field(frontier_row, "best_canonical_candidate_p_value"),
                    default=math.nan,
                ),
                "best_canonical_detected": canonical_detected,
                "scan_threshold": finite_float(_field(frontier_row, "scan_threshold"), default=math.nan),
                "candidate_null_count": int(_field(frontier_row, "candidate_null_count", "raw_candidate_count", default=0) or 0),
                "scan_null_count": scan_null_count,
                "raw_candidate_count": int(_field(frontier_row, "raw_candidate_count", default=0) or 0),
                "effective_candidate_count": finite_float(
                    _field(frontier_row, "effective_candidate_count", "effective_candidate_count_proxy"),
                    default=math.nan,
                ),
                "calibration_resolution_min_p": finite_float(
                    _field(frontier_row, "calibration_resolution_min_p", default=min_empirical_p),
                    default=math.nan,
                ),
            }
        )
    return pd.DataFrame(rows)


def _sort_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["event_id", "alpha", "rank_within_event"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return frame


def _parse_witness(value: str) -> tuple[str, str]:
    if "@" not in value:
        return value or "none", ""
    witness, projection = value.split("@", maxsplit=1)
    return witness, projection


def _frontier_witness_and_projection(row: Mapping[str, object]) -> tuple[str, str]:
    witness = str(row.get("witness_or_model", "") or "")
    projection = str(row.get("projection", "") or "")
    if witness:
        return witness, projection
    return _parse_witness(str(row.get("best_detection_witness", "")))


def _field(row: Mapping[str, object], *names: str, default: object = math.nan) -> object:
    for name in names:
        if name in row and pd.notna(row.get(name)):
            return row.get(name)
    return default


def _projection_size(projection: str) -> int:
    if not projection:
        return 0
    return len([part for part in projection.split("|") if part != ""])


def _is_canonical(group: EventGroup, witness: str, projection_size: int) -> bool:
    canonical = CANONICAL_DETECTION_WITNESSES.get(group.anomaly_type, ())
    if witness not in canonical:
        return False
    if "dependence" in group.constraint_tag or "relation" in group.semantic_scope:
        return projection_size >= 2
    return True


def _scope(
    *,
    canonical: bool,
    forbidden: bool,
    boundary_primary: bool,
    shortcut_status: str,
) -> str:
    if boundary_primary:
        return "boundary"
    if forbidden or shortcut_status == "shortcut_dominated":
        return "shortcut"
    if canonical:
        return "canonical"
    return "generic"


def _primary_cause(
    *,
    detected: bool,
    canonical: bool,
    canonical_detected: bool,
    forbidden: bool,
    boundary_primary: bool,
    shortcut_status: str,
) -> str:
    if not detected:
        return "observable_not_detected"
    if boundary_primary:
        return "boundary_artifact_suspected"
    if forbidden or shortcut_status == "shortcut_dominated":
        if canonical_detected:
            return "valid_with_shortcut"
        return "detected_wrong_reason"
    if canonical:
        return "valid_canonical_detection"
    if canonical_detected:
        if shortcut_status == "valid_with_shortcut":
            return "valid_with_shortcut"
        return "valid_canonical_detection"
    if shortcut_status == "valid_with_shortcut":
        return "valid_with_shortcut"
    return "valid_model_detection"


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


def _normalized_evidence(scan_p: float, min_empirical_p: float) -> float:
    if not math.isfinite(scan_p) or not math.isfinite(min_empirical_p):
        return math.nan
    return float(-math.log10(max(scan_p, min_empirical_p, 1e-300)))


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
        for column in ("event_id", "alpha", "witness_or_model", "projection", "scope")
        if column in frame.columns
    ]
    return table_content_hash(frame, sort_by=sort_columns or None)
