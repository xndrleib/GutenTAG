"""Dataset discovery and event grouping for capability analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .ontology import constraint_tag_for_anomaly, repair_operator_for_anomaly, semantic_scope_for_anomaly


@dataclass(frozen=True)
class EventGroup:
    """A grouped anomaly event with temporal and channel support."""

    group_id: str
    start: int
    end: int
    source_start: int
    source_end: int
    anomaly_type: str
    constraint_tag: str
    repair_operator: str
    semantic_scope: str
    intervention_channels: tuple[int, ...]
    context_channels: tuple[int, ...]
    group_channels: tuple[int, ...]
    primary_channels: tuple[int, ...]
    event_scope: str
    purity_hint: str
    raw_events: tuple[Mapping[str, Any], ...]
    genotype_id: str = ""
    contract_id: str = ""
    requested_effect_id: str = ""

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))

    @property
    def source_length(self) -> int:
        return max(0, int(self.source_end) - int(self.source_start))


@dataclass(frozen=True)
class InstanceRecord:
    """A generated instance consumed by the capability analyzer."""

    dataset_root: Path
    variant_id: str
    split: str
    instance_id: str
    instance_dir: Path
    clean_path: Path
    anomalous_path: Path
    events_path: Path
    summary_path: Path
    base_oscillation: str
    anomaly_type: str
    channels: int
    length: int
    event_groups: tuple[EventGroup, ...]


@dataclass(frozen=True)
class DatasetIndex:
    """Discovered dataset layout."""

    root: Path
    manifest: Mapping[str, Any]
    instances: tuple[InstanceRecord, ...]
    metadata_events: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    problem_genotypes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_timeseries_csv(path: Path) -> np.ndarray:
    """Read a generated time-series CSV into a float64 matrix."""

    frame = pd.read_csv(path)
    values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D time-series matrix in {path}")
    return values


def discover_dataset(root: Path, *, prefer_metadata_registry: bool = True) -> DatasetIndex:
    """Discover generated instances under a synth-gen dataset root."""

    dataset_root = Path(root).resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest: Mapping[str, Any] = {}
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    metadata_events = (
        _load_registry_events(dataset_root / "metadata" / "events.jsonl")
        if prefer_metadata_registry
        else {}
    )
    metadata_by_instance = _metadata_events_by_instance(metadata_events.values())
    problem_genotypes = _load_registry_by_id(
        dataset_root / "metadata" / "problem_genotypes.jsonl",
        id_field="genotype_id",
    )
    instances: list[InstanceRecord] = []
    for anomalous_path in sorted(dataset_root.glob("variants/*/*/instances/*/anomalous.csv")):
        instance_dir = anomalous_path.parent
        clean_path = instance_dir / "clean.csv"
        events_path = instance_dir / "events.json"
        summary_path = instance_dir / "instance_summary.json"
        if not clean_path.exists() or not summary_path.exists():
            continue
        summary = load_json(summary_path)
        clean_shape = pd.read_csv(clean_path, nrows=1).shape
        length = int(summary.get("length", summary.get("dataset_length", 0)) or 0)
        if length <= 0:
            length = int(sum(1 for _ in clean_path.open("r", encoding="utf-8")) - 1)
        channels = int(summary.get("channels", clean_shape[1]) or clean_shape[1])
        variant_id = str(summary.get("variant_id", anomalous_path.parents[3].name))
        split = str(summary.get("split", anomalous_path.parents[2].name))
        instance_id = str(summary.get("instance_id", instance_dir.name))
        anomaly_type = str(summary.get("anomaly_type", _parse_anomaly_type(variant_id)))
        base_oscillation = str(summary.get("base_oscillation", _parse_base_oscillation(variant_id)))
        events = _events_for_instance(
            metadata_by_instance=metadata_by_instance,
            variant_id=variant_id,
            split=split,
            instance_id=instance_id,
            fallback_path=events_path,
        )
        instances.append(
            InstanceRecord(
                dataset_root=dataset_root,
                variant_id=variant_id,
                split=split,
                instance_id=instance_id,
                instance_dir=instance_dir,
                clean_path=clean_path,
                anomalous_path=anomalous_path,
                events_path=events_path,
                summary_path=summary_path,
                base_oscillation=base_oscillation,
                anomaly_type=anomaly_type,
                channels=channels,
                length=length,
                event_groups=tuple(
                    group_events(
                        events,
                        anomaly_type=anomaly_type,
                        channels=channels,
                        length=length,
                    )
                ),
            )
        )
    return DatasetIndex(
        root=dataset_root,
        manifest=manifest,
        instances=tuple(instances),
        metadata_events=metadata_events,
        problem_genotypes=problem_genotypes,
    )


def instances_by_variant(instances: Sequence[InstanceRecord]) -> dict[str, list[InstanceRecord]]:
    """Group instances by variant identifier."""

    grouped: dict[str, list[InstanceRecord]] = {}
    for instance in instances:
        grouped.setdefault(instance.variant_id, []).append(instance)
    return grouped


def group_events(
    events: Sequence[Mapping[str, Any]],
    *,
    anomaly_type: str,
    channels: int,
    length: int,
) -> Iterable[EventGroup]:
    """Aggregate raw event rows into event groups."""

    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for index, event in enumerate(events):
        group_id = str(event.get("group_id", index))
        by_group.setdefault(group_id, []).append(event)
    for group_id, rows in sorted(by_group.items(), key=lambda item: _group_sort_key(item[0])):
        starts = [_clip_int(row.get("start", 0), 0, length) for row in rows]
        ends = [_clip_int(row.get("end", row.get("start", 0)), 0, length) for row in rows]
        source_starts = [_clip_int(row.get("source_start", row.get("start", 0)), 0, length) for row in rows]
        source_ends = [_clip_int(row.get("source_end", row.get("end", row.get("start", 0))), 0, length) for row in rows]
        group_anomaly_type = str(rows[0].get("anomaly_type", anomaly_type))
        intervention = _collect_channels(rows, ("operator_target_channels", "intervention_channels", "perturbed_channels", "channel"), channels)
        context = _collect_channels(rows, ("context_channels", "group_channels", "anchor_channel", "channel"), channels)
        grouped_channels = _collect_channels(rows, ("group_channels", "context_channels", "channel"), channels)
        primary = _collect_channels(rows, ("channel",), channels)
        if not grouped_channels:
            grouped_channels = context or intervention or tuple(range(channels))
        descriptor_constraint = constraint_tag_for_anomaly(group_anomaly_type)
        yield EventGroup(
            group_id=group_id,
            start=min(starts) if starts else 0,
            end=max(ends) if ends else 0,
            source_start=min(source_starts) if source_starts else 0,
            source_end=max(source_ends) if source_ends else 0,
            anomaly_type=group_anomaly_type,
            constraint_tag=descriptor_constraint,
            repair_operator=repair_operator_for_anomaly(group_anomaly_type),
            semantic_scope=semantic_scope_for_anomaly(group_anomaly_type),
            intervention_channels=intervention,
            context_channels=context,
            group_channels=grouped_channels,
            primary_channels=primary,
            event_scope=str(rows[0].get("event_scope", "unknown")),
            purity_hint=str(rows[0].get("purity_hint", "unknown")),
            raw_events=tuple(rows),
            genotype_id=str(rows[0].get("genotype_id", "")),
            contract_id=str(rows[0].get("contract_id", "")),
            requested_effect_id=str(rows[0].get("requested_effect_id", "")),
        )


def raw_events_for_instance(instance: InstanceRecord) -> tuple[Mapping[str, Any], ...]:
    """Return raw event records from the active dataset index representation."""

    rows: list[Mapping[str, Any]] = []
    for group in instance.event_groups:
        rows.extend(group.raw_events)
    if rows:
        return tuple(rows)
    if instance.events_path.exists():
        return tuple(load_json(instance.events_path))
    return ()


def event_uid(instance: InstanceRecord, group: EventGroup) -> str:
    """Stable event-group identifier used in output tables."""

    return f"{instance.variant_id}/{instance.split}/{instance.instance_id}/g{group.group_id}"


def _collect_channels(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    channels: int,
) -> tuple[int, ...]:
    found: set[int] = set()
    for row in rows:
        for key in keys:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, (list, tuple)):
                for item in value:
                    _add_channel(found, item, channels)
            else:
                _add_channel(found, value, channels)
    return tuple(sorted(found))


def _add_channel(found: set[int], value: Any, channels: int) -> None:
    try:
        channel = int(value)
    except (TypeError, ValueError):
        return
    if 0 <= channel < int(channels):
        found.add(channel)


def _clip_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(int(low), min(int(high), number))


def _group_sort_key(group_id: str) -> tuple[int, str]:
    try:
        return (int(group_id), group_id)
    except ValueError:
        return (10**9, group_id)


def _parse_anomaly_type(variant_id: str) -> str:
    parts = str(variant_id).split("__")
    return parts[1] if len(parts) >= 2 else "unknown"


def _parse_base_oscillation(variant_id: str) -> str:
    parts = str(variant_id).split("__")
    return parts[0] if parts else "unknown"


def _load_registry_events(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = _load_registry_by_id(path, id_field="event_id")
    return {key: value for key, value in rows.items() if key}


def _load_registry_by_id(path: Path, *, id_field: str) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            row_id = str(payload.get(id_field, ""))
            if row_id:
                rows[row_id] = payload
    return rows


def _metadata_events_by_instance(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("variant_id", "")),
            str(row.get("split", "")),
            str(row.get("instance_id", "")),
        )
        if all(key):
            grouped.setdefault(key, []).append(_registry_event_to_legacy_event(row))
    for values in grouped.values():
        values.sort(key=lambda item: _group_sort_key(str(item.get("group_id", ""))))
    return grouped


def _events_for_instance(
    *,
    metadata_by_instance: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    variant_id: str,
    split: str,
    instance_id: str,
    fallback_path: Path,
) -> Sequence[Mapping[str, Any]]:
    key = (str(variant_id), str(split), str(instance_id))
    rows = metadata_by_instance.get(key)
    if rows is not None:
        return tuple(rows)
    if fallback_path.exists():
        return load_json(fallback_path)
    return ()


def _registry_event_to_legacy_event(row: Mapping[str, Any]) -> dict[str, Any]:
    start = row.get("support_start", row.get("start", 0))
    end = row.get("support_end", row.get("end", start))
    source_start = row.get("source_start", start)
    source_end = row.get("source_end", end)
    primary = _tuple_from_registry_channels(row.get("primary_channels"))
    intervention = _tuple_from_registry_channels(row.get("intervention_channels"))
    context = _tuple_from_registry_channels(row.get("context_channels"))
    group_channels = _tuple_from_registry_channels(row.get("group_channels"))
    channel = primary[0] if primary else (intervention[0] if intervention else 0)
    return {
        **dict(row),
        "start": start,
        "end": end,
        "source_start": source_start,
        "source_end": source_end,
        "channel": channel,
        "operator_target_channels": intervention,
        "intervention_channels": intervention,
        "perturbed_channels": intervention,
        "context_channels": context,
        "group_channels": group_channels,
    }


def _tuple_from_registry_channels(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part for part in value.replace(",", "|").split("|") if part]
        return tuple(int(part) for part in parts)
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return tuple(result)
    try:
        return (int(value),)
    except (TypeError, ValueError):
        return ()
