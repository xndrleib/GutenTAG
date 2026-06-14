"""Generator-side v12 sidecar artifacts.

These helpers publish release-oriented tables that are derived from generated
paired instances. They intentionally avoid creating additional physical dataset
splits; consumers can decide whether to include the sidecar tables in a
capability run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .capabilities.dataset import DatasetIndex, InstanceRecord, discover_dataset, event_uid, load_json
from .io import sanitize_json_value, write_json
from .labels.annotation_channels import CHANNEL_ORDER, annotation_channel_manifest, build_annotation_channels


SIDECAR_VERSION = "synthgen.generator_sidecars.v12.2"
LAW_REPLICATE_VERSION = "synthgen.law_replicates.v12.2"


def write_v12_generator_sidecars(
    dataset_root: Path,
    *,
    annotation_channels: Mapping[str, Any] | None = None,
    law_level_replicates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write generator-side v12 labels and law-level replicate registries.

    Parameters
    ----------
    dataset_root:
        Root of a generated TS dataset.

    Returns
    -------
    dict[str, Any]
        Manifest additions for ``dataset_manifest.json``.
    """

    root = Path(dataset_root).resolve()
    dataset = discover_dataset(root)
    annotation_manifest = write_dataset_annotation_channels(
        dataset,
        config=annotation_channels,
    )
    law_manifest = write_law_level_replicates(
        dataset,
        config=law_level_replicates,
    )
    return {
        "generator_sidecar_version": SIDECAR_VERSION,
        "annotation_channels": annotation_manifest,
        "law_level_replicates": law_manifest,
    }


def write_dataset_annotation_channels(
    dataset: DatasetIndex,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write dataset-level annotation-channel CSV files under ``labels/``."""

    labels_dir = dataset.root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    selected_channels = _selected_annotation_channels(config)
    frames_by_channel: dict[str, list[pd.DataFrame]] = {name: [] for name in selected_channels}
    for instance in dataset.instances:
        events = load_json(instance.events_path)
        channels = build_annotation_channels(
            length=instance.length,
            channels=instance.channels,
            events=events,
        )
        for name in selected_channels:
            channel = channels[name]
            frames_by_channel[name].append(_flatten_label_table(instance, channel.values, channel.columns))

    table_paths: dict[str, str] = {}
    table_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for name in selected_channels:
        frame = _concat_frames(frames_by_channel[name])
        path = labels_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        table_paths[name] = _relative_path(path, dataset.root)
        table_hashes[name] = _file_hash(path)
        row_counts[name] = int(len(frame))

    manifest = annotation_channel_manifest()
    manifest["channels"] = [
        record
        for record in manifest["channels"]
        if str(record.get("name")) in set(selected_channels)
    ]
    manifest.update(
        {
            "generator_sidecar_version": SIDECAR_VERSION,
            "emit": list(selected_channels),
            "table_paths": table_paths,
            "table_hashes": table_hashes,
            "row_counts": row_counts,
            "instance_count": len(dataset.instances),
            "event_group_count": int(sum(len(instance.event_groups) for instance in dataset.instances)),
        }
    )
    manifest_path = labels_dir / "annotation_channel_manifest.json"
    write_json(manifest_path, manifest, sort_keys=True, indent=2)
    manifest["manifest_path"] = _relative_path(manifest_path, dataset.root)
    manifest["manifest_hash"] = _file_hash(manifest_path)
    write_json(manifest_path, manifest, sort_keys=True, indent=2)
    return manifest


def write_law_level_replicates(
    dataset: DatasetIndex,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write law-level replicate records derived from event groups."""

    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", True))
    output_split = str(cfg.get("output_split", "law_replicates"))
    paired_seed_policy = str(cfg.get("paired_seed_policy", "same_base_parameters"))
    replicas_per_genotype = cfg.get("replicas_per_genotype")
    requested_replicas = int(replicas_per_genotype) if replicas_per_genotype is not None else None
    metadata_dir = dataset.root / "metadata"
    law_dir = dataset.root / output_split
    metadata_dir.mkdir(parents=True, exist_ok=True)
    law_dir.mkdir(parents=True, exist_ok=True)
    if not enabled:
        return {
            "law_level_replicates_version": LAW_REPLICATE_VERSION,
            "enabled": False,
            "output_split": output_split,
            "paired_seed_policy": paired_seed_policy,
            "replicas_per_genotype_requested": requested_replicas,
            "replicate_count": 0,
            "genotype_count": 0,
        }
    event_metadata = _read_event_metadata(metadata_dir / "events.jsonl")
    replicate_indices: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for instance in dataset.instances:
        for group in instance.event_groups:
            event_id = event_uid(instance, group)
            metadata = event_metadata.get(event_id, {})
            genotype_id = str(metadata.get("genotype_id", f"genotype:{instance.variant_id}:unknown:v12"))
            replicate_index = replicate_indices.get(genotype_id, 0)
            if requested_replicas is not None and replicate_index >= requested_replicas:
                continue
            replicate_indices[genotype_id] = replicate_index + 1
            records.append(
                {
                    "law_level_replicate_version": LAW_REPLICATE_VERSION,
                    "replicate_id": f"law:{event_id}",
                    "replicate_index": replicate_index,
                    "replicate_source": "generated_paired_event_window",
                    "output_split": output_split,
                    "paired_seed_policy": paired_seed_policy,
                    "event_id": event_id,
                    "genotype_id": genotype_id,
                    "contract_id": metadata.get("contract_id"),
                    "requested_effect_id": metadata.get("requested_effect_id"),
                    "variant_id": instance.variant_id,
                    "base_oscillation": instance.base_oscillation,
                    "anomaly_type": group.anomaly_type,
                    "split": instance.split,
                    "instance_id": instance.instance_id,
                    "group_id": group.group_id,
                    "clean_path": _relative_path(instance.clean_path, dataset.root),
                    "anomalous_path": _relative_path(instance.anomalous_path, dataset.root),
                    "events_path": _relative_path(instance.events_path, dataset.root),
                    "support_start": int(group.start),
                    "support_end": int(group.end),
                    "source_start": int(group.source_start),
                    "source_end": int(group.source_end),
                    "intervention_channels": list(group.intervention_channels),
                    "context_channels": list(group.context_channels),
                    "group_channels": list(group.group_channels),
                    "semantic_scope": group.semantic_scope,
                    "constraint_tag": group.constraint_tag,
                }
            )

    records = sorted(records, key=lambda item: str(item["replicate_id"]))
    registry_path = metadata_dir / "law_level_replicates.jsonl"
    table_path = law_dir / "law_level_replicates.csv"
    _write_jsonl(registry_path, records)
    _write_replicate_csv(table_path, records)
    per_genotype = {
        genotype_id: int(count)
        for genotype_id, count in sorted(replicate_indices.items(), key=lambda item: item[0])
    }
    counts = list(per_genotype.values())
    return {
        "law_level_replicates_version": LAW_REPLICATE_VERSION,
        "enabled": True,
        "output_split": output_split,
        "paired_seed_policy": paired_seed_policy,
        "replicate_source": "generated_paired_event_window",
        "replicas_per_genotype_requested": requested_replicas,
        "replicate_registry_path": _relative_path(registry_path, dataset.root),
        "replicate_table_path": _relative_path(table_path, dataset.root),
        "replicate_registry_hash": _file_hash(registry_path),
        "replicate_table_hash": _file_hash(table_path),
        "replicate_count": len(records),
        "genotype_count": len(per_genotype),
        "min_observed_replicates_per_genotype": min(counts) if counts else 0,
        "max_observed_replicates_per_genotype": max(counts) if counts else 0,
        "replicas_per_genotype_observed": per_genotype,
    }


def _selected_annotation_channels(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    cfg = dict(config or {})
    raw_emit = cfg.get("emit")
    if raw_emit is None:
        return CHANNEL_ORDER
    selected: list[str] = []
    for value in raw_emit:
        for channel in _expand_annotation_channel(str(value)):
            if channel not in selected:
                selected.append(channel)
    if not selected:
        raise ValueError("annotation_channels.emit must select at least one channel")
    return tuple(selected)


def _expand_annotation_channel(name: str) -> tuple[str, ...]:
    normalized = name.strip().replace("-", "_")
    aliases = {
        "oracle": ("labels_oracle_any", "labels_oracle_intervention", "labels_oracle_context"),
        "oracle_any": ("labels_oracle_any",),
        "oracle_context": ("labels_oracle_context",),
        "event_only": ("labels_event_only",),
        "delayed": ("labels_delayed",),
        "weak_point": ("labels_weak_point",),
        "visible_only": ("labels_visible_only",),
        "noisy_boundary": ("labels_noisy_boundary",),
        "censored": ("labels_censored",),
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in CHANNEL_ORDER:
        return (normalized,)
    prefixed = f"labels_{normalized}"
    if prefixed in CHANNEL_ORDER:
        return (prefixed,)
    raise ValueError(f"Unknown annotation channel: {name}")


def _flatten_label_table(
    instance: InstanceRecord,
    values: np.ndarray,
    columns: Sequence[str],
) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=np.int8)
    frame = pd.DataFrame(matrix, columns=list(columns))
    frame.insert(0, "time_index", np.arange(instance.length, dtype=int))
    frame.insert(0, "instance_id", instance.instance_id)
    frame.insert(0, "split", instance.split)
    frame.insert(0, "variant_id", instance.variant_id)
    return frame


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if not frame.empty]
    return pd.concat(materialized, ignore_index=True) if materialized else pd.DataFrame()


def _read_event_metadata(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records[str(payload.get("event_id"))] = payload
    return records


def _write_replicate_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    csv_records: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for key in ("intervention_channels", "context_channels", "group_channels"):
            row[key] = " ".join(str(item) for item in row.get(key, []))
        csv_records.append(row)
    pd.DataFrame(csv_records).to_csv(path, index=False)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    sanitize_json_value(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
