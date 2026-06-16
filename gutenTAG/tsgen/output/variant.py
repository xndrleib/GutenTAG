"""Variant-level output metadata helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from ..io import sanitize_json_value
from .statistics import compute_dataset_statistics


class VariantLike(Protocol):
    """Variant fields required by output metadata."""

    @property
    def variant_id(self) -> str: ...

    @property
    def pair_id(self) -> str: ...

    @property
    def profile_id(self) -> str: ...

    @property
    def base_oscillation(self) -> str: ...

    @property
    def anomaly_type(self) -> str: ...


def build_variant_config(
    *,
    variant: VariantLike,
    base_parameter_template: Mapping[str, Any],
    fixed_base_parameters: Mapping[str, Any] | None,
    base_channel_parameter_template: Mapping[str, Any],
    fixed_base_channel_parameters: Sequence[Mapping[str, Any]] | None,
    effective_base_channel_correlation: Mapping[str, Any],
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Mapping[str, Any] | None,
    length: int,
    channels: int,
    splits: Sequence[str],
    instances_per_split: int,
    split_instance_counts: Mapping[str, Mapping[str, int]],
    density_range: Sequence[float],
    density_tolerance: float,
    segment_count_range: Sequence[int],
    placement_policy: str,
    channel_policy: str,
    effective_overlap_policy: str,
    variant_segment_planner: Mapping[str, Any],
    variant_anomaly_policy: Mapping[str, Any],
    length_normalization: str,
    base_parameter_policy: str,
    base_channel_parameter_policy: str,
    anomaly_parameter_policy: str,
    split_phase_shift: Mapping[str, Any],
) -> dict[str, Any]:
    """Build JSON-safe variant configuration metadata."""
    return sanitize_json_value(
        {
            "variant_id": variant.variant_id,
            "profile_id": variant.profile_id,
            "base_oscillation": {
                "kind": variant.base_oscillation,
                "parameter_template": base_parameter_template,
                "realized_parameters": fixed_base_parameters,
                "channel_parameter_template": base_channel_parameter_template,
                "channel_realized_parameters": fixed_base_channel_parameters,
                "channel_correlation": effective_base_channel_correlation,
                "split_phase_shift": split_phase_shift,
            },
            "anomaly_type": {
                "kind": variant.anomaly_type,
                "parameter_template": anomaly_parameter_template,
                "realized_parameters": fixed_anomaly_parameters,
            },
            "dataset": {
                "length": int(length),
                "channels": int(channels),
                "splits": list(splits),
                "instances_per_split": int(instances_per_split),
                "split_instance_counts": split_instance_counts,
            },
            "anomaly_policy": {
                "density_range": list(density_range),
                "density_tolerance": float(density_tolerance),
                "segment_count_range": list(segment_count_range),
                "placement_policy": placement_policy,
                "channel_policy": channel_policy,
                "overlap_policy": effective_overlap_policy,
                "segment_planner": variant_segment_planner,
                "variant_override": variant_anomaly_policy,
                "length_normalization": length_normalization,
            },
            "parameter_policies": {
                "base_parameter_policy": base_parameter_policy,
                "base_channel_parameter_policy": base_channel_parameter_policy,
                "anomaly_parameter_policy": anomaly_parameter_policy,
            },
            "split_phase_shift": split_phase_shift,
        }
    )


def write_variant_config(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``variant_config.yaml`` with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            sanitize_json_value(payload),
            handle,
            sort_keys=True,
            allow_unicode=False,
        )


def build_split_entry(
    *,
    split: str,
    instances: int,
    paired_instances: int,
    clean_only_instances: int,
) -> dict[str, Any]:
    """Build a variant manifest entry for one split."""
    return {
        "split": split,
        "instances": int(instances),
        "paired_instances": int(paired_instances),
        "clean_only_instances": int(clean_only_instances),
        "summary_file": str((Path(split) / "split_summary.json").as_posix()),
    }


def build_variant_manifest(
    *,
    variant: VariantLike,
    carrier_family: str,
    split_entries: Sequence[Mapping[str, Any]],
    instance_summaries: Sequence[Mapping[str, Any]],
    base_parameter_policy: str,
    base_channel_parameter_policy: str,
    anomaly_parameter_policy: str,
    split_phase_shift: Mapping[str, Any],
    effective_overlap_policy: str,
    variant_segment_planner: Mapping[str, Any],
    variant_anomaly_policy: Mapping[str, Any],
    length_normalization: str,
) -> dict[str, Any]:
    """Build JSON-safe variant manifest metadata."""
    return sanitize_json_value(
        {
            "variant_id": variant.variant_id,
            "pair_id": variant.pair_id,
            "profile_id": variant.profile_id,
            "base_oscillation": variant.base_oscillation,
            "carrier_family": carrier_family,
            "anomaly_type": variant.anomaly_type,
            "splits": list(split_entries),
            "instance_count": len(instance_summaries),
            "parameter_policies": {
                "base_parameter_policy": base_parameter_policy,
                "base_channel_parameter_policy": base_channel_parameter_policy,
                "anomaly_parameter_policy": anomaly_parameter_policy,
            },
            "split_phase_shift": split_phase_shift,
            "overlap_policy": effective_overlap_policy,
            "segment_planner": variant_segment_planner,
            "variant_anomaly_policy": variant_anomaly_policy,
            "length_normalization": length_normalization,
            "statistics": compute_dataset_statistics(instance_summaries),
        }
    )
