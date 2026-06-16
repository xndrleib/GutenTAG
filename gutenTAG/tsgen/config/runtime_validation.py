"""Runtime config validation helpers for ``TSGeneratorConfig``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np

from ..environment import validate_release_backend_policy
from .registry import available_base_names, validate_name_selections
from .runtime import (
    parse_pair as _parse_pair,
    validate_base_channel_correlation_mapping as _validate_base_channel_correlation_mapping,
)

if TYPE_CHECKING:
    from .runtime_model import TSGeneratorConfig


_CHANNEL_POLICIES = ("single-random", "paired-random", "all-channels")
_LENGTH_NORMALIZATION_POLICIES = ("none", "crop", "pad", "resample")
_SEGMENT_PLANNERS = (
    "uniform_segments",
    "point_events_from_density",
    "period_locked_frequency",
    "energy_aware_segments",
    "trend_parameter_aware_segments",
    "mode_boundary_segments",
    "mode_grid_segments",
    "fixed_first_onset_segments",
)


def validate_runtime_config(config: TSGeneratorConfig) -> None:
    """Validate runtime config values and cross-field invariants."""

    _validate_dataset_shape(config)
    _validate_split_counts(config)
    _validate_density_and_segments(config)
    _validate_anomaly_placement(config)
    _validate_variant_selection(config)
    _validate_parameter_policies(config)
    _validate_split_phase_shift(config)
    _validate_runtime_output_policy(config)
    _validate_plot_policy(config)
    _validate_segment_constraints(config)
    _validate_segment_planner(config)
    _validate_special_anomaly_policies(config)
    _validate_label_policy(config)
    _validate_sidecar_config(config)


def _validate_dataset_shape(config: TSGeneratorConfig) -> None:
    if config.length <= 0:
        raise ValueError("dataset.length must be > 0")
    if config.channels <= 0:
        raise ValueError("dataset.channels must be > 0")
    if config.instances_per_split <= 0:
        raise ValueError("dataset.instances_per_split must be > 0")
    if len(config.splits) == 0:
        raise ValueError("dataset.splits must contain at least one split name")
    if len(set(config.splits)) != len(config.splits):
        raise ValueError("dataset.splits must not contain duplicate split names")


def _validate_split_counts(config: TSGeneratorConfig) -> None:
    if not config.split_instance_counts:
        config.split_instance_counts = {
            split: {
                "paired_instances_per_variant": int(config.instances_per_split),
                "clean_only_instances_per_variant": 0,
            }
            for split in config.splits
        }

    for split in config.splits:
        if split not in config.split_instance_counts:
            raise ValueError(
                f"dataset.splits is missing count config for split {split!r}"
            )
        counts = config.split_instance_counts[split]
        paired_count = int(counts.get("paired_instances_per_variant", 0))
        clean_only_count = int(counts.get("clean_only_instances_per_variant", 0))
        if paired_count < 0 or clean_only_count < 0:
            raise ValueError("split instance counts must be >= 0")
        if paired_count + clean_only_count <= 0:
            raise ValueError(f"split {split!r} must request at least one instance")


def _validate_density_and_segments(config: TSGeneratorConfig) -> None:
    if config.density_range[0] <= 0 or config.density_range[1] >= 1:
        raise ValueError("anomaly density_range must stay inside (0, 1)")
    if config.density_range[0] > config.density_range[1]:
        raise ValueError("density_range must have min <= max")
    if config.density_tolerance < 0:
        raise ValueError("density_tolerance must be >= 0")
    if config.segment_count_range[0] <= 0 or config.segment_count_range[1] <= 0:
        raise ValueError("segment_count_range values must be > 0")
    if config.segment_count_range[0] > config.segment_count_range[1]:
        raise ValueError("segment_count_range must have min <= max")
    if config.segment_count_range[1] > config.length:
        raise ValueError(
            "segment_count_range max must be <= dataset.length so every segment "
            "can have at least one point."
        )


def _validate_anomaly_placement(config: TSGeneratorConfig) -> None:
    if config.placement_policy != "uniform":
        raise ValueError("Only placement_policy='uniform' is supported")
    if config.channel_policy not in _CHANNEL_POLICIES:
        raise ValueError(
            "channel_policy must be one of {'single-random','paired-random','all-channels'}"
        )
    if config.channel_policy == "paired-random" and config.channels < 2:
        raise ValueError(
            "channel_policy='paired-random' requires dataset.channels >= 2"
        )
    if config.overlap_policy not in ("global", "per_channel"):
        raise ValueError("overlap_policy must be one of {'global','per_channel'}")


def _validate_variant_selection(config: TSGeneratorConfig) -> None:
    if config.compatibility_mode not in ("hard", "recommended", "validated"):
        raise ValueError(
            "compatibility_mode must be one of {'hard','recommended','validated'}"
        )
    if config.profiles_per_pair <= 0:
        raise ValueError("profiles_per_pair must be > 0")
    validate_name_selections(
        base_oscillations=config.base_oscillations,
        anomaly_types=config.anomaly_types,
        skip_base_oscillations=config.skip_base_oscillations,
        skip_anomaly_types=config.skip_anomaly_types,
        disabled_anomaly_types=config.disabled_anomaly_types,
        pair_profiles=config.pair_profiles,
    )
    validate_release_backend_policy(
        base_oscillations=config.base_oscillations,
        all_base_oscillations=available_base_names(),
    )


def _validate_parameter_policies(config: TSGeneratorConfig) -> None:
    if config.base_parameter_policy not in (
        "fixed_per_variant",
        "random_per_instance",
    ):
        raise ValueError(
            "base_parameter_policy must be one of "
            "{'fixed_per_variant','random_per_instance'}"
        )
    if config.anomaly_parameter_policy not in (
        "fixed_per_variant",
        "random_per_instance",
        "random_per_segment",
    ):
        raise ValueError(
            "anomaly_parameter_policy must be one of "
            "{'fixed_per_variant','random_per_instance','random_per_segment'}"
        )
    if config.base_channel_parameter_policy not in (
        "fixed_per_variant",
        "random_per_instance",
    ):
        raise ValueError(
            "base_channel_parameter_policy must be one of "
            "{'fixed_per_variant','random_per_instance'}"
        )
    if not isinstance(config.base_channel_correlation, Mapping):
        raise ValueError("base_channel_correlation must be a mapping")
    _validate_base_channel_correlation_mapping(config.base_channel_correlation)


def _validate_split_phase_shift(config: TSGeneratorConfig) -> None:
    if not isinstance(config.split_phase_shift, Mapping):
        raise ValueError("split_phase_shift must be a mapping")
    split_phase_cfg = dict(config.split_phase_shift)
    split_phase_enabled = bool(split_phase_cfg.get("enabled", False))
    split_phase_mode = str(split_phase_cfg.get("mode", "fixed_map"))
    split_phase_modulo = float(split_phase_cfg.get("phase_modulo", float(2.0 * np.pi)))
    if split_phase_modulo <= 0.0:
        raise ValueError("split_phase_shift.phase_modulo must be > 0")
    if not split_phase_enabled:
        return
    if split_phase_mode != "fixed_map":
        raise ValueError(
            "split_phase_shift.mode must be 'fixed_map' when "
            "split_phase_shift.enabled=true"
        )
    values_cfg = split_phase_cfg.get("values", {})
    if not isinstance(values_cfg, Mapping):
        raise ValueError("split_phase_shift.values must be a mapping")
    missing_splits = [split for split in config.splits if split not in values_cfg]
    if missing_splits:
        raise ValueError(
            f"split_phase_shift.values is missing offsets for splits: {missing_splits}"
        )
    _validate_split_phase_values(values_cfg)


def _validate_split_phase_values(values_cfg: Mapping[str, object]) -> None:
    for split_name, offset_value in values_cfg.items():
        try:
            float(cast(Any, offset_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"split_phase_shift.values[{split_name}] must be numeric"
            ) from exc


def _validate_runtime_output_policy(config: TSGeneratorConfig) -> None:
    if config.length_normalization not in _LENGTH_NORMALIZATION_POLICIES:
        raise ValueError(
            "length_normalization must be one of {'none','crop','pad','resample'}"
        )
    if config.on_variant_failure not in ("skip", "fail_fast"):
        raise ValueError("on_variant_failure must be one of {'skip','fail_fast'}")


def _validate_plot_policy(config: TSGeneratorConfig) -> None:
    if config.zoom_count <= 0:
        raise ValueError("zoom_count must be > 0")
    if config.zoom_margin < 0 or config.zoom_margin_min < 0:
        raise ValueError("zoom_margin and zoom_margin_min must be >= 0")
    if config.zoom_margin_alpha < 0:
        raise ValueError("zoom_margin_alpha must be >= 0")
    if config.zoom_fill_policy not in ("repeat", "blank"):
        raise ValueError("zoom_fill_policy must be one of {'repeat','blank'}")


def _validate_segment_constraints(config: TSGeneratorConfig) -> None:
    for anomaly_type, min_length in config.min_segment_length_by_anomaly.items():
        if min_length <= 0:
            raise ValueError(
                f"min_segment_length_by_anomaly[{anomaly_type}] must be > 0"
            )


def _validate_segment_planner(config: TSGeneratorConfig) -> None:
    for planner_key, planner_cfg in config.segment_planner.items():
        if not isinstance(planner_cfg, Mapping):
            raise ValueError(f"segment_planner[{planner_key}] must be a mapping")
        planner_name = str(planner_cfg.get("planner", "uniform_segments"))
        if planner_name not in _SEGMENT_PLANNERS:
            raise ValueError(
                "segment_planner planner must be one of "
                "{'uniform_segments','point_events_from_density','period_locked_frequency',"
                "'energy_aware_segments','trend_parameter_aware_segments',"
                "'mode_boundary_segments','mode_grid_segments',"
                "'fixed_first_onset_segments'}"
            )
        if "density_range" in planner_cfg:
            _parse_pair(planner_cfg["density_range"], "segment_planner.density_range")


def _validate_special_anomaly_policies(config: TSGeneratorConfig) -> None:
    for anomaly_type, raw in config.special_anomaly_policies.items():
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"special_anomaly_policies[{anomaly_type}] must be a mapping"
            )
        special_channel_policy = raw.get("channel_policy")
        if special_channel_policy is None:
            continue
        special_channel_policy = str(special_channel_policy)
        if special_channel_policy not in _CHANNEL_POLICIES:
            raise ValueError(
                f"special_anomaly_policies[{anomaly_type}].channel_policy must be one of "
                "{'single-random','paired-random','all-channels'}"
            )
        if special_channel_policy == "paired-random" and config.channels < 2:
            raise ValueError(
                f"special_anomaly_policies[{anomaly_type}].channel_policy='paired-random' "
                "requires dataset.channels >= 2"
            )


def _validate_label_policy(config: TSGeneratorConfig) -> None:
    if config.support_label_mode not in ("strict_segment", "effective_support"):
        raise ValueError(
            "support_label_mode must be one of {'strict_segment','effective_support'}"
        )
    if config.support_eps_mode not in ("relative", "absolute"):
        raise ValueError("support_eps_mode must be one of {'relative','absolute'}")
    if config.support_eps_value < 0:
        raise ValueError("support_eps_value must be >= 0")
    if config.min_effective_label_length_non_extremum <= 0:
        raise ValueError("min_effective_label_length_non_extremum must be >= 1")
    if config.max_placement_attempts <= 0:
        raise ValueError("max_placement_attempts must be > 0")


def _validate_sidecar_config(config: TSGeneratorConfig) -> None:
    if not isinstance(config.annotation_channels, Mapping):
        raise ValueError("annotation_channels must be a mapping")
    raw_emit = config.annotation_channels.get("emit")
    if raw_emit is not None and not isinstance(raw_emit, (list, tuple)):
        raise ValueError("annotation_channels.emit must be a list")
    if not isinstance(config.law_level_replicates, Mapping):
        raise ValueError("law_level_replicates must be a mapping")
    law_enabled = config.law_level_replicates.get("enabled")
    if law_enabled is not None and not isinstance(law_enabled, bool):
        raise ValueError("law_level_replicates.enabled must be boolean")
    replicas_per_genotype = config.law_level_replicates.get("replicas_per_genotype")
    if replicas_per_genotype is not None and int(replicas_per_genotype) <= 0:
        raise ValueError("law_level_replicates.replicas_per_genotype must be > 0")
    paired_seed_policy = str(
        config.law_level_replicates.get(
            "paired_seed_policy",
            "same_base_parameters",
        )
    )
    if paired_seed_policy != "same_base_parameters":
        raise ValueError(
            "law_level_replicates.paired_seed_policy must be 'same_base_parameters'"
        )
    output_split = str(
        config.law_level_replicates.get("output_split", "law_replicates")
    )
    if not output_split:
        raise ValueError("law_level_replicates.output_split must be non-empty")


__all__ = ["validate_runtime_config"]
