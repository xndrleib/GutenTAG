"""Strict raw TS dataset config model validation.

The historical generator still parses into `TSGeneratorConfig`; this module is
the v11 boundary that rejects unknown keys before defaults and legacy aliases are
merged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratorSection(_StrictModel):
    output_root: Optional[str] = None
    master_seed: Optional[int] = None
    overwrite_output: Optional[bool] = None
    log_level: Optional[str] = None
    on_variant_failure: Optional[str] = None
    allow_empty_dataset: Optional[bool] = None


class DatasetSection(_StrictModel):
    length: Optional[int] = None
    channels: Optional[int] = None
    splits: Optional[Any] = None
    instances_per_split: Optional[int] = None


class VariantsSection(_StrictModel):
    base_oscillations: Optional[List[str]] = None
    anomaly_types: Optional[List[str]] = None
    compatibility_mode: Optional[str] = None
    skip_base_oscillations: Optional[List[str]] = None
    skip_anomaly_types: Optional[List[str]] = None
    disabled_anomaly_types: Optional[List[str]] = None
    profiles_per_pair: Optional[int] = None
    pair_profiles: Optional[Dict[str, List[str]]] = None
    base_parameter_policy: Optional[str] = None
    base_channel_parameter_policy: Optional[str] = None
    base_channel_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    base_channel_correlation: Optional[Dict[str, Any]] = None
    split_phase_shift: Optional[Dict[str, Any]] = None
    anomaly_parameter_policy: Optional[str] = None
    base_oscillation_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    anomaly_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    variant_overrides: Optional[Dict[str, Dict[str, Any]]] = None


class AnomalyPolicySection(_StrictModel):
    density_range: Optional[List[float]] = None
    density_tolerance: Optional[float] = None
    segment_count_range: Optional[List[int]] = None
    placement_policy: Optional[str] = None
    channel_policy: Optional[str] = None
    overlap_policy: Optional[str] = None
    length_normalization: Optional[str] = None
    max_placement_attempts: Optional[int] = None
    skip_density_incompatible_variants: Optional[bool] = None
    special_anomaly_policies: Optional[Dict[str, Dict[str, Any]]] = None
    segment_planner: Optional[Dict[str, Dict[str, Any]]] = None
    min_segment_length_by_anomaly: Optional[Dict[str, int]] = None
    support_label_mode: Optional[str] = None
    support_eps_mode: Optional[str] = None
    support_eps_value: Optional[float] = None
    min_effective_label_length_non_extremum: Optional[int] = None


class PlotSection(_StrictModel):
    enabled: Optional[bool] = None
    dpi: Optional[int] = None
    zoom_count: Optional[int] = None
    zoom_margin: Optional[int] = None
    zoom_margin_min: Optional[int] = None
    zoom_margin_alpha: Optional[float] = None
    zoom_fill_policy: Optional[str] = None


class IOSection(_StrictModel):
    csv_float_format: Optional[str] = None


class AnnotationChannelsSection(_StrictModel):
    emit: Optional[List[str]] = None


class LawLevelReplicatesSection(_StrictModel):
    enabled: Optional[bool] = None
    replicas_per_genotype: Optional[int] = None
    paired_seed_policy: Optional[str] = None
    output_split: Optional[str] = None


class TSRawConfig(_StrictModel):
    dataset_version: Optional[str] = None
    generator: Optional[GeneratorSection] = None
    dataset: Optional[DatasetSection] = None
    variants: Optional[VariantsSection] = None
    anomaly_policy: Optional[AnomalyPolicySection] = None
    plot: Optional[PlotSection] = None
    io: Optional[IOSection] = None
    annotation_channels: Optional[AnnotationChannelsSection] = None
    law_level_replicates: Optional[LawLevelReplicatesSection] = None

    # Legacy top-level aliases still accepted by `TSGeneratorConfig.from_dict`.
    output_root: Optional[str] = None
    master_seed: Optional[int] = None
    overwrite_output: Optional[bool] = None
    log_level: Optional[str] = None
    on_variant_failure: Optional[str] = None
    allow_empty_dataset: Optional[bool] = None
    length: Optional[int] = None
    channels: Optional[int] = None
    splits: Optional[Any] = None
    instances_per_split: Optional[int] = None
    density_range: Optional[List[float]] = None
    density_tolerance: Optional[float] = None
    segment_count_range: Optional[List[int]] = None
    placement_policy: Optional[str] = None
    channel_policy: Optional[str] = None
    overlap_policy: Optional[str] = None
    length_normalization: Optional[str] = None
    max_placement_attempts: Optional[int] = None
    skip_density_incompatible_variants: Optional[bool] = None
    base_oscillations: Optional[List[str]] = None
    anomaly_types: Optional[List[str]] = None
    compatibility_mode: Optional[str] = None
    skip_base_oscillations: Optional[List[str]] = None
    skip_anomaly_types: Optional[List[str]] = None
    disabled_anomaly_types: Optional[List[str]] = None
    profiles_per_pair: Optional[int] = None
    pair_profiles: Optional[Dict[str, List[str]]] = None
    base_parameter_policy: Optional[str] = None
    base_channel_parameter_policy: Optional[str] = None
    base_channel_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    base_channel_correlation: Optional[Dict[str, Any]] = None
    split_phase_shift: Optional[Dict[str, Any]] = None
    anomaly_parameter_policy: Optional[str] = None
    base_oscillation_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    anomaly_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    variant_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    special_anomaly_policies: Optional[Dict[str, Dict[str, Any]]] = None
    segment_planner: Optional[Dict[str, Dict[str, Any]]] = None
    min_segment_length_by_anomaly: Optional[Dict[str, int]] = None
    support_label_mode: Optional[str] = None
    support_eps_mode: Optional[str] = None
    support_eps_value: Optional[float] = None
    min_effective_label_length_non_extremum: Optional[int] = None
    generate_plots: Optional[bool] = None
    plot_dpi: Optional[int] = None
    zoom_count: Optional[int] = None
    zoom_margin: Optional[int] = None
    zoom_margin_min: Optional[int] = None
    zoom_margin_alpha: Optional[float] = None
    zoom_fill_policy: Optional[str] = None
    csv_float_format: Optional[str] = None


def validate_raw_ts_config(raw: dict[str, Any]) -> None:
    """Validate unknown-key policy for TS dataset configs.

    Legacy GutenTAG configs such as `{"timeseries": [...]}` are intentionally
    ignored here; they are parsed by the upstream generator stack, not by the TS
    dataset generator.
    """
    ts_section_keys = {"generator", "dataset", "variants", "anomaly_policy"}
    if not any(key in raw for key in ts_section_keys):
        return
    TSRawConfig.model_validate(raw)
