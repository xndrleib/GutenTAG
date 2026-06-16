"""Configuration validation and registry helpers for TS dataset generation."""

from .defaults import (
    ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY,
    BASES_REQUIRING_EXTRA_CONFIG,
    DEFAULT_ANOMALY_OVERRIDES,
    DEFAULT_BASE_OVERRIDES,
    DEFAULT_DENSITY_RANGE,
    DEFAULT_PROFILES_PER_PAIR,
    DEFAULT_SEGMENT_COUNT_RANGE,
    DEFAULT_SPLITS,
)
from .models import validate_raw_ts_config
from .registry import (
    available_anomaly_names,
    available_base_names,
    resolve_anomaly_names,
    resolve_base_names,
    validate_name_selections,
)
from .runtime_model import TSGeneratorConfig
from .runtime import (
    merge_dicts,
    optional_str_list,
    parse_pair,
    parse_pair_int,
    parse_pair_profiles,
    parse_segment_planner,
    parse_split_instance_counts,
    read_nested_dict,
    validate_base_channel_correlation_mapping,
)

__all__ = [
    "ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY",
    "BASES_REQUIRING_EXTRA_CONFIG",
    "DEFAULT_ANOMALY_OVERRIDES",
    "DEFAULT_BASE_OVERRIDES",
    "DEFAULT_DENSITY_RANGE",
    "DEFAULT_PROFILES_PER_PAIR",
    "DEFAULT_SEGMENT_COUNT_RANGE",
    "DEFAULT_SPLITS",
    "TSGeneratorConfig",
    "available_anomaly_names",
    "available_base_names",
    "merge_dicts",
    "optional_str_list",
    "parse_pair",
    "parse_pair_int",
    "parse_pair_profiles",
    "parse_segment_planner",
    "parse_split_instance_counts",
    "read_nested_dict",
    "resolve_anomaly_names",
    "resolve_base_names",
    "validate_base_channel_correlation_mapping",
    "validate_name_selections",
    "validate_raw_ts_config",
]
