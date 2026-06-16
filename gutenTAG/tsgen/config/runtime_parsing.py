"""Runtime config parsing helpers for ``TSGeneratorConfig``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .defaults import (
    DEFAULT_ANOMALY_OVERRIDES,
    DEFAULT_BASE_OVERRIDES,
    DEFAULT_DENSITY_RANGE,
    DEFAULT_PROFILES_PER_PAIR,
    DEFAULT_SEGMENT_COUNT_RANGE,
    DEFAULT_SPLITS,
)
from .runtime import (
    merge_dicts as _merge_dicts,
    optional_str_list as _optional_str_list,
    parse_pair as _parse_pair,
    parse_pair_profiles as _parse_pair_profiles,
    parse_segment_planner as _parse_segment_planner,
    parse_split_instance_counts as _parse_split_instance_counts,
    read_nested_dict as _read_nested_dict,
)


@dataclass(frozen=True)
class RuntimeConfigSections:
    """Nested runtime config sections with malformed sections normalized away."""

    generator: Mapping[str, Any]
    dataset: Mapping[str, Any]
    variants: Mapping[str, Any]
    anomaly_policy: Mapping[str, Any]
    plot: Mapping[str, Any]
    io: Mapping[str, Any]
    annotation_channels: Mapping[str, Any]
    law_level_replicates: Mapping[str, Any]


def build_runtime_config_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build constructor kwargs for ``TSGeneratorConfig``."""

    sections = read_runtime_config_sections(config)
    split_counts, legacy_instances = runtime_split_counts(config, sections)
    kwargs: dict[str, Any] = {}
    kwargs.update(generator_config_kwargs(config, sections))
    kwargs.update(
        dataset_config_kwargs(
            config,
            sections,
            split_counts=split_counts,
            legacy_instances_per_split=legacy_instances,
        )
    )
    kwargs.update(anomaly_policy_config_kwargs(config, sections))
    kwargs.update(variant_config_kwargs(config, sections))
    kwargs.update(plot_config_kwargs(config, sections))
    kwargs.update(io_config_kwargs(config, sections))
    kwargs.update(annotation_config_kwargs(sections))
    return kwargs


def read_runtime_config_sections(config: Mapping[str, Any]) -> RuntimeConfigSections:
    """Read recognized nested config sections."""

    return RuntimeConfigSections(
        generator=_read_nested_dict(config, "generator"),
        dataset=_read_nested_dict(config, "dataset"),
        variants=_read_nested_dict(config, "variants"),
        anomaly_policy=_read_nested_dict(config, "anomaly_policy"),
        plot=_read_nested_dict(config, "plot"),
        io=_read_nested_dict(config, "io"),
        annotation_channels=_read_nested_dict(config, "annotation_channels"),
        law_level_replicates=_read_nested_dict(config, "law_level_replicates"),
    )


def runtime_split_counts(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
) -> tuple[Dict[str, Dict[str, int]], int]:
    """Parse split names and paired/clean-only instance counts."""

    raw_splits = sections.dataset.get(
        "splits",
        config.get("splits", list(DEFAULT_SPLITS)),
    )
    legacy_instances = int(
        sections.dataset.get(
            "instances_per_split",
            config.get("instances_per_split", 5),
        )
    )
    return _parse_split_instance_counts(raw_splits, legacy_instances), legacy_instances


def generator_config_kwargs(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
) -> dict[str, Any]:
    """Build generator-level runtime kwargs."""

    generator = sections.generator
    return {
        "output_root": Path(
            generator.get("output_root", config.get("output_root", "data/ts_dataset"))
        ),
        "master_seed": int(generator.get("master_seed", config.get("master_seed", 42))),
        "overwrite_output": bool(
            generator.get("overwrite_output", config.get("overwrite_output", True))
        ),
        "allow_empty_dataset": bool(
            generator.get(
                "allow_empty_dataset",
                config.get("allow_empty_dataset", False),
            )
        ),
        "log_level": str(
            generator.get("log_level", config.get("log_level", "INFO"))
        ).upper(),
        "on_variant_failure": str(
            generator.get(
                "on_variant_failure",
                config.get("on_variant_failure", "skip"),
            )
        ),
    }


def dataset_config_kwargs(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
    *,
    split_counts: Dict[str, Dict[str, int]],
    legacy_instances_per_split: int,
) -> dict[str, Any]:
    """Build dataset-level runtime kwargs."""

    dataset = sections.dataset
    return {
        "dataset_version": str(config.get("dataset_version", "ts_dataset_v12")),
        "length": int(dataset.get("length", config.get("length", 10_000))),
        "channels": int(dataset.get("channels", config.get("channels", 5))),
        "splits": tuple(split_counts.keys()),
        "instances_per_split": legacy_instances_per_split,
        "split_instance_counts": split_counts,
    }


def anomaly_policy_config_kwargs(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
) -> dict[str, Any]:
    """Build anomaly-policy runtime kwargs."""

    anomaly = sections.anomaly_policy
    density_range, segment_count_range = _density_and_segment_ranges(config, anomaly)
    return {
        "density_range": (float(density_range[0]), float(density_range[1])),
        "density_tolerance": float(
            anomaly.get("density_tolerance", config.get("density_tolerance", 0.002))
        ),
        "segment_count_range": (
            int(segment_count_range[0]),
            int(segment_count_range[1]),
        ),
        "placement_policy": str(
            anomaly.get("placement_policy", config.get("placement_policy", "uniform"))
        ),
        "channel_policy": str(
            anomaly.get("channel_policy", config.get("channel_policy", "single-random"))
        ),
        "overlap_policy": str(
            anomaly.get("overlap_policy", config.get("overlap_policy", "global"))
        ),
        "special_anomaly_policies": dict(
            anomaly.get(
                "special_anomaly_policies",
                config.get("special_anomaly_policies", {}),
            )
        ),
        "segment_planner": _parse_segment_planner(
            anomaly.get("segment_planner", config.get("segment_planner", {}))
        ),
        "min_segment_length_by_anomaly": _min_segment_length_by_anomaly(
            config,
            anomaly,
        ),
        "length_normalization": str(
            anomaly.get(
                "length_normalization",
                config.get("length_normalization", "resample"),
            )
        ),
        "max_placement_attempts": int(
            anomaly.get(
                "max_placement_attempts",
                config.get("max_placement_attempts", 2_500),
            )
        ),
        "skip_density_incompatible_variants": bool(
            anomaly.get(
                "skip_density_incompatible_variants",
                config.get("skip_density_incompatible_variants", True),
            )
        ),
        "support_label_mode": str(
            anomaly.get(
                "support_label_mode",
                config.get("support_label_mode", "strict_segment"),
            )
        ),
        "support_eps_mode": str(
            anomaly.get("support_eps_mode", config.get("support_eps_mode", "relative"))
        ),
        "support_eps_value": float(
            anomaly.get("support_eps_value", config.get("support_eps_value", 0.05))
        ),
        "min_effective_label_length_non_extremum": int(
            anomaly.get(
                "min_effective_label_length_non_extremum",
                config.get("min_effective_label_length_non_extremum", 1),
            )
        ),
    }


def variant_config_kwargs(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
) -> dict[str, Any]:
    """Build variant-matrix and parameter-policy runtime kwargs."""

    variants = sections.variants
    kwargs: dict[str, Any] = {}
    kwargs.update(_variant_selection_kwargs(config, variants))
    kwargs.update(_variant_parameter_policy_kwargs(config, variants))
    kwargs.update(_variant_override_kwargs(config, variants))
    return kwargs


def _variant_selection_kwargs(
    config: Mapping[str, Any],
    variants: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "base_oscillations": _optional_list_or_none(
            variants.get("base_oscillations", config.get("base_oscillations", []))
        ),
        "anomaly_types": _optional_list_or_none(
            variants.get("anomaly_types", config.get("anomaly_types", []))
        ),
        "compatibility_mode": str(
            variants.get("compatibility_mode", config.get("compatibility_mode", "hard"))
        ),
        "skip_base_oscillations": _optional_str_list(
            variants.get(
                "skip_base_oscillations",
                config.get("skip_base_oscillations", []),
            ),
        ),
        "skip_anomaly_types": _optional_str_list(
            variants.get("skip_anomaly_types", config.get("skip_anomaly_types", [])),
        ),
        "disabled_anomaly_types": _optional_str_list(
            variants.get(
                "disabled_anomaly_types",
                config.get("disabled_anomaly_types", []),
            )
        ),
        "profiles_per_pair": int(
            variants.get(
                "profiles_per_pair",
                config.get("profiles_per_pair", DEFAULT_PROFILES_PER_PAIR),
            )
        ),
        "pair_profiles": _parse_pair_profiles(
            variants.get("pair_profiles", config.get("pair_profiles", {}))
        ),
    }


def _variant_parameter_policy_kwargs(
    config: Mapping[str, Any],
    variants: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "base_parameter_policy": str(
            variants.get(
                "base_parameter_policy",
                config.get("base_parameter_policy", "random_per_instance"),
            )
        ),
        "base_channel_parameter_policy": str(
            variants.get(
                "base_channel_parameter_policy",
                config.get("base_channel_parameter_policy", "random_per_instance"),
            )
        ),
        "anomaly_parameter_policy": str(
            variants.get(
                "anomaly_parameter_policy",
                config.get("anomaly_parameter_policy", "fixed_per_variant"),
            )
        ),
    }


def _variant_override_kwargs(
    config: Mapping[str, Any],
    variants: Mapping[str, Any],
) -> dict[str, Any]:
    base_channel_correlation = _mapping_or_empty(
        variants.get(
            "base_channel_correlation",
            config.get("base_channel_correlation", {}),
        )
    )
    split_phase_shift = _mapping_or_empty(
        variants.get("split_phase_shift", config.get("split_phase_shift", {}))
    )
    return {
        "base_channel_overrides": _merge_dicts(
            config.get("base_channel_overrides", {}),
            variants.get("base_channel_overrides", {}),
        ),
        "base_channel_correlation": copy.deepcopy(dict(base_channel_correlation)),
        "split_phase_shift": copy.deepcopy(dict(split_phase_shift)),
        "base_oscillation_overrides": _merge_dicts(
            DEFAULT_BASE_OVERRIDES,
            config.get("base_oscillation_overrides", {}),
            variants.get("base_oscillation_overrides", {}),
        ),
        "anomaly_overrides": _merge_dicts(
            DEFAULT_ANOMALY_OVERRIDES,
            config.get("anomaly_overrides", {}),
            variants.get("anomaly_overrides", {}),
        ),
        "variant_overrides": dict(
            variants.get("variant_overrides", config.get("variant_overrides", {}))
        ),
    }


def plot_config_kwargs(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
) -> dict[str, Any]:
    """Build plotting runtime kwargs."""

    plot = sections.plot
    return {
        "generate_plots": bool(plot.get("enabled", config.get("generate_plots", True))),
        "plot_dpi": int(plot.get("dpi", config.get("plot_dpi", 140))),
        "zoom_count": int(plot.get("zoom_count", config.get("zoom_count", 5))),
        "zoom_margin": int(plot.get("zoom_margin", config.get("zoom_margin", 100))),
        "zoom_margin_min": int(
            plot.get("zoom_margin_min", config.get("zoom_margin_min", 100))
        ),
        "zoom_margin_alpha": float(
            plot.get("zoom_margin_alpha", config.get("zoom_margin_alpha", 0.0))
        ),
        "zoom_fill_policy": str(
            plot.get("zoom_fill_policy", config.get("zoom_fill_policy", "repeat"))
        ),
    }


def io_config_kwargs(
    config: Mapping[str, Any],
    sections: RuntimeConfigSections,
) -> dict[str, Any]:
    """Build IO runtime kwargs."""

    return {
        "csv_float_format": str(
            sections.io.get(
                "csv_float_format",
                config.get("csv_float_format", "%.8f"),
            )
        )
    }


def annotation_config_kwargs(sections: RuntimeConfigSections) -> dict[str, Any]:
    """Build sidecar/annotation runtime kwargs."""

    return {
        "annotation_channels": copy.deepcopy(dict(sections.annotation_channels)),
        "law_level_replicates": copy.deepcopy(dict(sections.law_level_replicates)),
    }


def _density_and_segment_ranges(
    config: Mapping[str, Any],
    anomaly: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    density_range = _parse_pair(
        anomaly.get(
            "density_range", config.get("density_range", DEFAULT_DENSITY_RANGE)
        ),
        "density_range",
    )
    segment_count_range = _parse_pair(
        anomaly.get(
            "segment_count_range",
            config.get("segment_count_range", DEFAULT_SEGMENT_COUNT_RANGE),
        ),
        "segment_count_range",
    )
    return density_range, segment_count_range


def _min_segment_length_by_anomaly(
    config: Mapping[str, Any],
    anomaly: Mapping[str, Any],
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in dict(
            anomaly.get(
                "min_segment_length_by_anomaly",
                config.get("min_segment_length_by_anomaly", {}),
            )
        ).items()
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_list_or_none(value: Any) -> list[str] | None:
    return _optional_str_list(value) or None


__all__ = [
    "RuntimeConfigSections",
    "annotation_config_kwargs",
    "anomaly_policy_config_kwargs",
    "build_runtime_config_kwargs",
    "dataset_config_kwargs",
    "generator_config_kwargs",
    "io_config_kwargs",
    "plot_config_kwargs",
    "read_runtime_config_sections",
    "runtime_split_counts",
    "variant_config_kwargs",
]
