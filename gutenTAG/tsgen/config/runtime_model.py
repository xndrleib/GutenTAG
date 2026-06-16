"""Runtime configuration dataclass for TS dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..io import sanitize_json_value as _to_builtin_types
from .defaults import (
    DEFAULT_DENSITY_RANGE,
    DEFAULT_PROFILES_PER_PAIR,
    DEFAULT_SEGMENT_COUNT_RANGE,
    DEFAULT_SPLITS,
)
from .models import validate_raw_ts_config
from .runtime_parsing import build_runtime_config_kwargs
from .runtime_validation import validate_runtime_config


@dataclass
class TSGeneratorConfig:
    """Runtime configuration for synthetic TS dataset generation.

    Parameters
    ----------
    output_root : Path
        Output directory containing variant folders and the dataset manifest.
    master_seed : int
        Global seed used to derive all variant/split/instance seeds.
    length : int
        Time-series length :math:`T`.
    channels : int
        Number of channels :math:`C`.
    splits : Tuple[str, ...]
        Split names to generate.
    instances_per_split : int
        Number of instances generated per split and variant.
    density_range : Tuple[float, float]
        Inclusive min/max anomaly density range.
    density_tolerance : float
        Allowed absolute deviation between sampled and achieved density.
    segment_count_range : Tuple[int, int]
        Inclusive range for segment counts per instance.
    placement_policy : str
        Segment placement strategy. Currently only ``"uniform"`` is supported.
    channel_policy : str
        Channel assignment strategy. Currently only ``"single-random"`` is
        supported.
    base_oscillations : Optional[List[str]]
        Explicit base-oscillation list. ``None`` means all registered kinds.
    anomaly_types : Optional[List[str]]
        Explicit anomaly type list. ``None`` means all available kinds.
    compatibility_mode : str
        Compatibility filter used when enumerating base/anomaly pairs.
        ``"hard"`` keeps the full mechanically supported matrix, while
        ``"recommended"`` restricts generation to pairs that passed the
        multi-seed debug sweep, and ``"validated"`` keeps only the
        production-admitted subset of those pairs.
    skip_base_oscillations : List[str]
        Base oscillations removed from generation.
    skip_anomaly_types : List[str]
        Anomaly types removed from generation.
    base_oscillation_overrides : Dict[str, Dict[str, Any]]
        Per-base parameter overrides.
    base_channel_parameter_policy : str
        Per-channel base-parameter sampling policy. Supported values:
        ``"fixed_per_variant"`` and ``"random_per_instance"``.
    base_channel_overrides : Dict[str, Dict[str, Any]]
        Per-base parameter overrides sampled independently for each channel.
        Realized channel values overwrite shared base parameters.
    base_channel_correlation : Dict[str, Any]
        Channel-correlation controls for base generation. Currently supports
        ``shared_noise_weight``.
    split_phase_shift : Dict[str, Any]
        Optional deterministic phase offsets applied per split to channel base
        parameters that contain ``phase``.
    anomaly_overrides : Dict[str, Dict[str, Any]]
        Per-anomaly parameter overrides.
    variant_overrides : Dict[str, Dict[str, Any]]
        Per-variant overrides for base/anomaly parameters and anomaly policy.
    generate_plots : bool
        Whether to save per-instance plots (``plot_full.png`` and zoom images).
    plot_dpi : int
        Plot image resolution.
    csv_float_format : str
        Float formatting string used for CSV exports.
    overwrite_output : bool
        If True, the existing output directory is removed before generation.
    log_level : str
        Logging level string.
    max_placement_attempts : int
        Maximum attempts to place each segment without overlap.
    skip_density_incompatible_variants : bool
        Skip variants known to violate density constraints under current policy.
    support_label_mode : str
        Labeling strategy for event support. ``"strict_segment"`` marks the full
        planned segment. ``"effective_support"`` shrinks labels to points where
        ``|x_anom - x_clean|`` exceeds an event-specific epsilon.
    support_eps_mode : str
        Epsilon mode for ``effective_support``. Supported values:
        ``"relative"`` and ``"absolute"``.
    support_eps_value : float
        Relative multiplier (``relative`` mode) or absolute threshold
        (``absolute`` mode) used to determine effective support.
    min_effective_label_length_non_extremum : int
        Minimum label length enforced for non-point anomaly types when
        ``support_label_mode="effective_support"``.
    """

    output_root: Path
    master_seed: int
    dataset_version: str = "ts_dataset_v12"
    length: int = 10_000
    channels: int = 5
    splits: Tuple[str, ...] = DEFAULT_SPLITS
    instances_per_split: int = 5
    split_instance_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    density_range: Tuple[float, float] = DEFAULT_DENSITY_RANGE
    density_tolerance: float = 0.002
    segment_count_range: Tuple[int, int] = DEFAULT_SEGMENT_COUNT_RANGE
    placement_policy: str = "uniform"
    channel_policy: str = "single-random"
    overlap_policy: str = "global"
    base_oscillations: Optional[List[str]] = None
    anomaly_types: Optional[List[str]] = None
    compatibility_mode: str = "hard"
    skip_base_oscillations: List[str] = field(default_factory=list)
    skip_anomaly_types: List[str] = field(default_factory=list)
    disabled_anomaly_types: List[str] = field(default_factory=list)
    profiles_per_pair: int = DEFAULT_PROFILES_PER_PAIR
    pair_profiles: Dict[str, List[str]] = field(default_factory=dict)
    base_parameter_policy: str = "random_per_instance"
    base_channel_parameter_policy: str = "random_per_instance"
    base_channel_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    base_channel_correlation: Dict[str, Any] = field(default_factory=dict)
    split_phase_shift: Dict[str, Any] = field(default_factory=dict)
    anomaly_parameter_policy: str = "fixed_per_variant"
    base_oscillation_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomaly_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    variant_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    special_anomaly_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    segment_planner: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    min_segment_length_by_anomaly: Dict[str, int] = field(default_factory=dict)
    length_normalization: str = "resample"
    generate_plots: bool = True
    plot_dpi: int = 140
    zoom_count: int = 5
    zoom_margin: int = 100
    zoom_margin_min: int = 100
    zoom_margin_alpha: float = 0.0
    zoom_fill_policy: str = "repeat"
    csv_float_format: str = "%.8f"
    overwrite_output: bool = True
    allow_empty_dataset: bool = False
    log_level: str = "INFO"
    max_placement_attempts: int = 2_500
    skip_density_incompatible_variants: bool = True
    on_variant_failure: str = "skip"
    support_label_mode: str = "strict_segment"
    support_eps_mode: str = "relative"
    support_eps_value: float = 0.05
    min_effective_label_length_non_extremum: int = 1
    annotation_channels: Dict[str, Any] = field(default_factory=dict)
    law_level_replicates: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> TSGeneratorConfig:
        """Build a configuration object from a Python dictionary.

        Parameters
        ----------
        config : Mapping[str, Any]
            Raw configuration dictionary loaded from YAML/JSON.

        Returns
        -------
        TSGeneratorConfig
            Validated configuration object.
        """
        validate_raw_ts_config(dict(config))
        ts_config = cls(**build_runtime_config_kwargs(config))
        ts_config.validate()
        return ts_config

    def validate(self) -> None:
        """Validate this config and raise if any value is invalid.

        Raises
        ------
        ValueError
            If any configuration value is outside the supported range.
        """
        validate_runtime_config(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to a JSON/YAML-safe dictionary."""
        return _to_builtin_types(
            {
                "generator": {
                    "output_root": str(self.output_root),
                    "master_seed": self.master_seed,
                    "overwrite_output": self.overwrite_output,
                    "allow_empty_dataset": self.allow_empty_dataset,
                    "log_level": self.log_level,
                    "on_variant_failure": self.on_variant_failure,
                },
                "dataset": {
                    "dataset_version": self.dataset_version,
                    "length": self.length,
                    "channels": self.channels,
                    "splits": list(self.splits),
                    "instances_per_split": self.instances_per_split,
                    "split_instance_counts": self.split_instance_counts,
                },
                "anomaly_policy": {
                    "density_range": list(self.density_range),
                    "density_tolerance": self.density_tolerance,
                    "segment_count_range": list(self.segment_count_range),
                    "placement_policy": self.placement_policy,
                    "channel_policy": self.channel_policy,
                    "overlap_policy": self.overlap_policy,
                    "length_normalization": self.length_normalization,
                    "max_placement_attempts": self.max_placement_attempts,
                    "skip_density_incompatible_variants": self.skip_density_incompatible_variants,
                    "special_anomaly_policies": self.special_anomaly_policies,
                    "segment_planner": self.segment_planner,
                    "min_segment_length_by_anomaly": self.min_segment_length_by_anomaly,
                    "support_label_mode": self.support_label_mode,
                    "support_eps_mode": self.support_eps_mode,
                    "support_eps_value": self.support_eps_value,
                    "min_effective_label_length_non_extremum": (
                        self.min_effective_label_length_non_extremum
                    ),
                },
                "variants": {
                    "base_oscillations": self.base_oscillations,
                    "anomaly_types": self.anomaly_types,
                    "compatibility_mode": self.compatibility_mode,
                    "skip_base_oscillations": self.skip_base_oscillations,
                    "skip_anomaly_types": self.skip_anomaly_types,
                    "disabled_anomaly_types": self.disabled_anomaly_types,
                    "profiles_per_pair": self.profiles_per_pair,
                    "pair_profiles": self.pair_profiles,
                    "base_parameter_policy": self.base_parameter_policy,
                    "base_channel_parameter_policy": self.base_channel_parameter_policy,
                    "anomaly_parameter_policy": self.anomaly_parameter_policy,
                    "base_oscillation_overrides": self.base_oscillation_overrides,
                    "base_channel_overrides": self.base_channel_overrides,
                    "base_channel_correlation": self.base_channel_correlation,
                    "split_phase_shift": self.split_phase_shift,
                    "anomaly_overrides": self.anomaly_overrides,
                    "variant_overrides": self.variant_overrides,
                },
                "plot": {
                    "enabled": self.generate_plots,
                    "dpi": self.plot_dpi,
                    "zoom_count": self.zoom_count,
                    "zoom_margin": self.zoom_margin,
                    "zoom_margin_min": self.zoom_margin_min,
                    "zoom_margin_alpha": self.zoom_margin_alpha,
                    "zoom_fill_policy": self.zoom_fill_policy,
                },
                "io": {"csv_float_format": self.csv_float_format},
                "annotation_channels": self.annotation_channels,
                "law_level_replicates": self.law_level_replicates,
            }
        )
