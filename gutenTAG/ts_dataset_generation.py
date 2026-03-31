from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml
from numpy.random import SeedSequence
from tqdm import tqdm

from ._version import __version__
from .anomalies import Anomaly, AnomalyKind, Position
from .base_oscillations import BaseOscillation, RandomModeJump
from .base_oscillations.utils.math_func_support import SAMPLING_F
from .config.parser import decode_trend_obj
from .utils.compatibility import Compatibility
from .utils.global_variables import PARAMETERS
from .utils.types import GenerationContext


DEFAULT_SPLITS: Tuple[str, ...] = ("train", "val", "test")
DEFAULT_DENSITY_RANGE: Tuple[float, float] = (0.05, 0.10)
DEFAULT_SEGMENT_COUNT_RANGE: Tuple[int, int] = (20, 50)
DEFAULT_PROFILES_PER_PAIR = 1
DEFAULT_BASE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "random-mode-jump": {"frequency": 250, "variance": 0.05, "random-seed": 7},
    "sine": {"frequency": 8.0, "variance": 0.03},
    "cosine": {"frequency": 8.0, "variance": 0.03},
    "square": {"frequency": 8.0, "variance": 0.03},
    "sawtooth": {"frequency": 8.0, "variance": 0.03},
    "dirichlet": {"frequency": 10.0, "variance": 0.03},
    "ecg": {"frequency": 8.0, "variance": 0.03},
    "random-walk": {"smoothing": 0.01, "variance": 0.03},
    "polynomial": {"polynomial": [0.05, 0.4], "variance": 0.03},
    "cylinder-bell-funnel": {"variance": 0.03},
    "mls": {"complexity": 7, "variance": 0.03},
}
DEFAULT_ANOMALY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "amplitude": {"amplitude_factor": 2.0},
    "frequency": {"frequency_factor": 2.0},
    "mean": {"offset": 1.0},
    "pattern": {
        "sinusoid_k": 10.0,
        "cbf_pattern_factor": 2.0,
        "square_duty": 0.8,
        "sawtooth_width": 0.5,
    },
    "pattern-shift": {"shift_by": 4, "transition_window": 10},
    "platform": {"value": 3.0},
    "trend": {"oscillation": {"kind": "sine", "frequency": 2.0, "amplitude": 1.0}},
    "variance": {"variance": 1.0},
    "extremum": {"min": False, "local": False, "context_window": 10},
    "mode-correlation": {},
}
BASES_REQUIRING_EXTRA_CONFIG: Tuple[str, ...] = ("custom-input", "formula")
ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY: Tuple[str, ...] = ("extremum",)


@dataclass
class VariantSpec:
    """Single base-oscillation/anomaly/profile combination."""

    base_oscillation: str
    anomaly_type: str
    profile_id: str = "p00"

    @property
    def pair_id(self) -> str:
        """Return base/anomaly pair identifier."""
        return f"{self.base_oscillation}__{self.anomaly_type}"

    @property
    def variant_id(self) -> str:
        """Return the deterministic and human-readable identifier."""
        return f"{self.base_oscillation}__{self.anomaly_type}__{self.profile_id}"


@dataclass
class SegmentPlan:
    """Specification of a single anomaly segment."""

    start: int
    end: int
    length: int
    channel: int
    attrs: Dict[str, Any] = field(default_factory=dict)


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
    length: int = 10_000
    channels: int = 5
    splits: Tuple[str, ...] = DEFAULT_SPLITS
    instances_per_split: int = 5
    density_range: Tuple[float, float] = DEFAULT_DENSITY_RANGE
    density_tolerance: float = 0.002
    segment_count_range: Tuple[int, int] = DEFAULT_SEGMENT_COUNT_RANGE
    placement_policy: str = "uniform"
    channel_policy: str = "single-random"
    overlap_policy: str = "global"
    base_oscillations: Optional[List[str]] = None
    anomaly_types: Optional[List[str]] = None
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
    log_level: str = "INFO"
    max_placement_attempts: int = 2_500
    skip_density_incompatible_variants: bool = True
    on_variant_failure: str = "skip"
    support_label_mode: str = "strict_segment"
    support_eps_mode: str = "relative"
    support_eps_value: float = 0.05
    min_effective_label_length_non_extremum: int = 1

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
        generator_cfg = _read_nested_dict(config, "generator")
        dataset_cfg = _read_nested_dict(config, "dataset")
        variants_cfg = _read_nested_dict(config, "variants")
        anomaly_cfg = _read_nested_dict(config, "anomaly_policy")
        plot_cfg = _read_nested_dict(config, "plot")
        io_cfg = _read_nested_dict(config, "io")

        density_range_raw = anomaly_cfg.get(
            "density_range",
            config.get("density_range", DEFAULT_DENSITY_RANGE),
        )
        segment_count_range_raw = anomaly_cfg.get(
            "segment_count_range",
            config.get("segment_count_range", DEFAULT_SEGMENT_COUNT_RANGE),
        )
        density_range = _parse_pair(density_range_raw, "density_range")
        segment_count_range = _parse_pair(
            segment_count_range_raw, "segment_count_range"
        )

        merged_base_overrides = _merge_dicts(
            DEFAULT_BASE_OVERRIDES,
            config.get("base_oscillation_overrides", {}),
            variants_cfg.get("base_oscillation_overrides", {}),
        )
        merged_anomaly_overrides = _merge_dicts(
            DEFAULT_ANOMALY_OVERRIDES,
            config.get("anomaly_overrides", {}),
            variants_cfg.get("anomaly_overrides", {}),
        )
        merged_base_channel_overrides = _merge_dicts(
            config.get("base_channel_overrides", {}),
            variants_cfg.get("base_channel_overrides", {}),
        )
        raw_base_channel_correlation = variants_cfg.get(
            "base_channel_correlation",
            config.get("base_channel_correlation", {}),
        )
        if not isinstance(raw_base_channel_correlation, Mapping):
            raw_base_channel_correlation = {}
        raw_split_phase_shift = variants_cfg.get(
            "split_phase_shift",
            config.get("split_phase_shift", {}),
        )
        if not isinstance(raw_split_phase_shift, Mapping):
            raw_split_phase_shift = {}
        pair_profiles = _parse_pair_profiles(
            variants_cfg.get("pair_profiles", config.get("pair_profiles", {}))
        )
        segment_planner = _parse_segment_planner(
            anomaly_cfg.get("segment_planner", config.get("segment_planner", {}))
        )

        ts_config = cls(
            output_root=Path(
                generator_cfg.get(
                    "output_root", config.get("output_root", "data/ts_dataset")
                )
            ),
            master_seed=int(
                generator_cfg.get("master_seed", config.get("master_seed", 42))
            ),
            length=int(dataset_cfg.get("length", config.get("length", 10_000))),
            channels=int(dataset_cfg.get("channels", config.get("channels", 5))),
            splits=tuple(
                dataset_cfg.get("splits", config.get("splits", list(DEFAULT_SPLITS)))
            ),
            instances_per_split=int(
                dataset_cfg.get(
                    "instances_per_split", config.get("instances_per_split", 5)
                )
            ),
            density_range=(float(density_range[0]), float(density_range[1])),
            density_tolerance=float(
                anomaly_cfg.get(
                    "density_tolerance", config.get("density_tolerance", 0.002)
                )
            ),
            segment_count_range=(
                int(segment_count_range[0]),
                int(segment_count_range[1]),
            ),
            placement_policy=str(
                anomaly_cfg.get(
                    "placement_policy", config.get("placement_policy", "uniform")
                )
            ),
            channel_policy=str(
                anomaly_cfg.get(
                    "channel_policy", config.get("channel_policy", "single-random")
                )
            ),
            overlap_policy=str(
                anomaly_cfg.get(
                    "overlap_policy", config.get("overlap_policy", "global")
                )
            ),
            base_oscillations=(
                _optional_str_list(
                    variants_cfg.get(
                        "base_oscillations", config.get("base_oscillations", [])
                    )
                )
                or None
            ),
            anomaly_types=(
                _optional_str_list(
                    variants_cfg.get("anomaly_types", config.get("anomaly_types", []))
                )
                or None
            ),
            skip_base_oscillations=_optional_str_list(
                variants_cfg.get(
                    "skip_base_oscillations",
                    config.get("skip_base_oscillations", []),
                ),
            ),
            skip_anomaly_types=_optional_str_list(
                variants_cfg.get(
                    "skip_anomaly_types",
                    config.get("skip_anomaly_types", []),
                ),
            ),
            disabled_anomaly_types=_optional_str_list(
                variants_cfg.get(
                    "disabled_anomaly_types",
                    config.get("disabled_anomaly_types", []),
                )
            ),
            profiles_per_pair=int(
                variants_cfg.get(
                    "profiles_per_pair",
                    config.get("profiles_per_pair", DEFAULT_PROFILES_PER_PAIR),
                )
            ),
            pair_profiles=pair_profiles,
            base_parameter_policy=str(
                variants_cfg.get(
                    "base_parameter_policy",
                    config.get("base_parameter_policy", "random_per_instance"),
                )
            ),
            base_channel_parameter_policy=str(
                variants_cfg.get(
                    "base_channel_parameter_policy",
                    config.get("base_channel_parameter_policy", "random_per_instance"),
                )
            ),
            base_channel_overrides=merged_base_channel_overrides,
            base_channel_correlation=copy.deepcopy(dict(raw_base_channel_correlation)),
            split_phase_shift=copy.deepcopy(dict(raw_split_phase_shift)),
            anomaly_parameter_policy=str(
                variants_cfg.get(
                    "anomaly_parameter_policy",
                    config.get("anomaly_parameter_policy", "fixed_per_variant"),
                )
            ),
            base_oscillation_overrides=merged_base_overrides,
            anomaly_overrides=merged_anomaly_overrides,
            variant_overrides=dict(
                variants_cfg.get(
                    "variant_overrides", config.get("variant_overrides", {})
                )
            ),
            special_anomaly_policies=dict(
                anomaly_cfg.get(
                    "special_anomaly_policies",
                    config.get("special_anomaly_policies", {}),
                )
            ),
            segment_planner=segment_planner,
            min_segment_length_by_anomaly={
                str(k): int(v)
                for k, v in dict(
                    anomaly_cfg.get(
                        "min_segment_length_by_anomaly",
                        config.get("min_segment_length_by_anomaly", {}),
                    )
                ).items()
            },
            length_normalization=str(
                anomaly_cfg.get(
                    "length_normalization",
                    config.get("length_normalization", "resample"),
                )
            ),
            generate_plots=bool(
                plot_cfg.get("enabled", config.get("generate_plots", True))
            ),
            plot_dpi=int(plot_cfg.get("dpi", config.get("plot_dpi", 140))),
            zoom_count=int(plot_cfg.get("zoom_count", config.get("zoom_count", 5))),
            zoom_margin=int(
                plot_cfg.get("zoom_margin", config.get("zoom_margin", 100))
            ),
            zoom_margin_min=int(
                plot_cfg.get("zoom_margin_min", config.get("zoom_margin_min", 100))
            ),
            zoom_margin_alpha=float(
                plot_cfg.get("zoom_margin_alpha", config.get("zoom_margin_alpha", 0.0))
            ),
            zoom_fill_policy=str(
                plot_cfg.get(
                    "zoom_fill_policy", config.get("zoom_fill_policy", "repeat")
                )
            ),
            csv_float_format=str(
                io_cfg.get("csv_float_format", config.get("csv_float_format", "%.8f"))
            ),
            overwrite_output=bool(
                generator_cfg.get(
                    "overwrite_output", config.get("overwrite_output", True)
                )
            ),
            log_level=str(
                generator_cfg.get("log_level", config.get("log_level", "INFO"))
            ).upper(),
            on_variant_failure=str(
                generator_cfg.get(
                    "on_variant_failure", config.get("on_variant_failure", "skip")
                )
            ),
            max_placement_attempts=int(
                anomaly_cfg.get(
                    "max_placement_attempts",
                    config.get("max_placement_attempts", 2_500),
                )
            ),
            skip_density_incompatible_variants=bool(
                anomaly_cfg.get(
                    "skip_density_incompatible_variants",
                    config.get("skip_density_incompatible_variants", True),
                )
            ),
            support_label_mode=str(
                anomaly_cfg.get(
                    "support_label_mode",
                    config.get("support_label_mode", "strict_segment"),
                )
            ),
            support_eps_mode=str(
                anomaly_cfg.get(
                    "support_eps_mode",
                    config.get("support_eps_mode", "relative"),
                )
            ),
            support_eps_value=float(
                anomaly_cfg.get(
                    "support_eps_value",
                    config.get("support_eps_value", 0.05),
                )
            ),
            min_effective_label_length_non_extremum=int(
                anomaly_cfg.get(
                    "min_effective_label_length_non_extremum",
                    config.get("min_effective_label_length_non_extremum", 1),
                )
            ),
        )
        ts_config.validate()
        return ts_config

    def validate(self) -> None:
        """Validate this config and raise if any value is invalid.

        Raises
        ------
        ValueError
            If any configuration value is outside the supported range.
        """
        if self.length <= 0:
            raise ValueError("dataset.length must be > 0")
        if self.channels <= 0:
            raise ValueError("dataset.channels must be > 0")
        if self.instances_per_split <= 0:
            raise ValueError("dataset.instances_per_split must be > 0")
        if len(self.splits) == 0:
            raise ValueError("dataset.splits must contain at least one split name")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("dataset.splits must not contain duplicate split names")
        if self.density_range[0] <= 0 or self.density_range[1] >= 1:
            raise ValueError("anomaly density_range must stay inside (0, 1)")
        if self.density_range[0] > self.density_range[1]:
            raise ValueError("density_range must have min <= max")
        if self.density_tolerance < 0:
            raise ValueError("density_tolerance must be >= 0")
        if self.segment_count_range[0] <= 0 or self.segment_count_range[1] <= 0:
            raise ValueError("segment_count_range values must be > 0")
        if self.segment_count_range[0] > self.segment_count_range[1]:
            raise ValueError("segment_count_range must have min <= max")
        if self.segment_count_range[1] > self.length:
            raise ValueError(
                "segment_count_range max must be <= dataset.length so every segment "
                "can have at least one point."
            )
        if self.placement_policy != "uniform":
            raise ValueError("Only placement_policy='uniform' is supported")
        allowed_channel_policies = ("single-random", "paired-random", "all-channels")
        if self.channel_policy not in allowed_channel_policies:
            raise ValueError(
                "channel_policy must be one of {'single-random','paired-random','all-channels'}"
            )
        if self.channel_policy == "paired-random" and self.channels < 2:
            raise ValueError("channel_policy='paired-random' requires dataset.channels >= 2")
        if self.overlap_policy not in ("global", "per_channel"):
            raise ValueError("overlap_policy must be one of {'global','per_channel'}")
        if self.profiles_per_pair <= 0:
            raise ValueError("profiles_per_pair must be > 0")
        if self.base_parameter_policy not in (
            "fixed_per_variant",
            "random_per_instance",
        ):
            raise ValueError(
                "base_parameter_policy must be one of "
                "{'fixed_per_variant','random_per_instance'}"
            )
        if self.anomaly_parameter_policy not in (
            "fixed_per_variant",
            "random_per_instance",
            "random_per_segment",
        ):
            raise ValueError(
                "anomaly_parameter_policy must be one of "
                "{'fixed_per_variant','random_per_instance','random_per_segment'}"
            )
        if self.base_channel_parameter_policy not in (
            "fixed_per_variant",
            "random_per_instance",
        ):
            raise ValueError(
                "base_channel_parameter_policy must be one of "
                "{'fixed_per_variant','random_per_instance'}"
            )
        if not isinstance(self.base_channel_correlation, Mapping):
            raise ValueError("base_channel_correlation must be a mapping")
        shared_noise_weight = float(
            dict(self.base_channel_correlation).get("shared_noise_weight", 0.0)
        )
        if shared_noise_weight < 0.0 or shared_noise_weight > 1.0:
            raise ValueError("base_channel_correlation.shared_noise_weight must be in [0, 1]")
        if not isinstance(self.split_phase_shift, Mapping):
            raise ValueError("split_phase_shift must be a mapping")
        split_phase_cfg = dict(self.split_phase_shift)
        split_phase_enabled = bool(split_phase_cfg.get("enabled", False))
        split_phase_mode = str(split_phase_cfg.get("mode", "fixed_map"))
        split_phase_modulo = float(split_phase_cfg.get("phase_modulo", float(2.0 * np.pi)))
        if split_phase_modulo <= 0.0:
            raise ValueError("split_phase_shift.phase_modulo must be > 0")
        if split_phase_enabled:
            if split_phase_mode != "fixed_map":
                raise ValueError(
                    "split_phase_shift.mode must be 'fixed_map' when split_phase_shift.enabled=true"
                )
            values_cfg = split_phase_cfg.get("values", {})
            if not isinstance(values_cfg, Mapping):
                raise ValueError("split_phase_shift.values must be a mapping")
            missing_splits = [split for split in self.splits if split not in values_cfg]
            if missing_splits:
                raise ValueError(
                    f"split_phase_shift.values is missing offsets for splits: {missing_splits}"
                )
            for split_name, offset_value in values_cfg.items():
                try:
                    float(offset_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"split_phase_shift.values[{split_name}] must be numeric"
                    ) from exc
        if self.length_normalization not in ("none", "crop", "pad", "resample"):
            raise ValueError(
                "length_normalization must be one of {'none','crop','pad','resample'}"
            )
        if self.on_variant_failure not in ("skip", "fail_fast"):
            raise ValueError("on_variant_failure must be one of {'skip','fail_fast'}")
        if self.zoom_count <= 0:
            raise ValueError("zoom_count must be > 0")
        if self.zoom_margin < 0 or self.zoom_margin_min < 0:
            raise ValueError("zoom_margin and zoom_margin_min must be >= 0")
        if self.zoom_margin_alpha < 0:
            raise ValueError("zoom_margin_alpha must be >= 0")
        if self.zoom_fill_policy not in ("repeat", "blank"):
            raise ValueError("zoom_fill_policy must be one of {'repeat','blank'}")
        for anomaly_type, min_length in self.min_segment_length_by_anomaly.items():
            if min_length <= 0:
                raise ValueError(
                    f"min_segment_length_by_anomaly[{anomaly_type}] must be > 0"
                )
        for planner_key, planner_cfg in self.segment_planner.items():
            if not isinstance(planner_cfg, Mapping):
                raise ValueError(f"segment_planner[{planner_key}] must be a mapping")
            planner_name = str(planner_cfg.get("planner", "uniform_segments"))
            if planner_name not in (
                "uniform_segments",
                "point_events_from_density",
                "period_locked_frequency",
                "energy_aware_segments",
                "trend_parameter_aware_segments",
                "mode_boundary_segments",
            ):
                raise ValueError(
                    "segment_planner planner must be one of "
                    "{'uniform_segments','point_events_from_density','period_locked_frequency',"
                    "'energy_aware_segments','trend_parameter_aware_segments','mode_boundary_segments'}"
                )
            if "density_range" in planner_cfg:
                _parse_pair(
                    planner_cfg["density_range"], "segment_planner.density_range"
                )
        for anomaly_type, raw in self.special_anomaly_policies.items():
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"special_anomaly_policies[{anomaly_type}] must be a mapping"
                )
            special_channel_policy = raw.get("channel_policy")
            if special_channel_policy is None:
                continue
            special_channel_policy = str(special_channel_policy)
            if special_channel_policy not in allowed_channel_policies:
                raise ValueError(
                    f"special_anomaly_policies[{anomaly_type}].channel_policy must be one of "
                    "{'single-random','paired-random','all-channels'}"
                )
            if special_channel_policy == "paired-random" and self.channels < 2:
                raise ValueError(
                    f"special_anomaly_policies[{anomaly_type}].channel_policy='paired-random' "
                    "requires dataset.channels >= 2"
                )
        if self.support_label_mode not in ("strict_segment", "effective_support"):
            raise ValueError(
                "support_label_mode must be one of {'strict_segment','effective_support'}"
            )
        if self.support_eps_mode not in ("relative", "absolute"):
            raise ValueError("support_eps_mode must be one of {'relative','absolute'}")
        if self.support_eps_value < 0:
            raise ValueError("support_eps_value must be >= 0")
        if self.min_effective_label_length_non_extremum <= 0:
            raise ValueError(
                "min_effective_label_length_non_extremum must be >= 1"
            )
        if self.max_placement_attempts <= 0:
            raise ValueError("max_placement_attempts must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to a JSON/YAML-safe dictionary."""
        return _to_builtin_types(
            {
                "generator": {
                    "output_root": str(self.output_root),
                    "master_seed": self.master_seed,
                    "overwrite_output": self.overwrite_output,
                    "log_level": self.log_level,
                    "on_variant_failure": self.on_variant_failure,
                },
                "dataset": {
                    "length": self.length,
                    "channels": self.channels,
                    "splits": list(self.splits),
                    "instances_per_split": self.instances_per_split,
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
            }
        )


class TSDatasetGenerator:
    """Generate a paired clean/anomalous TS dataset from one config.

    Notes
    -----
    - The generator writes one dataset root containing variants and manifests.
    - Every instance is reproducible through deterministic seed derivation.
    - All normal progress output is logged through ``logging``.
    """

    def __init__(self, config: TSGeneratorConfig):
        self.config = config
        self.logger = logging.getLogger("GutenTAG.ts_dataset")

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> TSDatasetGenerator:
        """Create a generator from a dictionary configuration."""
        return cls(TSGeneratorConfig.from_dict(config))

    @classmethod
    def from_yaml(cls, path: Path) -> TSDatasetGenerator:
        """Create a generator from a YAML configuration file."""
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("YAML config must be a mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_json(cls, path: Path) -> TSDatasetGenerator:
        """Create a generator from a JSON configuration file."""
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("JSON config must be a mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_file(cls, path: Path) -> TSDatasetGenerator:
        """Create a generator from either YAML or JSON config file.

        Parameters
        ----------
        path : Path
            Path to a configuration file.

        Returns
        -------
        TSDatasetGenerator
            Initialized generator.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        """
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if suffix == ".json":
            return cls.from_json(path)
        raise ValueError(f"Unsupported config format for file: {path}")

    def run(self) -> Dict[str, Any]:
        """Generate the complete dataset and write all artifacts.

        Returns
        -------
        Dict[str, Any]
            Dataset-level manifest dictionary that is also written to disk.
        """
        output_root = self.config.output_root
        self._prepare_output_directory(output_root)
        self._configure_logging(output_root)
        self.logger.info(
            "Starting TS dataset generation | output_root=%s | master_seed=%d",
            output_root,
            self.config.master_seed,
        )

        variants_root = output_root / "variants"
        variants_root.mkdir(parents=True, exist_ok=True)

        variants, skipped_variants, disabled_anomaly_types = self._resolve_variants()
        generated_variant_entries: List[Dict[str, Any]] = []
        all_instance_summaries: List[Dict[str, Any]] = []
        all_seed_audit: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}

        for variant in tqdm(variants, desc="Generating variants", total=len(variants)):
            try:
                variant_result = self._generate_variant(variant, variants_root)
                generated_variant_entries.append(variant_result["variant_manifest"])
                all_instance_summaries.extend(variant_result["instance_summaries"])
                all_seed_audit[variant.variant_id] = variant_result["seed_audit"]
                self.logger.info(
                    "Finished variant %s | instances=%d",
                    variant.variant_id,
                    len(variant_result["instance_summaries"]),
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                variant_dir = variants_root / variant.variant_id
                if variant_dir.exists():
                    shutil.rmtree(variant_dir, ignore_errors=True)
                skipped_variants.append(
                    {"variant_id": variant.variant_id, "reason": reason}
                )
                self.logger.exception(
                    "Variant %s failed and was skipped", variant.variant_id
                )
                if self.config.on_variant_failure == "fail_fast":
                    raise

        generated_variant_entries = sorted(
            generated_variant_entries, key=lambda entry: entry["variant_id"]
        )
        skipped_variants = sorted(
            skipped_variants, key=lambda entry: entry["variant_id"]
        )

        dataset_stats = _compute_dataset_statistics(all_instance_summaries)
        manifest: Dict[str, Any] = {
            "generator": {
                "library_version": __version__,
                "git_commit": _try_resolve_git_commit(),
            },
            "config": self.config.to_dict(),
            "generated_variants": [
                entry["variant_id"] for entry in generated_variant_entries
            ],
            "variant_manifests": generated_variant_entries,
            "skipped_variants": skipped_variants,
            "disabled_anomaly_types": disabled_anomaly_types,
            "aggregated_statistics": dataset_stats,
            "derived_seeds": all_seed_audit,
        }
        manifest = _to_builtin_types(manifest)
        _write_json(output_root / "dataset_manifest.json", manifest)

        self.logger.info(
            "Generation complete | generated_variants=%d | skipped_variants=%d | total_instances=%d",
            len(generated_variant_entries),
            len(skipped_variants),
            dataset_stats["instance_count"],
        )
        return manifest

    def _prepare_output_directory(self, output_root: Path) -> None:
        if output_root.exists():
            if not self.config.overwrite_output:
                raise FileExistsError(
                    f"Output folder already exists and overwrite_output=false: {output_root}"
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    def _configure_logging(self, output_root: Path) -> None:
        level = getattr(logging, self.config.log_level, logging.INFO)
        self.logger.handlers.clear()
        self.logger.setLevel(level)
        self.logger.propagate = False

        formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            output_root / "generation.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def _resolve_variants(
        self,
    ) -> Tuple[List[VariantSpec], List[Dict[str, str]], List[Dict[str, str]]]:
        base_kinds = self._resolve_base_oscillation_kinds()
        anomaly_kinds = self._resolve_anomaly_kinds()
        variants: List[VariantSpec] = []
        skipped: List[Dict[str, str]] = []
        disabled: List[Dict[str, str]] = []

        for base_kind in base_kinds:
            for anomaly_kind in anomaly_kinds:
                pair_variant = VariantSpec(base_kind, anomaly_kind, profile_id="p00")
                pair_id = pair_variant.pair_id
                profile_ids = self._resolve_profile_ids(pair_id)
                for profile_id in profile_ids:
                    variant = VariantSpec(
                        base_kind, anomaly_kind, profile_id=profile_id
                    )
                    if anomaly_kind in self.config.disabled_anomaly_types:
                        reason = "Disabled by configuration."
                        skipped.append(
                            {"variant_id": variant.variant_id, "reason": reason}
                        )
                        disabled.append(
                            {"anomaly_type": anomaly_kind, "reason": reason}
                        )
                        continue

                    compatible = Compatibility.check(anomaly_kind, base_kind)
                    if not compatible:
                        skipped.append(
                            {
                                "variant_id": variant.variant_id,
                                "reason": "Incompatible (base_oscillation, anomaly_type) pair.",
                            }
                        )
                        continue

                    has_special_policy = (
                        anomaly_kind in self.config.special_anomaly_policies
                    )
                    planner_cfg = self._resolve_segment_planner(anomaly_kind)
                    planner_name = str(
                        planner_cfg.get("planner", "uniform_segments")
                    ).lower()
                    has_special_planner = planner_name == "point_events_from_density"
                    if (
                        self.config.skip_density_incompatible_variants
                        and anomaly_kind in ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY
                        and not has_special_policy
                        and not has_special_planner
                    ):
                        reason = (
                            f"Anomaly '{anomaly_kind}' requires special_anomaly_policies "
                            "or a dedicated segment_planner under current density "
                            "constraints."
                        )
                        skipped.append(
                            {"variant_id": variant.variant_id, "reason": reason}
                        )
                        disabled.append(
                            {"anomaly_type": anomaly_kind, "reason": reason}
                        )
                        continue

                    if base_kind in BASES_REQUIRING_EXTRA_CONFIG:
                        base_params = self._resolve_base_parameters(variant)
                        if (
                            base_kind == "formula"
                            and PARAMETERS.FORMULA not in base_params
                        ):
                            skipped.append(
                                {
                                    "variant_id": variant.variant_id,
                                    "reason": (
                                        "Base oscillation 'formula' requires explicit "
                                        "'formula' configuration."
                                    ),
                                }
                            )
                            continue
                        if (
                            base_kind == "custom-input"
                            and PARAMETERS.INPUT_TIMESERIES_PATH_TEST not in base_params
                        ):
                            skipped.append(
                                {
                                    "variant_id": variant.variant_id,
                                    "reason": (
                                        "Base oscillation 'custom-input' requires "
                                        "'input-timeseries-path-test'."
                                    ),
                                }
                            )
                            continue

                    variants.append(variant)

        variants = sorted(variants, key=lambda v: v.variant_id)
        skipped = sorted(skipped, key=lambda entry: entry["variant_id"])
        disabled_unique = _unique_records(
            disabled, key_fields=("anomaly_type", "reason")
        )
        return variants, skipped, disabled_unique

    def _resolve_profile_ids(self, pair_id: str) -> List[str]:
        explicit_ids = self.config.pair_profiles.get(pair_id)
        if explicit_ids:
            return sorted({str(pid) for pid in explicit_ids})
        return [f"p{i:02d}" for i in range(self.config.profiles_per_pair)]

    def _resolve_base_oscillation_kinds(self) -> List[str]:
        available = sorted(BaseOscillation.key_mapping.keys())
        requested = (
            sorted(set(self.config.base_oscillations))
            if self.config.base_oscillations is not None
            else available
        )
        excluded = set(self.config.skip_base_oscillations)
        selected = [
            kind for kind in requested if kind in available and kind not in excluded
        ]
        return sorted(selected)

    def _resolve_anomaly_kinds(self) -> List[str]:
        available = sorted(kind.value for kind in AnomalyKind)
        requested = (
            sorted(set(self.config.anomaly_types))
            if self.config.anomaly_types is not None
            else available
        )
        excluded = set(self.config.skip_anomaly_types)
        selected = [
            kind for kind in requested if kind in available and kind not in excluded
        ]
        return sorted(selected)

    def _generate_variant(
        self, variant: VariantSpec, variants_root: Path
    ) -> Dict[str, Any]:
        variant_dir = variants_root / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        base_parameter_template = self._resolve_base_parameters(variant)
        base_channel_parameter_template = self._resolve_base_channel_parameters(variant)
        anomaly_parameter_template = self._resolve_anomaly_parameters(variant)
        variant_anomaly_policy = self._resolve_variant_anomaly_policy(variant)
        segment_planner_override = variant_anomaly_policy.get("segment_planner", {})
        if not isinstance(segment_planner_override, Mapping):
            segment_planner_override = {}
        variant_segment_planner = self._resolve_segment_planner(
            variant.anomaly_type, segment_planner_override
        )
        effective_overlap_policy = str(
            variant_anomaly_policy.get(
                "overlap_policy",
                variant_segment_planner.get(
                    "overlap_policy", self.config.overlap_policy
                ),
            )
        )
        variant_param_rng = np.random.default_rng(
            _derive_seed(self.config.master_seed, variant.variant_id, "variant-params")
        )
        fixed_base_parameters = (
            self._realize_parameters(base_parameter_template, variant_param_rng)
            if self.config.base_parameter_policy == "fixed_per_variant"
            else None
        )
        fixed_base_channel_parameters = (
            self._realize_base_channel_parameters(
                base_channel_parameter_template, variant_param_rng
            )
            if self.config.base_channel_parameter_policy == "fixed_per_variant"
            else None
        )
        fixed_anomaly_parameters = (
            self._realize_parameters(anomaly_parameter_template, variant_param_rng)
            if self.config.anomaly_parameter_policy == "fixed_per_variant"
            else None
        )

        variant_config = {
            "variant_id": variant.variant_id,
            "profile_id": variant.profile_id,
            "base_oscillation": {
                "kind": variant.base_oscillation,
                "parameter_template": _to_builtin_types(base_parameter_template),
                "realized_parameters": _to_builtin_types(fixed_base_parameters),
                "channel_parameter_template": _to_builtin_types(
                    base_channel_parameter_template
                ),
                "channel_realized_parameters": _to_builtin_types(
                    fixed_base_channel_parameters
                ),
                "channel_correlation": _to_builtin_types(
                    self.config.base_channel_correlation
                ),
                "split_phase_shift": _to_builtin_types(self.config.split_phase_shift),
            },
            "anomaly_type": {
                "kind": variant.anomaly_type,
                "parameter_template": _to_builtin_types(anomaly_parameter_template),
                "realized_parameters": _to_builtin_types(fixed_anomaly_parameters),
            },
            "dataset": {
                "length": self.config.length,
                "channels": self.config.channels,
                "splits": list(self.config.splits),
                "instances_per_split": self.config.instances_per_split,
            },
            "anomaly_policy": {
                "density_range": list(self.config.density_range),
                "density_tolerance": self.config.density_tolerance,
                "segment_count_range": list(self.config.segment_count_range),
                "placement_policy": self.config.placement_policy,
                "channel_policy": self.config.channel_policy,
                "overlap_policy": effective_overlap_policy,
                "segment_planner": _to_builtin_types(variant_segment_planner),
                "variant_override": _to_builtin_types(variant_anomaly_policy),
                "length_normalization": self.config.length_normalization,
            },
            "parameter_policies": {
                "base_parameter_policy": self.config.base_parameter_policy,
                "base_channel_parameter_policy": self.config.base_channel_parameter_policy,
                "anomaly_parameter_policy": self.config.anomaly_parameter_policy,
            },
            "split_phase_shift": _to_builtin_types(self.config.split_phase_shift),
        }
        with (variant_dir / "variant_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(
                _to_builtin_types(variant_config),
                handle,
                sort_keys=True,
                allow_unicode=False,
            )

        variant_seed_audit: Dict[str, Dict[str, Dict[str, int]]] = {}
        variant_instance_summaries: List[Dict[str, Any]] = []
        split_entries: List[Dict[str, Any]] = []

        for split in self.config.splits:
            split_dir = variant_dir / split
            instances_dir = split_dir / "instances"
            instances_dir.mkdir(parents=True, exist_ok=True)
            split_seed_audit: Dict[str, Dict[str, int]] = {}
            split_instance_summaries: List[Dict[str, Any]] = []

            for instance_index in tqdm(
                range(self.config.instances_per_split),
                desc=f"{variant.variant_id}/{split}",
                leave=False,
            ):
                instance_name = f"instance_{instance_index:03d}"
                instance_dir = instances_dir / instance_name
                instance_dir.mkdir(parents=True, exist_ok=True)
                seeds = self._derive_instance_seeds(
                    variant.variant_id, split, instance_index
                )
                split_seed_audit[instance_name] = seeds
                summary = self._generate_instance(
                    variant=variant,
                    split=split,
                    instance_dir=instance_dir,
                    seeds=seeds,
                    base_parameter_template=base_parameter_template,
                    base_channel_parameter_template=base_channel_parameter_template,
                    anomaly_parameter_template=anomaly_parameter_template,
                    fixed_base_parameters=fixed_base_parameters,
                    fixed_base_channel_parameters=fixed_base_channel_parameters,
                    fixed_anomaly_parameters=fixed_anomaly_parameters,
                    variant_anomaly_policy=variant_anomaly_policy,
                    variant_segment_planner=variant_segment_planner,
                )
                split_instance_summaries.append(summary)
                variant_instance_summaries.append(summary)

            split_summary = _compute_split_statistics(
                split,
                split_instance_summaries,
                self.config.length,
                self.config.channels,
            )
            _write_json(split_dir / "split_summary.json", split_summary)
            split_entries.append(
                {
                    "split": split,
                    "instances": self.config.instances_per_split,
                    "summary_file": str(
                        (Path(split) / "split_summary.json").as_posix()
                    ),
                }
            )
            variant_seed_audit[split] = split_seed_audit

        variant_manifest = {
            "variant_id": variant.variant_id,
            "pair_id": variant.pair_id,
            "profile_id": variant.profile_id,
            "base_oscillation": variant.base_oscillation,
            "anomaly_type": variant.anomaly_type,
            "splits": split_entries,
            "instance_count": len(variant_instance_summaries),
            "parameter_policies": {
                "base_parameter_policy": self.config.base_parameter_policy,
                "base_channel_parameter_policy": self.config.base_channel_parameter_policy,
                "anomaly_parameter_policy": self.config.anomaly_parameter_policy,
            },
            "split_phase_shift": _to_builtin_types(self.config.split_phase_shift),
            "overlap_policy": effective_overlap_policy,
            "segment_planner": _to_builtin_types(variant_segment_planner),
            "variant_anomaly_policy": _to_builtin_types(variant_anomaly_policy),
            "length_normalization": self.config.length_normalization,
            "statistics": _compute_dataset_statistics(variant_instance_summaries),
        }

        return {
            "variant_manifest": _to_builtin_types(variant_manifest),
            "instance_summaries": _to_builtin_types(variant_instance_summaries),
            "seed_audit": _to_builtin_types(variant_seed_audit),
        }

    def _resolve_base_parameters(self, variant: VariantSpec) -> Dict[str, Any]:
        base_kind = variant.base_oscillation
        parameters: Dict[str, Any] = {}
        parameters.update(copy.deepcopy(DEFAULT_BASE_OVERRIDES.get(base_kind, {})))
        parameters.update(
            copy.deepcopy(self.config.base_oscillation_overrides.get(base_kind, {}))
        )
        pair_override = self.config.variant_overrides.get(variant.pair_id, {})
        if not isinstance(pair_override, Mapping):
            pair_override = {}
        parameters.update(copy.deepcopy(pair_override.get("base_oscillation", {})))
        variant_override = self.config.variant_overrides.get(variant.variant_id, {})
        if not isinstance(variant_override, Mapping):
            variant_override = {}
        parameters.update(copy.deepcopy(variant_override.get("base_oscillation", {})))
        parameters[PARAMETERS.LENGTH] = self.config.length
        return parameters

    def _resolve_base_channel_parameters(self, variant: VariantSpec) -> Dict[str, Any]:
        base_kind = variant.base_oscillation
        parameters: Dict[str, Any] = {}
        parameters.update(copy.deepcopy(self.config.base_channel_overrides.get(base_kind, {})))
        pair_override = self.config.variant_overrides.get(variant.pair_id, {})
        if not isinstance(pair_override, Mapping):
            pair_override = {}
        parameters.update(copy.deepcopy(pair_override.get("base_channel", {})))
        variant_override = self.config.variant_overrides.get(variant.variant_id, {})
        if not isinstance(variant_override, Mapping):
            variant_override = {}
        parameters.update(copy.deepcopy(variant_override.get("base_channel", {})))
        return parameters

    def _resolve_anomaly_parameters(self, variant: VariantSpec) -> Dict[str, Any]:
        anomaly_type = variant.anomaly_type
        parameters: Dict[str, Any] = {}
        parameters.update(
            copy.deepcopy(DEFAULT_ANOMALY_OVERRIDES.get(anomaly_type, {}))
        )
        parameters.update(
            copy.deepcopy(self.config.anomaly_overrides.get(anomaly_type, {}))
        )
        pair_override = self.config.variant_overrides.get(variant.pair_id, {})
        if not isinstance(pair_override, Mapping):
            pair_override = {}
        parameters.update(copy.deepcopy(pair_override.get("anomaly", {})))
        variant_override = self.config.variant_overrides.get(variant.variant_id, {})
        if not isinstance(variant_override, Mapping):
            variant_override = {}
        parameters.update(copy.deepcopy(variant_override.get("anomaly", {})))
        return parameters

    def _resolve_variant_anomaly_policy(self, variant: VariantSpec) -> Dict[str, Any]:
        policy: Dict[str, Any] = {}
        pair_override = self.config.variant_overrides.get(variant.pair_id, {})
        if isinstance(pair_override, Mapping) and isinstance(
            pair_override.get("anomaly_policy"), Mapping
        ):
            policy = _merge_dicts(policy, pair_override["anomaly_policy"])
        variant_override = self.config.variant_overrides.get(variant.variant_id, {})
        if isinstance(variant_override, Mapping) and isinstance(
            variant_override.get("anomaly_policy"), Mapping
        ):
            policy = _merge_dicts(policy, variant_override["anomaly_policy"])
        return policy

    def _derive_instance_seeds(
        self, variant_id: str, split: str, instance_index: int
    ) -> Dict[str, int]:
        instance_label = f"instance_{instance_index:03d}"
        instance_seed = _derive_seed(
            self.config.master_seed, variant_id, split, instance_label
        )
        split_stable_instance_seed = _derive_seed(
            self.config.master_seed, variant_id, instance_label
        )
        return {
            "instance_seed": instance_seed,
            "split_stable_instance_seed": split_stable_instance_seed,
            "base_seed": _derive_seed(instance_seed, "base"),
            "base_shared_noise_seed": _derive_seed(instance_seed, "base-shared-noise"),
            "base_params_seed": _derive_seed(
                split_stable_instance_seed, "base-parameter-sampling"
            ),
            "base_channel_params_seed": _derive_seed(
                split_stable_instance_seed, "base-channel-parameter-sampling"
            ),
            "plan_seed": _derive_seed(instance_seed, "segment-plan"),
            "anomaly_seed": _derive_seed(instance_seed, "anomaly-transform"),
            "params_seed": _derive_seed(instance_seed, "parameter-sampling"),
            "zoom_seed": _derive_seed(instance_seed, "zoom-selection"),
        }

    def _generate_instance(
        self,
        variant: VariantSpec,
        split: str,
        instance_dir: Path,
        seeds: Mapping[str, int],
        base_parameter_template: Mapping[str, Any],
        base_channel_parameter_template: Mapping[str, Any],
        anomaly_parameter_template: Mapping[str, Any],
        fixed_base_parameters: Optional[Mapping[str, Any]],
        fixed_base_channel_parameters: Optional[List[Dict[str, Any]]],
        fixed_anomaly_parameters: Optional[Mapping[str, Any]],
        variant_anomaly_policy: Optional[Mapping[str, Any]],
        variant_segment_planner: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        base_parameter_rng = np.random.default_rng(int(seeds["base_params_seed"]))
        base_channel_parameter_rng = np.random.default_rng(
            int(seeds["base_channel_params_seed"])
        )
        anomaly_parameter_rng = np.random.default_rng(int(seeds["params_seed"]))
        if self.config.base_parameter_policy == "fixed_per_variant":
            if fixed_base_parameters is None:
                base_parameters = self._realize_parameters(
                    base_parameter_template, base_parameter_rng
                )
            else:
                base_parameters = copy.deepcopy(dict(fixed_base_parameters))
        else:
            base_parameters = self._realize_parameters(
                base_parameter_template, base_parameter_rng
            )

        if self.config.base_channel_parameter_policy == "fixed_per_variant":
            if fixed_base_channel_parameters is None:
                base_channel_parameters = self._realize_base_channel_parameters(
                    base_channel_parameter_template, base_channel_parameter_rng
                )
            else:
                base_channel_parameters = copy.deepcopy(fixed_base_channel_parameters)
        else:
            base_channel_parameters = self._realize_base_channel_parameters(
                base_channel_parameter_template, base_channel_parameter_rng
            )
        base_parameters_per_channel_raw = self._compose_base_parameters_per_channel(
            base_parameters=base_parameters,
            base_channel_parameters=base_channel_parameters,
        )
        (
            base_parameters_per_channel,
            split_phase_shift_info,
        ) = self._apply_split_phase_shift(
            base_parameters_per_channel=base_parameters_per_channel_raw,
            split=split,
        )

        if self.config.anomaly_parameter_policy == "fixed_per_variant":
            if fixed_anomaly_parameters is None:
                anomaly_parameters_instance = self._realize_parameters(
                    anomaly_parameter_template, anomaly_parameter_rng
                )
            else:
                anomaly_parameters_instance = copy.deepcopy(
                    dict(fixed_anomaly_parameters)
                )
        elif self.config.anomaly_parameter_policy == "random_per_instance":
            anomaly_parameters_instance = self._realize_parameters(
                anomaly_parameter_template, anomaly_parameter_rng
            )
        else:
            anomaly_parameters_instance = None

        clean_bos = self._generate_base_channels(
            base_kind=variant.base_oscillation,
            base_parameters_per_channel=base_parameters_per_channel,
            seed=int(seeds["base_seed"]),
        )
        self._apply_shared_noise_correlation(
            clean_bos, seed=int(seeds["base_shared_noise_seed"])
        )
        clean_base = self._stack_channel_timeseries(clean_bos)
        clean = self._apply_variations(clean_base, clean_bos)

        anomalous_bos = self._generate_base_channels(
            base_kind=variant.base_oscillation,
            base_parameters_per_channel=base_parameters_per_channel,
            seed=int(seeds["base_seed"]),
        )
        self._apply_shared_noise_correlation(
            anomalous_bos, seed=int(seeds["base_shared_noise_seed"])
        )
        anomalous_base = self._stack_channel_timeseries(anomalous_bos)

        plan_rng = np.random.default_rng(int(seeds["plan_seed"]))
        target_density = float(
            plan_rng.uniform(self.config.density_range[0], self.config.density_range[1])
        )
        special_policy = _merge_dicts(
            self._special_policy(variant.anomaly_type),
            dict(variant_anomaly_policy or {}),
        )
        if isinstance(variant_segment_planner, Mapping):
            planner_cfg = copy.deepcopy(dict(variant_segment_planner))
        else:
            planner_cfg = self._resolve_segment_planner(variant.anomaly_type)
        base_period_size = anomalous_bos[0].get_period_size()
        if base_period_size is not None:
            base_period_size = int(base_period_size)
        if variant.anomaly_type == "trend" and base_period_size is not None:
            if base_period_size > 1:
                existing_min = int(
                    special_policy.get(
                        "min_segment_length", self._minimum_segment_length("trend")
                    )
                )
                special_policy["min_segment_length"] = int(
                    max(existing_min, 2 * base_period_size)
                )
        base_frequency = getattr(anomalous_bos[0], "frequency", None)
        period_boundaries = self._compute_period_boundaries(base_frequency)
        period_boundaries_by_channel = {
            channel: self._compute_period_boundaries(getattr(bo, "frequency", None))
            for channel, bo in enumerate(anomalous_bos)
        }
        active_overlap_policy = str(
            special_policy.get(
                "overlap_policy",
                planner_cfg.get("overlap_policy", self.config.overlap_policy),
            )
        )
        active_channel_policy = str(
            special_policy.get("channel_policy", self.config.channel_policy)
        )
        segment_plan = self._sample_segment_plan(
            rng=plan_rng,
            target_density=target_density,
            anomaly_type=variant.anomaly_type,
            clean_values=clean_base,
            anomaly_policy=special_policy,
            planner_cfg=planner_cfg,
            base_period_size=base_period_size,
            period_boundaries=period_boundaries,
            period_boundaries_by_channel=period_boundaries_by_channel,
            anomaly_parameter_template=anomaly_parameter_template,
            parameter_seed=int(seeds["params_seed"]),
        )
        anomaly_params_per_segment = self._resolve_anomaly_parameters_for_segments(
            anomaly_type=variant.anomaly_type,
            anomaly_parameter_template=anomaly_parameter_template,
            fixed_anomaly_parameters=fixed_anomaly_parameters,
            anomaly_parameters_instance=anomaly_parameters_instance,
            segment_plan=segment_plan,
            parameter_seed=int(seeds["params_seed"]),
            planner_cfg=planner_cfg,
        )

        anomaly_objects = self._build_anomalies(
            variant.anomaly_type,
            anomaly_params_per_segment,
            segment_plan,
        )
        labels, events = self._apply_anomalies(
            anomaly_objects=anomaly_objects,
            segment_plan=segment_plan,
            base=anomalous_base,
            channel_bos=anomalous_bos,
            anomaly_seed=int(seeds["anomaly_seed"]),
            anomaly_type=variant.anomaly_type,
            anomaly_parameters_per_segment=anomaly_params_per_segment,
        )
        anomalous = self._apply_variations(anomalous_base, anomalous_bos)

        achieved_density_labeled = float(labels.max(axis=1).mean())
        achieved_density_source = achieved_density_labeled
        if self.config.support_label_mode == "effective_support":
            source_labels = np.zeros_like(labels, dtype=np.int8)
            for event in events:
                source_start = int(event.get("source_start", event["start"]))
                source_end = int(event.get("source_end", event["end"]))
                source_channel = int(event["channel"])
                if source_end > source_start:
                    source_labels[source_start:source_end, source_channel] = 1
            achieved_density_source = float(source_labels.max(axis=1).mean())
            density_for_validation = achieved_density_source
            density_validation_mode = "source_support"
        else:
            density_for_validation = achieved_density_labeled
            density_validation_mode = "labeled_support"
        active_density_tolerance = float(
            special_policy.get(
                "density_tolerance",
                planner_cfg.get("density_tolerance", self.config.density_tolerance),
            )
        )
        if active_density_tolerance < 0:
            raise ValueError("density_tolerance must be >= 0")
        lower_with_tolerance = self.config.density_range[0] - active_density_tolerance
        upper_with_tolerance = self.config.density_range[1] + active_density_tolerance
        if not (lower_with_tolerance <= density_for_validation <= upper_with_tolerance):
            raise ValueError(
                f"Achieved density {density_for_validation:.6f} outside target range "
                f"{self.config.density_range} | mode={density_validation_mode} "
                f"| tolerance={active_density_tolerance:.6f}"
                f"| labeled={achieved_density_labeled:.6f} | source={achieved_density_source:.6f}."
            )
        if abs(density_for_validation - target_density) > active_density_tolerance:
            raise ValueError(
                f"Density mismatch | target={target_density:.6f} | achieved={density_for_validation:.6f} "
                f"| tolerance={active_density_tolerance:.6f} | mode={density_validation_mode} "
                f"| labeled={achieved_density_labeled:.6f} | source={achieved_density_source:.6f}"
            )

        self._check_labels_and_events_consistency(labels, events)

        self._write_timeseries_csv(instance_dir / "clean.csv", clean)
        self._write_timeseries_csv(instance_dir / "anomalous.csv", anomalous)
        self._write_labels_csv(instance_dir / "labels_pointwise.csv", labels)
        _write_json(instance_dir / "events.json", events)

        segment_lengths = [int(event["length"]) for event in events]
        source_segment_lengths = [
            int(event.get("source_end", event["end"])) - int(event.get("source_start", event["start"]))
            for event in events
        ]
        effective_support_shrink_count = int(
            sum(
                1
                for event in events
                if int(event.get("source_start", event["start"])) != int(event["start"])
                or int(event.get("source_end", event["end"])) != int(event["end"])
            )
        )
        energy_fallback_count = int(
            sum(1 for segment in segment_plan if bool(segment.attrs.get("energy_fallback", False)))
        )
        per_channel_counts = {
            str(channel): int(sum(1 for event in events if event["channel"] == channel))
            for channel in range(self.config.channels)
        }
        unique_group_ids = sorted(
            {
                int(event.get("group_id", idx))
                for idx, event in enumerate(events)
            }
        )
        instance_summary: Dict[str, Any] = {
            "instance_id": instance_dir.name,
            "split": split,
            "variant_id": variant.variant_id,
            "profile_id": variant.profile_id,
            "base_oscillation": variant.base_oscillation,
            "anomaly_type": variant.anomaly_type,
            "target_density": target_density,
            "achieved_density": density_for_validation,
            "achieved_density_labeled": achieved_density_labeled,
            "achieved_density_source": achieved_density_source,
            "density_validation_mode": density_validation_mode,
            "density_error": abs(density_for_validation - target_density),
            "density_tolerance": active_density_tolerance,
            "n_segments": len(events),
            "n_event_groups": len(unique_group_ids),
            "segment_length_mean": float(np.mean(segment_lengths)),
            "segment_length_median": float(np.median(segment_lengths)),
            "segment_length_std": float(np.std(segment_lengths)),
            "segment_length_min": int(np.min(segment_lengths)),
            "segment_length_max": int(np.max(segment_lengths)),
            "segment_lengths": segment_lengths,
            "source_segment_lengths": source_segment_lengths,
            "effective_support_shrink_count": effective_support_shrink_count,
            "energy_fallback_count": energy_fallback_count,
            "per_channel_segment_counts": per_channel_counts,
            "channel_policy": active_channel_policy,
            "overlap_policy": active_overlap_policy,
            "segment_planner": _to_builtin_types(planner_cfg),
            "variant_anomaly_policy": _to_builtin_types(special_policy),
            "length_normalization": self.config.length_normalization,
            "support_label_mode": self.config.support_label_mode,
            "support_eps_mode": self.config.support_eps_mode,
            "support_eps_value": self.config.support_eps_value,
            "base_parameter_policy": self.config.base_parameter_policy,
            "base_channel_parameter_policy": self.config.base_channel_parameter_policy,
            "anomaly_parameter_policy": self.config.anomaly_parameter_policy,
            "base_parameters": _to_builtin_types(base_parameters),
            "base_channel_parameters": _to_builtin_types(base_channel_parameters),
            "base_parameters_per_channel": _to_builtin_types(base_parameters_per_channel),
            "split_phase_shift": _to_builtin_types(split_phase_shift_info),
            "base_channel_correlation": _to_builtin_types(
                self.config.base_channel_correlation
            ),
            "anomaly_parameters_instance": _to_builtin_types(
                anomaly_parameters_instance
            ),
            "seeds": dict(seeds),
        }
        _write_json(instance_dir / "instance_summary.json", instance_summary)

        if self.config.generate_plots:
            self._write_instance_plots(
                instance_dir=instance_dir,
                anomalous=anomalous,
                events=events,
                zoom_seed=int(seeds["zoom_seed"]),
            )

        return _to_builtin_types(instance_summary)

    def _realize_base_channel_parameters(
        self, template: Mapping[str, Any], rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        if len(template) == 0:
            return [{} for _ in range(self.config.channels)]
        return [
            self._realize_parameters(template, rng) for _ in range(self.config.channels)
        ]

    def _compose_base_parameters_per_channel(
        self,
        base_parameters: Mapping[str, Any],
        base_channel_parameters: List[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        channel_parameters: List[Dict[str, Any]] = []
        for channel in range(self.config.channels):
            params = copy.deepcopy(dict(base_parameters))
            if channel < len(base_channel_parameters):
                params.update(copy.deepcopy(dict(base_channel_parameters[channel])))
            params[PARAMETERS.LENGTH] = self.config.length
            channel_parameters.append(params)
        return channel_parameters

    def _apply_split_phase_shift(
        self,
        base_parameters_per_channel: List[Mapping[str, Any]],
        split: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply deterministic split-specific phase shifts to base parameters."""
        cfg = dict(self.config.split_phase_shift)
        enabled = bool(cfg.get("enabled", False))
        phase_modulo = float(cfg.get("phase_modulo", float(2.0 * np.pi)))
        offset = 0.0
        if enabled:
            offset = float(dict(cfg.get("values", {}))[split])

        realized: List[Dict[str, Any]] = [
            copy.deepcopy(dict(params)) for params in base_parameters_per_channel
        ]
        channels_with_phase: List[int] = []
        for channel, params in enumerate(realized):
            if "phase" not in params:
                continue
            channels_with_phase.append(channel)
            if not enabled or abs(offset) <= 0.0:
                continue
            shifted_phase = float(params["phase"]) + offset
            params["phase"] = float(np.mod(shifted_phase, phase_modulo))

        info: Dict[str, Any] = {
            "enabled": enabled,
            "split": split,
            "offset": float(offset),
            "phase_modulo": phase_modulo,
            "channels_with_phase": channels_with_phase,
            "phase_shift_applied": bool(enabled and abs(offset) > 0.0),
        }
        return realized, info

    def _generate_base_channels(
        self,
        base_kind: str,
        base_parameters_per_channel: List[Mapping[str, Any]],
        seed: int,
    ) -> List[Any]:
        ctx = GenerationContext(SeedSequence(seed))
        channels: List[Any] = []
        previous_channels: List[np.ndarray] = []
        for channel, channel_parameters in enumerate(base_parameters_per_channel):
            bo = BaseOscillation.from_key(
                base_kind, **copy.deepcopy(dict(channel_parameters))
            )
            bo.generate_timeseries_and_variations(
                ctx.to_bo(channel=channel, previous_channels=previous_channels)
            )
            self._sanitize_generated_base_channel(base_kind, bo)
            channels.append(bo)
            previous_channels.append(np.array(bo.timeseries, copy=True))
        return channels

    def _sanitize_generated_base_channel(self, base_kind: str, bo: Any) -> None:
        if bo.timeseries is None:
            raise ValueError(f"Base oscillation '{base_kind}' produced no timeseries.")
        if bo.timeseries.shape[0] != self.config.length:
            raise ValueError(
                f"Base oscillation '{base_kind}' produced length {bo.timeseries.shape[0]}, "
                f"expected {self.config.length}."
            )
        if bo.noise is None:
            bo.noise = np.zeros(self.config.length, dtype=np.float64)
        if bo.trend_series is None:
            bo.trend_series = np.zeros(self.config.length, dtype=np.float64)
        bo.timeseries = bo.timeseries.astype(np.float64)
        bo.noise = bo.noise.astype(np.float64)
        bo.trend_series = bo.trend_series.astype(np.float64)

    def _stack_channel_timeseries(self, channel_bos: List[Any]) -> np.ndarray:
        return np.column_stack([bo.timeseries for bo in channel_bos]).astype(np.float64)

    def _shared_noise_weight(self) -> float:
        weight = float(self.config.base_channel_correlation.get("shared_noise_weight", 0.0))
        return float(np.clip(weight, 0.0, 1.0))

    def _apply_shared_noise_correlation(self, channel_bos: List[Any], seed: int) -> None:
        shared_weight = self._shared_noise_weight()
        if shared_weight <= 0.0 or len(channel_bos) <= 1:
            return
        noise_lengths = {
            int(np.asarray(bo.noise).shape[0])
            for bo in channel_bos
            if bo.noise is not None
        }
        if len(noise_lengths) == 0:
            return
        if len(noise_lengths) > 1:
            raise ValueError(
                "Cannot apply shared-noise correlation to channel noises with different lengths."
            )
        noise_length = next(iter(noise_lengths))
        rng = np.random.default_rng(seed)
        shared_noise = rng.normal(0.0, 1.0, noise_length).astype(np.float64)
        shared_std = float(np.std(shared_noise))
        if shared_std > 0.0:
            shared_noise = (shared_noise - float(np.mean(shared_noise))) / shared_std
        else:
            shared_noise = np.zeros(noise_length, dtype=np.float64)
        residual_weight = float(np.sqrt(max(0.0, 1.0 - shared_weight**2)))
        for bo in channel_bos:
            if bo.noise is None:
                continue
            noise = np.asarray(bo.noise, dtype=np.float64)
            mean = float(np.mean(noise))
            centered = noise - mean
            channel_std = float(np.std(centered))
            if channel_std <= 1e-12:
                bo.noise = np.array(noise, copy=True)
                continue
            shared_component = shared_noise * channel_std
            mixed = residual_weight * centered + shared_weight * shared_component
            bo.noise = (mixed + mean).astype(np.float64)

    def _apply_variations(self, base: np.ndarray, channel_bos: List[Any]) -> np.ndarray:
        result = np.array(base, dtype=np.float64, copy=True)
        for channel, bo in enumerate(channel_bos):
            if bo.noise is not None:
                result[:, channel] = result[:, channel] + bo.noise
            if bo.trend_series is not None:
                result[:, channel] = result[:, channel] + bo.trend_series
            if bo.offset is not None:
                result[:, channel] = result[:, channel] + bo.offset
        return result

    def _compose_channel_window_with_variations(
        self, base: np.ndarray, bo: Any, channel: int, start: int, end: int
    ) -> np.ndarray:
        """Compose a single-channel window including variations.

        Parameters
        ----------
        base : np.ndarray
            Base (pre-variation) multichannel timeseries.
        bo : Any
            Base oscillation object for the channel.
        channel : int
            Channel index.
        start : int
            Window start index (inclusive).
        end : int
            Window end index (exclusive).

        Returns
        -------
        np.ndarray
            Composed window values.
        """
        start_i = max(0, min(int(start), self.config.length))
        end_i = max(start_i, min(int(end), self.config.length))
        window = np.array(base[start_i:end_i, channel], dtype=np.float64, copy=True)
        if window.size == 0:
            return window
        if bo.noise is not None:
            window = window + np.asarray(bo.noise[start_i:end_i], dtype=np.float64)
        if bo.trend_series is not None:
            window = window + np.asarray(
                bo.trend_series[start_i:end_i], dtype=np.float64
            )
        if bo.offset is not None:
            offset_array = np.asarray(bo.offset, dtype=np.float64)
            if offset_array.ndim == 0:
                window = window + float(offset_array)
            elif offset_array.shape[0] == self.config.length:
                window = window + offset_array[start_i:end_i]
            else:
                window = window + float(offset_array.reshape(-1)[0])
        return window

    def _sample_segment_plan(
        self,
        rng: np.random.Generator,
        target_density: float,
        anomaly_type: str,
        clean_values: Optional[np.ndarray] = None,
        anomaly_policy: Optional[Mapping[str, Any]] = None,
        planner_cfg: Optional[Mapping[str, Any]] = None,
        base_period_size: Optional[int] = None,
        period_boundaries: Optional[Iterable[int]] = None,
        period_boundaries_by_channel: Optional[Mapping[int, Optional[Iterable[int]]]] = None,
        anomaly_parameter_template: Optional[Mapping[str, Any]] = None,
        parameter_seed: Optional[int] = None,
    ) -> List[SegmentPlan]:
        if planner_cfg is None:
            planner_cfg = self._resolve_segment_planner(anomaly_type)
        else:
            planner_cfg = copy.deepcopy(dict(planner_cfg))
        if anomaly_policy is None:
            special_policy = self._special_policy(anomaly_type)
        else:
            special_policy = copy.deepcopy(dict(anomaly_policy))
        planner_name = str(planner_cfg.get("planner", "uniform_segments")).lower()
        overlap_policy = str(
            special_policy.get(
                "overlap_policy",
                planner_cfg.get("overlap_policy", self.config.overlap_policy),
            )
        )
        active_channel_policy = str(
            special_policy.get("channel_policy", self.config.channel_policy)
        )
        raw_segments: List[SegmentPlan]

        if planner_name == "point_events_from_density":
            raw_segments = self._sample_point_event_segments(
                rng=rng,
                target_density=target_density,
                overlap_policy=overlap_policy,
                planner_cfg=planner_cfg,
            )
            return self._apply_channel_policy_to_segments(
                raw_segments, rng, channel_policy=active_channel_policy
            )
        if planner_name == "period_locked_frequency":
            if anomaly_type != "frequency":
                raise ValueError(
                    "period_locked_frequency planner is only supported for anomaly_type='frequency'."
                )
            raw_segments = self._sample_period_locked_frequency_segments(
                rng=rng,
                target_density=target_density,
                overlap_policy=overlap_policy,
                planner_cfg=planner_cfg,
                anomaly_policy=special_policy,
                base_period_size=base_period_size,
                period_boundaries=period_boundaries,
                period_boundaries_by_channel=period_boundaries_by_channel,
            )
            return self._apply_channel_policy_to_segments(
                raw_segments, rng, channel_policy=active_channel_policy
            )
        if planner_name == "energy_aware_segments":
            raw_segments = self._sample_energy_aware_segments(
                rng=rng,
                target_density=target_density,
                overlap_policy=overlap_policy,
                planner_cfg=planner_cfg,
                anomaly_policy=special_policy,
                anomaly_type=anomaly_type,
                clean_values=clean_values,
            )
            return self._apply_channel_policy_to_segments(
                raw_segments, rng, channel_policy=active_channel_policy
            )
        if planner_name == "trend_parameter_aware_segments":
            if anomaly_type != "trend":
                raise ValueError(
                    "trend_parameter_aware_segments planner is only supported for anomaly_type='trend'."
                )
            if anomaly_parameter_template is None or parameter_seed is None:
                raise ValueError(
                    "trend_parameter_aware_segments planner requires anomaly_parameter_template "
                    "and parameter_seed."
                )
            raw_segments = self._sample_trend_parameter_aware_segments(
                rng=rng,
                target_density=target_density,
                overlap_policy=overlap_policy,
                planner_cfg=planner_cfg,
                anomaly_policy=special_policy,
                anomaly_parameter_template=anomaly_parameter_template,
                parameter_seed=int(parameter_seed),
                clean_values=clean_values,
            )
            return self._apply_channel_policy_to_segments(
                raw_segments, rng, channel_policy=active_channel_policy
            )
        if planner_name == "mode_boundary_segments":
            raw_segments = self._sample_mode_boundary_segments(
                rng=rng,
                target_density=target_density,
                overlap_policy=overlap_policy,
                planner_cfg=planner_cfg,
                anomaly_policy=special_policy,
                clean_values=clean_values,
                anomaly_type=anomaly_type,
            )
            return self._apply_channel_policy_to_segments(
                raw_segments, rng, channel_policy=active_channel_policy
            )

        segment_count_range = _parse_pair_int(
            special_policy.get(
                "segment_count_range",
                planner_cfg.get("segment_count_range", self.config.segment_count_range),
            ),
            "segment_count_range",
        )
        if segment_count_range[0] > segment_count_range[1]:
            segment_count_range = (segment_count_range[1], segment_count_range[0])
        n_segments = int(
            rng.integers(segment_count_range[0], segment_count_range[1] + 1)
        )
        if n_segments > self.config.length:
            raise ValueError(
                f"Requested n_segments={n_segments} exceeds series length={self.config.length}."
            )
        target_points = int(round(target_density * self.config.length))
        target_points = max(target_points, n_segments)
        target_points = min(target_points, self.config.length)

        min_segment_length = int(
            special_policy.get(
                "min_segment_length",
                planner_cfg.get(
                    "min_segment_length", self._minimum_segment_length(anomaly_type)
                ),
            )
        )
        max_segments_for_min_length = max(1, target_points // max(1, min_segment_length))
        if n_segments > max_segments_for_min_length:
            self.logger.warning(
                "Reducing n_segments from %s to %s to satisfy min_segment_length=%s "
                "for target_points=%s.",
                n_segments,
                max_segments_for_min_length,
                min_segment_length,
                target_points,
            )
            n_segments = max_segments_for_min_length
        lengths = self._sample_segment_lengths(
            rng, target_points, n_segments, min_segment_length=min_segment_length
        )
        occupied_global = np.zeros(self.config.length, dtype=np.int8)
        occupied_per_channel = np.zeros(
            (self.config.channels, self.config.length), dtype=np.int8
        )
        segments: List[SegmentPlan] = []

        for length in lengths:
            placed = False
            max_start = self.config.length - length
            for _ in range(self.config.max_placement_attempts):
                channel = int(rng.integers(0, self.config.channels))
                start = int(rng.integers(0, max_start + 1))
                end = start + length
                if self._is_slot_available(
                    start,
                    end,
                    channel,
                    overlap_policy,
                    occupied_global,
                    occupied_per_channel,
                ):
                    self._occupy_slot(
                        start,
                        end,
                        channel,
                        overlap_policy,
                        occupied_global,
                        occupied_per_channel,
                    )
                    segments.append(
                        SegmentPlan(
                            start=start, end=end, length=length, channel=channel
                        )
                    )
                    placed = True
                    break
            if not placed:
                channel_order = rng.permutation(self.config.channels).tolist()
                for channel in channel_order:
                    for start in range(max_start + 1):
                        end = start + length
                        if self._is_slot_available(
                            start,
                            end,
                            channel,
                            overlap_policy,
                            occupied_global,
                            occupied_per_channel,
                        ):
                            self._occupy_slot(
                                start,
                                end,
                                channel,
                                overlap_policy,
                                occupied_global,
                                occupied_per_channel,
                            )
                            segments.append(
                                SegmentPlan(
                                    start=start, end=end, length=length, channel=channel
                                )
                            )
                            placed = True
                            break
                    if placed:
                        break
            if not placed:
                raise ValueError(
                    f"Failed to place segment of length {length} without overlap."
                )

        segments.sort(
            key=lambda segment: (segment.start, segment.channel, segment.length)
        )
        return self._apply_channel_policy_to_segments(
            segments, rng, channel_policy=active_channel_policy
        )

    def _apply_channel_policy_to_segments(
        self,
        segments: List[SegmentPlan],
        rng: np.random.Generator,
        channel_policy: Optional[str] = None,
    ) -> List[SegmentPlan]:
        """Expand a single-channel segment plan according to channel policy."""
        active_channel_policy = str(channel_policy or self.config.channel_policy)
        if active_channel_policy == "single-random":
            expanded: List[SegmentPlan] = []
            for group_id, segment in enumerate(segments):
                attrs = copy.deepcopy(dict(segment.attrs))
                attrs.setdefault("group_id", int(group_id))
                attrs.setdefault("group_channels", [int(segment.channel)])
                expanded.append(
                    SegmentPlan(
                        start=segment.start,
                        end=segment.end,
                        length=segment.length,
                        channel=segment.channel,
                        attrs=attrs,
                    )
                )
            return expanded

        expanded = []
        all_channels = list(range(self.config.channels))
        for group_id, segment in enumerate(segments):
            if active_channel_policy == "all-channels":
                group_channels = all_channels
            else:
                remaining = [ch for ch in all_channels if ch != int(segment.channel)]
                partner = int(remaining[int(rng.integers(0, len(remaining)))])
                group_channels = sorted({int(segment.channel), partner})
            for channel in group_channels:
                attrs = copy.deepcopy(dict(segment.attrs))
                attrs["group_id"] = int(group_id)
                attrs["group_channels"] = [int(ch) for ch in group_channels]
                expanded.append(
                    SegmentPlan(
                        start=segment.start,
                        end=segment.end,
                        length=segment.length,
                        channel=int(channel),
                        attrs=attrs,
                    )
                )

        expanded.sort(key=lambda segment: (segment.start, segment.channel, segment.length))
        return expanded

    def _sample_period_locked_frequency_segments(
        self,
        rng: np.random.Generator,
        target_density: float,
        overlap_policy: str,
        planner_cfg: Mapping[str, Any],
        anomaly_policy: Mapping[str, Any],
        base_period_size: Optional[int],
        period_boundaries: Optional[Iterable[int]] = None,
        period_boundaries_by_channel: Optional[
            Mapping[int, Optional[Iterable[int]]]
        ] = None,
    ) -> List[SegmentPlan]:
        channel_boundaries: Dict[int, np.ndarray] = {}
        if period_boundaries_by_channel is not None:
            for channel in range(self.config.channels):
                raw_boundaries = period_boundaries_by_channel.get(channel)
                sanitized = self._sanitize_period_boundaries(raw_boundaries)
                if sanitized is not None and sanitized.shape[0] >= 2:
                    channel_boundaries[channel] = sanitized
        if len(channel_boundaries) == 0:
            boundaries = self._sanitize_period_boundaries(period_boundaries)
            if boundaries is None:
                if base_period_size is None or int(base_period_size) <= 1:
                    raise ValueError(
                        "period_locked_frequency planner requires either period boundaries "
                        "or period_size > 1."
                    )
                period_size = int(base_period_size)
                boundaries = np.arange(0, self.config.length + 1, period_size, dtype=int)
                if boundaries[-1] != self.config.length:
                    boundaries = np.append(boundaries, self.config.length)
            for channel in range(self.config.channels):
                channel_boundaries[channel] = np.array(boundaries, copy=True)

        channel_period_counts = {
            channel: int(boundaries.shape[0] - 1)
            for channel, boundaries in channel_boundaries.items()
        }
        channel_period_counts = {
            channel: count for channel, count in channel_period_counts.items() if count > 0
        }
        if len(channel_period_counts) == 0:
            raise ValueError(
                "period_locked_frequency planner requires at least one channel with period boundaries."
            )
        max_whole_periods = int(max(channel_period_counts.values()))
        avg_period_size = float(
            np.median(
                [
                    float(self.config.length) / float(period_count)
                    for period_count in channel_period_counts.values()
                ]
            )
        )

        periods_per_segment_range = _parse_pair_int(
            planner_cfg.get("periods_per_segment_range", [2, 4]),
            "periods_per_segment_range",
        )
        if periods_per_segment_range[0] > periods_per_segment_range[1]:
            periods_per_segment_range = (
                periods_per_segment_range[1],
                periods_per_segment_range[0],
            )
        min_periods = max(1, int(periods_per_segment_range[0]))
        max_periods = max(min_periods, int(periods_per_segment_range[1]))
        max_periods = min(max_periods, max_whole_periods)
        if min_periods > max_periods:
            raise ValueError(
                "periods_per_segment_range is infeasible for available period boundaries."
            )

        segment_count_range = _parse_pair_int(
            anomaly_policy.get(
                "segment_count_range",
                planner_cfg.get("segment_count_range", self.config.segment_count_range),
            ),
            "segment_count_range",
        )
        if segment_count_range[0] > segment_count_range[1]:
            segment_count_range = (segment_count_range[1], segment_count_range[0])

        target_points = int(round(target_density * self.config.length))
        target_points = max(1, min(target_points, self.config.length))
        total_period_points = int(
            np.clip(
                int(round(target_points / avg_period_size)),
                1,
                max_whole_periods,
            )
        )
        total_points = int(round(total_period_points * avg_period_size))

        effective_min_periods = min(min_periods, total_period_points)
        effective_max_periods = min(max_periods, total_period_points)
        if effective_min_periods > effective_max_periods:
            effective_min_periods = effective_max_periods
        min_required_segments = int(
            np.ceil(total_period_points / float(effective_max_periods))
        )
        max_allowed_segments = int(
            np.floor(total_period_points / float(effective_min_periods))
        )
        low = max(segment_count_range[0], min_required_segments)
        high = min(segment_count_range[1], max_allowed_segments)
        if low > high:
            low = min_required_segments
            high = max_allowed_segments
            self.logger.warning(
                "period_locked_frequency planner adjusted segment_count_range "
                "from requested=%s to feasible=%s for target_points=%s.",
                list(segment_count_range),
                [low, high],
                total_points,
            )
        n_segments = int(rng.integers(low, high + 1))

        periods_per_segment = self._sample_bounded_integer_lengths(
            rng=rng,
            total_points=total_period_points,
            n_segments=n_segments,
            min_value=effective_min_periods,
            max_value=effective_max_periods,
        )

        align_to_period_start = bool(planner_cfg.get("align_to_period_start", True))
        occupied_global = np.zeros(self.config.length, dtype=np.int8)
        occupied_per_channel = np.zeros(
            (self.config.channels, self.config.length), dtype=np.int8
        )
        segments: List[SegmentPlan] = []

        for period_count in periods_per_segment:
            eligible_channels = [
                channel
                for channel, boundaries in channel_boundaries.items()
                if int(boundaries.shape[0] - 1) >= int(period_count)
            ]
            if len(eligible_channels) == 0:
                raise ValueError(
                    "No channels can satisfy period_count=%s under period-locked planner."
                    % period_count
                )
            placed = False
            for _ in range(self.config.max_placement_attempts):
                channel = int(eligible_channels[int(rng.integers(0, len(eligible_channels)))])
                boundaries = channel_boundaries[channel]
                candidate_start_period_idxs = np.arange(
                    0, int(boundaries.shape[0] - 1) - int(period_count) + 1, dtype=int
                )
                if candidate_start_period_idxs.size == 0:
                    continue
                if align_to_period_start:
                    start_period_idx = int(
                        candidate_start_period_idxs[
                            int(rng.integers(0, candidate_start_period_idxs.size))
                        ]
                    )
                    start = int(boundaries[start_period_idx])
                    end = int(boundaries[start_period_idx + period_count])
                else:
                    period_size = int(np.median(np.diff(boundaries)))
                    if period_size <= 0:
                        continue
                    length = int(period_count * period_size)
                    max_start = self.config.length - length
                    if max_start < 0:
                        continue
                    candidate_starts = np.arange(0, max_start + 1, dtype=int)
                    start = int(
                        candidate_starts[int(rng.integers(0, candidate_starts.size))]
                    )
                    end = start + int(period_count * period_size)
                if self._is_slot_available(
                    start,
                    end,
                    channel,
                    overlap_policy,
                    occupied_global,
                    occupied_per_channel,
                ):
                    self._occupy_slot(
                        start,
                        end,
                        channel,
                        overlap_policy,
                        occupied_global,
                        occupied_per_channel,
                    )
                    segments.append(
                        SegmentPlan(
                            start=start,
                            end=end,
                            length=end - start,
                            channel=channel,
                            attrs={"period_count": int(period_count)},
                        )
                    )
                    placed = True
                    break
            if not placed:
                channel_order = rng.permutation(eligible_channels).tolist()
                for channel in channel_order:
                    boundaries = channel_boundaries[int(channel)]
                    candidate_start_period_idxs = np.arange(
                        0, int(boundaries.shape[0] - 1) - int(period_count) + 1, dtype=int
                    )
                    if candidate_start_period_idxs.size == 0:
                        continue
                    if align_to_period_start:
                        start_order = candidate_start_period_idxs[
                            rng.permutation(candidate_start_period_idxs.size)
                        ]
                        candidate_values = [int(value) for value in start_order.tolist()]
                    else:
                        period_size = int(np.median(np.diff(boundaries)))
                        if period_size <= 0:
                            continue
                        length = int(period_count * period_size)
                        max_start = self.config.length - length
                        if max_start < 0:
                            continue
                        candidate_starts = np.arange(0, max_start + 1, dtype=int)
                        start_order = candidate_starts[
                            rng.permutation(candidate_starts.size)
                        ]
                        candidate_values = [int(value) for value in start_order.tolist()]
                    for start_candidate in candidate_values:
                        if align_to_period_start:
                            start = int(boundaries[start_candidate])
                            end = int(boundaries[start_candidate + period_count])
                        else:
                            start = int(start_candidate)
                            end = int(start + period_count * period_size)
                        if self._is_slot_available(
                            start,
                            end,
                            int(channel),
                            overlap_policy,
                            occupied_global,
                            occupied_per_channel,
                        ):
                            self._occupy_slot(
                                start,
                                end,
                                int(channel),
                                overlap_policy,
                                occupied_global,
                                occupied_per_channel,
                            )
                            segments.append(
                                SegmentPlan(
                                    start=start,
                                    end=end,
                                    length=end - start,
                                    channel=int(channel),
                                    attrs={"period_count": int(period_count)},
                                )
                            )
                            placed = True
                            break
                    if placed:
                        break
            if not placed:
                raise ValueError(
                    "Failed to place period-locked segment without overlap "
                    f"(period_count={period_count})."
                )

        segments.sort(
            key=lambda segment: (segment.start, segment.channel, segment.length)
        )
        return segments

    def _sample_energy_aware_segments(
        self,
        rng: np.random.Generator,
        target_density: float,
        overlap_policy: str,
        planner_cfg: Mapping[str, Any],
        anomaly_policy: Mapping[str, Any],
        anomaly_type: str,
        clean_values: Optional[np.ndarray],
    ) -> List[SegmentPlan]:
        if clean_values is None:
            raise ValueError(
                "energy_aware_segments planner requires clean_values for window-energy checks."
            )
        if clean_values.shape != (self.config.length, self.config.channels):
            raise ValueError(
                "energy_aware_segments expected clean_values shape "
                f"({self.config.length}, {self.config.channels}), got {clean_values.shape}."
            )

        metric_mode = str(planner_cfg.get("energy_metric", "rms")).lower()
        if metric_mode not in ("rms", "peak", "rms_and_peak"):
            raise ValueError(
                "energy_aware_segments.energy_metric must be one of "
                "{'rms','peak','rms_and_peak'}"
            )
        rms_quantile = float(np.clip(float(planner_cfg.get("rms_quantile", 0.60)), 0.0, 1.0))
        peak_quantile = float(np.clip(float(planner_cfg.get("peak_quantile", 0.55)), 0.0, 1.0))
        weighted_sampling = bool(planner_cfg.get("weighted_sampling", True))
        fallback_mode = str(planner_cfg.get("fallback", "uniform_segments")).lower()
        if fallback_mode not in ("uniform_segments", "error"):
            raise ValueError(
                "energy_aware_segments.fallback must be one of {'uniform_segments','error'}"
            )

        segment_count_range = _parse_pair_int(
            anomaly_policy.get(
                "segment_count_range",
                planner_cfg.get("segment_count_range", self.config.segment_count_range),
            ),
            "segment_count_range",
        )
        if segment_count_range[0] > segment_count_range[1]:
            segment_count_range = (segment_count_range[1], segment_count_range[0])
        n_segments = int(
            rng.integers(segment_count_range[0], segment_count_range[1] + 1)
        )
        if n_segments > self.config.length:
            raise ValueError(
                f"Requested n_segments={n_segments} exceeds series length={self.config.length}."
            )

        target_points = int(round(target_density * self.config.length))
        target_points = max(target_points, n_segments)
        target_points = min(target_points, self.config.length)
        min_segment_length = int(
            anomaly_policy.get(
                "min_segment_length",
                planner_cfg.get(
                    "min_segment_length", self._minimum_segment_length(anomaly_type)
                ),
            )
        )
        max_segments_for_min_length = max(1, target_points // max(1, min_segment_length))
        if n_segments > max_segments_for_min_length:
            self.logger.warning(
                "Reducing n_segments from %s to %s to satisfy min_segment_length=%s "
                "for target_points=%s (energy_aware_segments).",
                n_segments,
                max_segments_for_min_length,
                min_segment_length,
                target_points,
            )
            n_segments = max_segments_for_min_length
        lengths = self._sample_segment_lengths(
            rng, target_points, n_segments, min_segment_length=min_segment_length
        )

        occupied_global = np.zeros(self.config.length, dtype=np.int8)
        occupied_per_channel = np.zeros(
            (self.config.channels, self.config.length), dtype=np.int8
        )
        sq_prefix_by_channel = [
            np.concatenate(
                [[0.0], np.cumsum(np.square(clean_values[:, channel]), dtype=np.float64)]
            )
            for channel in range(self.config.channels)
        ]
        abs_by_channel = [
            np.abs(clean_values[:, channel]).astype(np.float64)
            for channel in range(self.config.channels)
        ]
        rms_cache: Dict[Tuple[int, int], np.ndarray] = {}
        peak_cache: Dict[Tuple[int, int], np.ndarray] = {}
        segments: List[SegmentPlan] = []

        for length in lengths:
            n_starts = self.config.length - int(length) + 1
            if n_starts <= 0:
                raise ValueError(
                    f"Segment length {length} is infeasible for series length {self.config.length}."
                )
            if overlap_policy == "global":
                availability_by_channel = {
                    channel: self._available_start_mask(occupied_global, int(length))
                    for channel in range(self.config.channels)
                }
            else:
                availability_by_channel = {
                    channel: self._available_start_mask(
                        occupied_per_channel[channel, :], int(length)
                    )
                    for channel in range(self.config.channels)
                }

            thresholds_rms: Dict[int, float] = {}
            thresholds_peak: Dict[int, float] = {}
            if metric_mode in ("rms", "rms_and_peak"):
                for channel in range(self.config.channels):
                    values = self._rms_values_for_channel_length(
                        sq_prefix=sq_prefix_by_channel[channel],
                        length=int(length),
                    )
                    rms_cache[(channel, int(length))] = values
                    thresholds_rms[channel] = float(np.quantile(values, rms_quantile))
            if metric_mode in ("peak", "rms_and_peak"):
                for channel in range(self.config.channels):
                    values = self._peak_values_for_channel_length(
                        abs_values=abs_by_channel[channel], length=int(length)
                    )
                    peak_cache[(channel, int(length))] = values
                    thresholds_peak[channel] = float(np.quantile(values, peak_quantile))

            candidate_channels: List[np.ndarray] = []
            candidate_starts: List[np.ndarray] = []
            candidate_weights: List[np.ndarray] = []
            for channel in range(self.config.channels):
                energy_mask = np.ones(n_starts, dtype=bool)
                if metric_mode in ("rms", "rms_and_peak"):
                    rms_values = rms_cache[(channel, int(length))]
                    energy_mask &= rms_values >= thresholds_rms[channel]
                if metric_mode in ("peak", "rms_and_peak"):
                    peak_values = peak_cache[(channel, int(length))]
                    energy_mask &= peak_values >= thresholds_peak[channel]
                mask = energy_mask & availability_by_channel[channel]
                starts = np.flatnonzero(mask)
                if starts.size == 0:
                    continue
                candidate_channels.append(
                    np.full(starts.shape[0], int(channel), dtype=int)
                )
                candidate_starts.append(starts.astype(int))
                if weighted_sampling:
                    weights = np.ones(starts.shape[0], dtype=np.float64)
                    if metric_mode in ("rms", "rms_and_peak"):
                        rms_part = np.maximum(
                            rms_cache[(channel, int(length))][starts]
                            - thresholds_rms[channel],
                            0.0,
                        )
                        weights += rms_part
                    if metric_mode in ("peak", "rms_and_peak"):
                        peak_part = np.maximum(
                            peak_cache[(channel, int(length))][starts]
                            - thresholds_peak[channel],
                            0.0,
                        )
                        weights += peak_part
                    candidate_weights.append(weights)

            selected: Optional[SegmentPlan] = None
            used_fallback = False
            if len(candidate_starts) > 0:
                all_channels = np.concatenate(candidate_channels)
                all_starts = np.concatenate(candidate_starts)
                selected_idx: int
                if weighted_sampling and len(candidate_weights) > 0:
                    all_weights = np.concatenate(candidate_weights).astype(np.float64)
                    weight_sum = float(np.sum(all_weights))
                    if weight_sum > 0.0:
                        probs = all_weights / weight_sum
                        selected_idx = int(rng.choice(np.arange(all_starts.size), p=probs))
                    else:
                        selected_idx = int(rng.integers(0, all_starts.size))
                else:
                    selected_idx = int(rng.integers(0, all_starts.size))
                selected_channel = int(all_channels[selected_idx])
                selected_start = int(all_starts[selected_idx])
                selected_end = int(selected_start + int(length))
                selected = SegmentPlan(
                    start=selected_start,
                    end=selected_end,
                    length=int(length),
                    channel=selected_channel,
                )
            elif fallback_mode == "uniform_segments":
                selected = self._sample_uniform_slot(
                    rng=rng,
                    length=int(length),
                    overlap_policy=overlap_policy,
                    occupied_global=occupied_global,
                    occupied_per_channel=occupied_per_channel,
                )
                used_fallback = selected is not None

            if selected is None:
                raise ValueError(
                    "Failed to place an energy-aware segment without overlap "
                    f"(length={length}, fallback={fallback_mode})."
                )

            self._occupy_slot(
                selected.start,
                selected.end,
                selected.channel,
                overlap_policy,
                occupied_global,
                occupied_per_channel,
            )
            channel_sq_prefix = sq_prefix_by_channel[selected.channel]
            window_rms = self._window_rms(channel_sq_prefix, selected.start, selected.end)
            window_peak = float(
                np.max(abs_by_channel[selected.channel][selected.start : selected.end])
            )
            selected.attrs.update(
                {
                    "window_rms": float(window_rms),
                    "window_peak": float(window_peak),
                    "energy_metric": metric_mode,
                    "energy_fallback": bool(used_fallback),
                }
            )
            segments.append(selected)

        segments.sort(
            key=lambda segment: (segment.start, segment.channel, segment.length)
        )
        return segments

    @staticmethod
    def _available_start_mask(occupied: np.ndarray, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros(0, dtype=bool)
        prefix = np.concatenate([[0], np.cumsum(occupied, dtype=np.int64)])
        return (prefix[length:] - prefix[:-length]) == 0

    @staticmethod
    def _rms_values_for_channel_length(sq_prefix: np.ndarray, length: int) -> np.ndarray:
        sums = sq_prefix[length:] - sq_prefix[:-length]
        means = np.maximum(sums / float(length), 0.0)
        return np.sqrt(means)

    @staticmethod
    def _peak_values_for_channel_length(abs_values: np.ndarray, length: int) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(abs_values, window_shape=length)
        return np.max(windows, axis=1)

    @staticmethod
    def _window_rms(sq_prefix: np.ndarray, start: int, end: int) -> float:
        length = max(1, int(end - start))
        sum_sq = float(sq_prefix[end] - sq_prefix[start])
        mean_sq = max(sum_sq / float(length), 0.0)
        return float(np.sqrt(mean_sq))

    def _sample_uniform_slot(
        self,
        rng: np.random.Generator,
        length: int,
        overlap_policy: str,
        occupied_global: np.ndarray,
        occupied_per_channel: np.ndarray,
    ) -> Optional[SegmentPlan]:
        max_start = self.config.length - length
        if max_start < 0:
            return None
        for _ in range(self.config.max_placement_attempts):
            channel = int(rng.integers(0, self.config.channels))
            start = int(rng.integers(0, max_start + 1))
            end = start + length
            if self._is_slot_available(
                start,
                end,
                channel,
                overlap_policy,
                occupied_global,
                occupied_per_channel,
            ):
                return SegmentPlan(start=start, end=end, length=length, channel=channel)

        channel_order = rng.permutation(self.config.channels).tolist()
        for channel in channel_order:
            for start in range(max_start + 1):
                end = start + length
                if self._is_slot_available(
                    start,
                    end,
                    channel,
                    overlap_policy,
                    occupied_global,
                    occupied_per_channel,
                ):
                    return SegmentPlan(start=start, end=end, length=length, channel=channel)
        return None

    def _sample_segment_lengths_with_minima(
        self,
        rng: np.random.Generator,
        target_points: int,
        minimum_lengths: List[int],
    ) -> List[int]:
        """Sample segment lengths subject to per-segment minimum lengths."""
        if len(minimum_lengths) == 0:
            return []
        minima = np.asarray([max(1, int(value)) for value in minimum_lengths], dtype=int)
        total_min = int(np.sum(minima))
        total_points = int(max(int(target_points), total_min))
        total_points = int(min(total_points, self.config.length))
        if total_points < total_min:
            raise ValueError(
                "Requested target points are infeasible under per-segment minimum lengths "
                f"(target={total_points}, min_total={total_min})."
            )
        extra_points = int(total_points - total_min)
        if extra_points > 0:
            weights = rng.random(minima.shape[0]).astype(np.float64)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0:
                weights = np.full(minima.shape[0], 1.0 / float(minima.shape[0]))
            else:
                weights = weights / weight_sum
            extra = rng.multinomial(extra_points, weights)
            minima = minima + extra
        return [int(value) for value in minima.tolist()]

    def _sample_trend_parameter_aware_segments(
        self,
        rng: np.random.Generator,
        target_density: float,
        overlap_policy: str,
        planner_cfg: Mapping[str, Any],
        anomaly_policy: Mapping[str, Any],
        anomaly_parameter_template: Mapping[str, Any],
        parameter_seed: int,
        clean_values: Optional[np.ndarray] = None,
    ) -> List[SegmentPlan]:
        """Sample trend segments with parameter-driven per-segment length constraints."""
        segment_count_range = _parse_pair_int(
            anomaly_policy.get(
                "segment_count_range",
                planner_cfg.get("segment_count_range", self.config.segment_count_range),
            ),
            "segment_count_range",
        )
        if segment_count_range[0] > segment_count_range[1]:
            segment_count_range = (segment_count_range[1], segment_count_range[0])
        n_segments = int(
            rng.integers(segment_count_range[0], segment_count_range[1] + 1)
        )
        if n_segments > self.config.length:
            raise ValueError(
                f"Requested n_segments={n_segments} exceeds series length={self.config.length}."
            )

        target_points = int(round(target_density * self.config.length))
        target_points = max(target_points, n_segments)
        target_points = min(target_points, self.config.length)

        base_min_segment_length = int(
            anomaly_policy.get(
                "min_segment_length",
                planner_cfg.get(
                    "min_segment_length", self._minimum_segment_length("trend")
                ),
            )
        )
        sine_min_cycles = float(planner_cfg.get("sine_min_cycles", 0.30))
        if sine_min_cycles <= 0.0:
            raise ValueError("trend_parameter_aware_segments.sine_min_cycles must be > 0")
        random_walk_min_segment_length = int(
            max(1, int(planner_cfg.get("random_walk_min_segment_length", 8)))
        )

        per_segment_min_lengths: List[int] = []
        per_segment_attrs: List[Dict[str, Any]] = []
        for idx in range(n_segments):
            params_rng = np.random.default_rng(
                _derive_seed(parameter_seed, "trend-planner-params", str(idx))
            )
            trend_params = self._sanitize_anomaly_parameters(
                "trend",
                self._realize_parameters(anomaly_parameter_template, params_rng),
            )
            attrs: Dict[str, Any] = {"trend_params": copy.deepcopy(trend_params)}
            min_segment_length = int(max(1, base_min_segment_length))
            oscillation = trend_params.get("oscillation")
            if isinstance(oscillation, Mapping):
                osc_kind = str(oscillation.get("kind", "")).lower()
                attrs["trend_oscillation_kind"] = osc_kind
                if osc_kind == "sine":
                    frequency = float(oscillation.get("frequency", 0.0))
                    if frequency > 0.0:
                        min_from_cycles = int(
                            np.ceil((float(SAMPLING_F) * sine_min_cycles) / frequency)
                        )
                        min_segment_length = max(min_segment_length, min_from_cycles)
                        attrs["trend_sine_frequency"] = float(frequency)
                        attrs["trend_sine_min_cycles"] = float(sine_min_cycles)
                elif osc_kind == "random-walk":
                    min_segment_length = max(
                        min_segment_length, random_walk_min_segment_length
                    )
                    attrs["trend_random_walk_min_segment_length"] = int(
                        random_walk_min_segment_length
                    )

            min_segment_length = int(np.clip(min_segment_length, 1, self.config.length))
            per_segment_min_lengths.append(min_segment_length)
            attrs["trend_min_segment_length"] = int(min_segment_length)
            per_segment_attrs.append(attrs)

        dropped_segments: List[Dict[str, Any]] = []

        def drop_largest_minimum(reason: str) -> None:
            drop_idx = int(np.argmax(np.asarray(per_segment_min_lengths, dtype=int)))
            dropped = int(per_segment_min_lengths.pop(drop_idx))
            dropped_attrs = per_segment_attrs.pop(drop_idx)
            dropped_kind = str(dropped_attrs.get("trend_oscillation_kind", "unknown"))
            dropped_segments.append(
                {
                    "reason": reason,
                    "min_length": dropped,
                    "kind": dropped_kind,
                }
            )

        while sum(per_segment_min_lengths) > self.config.length and len(
            per_segment_min_lengths
        ) > 1:
            drop_largest_minimum("series_length_infeasibility")

        while sum(per_segment_min_lengths) > target_points and len(
            per_segment_min_lengths
        ) > 1:
            drop_largest_minimum("target_density_infeasibility")

        if (
            len(per_segment_min_lengths) == 1
            and per_segment_min_lengths[0] > target_points
        ):
            original = int(per_segment_min_lengths[0])
            per_segment_min_lengths[0] = int(target_points)
            per_segment_attrs[0]["trend_min_segment_length_original"] = int(original)
            per_segment_attrs[0]["trend_min_segment_length_relaxed"] = int(
                target_points
            )
            self.logger.warning(
                "Relaxing single trend minimum length to preserve target density "
                "(original=%s, relaxed=%s).",
                original,
                target_points,
            )

        if dropped_segments:
            reason_counts: Dict[str, int] = {}
            for item in dropped_segments:
                key = str(item["reason"])
                reason_counts[key] = int(reason_counts.get(key, 0) + 1)
            max_dropped = int(max(item["min_length"] for item in dropped_segments))
            self.logger.warning(
                "trend_parameter_aware_segments dropped segments to satisfy constraints "
                "(dropped=%s, remaining=%s, target_points=%s, max_dropped_min=%s, reasons=%s).",
                len(dropped_segments),
                len(per_segment_min_lengths),
                target_points,
                max_dropped,
                reason_counts,
            )

        target_points = min(target_points, self.config.length)
        lengths = self._sample_segment_lengths_with_minima(
            rng=rng,
            target_points=target_points,
            minimum_lengths=per_segment_min_lengths,
        )

        occupied_global = np.zeros(self.config.length, dtype=np.int8)
        occupied_per_channel = np.zeros(
            (self.config.channels, self.config.length), dtype=np.int8
        )
        segments: List[SegmentPlan] = []
        for length, attrs in zip(lengths, per_segment_attrs):
            selected = self._sample_uniform_slot(
                rng=rng,
                length=int(length),
                overlap_policy=overlap_policy,
                occupied_global=occupied_global,
                occupied_per_channel=occupied_per_channel,
            )
            if selected is None:
                raise ValueError(
                    "Failed to place a trend_parameter_aware segment without overlap "
                    f"(length={length})."
                )
            self._occupy_slot(
                selected.start,
                selected.end,
                selected.channel,
                overlap_policy,
                occupied_global,
                occupied_per_channel,
            )
            selected.attrs.update(copy.deepcopy(attrs))
            segments.append(selected)

        if clean_values is not None:
            if clean_values.shape != (self.config.length, self.config.channels):
                raise ValueError(
                    "trend_parameter_aware_segments expected clean_values shape "
                    f"({self.config.length}, {self.config.channels}), got {clean_values.shape}."
                )
            sq_prefix_by_channel = [
                np.concatenate(
                    [[0.0], np.cumsum(np.square(clean_values[:, channel]), dtype=np.float64)]
                )
                for channel in range(self.config.channels)
            ]
            abs_by_channel = [
                np.abs(clean_values[:, channel]).astype(np.float64)
                for channel in range(self.config.channels)
            ]
            for segment in segments:
                channel = int(segment.channel)
                window_rms = self._window_rms(
                    sq_prefix_by_channel[channel], int(segment.start), int(segment.end)
                )
                window_peak = float(
                    np.max(abs_by_channel[channel][int(segment.start) : int(segment.end)])
                )
                segment.attrs.update(
                    {
                        "window_rms": float(window_rms),
                        "window_peak": float(window_peak),
                    }
                )

        segments.sort(
            key=lambda segment: (segment.start, segment.channel, segment.length)
        )
        return segments

    def _sample_mode_boundary_segments(
        self,
        rng: np.random.Generator,
        target_density: float,
        overlap_policy: str,
        planner_cfg: Mapping[str, Any],
        anomaly_policy: Mapping[str, Any],
        clean_values: Optional[np.ndarray],
        anomaly_type: str,
    ) -> List[SegmentPlan]:
        """Sample segments aligned to realized mode-change boundaries.

        This planner is intended for `mode-correlation` on `random-mode-jump`
        bases so the relation anomaly does not degrade into a boundary seam.
        """
        if anomaly_type != "mode-correlation":
            raise ValueError(
                "mode_boundary_segments planner is only supported for anomaly_type='mode-correlation'."
            )
        if clean_values is None:
            raise ValueError(
                "mode_boundary_segments planner requires clean_values."
            )
        if clean_values.shape != (self.config.length, self.config.channels):
            raise ValueError(
                "mode_boundary_segments expected clean_values shape "
                f"({self.config.length}, {self.config.channels}), got {clean_values.shape}."
            )

        def stable_sign(values: np.ndarray) -> np.ndarray:
            signs = np.sign(np.asarray(values, dtype=np.float64)).astype(np.int8)
            last = 1
            for idx, value in enumerate(signs):
                if value == 0:
                    signs[idx] = last
                else:
                    last = int(value)
            return signs

        reference = np.asarray(clean_values[:, 0], dtype=np.float64)
        sign_trace = stable_sign(reference)
        change_boundaries = (
            np.flatnonzero(sign_trace[1:] != sign_trace[:-1]).astype(int) + 1
        )
        change_boundaries = change_boundaries[
            (change_boundaries > 0) & (change_boundaries < self.config.length)
        ]
        if change_boundaries.shape[0] < 2:
            self.logger.warning(
                "mode_boundary_segments found too few mode changes; falling back to uniform placement."
            )
            return self._sample_segment_plan(
                rng=rng,
                target_density=target_density,
                anomaly_type=anomaly_type,
                clean_values=clean_values,
                anomaly_policy=_merge_dicts(anomaly_policy, {"channel_policy": "single-random"}),
                planner_cfg={"planner": "uniform_segments"},
            )

        segment_count_range = _parse_pair_int(
            anomaly_policy.get(
                "segment_count_range",
                planner_cfg.get("segment_count_range", self.config.segment_count_range),
            ),
            "segment_count_range",
        )
        if segment_count_range[0] > segment_count_range[1]:
            segment_count_range = (segment_count_range[1], segment_count_range[0])
        n_segments = int(
            rng.integers(segment_count_range[0], segment_count_range[1] + 1)
        )
        target_points = int(round(target_density * self.config.length))
        target_points = max(target_points, n_segments)
        target_points = min(target_points, self.config.length)

        min_segment_length = int(
            anomaly_policy.get(
                "min_segment_length",
                planner_cfg.get(
                    "min_segment_length", self._minimum_segment_length(anomaly_type)
                ),
            )
        )
        max_segments_for_min_length = max(1, target_points // max(1, min_segment_length))
        if n_segments > max_segments_for_min_length:
            n_segments = max_segments_for_min_length
        lengths = self._sample_segment_lengths(
            rng, target_points, n_segments, min_segment_length=min_segment_length
        )

        occupied_global = np.zeros(self.config.length, dtype=np.int8)
        occupied_per_channel = np.zeros(
            (self.config.channels, self.config.length), dtype=np.int8
        )
        segments: List[SegmentPlan] = []

        for length in lengths:
            placed = False
            for _ in range(self.config.max_placement_attempts):
                channel = int(rng.integers(0, self.config.channels))
                start = int(change_boundaries[int(rng.integers(0, len(change_boundaries) - 1))])
                end_candidates = change_boundaries[
                    change_boundaries >= start + max(1, min_segment_length)
                ]
                if end_candidates.size == 0:
                    continue
                candidate_lengths = end_candidates - start
                end = int(
                    end_candidates[int(np.argmin(np.abs(candidate_lengths - int(length))))]
                )
                if not self._is_slot_available(
                    start,
                    end,
                    channel,
                    overlap_policy,
                    occupied_global,
                    occupied_per_channel,
                ):
                    continue
                self._occupy_slot(
                    start,
                    end,
                    channel,
                    overlap_policy,
                    occupied_global,
                    occupied_per_channel,
                )
                segments.append(
                    SegmentPlan(
                        start=start,
                        end=end,
                        length=int(end - start),
                        channel=channel,
                        attrs={
                            "planner": "mode_boundary_segments",
                            "mode_change_aligned": True,
                        },
                    )
                )
                placed = True
                break

            if placed:
                continue

            channel_order = rng.permutation(self.config.channels).tolist()
            for channel in channel_order:
                for start in change_boundaries[:-1]:
                    end_candidates = change_boundaries[
                        change_boundaries >= int(start) + max(1, min_segment_length)
                    ]
                    if end_candidates.size == 0:
                        continue
                    candidate_lengths = end_candidates - int(start)
                    end = int(end_candidates[int(np.argmin(np.abs(candidate_lengths - int(length))))])
                    if not self._is_slot_available(
                        int(start),
                        end,
                        int(channel),
                        overlap_policy,
                        occupied_global,
                        occupied_per_channel,
                    ):
                        continue
                    self._occupy_slot(
                        int(start),
                        end,
                        int(channel),
                        overlap_policy,
                        occupied_global,
                        occupied_per_channel,
                    )
                    segments.append(
                        SegmentPlan(
                            start=int(start),
                            end=end,
                            length=int(end - int(start)),
                            channel=int(channel),
                            attrs={
                                "planner": "mode_boundary_segments",
                                "mode_change_aligned": True,
                            },
                        )
                    )
                    placed = True
                    break
                if placed:
                    break

            if not placed:
                raise ValueError(
                    f"Failed to place a mode_boundary segment of target length {length}."
                )

        segments.sort(
            key=lambda segment: (segment.start, segment.channel, segment.length)
        )
        return segments

    def _sample_point_event_segments(
        self,
        rng: np.random.Generator,
        target_density: float,
        overlap_policy: str,
        planner_cfg: Mapping[str, Any],
    ) -> List[SegmentPlan]:
        density_range_raw = planner_cfg.get("density_range")
        if density_range_raw is not None:
            planner_density_range = _parse_pair(density_range_raw, "density_range")
            density = float(
                np.clip(
                    target_density, planner_density_range[0], planner_density_range[1]
                )
            )
        else:
            density = float(target_density)

        n_points = int(np.floor(density * self.config.length))
        n_points = max(1, min(n_points, self.config.length))
        unique_timestamps = bool(planner_cfg.get("unique_timestamps", True))

        if unique_timestamps:
            timestamps = rng.choice(
                self.config.length, size=n_points, replace=False
            ).astype(int)
        else:
            timestamps = rng.integers(0, self.config.length, size=n_points, dtype=int)
        timestamps = np.sort(timestamps)

        segments: List[SegmentPlan] = []
        occupied_global = np.zeros(self.config.length, dtype=np.int8)
        occupied_per_channel = np.zeros(
            (self.config.channels, self.config.length), dtype=np.int8
        )
        for timestamp in timestamps.tolist():
            channel = int(rng.integers(0, self.config.channels))
            start = int(timestamp)
            end = start + 1
            if not self._is_slot_available(
                start,
                end,
                channel,
                overlap_policy,
                occupied_global,
                occupied_per_channel,
            ):
                placed = False
                for fallback_channel in rng.permutation(self.config.channels).tolist():
                    if self._is_slot_available(
                        start,
                        end,
                        fallback_channel,
                        overlap_policy,
                        occupied_global,
                        occupied_per_channel,
                    ):
                        channel = int(fallback_channel)
                        placed = True
                        break
                if not placed:
                    continue
            self._occupy_slot(
                start,
                end,
                channel,
                overlap_policy,
                occupied_global,
                occupied_per_channel,
            )
            segments.append(
                SegmentPlan(start=start, end=end, length=1, channel=channel)
            )

        segments.sort(
            key=lambda segment: (segment.start, segment.channel, segment.length)
        )
        return segments

    def _compute_period_boundaries(self, base_frequency: Any) -> Optional[np.ndarray]:
        if not isinstance(base_frequency, (int, float)):
            return None
        frequency = float(base_frequency)
        if frequency <= 0:
            return None
        if self.config.length <= 1:
            return None
        # Base oscillators are sampled with np.linspace(..., endpoint=True), therefore
        # the effective period in sample indices is slightly shorter than SAMPLING_F / f.
        period_size = (float(SAMPLING_F) * float(self.config.length - 1)) / (
            frequency * float(self.config.length)
        )
        if not np.isfinite(period_size) or period_size <= 1:
            return None
        max_periods = int(np.floor(self.config.length / period_size))
        if max_periods <= 0:
            return None
        boundaries = np.round(
            np.arange(max_periods + 1, dtype=np.float64) * period_size
        ).astype(int)
        boundaries = np.clip(boundaries, 0, self.config.length)
        boundaries = np.unique(boundaries)
        if boundaries.size == 0 or boundaries[0] != 0:
            boundaries = np.concatenate([[0], boundaries])
        if boundaries[-1] != self.config.length:
            boundaries = np.concatenate([boundaries, [self.config.length]])
        boundaries = np.unique(boundaries)
        if boundaries.size < 2:
            return None
        return boundaries

    def _sanitize_period_boundaries(
        self, period_boundaries: Optional[Iterable[int]]
    ) -> Optional[np.ndarray]:
        if period_boundaries is None:
            return None
        try:
            boundaries = np.array([int(value) for value in period_boundaries], dtype=int)
        except TypeError:
            return None
        if boundaries.size == 0:
            return None
        boundaries = np.clip(boundaries, 0, self.config.length)
        boundaries = np.unique(boundaries)
        if boundaries.size == 0 or boundaries[0] != 0:
            boundaries = np.concatenate([[0], boundaries])
        if boundaries[-1] != self.config.length:
            boundaries = np.concatenate([boundaries, [self.config.length]])
        boundaries = np.unique(boundaries)
        if boundaries.size < 2:
            return None
        if np.any(np.diff(boundaries) <= 0):
            return None
        return boundaries

    def _sample_segment_lengths(
        self,
        rng: np.random.Generator,
        total_points: int,
        n_segments: int,
        min_segment_length: int = 1,
    ) -> List[int]:
        effective_min = max(1, min_segment_length)
        if effective_min * n_segments > total_points:
            effective_min = max(1, total_points // n_segments)

        raw = rng.random(n_segments)
        raw_sum = float(raw.sum())
        if raw_sum == 0:
            raw = np.ones(n_segments)
            raw_sum = float(raw.sum())
        lengths = np.floor(raw / raw_sum * total_points).astype(int)
        lengths = np.maximum(lengths, effective_min)

        diff = int(total_points - lengths.sum())
        if diff > 0:
            for idx in rng.permutation(n_segments):
                lengths[idx] += 1
                diff -= 1
                if diff == 0:
                    break
            while diff > 0:
                idx = int(rng.integers(0, n_segments))
                lengths[idx] += 1
                diff -= 1
        elif diff < 0:
            while diff < 0:
                candidates = np.where(lengths > effective_min)[0]
                if len(candidates) == 0:
                    break
                idx = int(rng.choice(candidates))
                lengths[idx] -= 1
                diff += 1

        lengths_list = [int(x) for x in lengths]
        assert sum(lengths_list) == total_points
        return lengths_list

    def _sample_bounded_integer_lengths(
        self,
        rng: np.random.Generator,
        total_points: int,
        n_segments: int,
        min_value: int,
        max_value: int,
    ) -> List[int]:
        min_value = max(1, int(min_value))
        max_value = max(min_value, int(max_value))
        if n_segments <= 0:
            raise ValueError("n_segments must be > 0 for bounded integer sampling.")
        if n_segments * min_value > total_points:
            raise ValueError(
                "Cannot satisfy bounded integer sampling: "
                f"n_segments * min_value > total_points ({n_segments}*{min_value}>{total_points})."
            )
        if n_segments * max_value < total_points:
            raise ValueError(
                "Cannot satisfy bounded integer sampling: "
                f"n_segments * max_value < total_points ({n_segments}*{max_value}<{total_points})."
            )

        values = np.full(n_segments, min_value, dtype=int)
        capacities = np.full(n_segments, max_value - min_value, dtype=int)
        remaining = int(total_points - n_segments * min_value)

        while remaining > 0:
            candidates = np.where(capacities > 0)[0]
            if candidates.size == 0:
                raise ValueError(
                    "Bounded integer sampler ran out of capacity before reaching target total."
                )
            idx = int(rng.choice(candidates))
            values[idx] += 1
            capacities[idx] -= 1
            remaining -= 1

        return [int(v) for v in values.tolist()]

    def _minimum_segment_length(self, anomaly_type: str) -> int:
        if anomaly_type in self.config.min_segment_length_by_anomaly:
            return int(self.config.min_segment_length_by_anomaly[anomaly_type])
        minimums = {
            "amplitude": 5,
            "mean": 5,
            "variance": 5,
            "platform": 5,
            "pattern": 5,
            "pattern-shift": 5,
            "trend": 5,
            "mode-correlation": 5,
        }
        return minimums.get(anomaly_type, 1)

    def _special_policy(self, anomaly_type: str) -> Dict[str, Any]:
        raw = self.config.special_anomaly_policies.get(anomaly_type, {})
        if isinstance(raw, Mapping):
            special = copy.deepcopy(dict(raw))
        else:
            special = {}
        if anomaly_type == "mode-correlation":
            special.setdefault("channel_policy", "paired-random")
        return special

    def _resolve_segment_planner(
        self,
        anomaly_type: str,
        planner_override: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        default_cfg = {"planner": "uniform_segments"}
        configured_default = self.config.segment_planner.get("default", {})
        planner_cfg = _merge_dicts(default_cfg, configured_default)

        if anomaly_type in self.config.segment_planner:
            planner_cfg = _merge_dicts(
                planner_cfg, self.config.segment_planner[anomaly_type]
            )
        elif anomaly_type == "extremum" and "extremum" in self.config.segment_planner:
            planner_cfg = _merge_dicts(
                planner_cfg, self.config.segment_planner["extremum"]
            )

        special = self._special_policy(anomaly_type)
        special_planner = special.get("segment_planner", {})
        if isinstance(special_planner, Mapping):
            planner_cfg = _merge_dicts(planner_cfg, special_planner)
        if isinstance(planner_override, Mapping):
            planner_cfg = _merge_dicts(planner_cfg, planner_override)

        planner_name = str(planner_cfg.get("planner", "uniform_segments")).lower()
        planner_cfg["planner"] = planner_name
        return planner_cfg

    @staticmethod
    def _is_slot_available(
        start: int,
        end: int,
        channel: int,
        overlap_policy: str,
        occupied_global: np.ndarray,
        occupied_per_channel: np.ndarray,
    ) -> bool:
        if overlap_policy == "global":
            return int(occupied_global[start:end].sum()) == 0
        return int(occupied_per_channel[channel, start:end].sum()) == 0

    @staticmethod
    def _occupy_slot(
        start: int,
        end: int,
        channel: int,
        overlap_policy: str,
        occupied_global: np.ndarray,
        occupied_per_channel: np.ndarray,
    ) -> None:
        if overlap_policy == "global":
            occupied_global[start:end] = 1
        else:
            occupied_per_channel[channel, start:end] = 1

    def _resolve_anomaly_parameters_for_segments(
        self,
        anomaly_type: str,
        anomaly_parameter_template: Mapping[str, Any],
        fixed_anomaly_parameters: Optional[Mapping[str, Any]],
        anomaly_parameters_instance: Optional[Mapping[str, Any]],
        segment_plan: List[SegmentPlan],
        parameter_seed: int,
        planner_cfg: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        planner_name = str(
            (planner_cfg or {}).get("planner", "uniform_segments")
        ).lower()

        if self.config.anomaly_parameter_policy == "fixed_per_variant":
            if fixed_anomaly_parameters is None:
                rng = np.random.default_rng(
                    _derive_seed(parameter_seed, "fixed-anomaly")
                )
                resolved = self._realize_parameters(anomaly_parameter_template, rng)
            else:
                resolved = copy.deepcopy(dict(fixed_anomaly_parameters))
            resolved = self._sanitize_anomaly_parameters(anomaly_type, resolved)
            segment_params = [copy.deepcopy(resolved) for _ in segment_plan]
        elif self.config.anomaly_parameter_policy == "random_per_instance":
            if anomaly_parameters_instance is None:
                rng = np.random.default_rng(
                    _derive_seed(parameter_seed, "instance-anomaly")
                )
                resolved = self._realize_parameters(anomaly_parameter_template, rng)
            else:
                resolved = copy.deepcopy(dict(anomaly_parameters_instance))
            resolved = self._sanitize_anomaly_parameters(anomaly_type, resolved)
            segment_params = [copy.deepcopy(resolved) for _ in segment_plan]
        else:
            segment_params = []
            for idx, _segment in enumerate(segment_plan):
                rng = np.random.default_rng(
                    _derive_seed(parameter_seed, "segment", str(idx))
                )
                resolved = self._realize_parameters(anomaly_parameter_template, rng)
                segment_params.append(
                    self._sanitize_anomaly_parameters(anomaly_type, resolved)
                )

        if anomaly_type == "trend" and planner_name == "trend_parameter_aware_segments":
            for idx, segment in enumerate(segment_plan):
                planned = segment.attrs.get("trend_params")
                if isinstance(planned, Mapping):
                    segment_params[idx] = self._sanitize_anomaly_parameters(
                        "trend", copy.deepcopy(dict(planned))
                    )

        if anomaly_type == "frequency" and planner_name == "period_locked_frequency":
            offsets_raw = (planner_cfg or {}).get("period_ratio_offsets", [-1, 1])
            if not isinstance(offsets_raw, (list, tuple)):
                offsets_raw = [-1, 1]
            offsets = sorted(
                {
                    int(offset)
                    for offset in offsets_raw
                    if isinstance(offset, (int, float)) and int(offset) != 0
                }
            )
            if len(offsets) == 0:
                raise ValueError(
                    "period_locked_frequency requires non-zero 'period_ratio_offsets'."
                )
            factor_bounds_raw = (planner_cfg or {}).get("frequency_factor_bounds")
            factor_bounds: Optional[Tuple[float, float]] = None
            if factor_bounds_raw is not None:
                factor_bounds = _parse_pair(
                    factor_bounds_raw,
                    "segment_planner.frequency_factor_bounds",
                )
                if factor_bounds[0] > factor_bounds[1]:
                    factor_bounds = (factor_bounds[1], factor_bounds[0])

            for idx, segment in enumerate(segment_plan):
                period_count = int(segment.attrs.get("period_count", 0))
                if period_count <= 0:
                    raise ValueError(
                        "period_locked_frequency requires segment attrs['period_count']."
                    )
                candidate_factors = [
                    float(period_count + offset) / float(period_count)
                    for offset in offsets
                    if period_count + offset > 0
                ]
                if factor_bounds is not None:
                    candidate_factors = [
                        factor
                        for factor in candidate_factors
                        if factor_bounds[0] <= factor <= factor_bounds[1]
                    ]
                if len(candidate_factors) == 0:
                    raise ValueError(
                        "No valid frequency_factor candidates for period-locked "
                        f"segment with period_count={period_count} and offsets={offsets}."
                    )
                segment_rng = np.random.default_rng(
                    _derive_seed(parameter_seed, "period-locked-frequency", str(idx))
                )
                chosen_idx = int(segment_rng.integers(0, len(candidate_factors)))
                segment_params[idx]["frequency_factor"] = float(
                    candidate_factors[chosen_idx]
                )

        if anomaly_type == "amplitude":
            planner = dict(planner_cfg or {})
            adaptive_strength = bool(planner.get("adaptive_strength", False))
            min_effect_delta = max(0.0, float(planner.get("min_effect_delta", 0.0)))
            factor_bounds_raw = planner.get("amplitude_factor_bounds")
            factor_bounds: Optional[Tuple[float, float]] = None
            if factor_bounds_raw is not None:
                factor_bounds = _parse_pair(
                    factor_bounds_raw, "segment_planner.amplitude_factor_bounds"
                )
                if factor_bounds[0] > factor_bounds[1]:
                    factor_bounds = (factor_bounds[1], factor_bounds[0])

            deadzone_raw = planner.get("amplitude_factor_deadzone")
            deadzone: Optional[Tuple[float, float]] = None
            if deadzone_raw is not None:
                deadzone = _parse_pair(
                    deadzone_raw, "segment_planner.amplitude_factor_deadzone"
                )
                if deadzone[0] > deadzone[1]:
                    deadzone = (deadzone[1], deadzone[0])

            for idx, segment in enumerate(segment_plan):
                if "amplitude_factor" not in segment_params[idx]:
                    continue
                factor = float(segment_params[idx]["amplitude_factor"])
                if adaptive_strength and min_effect_delta > 0.0:
                    window_rms = float(segment.attrs.get("window_rms", 0.0))
                    if window_rms > 1e-12:
                        required_offset = min_effect_delta / window_rms
                        current_offset = abs(factor - 1.0)
                        if required_offset > current_offset:
                            direction = -1.0 if factor < 1.0 else 1.0
                            if current_offset <= 1e-12:
                                direction = -1.0 if (idx % 2 == 0) else 1.0
                            factor = 1.0 + direction * required_offset

                if deadzone is not None and deadzone[0] <= factor <= deadzone[1]:
                    if factor >= 1.0:
                        factor = deadzone[1]
                    else:
                        factor = deadzone[0]

                if factor_bounds is not None:
                    factor = float(np.clip(factor, factor_bounds[0], factor_bounds[1]))
                segment_params[idx]["amplitude_factor"] = float(factor)

        if anomaly_type == "trend":
            planner = dict(planner_cfg or {})
            adaptive_strength = bool(planner.get("adaptive_strength", False))
            min_effect_delta = max(0.0, float(planner.get("min_effect_delta", 0.0)))
            for idx, segment in enumerate(segment_plan):
                if adaptive_strength and min_effect_delta > 0.0:
                    current_floor = float(segment_params[idx].get("min_effect_delta", 0.0))
                    segment_params[idx]["min_effect_delta"] = float(
                        max(current_floor, min_effect_delta)
                    )
                if "window_rms" in segment.attrs:
                    segment_params[idx]["window_rms"] = float(segment.attrs["window_rms"])
                if "window_peak" in segment.attrs:
                    segment_params[idx]["window_peak"] = float(segment.attrs["window_peak"])

        return segment_params

    def _realize_parameters(
        self, template: Mapping[str, Any], rng: np.random.Generator
    ) -> Dict[str, Any]:
        def realize(value: Any) -> Any:
            if isinstance(value, Mapping):
                mapping = dict(value)
                if "distribution" in mapping:
                    return sample_distribution(mapping)
                if "min" in mapping and "max" in mapping and len(mapping) == 2:
                    return _sample_between(mapping["min"], mapping["max"], rng)
                return {str(k): realize(v) for k, v in mapping.items()}
            if isinstance(value, (list, tuple)):
                if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                    return _sample_between(value[0], value[1], rng)
                return [realize(v) for v in value]
            return value

        def sample_distribution(spec: Mapping[str, Any]) -> Any:
            distribution = str(spec.get("distribution", "")).lower()
            if distribution == "uniform":
                low = float(spec.get("low", spec.get("min")))
                high = float(spec.get("high", spec.get("max")))
                sampled = float(rng.uniform(min(low, high), max(low, high)))
            elif distribution == "int_uniform":
                low = int(spec.get("low", spec.get("min")))
                high = int(spec.get("high", spec.get("max")))
                if low > high:
                    low, high = high, low
                sampled = int(rng.integers(low, high + 1))
            elif distribution == "choice":
                values = spec.get("values", spec.get("choices"))
                if not isinstance(values, (list, tuple)) or len(values) == 0:
                    raise ValueError(
                        "choice distribution requires non-empty 'values' list"
                    )
                idx = int(rng.integers(0, len(values)))
                sampled = realize(copy.deepcopy(values[idx]))
            elif distribution == "bernoulli":
                p = float(spec.get("p", 0.5))
                if p < 0.0 or p > 1.0:
                    raise ValueError("bernoulli distribution requires p in [0, 1]")
                sampled = bool(rng.random() < p)
            elif distribution == "reject_if_abs_lt":
                threshold = float(
                    spec.get("threshold", spec.get("reject_if_abs_lt", 0.0))
                )
                base_spec = spec.get("base")
                if base_spec is None:
                    raise ValueError(
                        "reject_if_abs_lt distribution requires nested 'base' specification"
                    )
                sampled = sample_with_abs_rejection(base_spec, threshold)
            else:
                raise ValueError(
                    f"Unsupported distribution primitive: '{distribution}'"
                )

            if "reject_if_abs_lt" in spec and distribution != "reject_if_abs_lt":
                threshold = float(spec["reject_if_abs_lt"])
                sampled = sample_with_abs_rejection(spec, threshold, nested=False)
            return sampled

        def sample_with_abs_rejection(
            base_spec: Any,
            threshold: float,
            nested: bool = True,
            max_tries: int = 512,
        ) -> Any:
            threshold = max(0.0, float(threshold))
            for _ in range(max_tries):
                if isinstance(base_spec, Mapping):
                    spec_dict = dict(base_spec)
                    if not nested:
                        spec_dict = {
                            key: value
                            for key, value in spec_dict.items()
                            if key != "reject_if_abs_lt"
                        }
                    candidate = sample_distribution(spec_dict)
                else:
                    candidate = realize(base_spec)
                if isinstance(candidate, bool):
                    return candidate
                if (
                    isinstance(candidate, (int, float))
                    and abs(float(candidate)) >= threshold
                ):
                    return candidate
            sign = -1.0 if rng.random() < 0.5 else 1.0
            return float(sign * threshold)

        return {str(k): realize(v) for k, v in dict(template).items()}

    def _sanitize_anomaly_parameters(
        self, anomaly_type: str, parameters: Mapping[str, Any]
    ) -> Dict[str, Any]:
        resolved = copy.deepcopy(dict(parameters))
        if anomaly_type in {"amplitude", "trend", "mean", "platform", "variance", "pattern"} and "transition_length" in resolved:
            transition_length = int(resolved["transition_length"])
            resolved["transition_length"] = max(0, transition_length)
        if anomaly_type == "pattern":
            if "min_effect_delta" in resolved:
                resolved["min_effect_delta"] = max(
                    0.0, float(resolved["min_effect_delta"])
                )
            if "min_window_ptp" in resolved:
                resolved["min_window_ptp"] = max(0.0, float(resolved["min_window_ptp"]))
            if "adaptive_blend" in resolved:
                resolved["adaptive_blend"] = bool(resolved["adaptive_blend"])
            if "blend_strength" in resolved:
                resolved["blend_strength"] = max(0.0, float(resolved["blend_strength"]))
        if anomaly_type == "trend":
            if "boundary_mode" in resolved:
                resolved["boundary_mode"] = str(resolved["boundary_mode"]).lower()
            if "envelope_kind" in resolved:
                resolved["envelope_kind"] = str(resolved["envelope_kind"]).lower()
            if "min_effect_delta" in resolved:
                resolved["min_effect_delta"] = max(0.0, float(resolved["min_effect_delta"]))
        if anomaly_type == "pattern-shift":
            transition_window = int(resolved.get("transition_window", 10))
            transition_window = max(1, abs(transition_window))
            shift_by = int(resolved.get("shift_by", 0))
            shift_by = int(np.clip(shift_by, -transition_window, transition_window))
            if shift_by == 0 and transition_window > 0:
                shift_by = 1
            resolved["transition_window"] = transition_window
            resolved["shift_by"] = shift_by
            if "crossfade_mode" in resolved:
                resolved["crossfade_mode"] = str(resolved["crossfade_mode"]).lower()
            if "min_effect_delta" in resolved:
                resolved["min_effect_delta"] = max(0.0, float(resolved["min_effect_delta"]))
        if anomaly_type == "platform" and "min_effect_delta" in resolved:
            resolved["min_effect_delta"] = max(
                0.0, float(resolved["min_effect_delta"])
            )
        if anomaly_type == "variance" and "min_effect_delta" in resolved:
            resolved["min_effect_delta"] = max(
                0.0, float(resolved["min_effect_delta"])
            )
        if anomaly_type == "extremum":
            context_window = int(resolved.get("context_window", 10))
            resolved["context_window"] = max(1, abs(context_window))
            if "min" in resolved:
                resolved["min"] = bool(resolved["min"])
            if "local" in resolved:
                resolved["local"] = bool(resolved["local"])
        return resolved

    def _build_anomalies(
        self,
        anomaly_type: str,
        anomaly_parameters_per_segment: List[Dict[str, Any]],
        segment_plan: Iterable[SegmentPlan],
    ) -> List[Anomaly]:
        segments = list(segment_plan)
        if len(segments) != len(anomaly_parameters_per_segment):
            raise ValueError(
                "segment_plan and anomaly_parameters_per_segment must have equal length"
            )
        anomalies: List[Anomaly] = []
        for segment, anomaly_parameters in zip(
            segments, anomaly_parameters_per_segment
        ):
            anomaly = Anomaly(
                position=Position.Middle,
                exact_position=segment.start,
                anomaly_length=segment.length,
                channel=segment.channel,
                creeping_length=0,
            )
            anomaly_kind_object = self._build_single_anomaly_kind(
                anomaly_type=anomaly_type,
                parameters=anomaly_parameters,
                anomaly_length=segment.length,
            )
            anomaly.set_anomaly(anomaly_kind_object)
            anomalies.append(anomaly)
        return anomalies

    def _build_single_anomaly_kind(
        self, anomaly_type: str, parameters: Mapping[str, Any], anomaly_length: int
    ) -> Any:
        if anomaly_type == "trend":
            raw = copy.deepcopy(dict(parameters))
            oscillation = raw.get(
                PARAMETERS.OSCILLATION,
                {"kind": "sine", "frequency": 2.0, "amplitude": 1.0},
            )
            trend = decode_trend_obj(copy.deepcopy(oscillation), anomaly_length)
            trend_parameters: Dict[str, Any] = {PARAMETERS.TREND: trend}
            if "transition_length" in raw:
                trend_parameters["transition_length"] = int(raw["transition_length"])
            if "boundary_mode" in raw:
                trend_parameters["boundary_mode"] = str(raw["boundary_mode"])
            if "envelope_kind" in raw:
                trend_parameters["envelope_kind"] = str(raw["envelope_kind"])
            if "min_effect_delta" in raw:
                trend_parameters["min_effect_delta"] = float(raw["min_effect_delta"])
            return AnomalyKind(anomaly_type).create(trend_parameters)
        return AnomalyKind(anomaly_type).create(copy.deepcopy(dict(parameters)))

    def _apply_anomalies(
        self,
        anomaly_objects: List[Anomaly],
        segment_plan: List[SegmentPlan],
        base: np.ndarray,
        channel_bos: List[Any],
        anomaly_seed: int,
        anomaly_type: str,
        anomaly_parameters_per_segment: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        labels = np.zeros((self.config.length, self.config.channels), dtype=np.int8)
        events: List[Dict[str, Any]] = []
        ctx = GenerationContext(SeedSequence(anomaly_seed))
        used_positions: Dict[int, List[Tuple[int, int]]] = {
            channel: [] for channel in range(self.config.channels)
        }
        processed_group_ids = set()

        for segment_idx, anomaly in enumerate(anomaly_objects):
            segment_metadata = segment_plan[segment_idx]
            group_id = int(segment_metadata.attrs.get("group_id", segment_idx))
            group_channels = [
                int(ch)
                for ch in segment_metadata.attrs.get(
                    "group_channels", [int(segment_metadata.channel)]
                )
            ]
            if (
                anomaly_type == "mode-correlation"
                and len(group_channels) > 1
                and all(
                    channel_bos[ch].get_base_oscillation_kind() == RandomModeJump.KIND
                    for ch in group_channels
                )
            ):
                if group_id in processed_group_ids:
                    continue
                processed_group_ids.add(group_id)
                group_indices = [
                    idx
                    for idx, plan in enumerate(segment_plan)
                    if int(plan.attrs.get("group_id", idx)) == group_id
                ]
                group_events = self._apply_mode_correlation_group(
                    group_indices=group_indices,
                    segment_plan=segment_plan,
                    base=base,
                    channel_bos=channel_bos,
                    labels=labels,
                    used_positions=used_positions,
                    anomaly_type=anomaly_type,
                    anomaly_parameters_per_segment=anomaly_parameters_per_segment,
                )
                events.extend(group_events)
                continue
            channel = anomaly.channel
            bo = channel_bos[channel]
            planned_start = (
                int(anomaly.exact_position)
                if getattr(anomaly, "exact_position", None) is not None
                else None
            )
            planned_end = (
                planned_start + int(anomaly.anomaly_length)
                if planned_start is not None
                else None
            )
            before_window_with_variations = (
                self._compose_channel_window_with_variations(
                    base=base,
                    bo=bo,
                    channel=channel,
                    start=planned_start,
                    end=planned_end,
                )
                if planned_start is not None and planned_end is not None
                else None
            )
            protocol = anomaly.generate(ctx.to_anomaly(bo, used_positions[channel]))
            original_segment = np.array(
                base[protocol.start : protocol.end, channel], copy=True
            )
            expected_length = int(protocol.end - protocol.start)
            effective_delta = np.zeros(expected_length, dtype=np.float64)
            if protocol.subsequences:
                subsequence = np.vstack(protocol.subsequences).sum(axis=0)
                subsequence = self._normalize_subsequence_length(
                    subsequence, expected_length
                )
                base[protocol.start : protocol.end, channel] = subsequence
                if (
                    planned_start == int(protocol.start)
                    and planned_end == int(protocol.end)
                    and before_window_with_variations is not None
                    and before_window_with_variations.shape[0] == expected_length
                ):
                    after_window_with_variations = (
                        self._compose_channel_window_with_variations(
                            base=base,
                            bo=bo,
                            channel=channel,
                            start=protocol.start,
                            end=protocol.end,
                        )
                    )
                    effective_delta = np.abs(
                        after_window_with_variations - before_window_with_variations
                    )
                elif subsequence.shape[0] == original_segment.shape[0]:
                    effective_delta = np.abs(subsequence - original_segment)
            elif (
                planned_start == int(protocol.start)
                and planned_end == int(protocol.end)
                and before_window_with_variations is not None
                and before_window_with_variations.shape[0] == expected_length
            ):
                after_window_with_variations = (
                    self._compose_channel_window_with_variations(
                        base=base,
                        bo=bo,
                        channel=channel,
                        start=protocol.start,
                        end=protocol.end,
                    )
                )
                effective_delta = np.abs(
                    after_window_with_variations - before_window_with_variations
                )

            label_start, label_end = self._resolve_label_bounds_from_effect(
                protocol_start=int(protocol.start),
                protocol_end=int(protocol.end),
                delta=effective_delta,
                anomaly_type=anomaly_type,
            )
            labels[label_start:label_end, channel] = 1
            used_positions[channel].append((protocol.start, protocol.end))
            events.append(
                {
                    "start": int(label_start),
                    "end": int(label_end),
                    "channel": int(channel),
                    "anomaly_type": anomaly_type,
                    "group_id": int(segment_metadata.attrs.get("group_id", segment_idx)),
                    "group_channels": [
                        int(ch)
                        for ch in segment_metadata.attrs.get(
                            "group_channels", [int(channel)]
                        )
                    ],
                    "params": _to_builtin_types(
                        anomaly_parameters_per_segment[segment_idx]
                    ),
                    "length": int(label_end - label_start),
                    "source_start": int(protocol.start),
                    "source_end": int(protocol.end),
                }
            )

        events.sort(key=lambda event: (event["start"], event["channel"], event["end"]))
        return labels, events

    def _apply_mode_correlation_group(
        self,
        group_indices: List[int],
        segment_plan: List[SegmentPlan],
        base: np.ndarray,
        channel_bos: List[Any],
        labels: np.ndarray,
        used_positions: Dict[int, List[Tuple[int, int]]],
        anomaly_type: str,
        anomaly_parameters_per_segment: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(group_indices) == 0:
            return []
        group_segments = [segment_plan[idx] for idx in group_indices]
        source_start = int(min(segment.start for segment in group_segments))
        source_end = int(max(segment.end for segment in group_segments))
        if source_end <= source_start:
            return []

        group_id = int(group_segments[0].attrs.get("group_id", group_indices[0]))
        group_channels = sorted(
            {
                int(ch)
                for segment in group_segments
                for ch in segment.attrs.get("group_channels", [int(segment.channel)])
            }
        )
        segment_idx_by_channel = {
            int(segment_plan[idx].channel): int(idx) for idx in group_indices
        }
        if len(group_channels) < 2:
            return []
        anchor_channel = int(group_channels[0])
        flipped_channels = [int(ch) for ch in group_channels[1:]]
        if len(flipped_channels) == 0:
            return []

        before_windows = {
            int(channel): self._compose_channel_window_with_variations(
                base=base,
                bo=channel_bos[int(channel)],
                channel=int(channel),
                start=source_start,
                end=source_end,
            )
            for channel in flipped_channels
        }

        for channel in flipped_channels:
            base[source_start:source_end, channel] = (
                -1.0 * np.asarray(base[source_start:source_end, channel], dtype=np.float64)
            )

        events: List[Dict[str, Any]] = []
        mode_change_aligned = bool(
            group_segments[0].attrs.get("mode_change_aligned", False)
        )
        for channel in flipped_channels:
            segment_idx = int(segment_idx_by_channel[int(channel)])
            after_window = self._compose_channel_window_with_variations(
                base=base,
                bo=channel_bos[int(channel)],
                channel=int(channel),
                start=source_start,
                end=source_end,
            )
            effective_delta = np.abs(after_window - before_windows[int(channel)])
            label_start, label_end = self._resolve_label_bounds_from_effect(
                protocol_start=source_start,
                protocol_end=source_end,
                delta=effective_delta,
                anomaly_type=anomaly_type,
            )
            labels[label_start:label_end, int(channel)] = 1
            used_positions[int(channel)].append((source_start, source_end))
            events.append(
                {
                    "start": int(label_start),
                    "end": int(label_end),
                    "channel": int(channel),
                    "anomaly_type": anomaly_type,
                    "group_id": int(group_id),
                    "group_channels": [int(ch) for ch in group_channels],
                    "params": _to_builtin_types(
                        anomaly_parameters_per_segment[int(segment_idx)]
                    ),
                    "length": int(label_end - label_start),
                    "source_start": int(source_start),
                    "source_end": int(source_end),
                    "anomaly_object": "relation_sign_flip",
                    "anchor_channel": int(anchor_channel),
                    "flipped_channels": [int(ch) for ch in flipped_channels],
                    "mode_change_aligned": bool(mode_change_aligned),
                }
            )
        return events

    def _resolve_label_bounds_from_effect(
        self,
        protocol_start: int,
        protocol_end: int,
        delta: np.ndarray,
        anomaly_type: str,
    ) -> Tuple[int, int]:
        if self.config.support_label_mode == "strict_segment":
            return int(protocol_start), int(protocol_end)
        if delta.size == 0:
            return int(protocol_start), int(protocol_end)
        if self.config.support_eps_mode == "relative":
            peak_delta = float(np.max(delta))
            epsilon = float(self.config.support_eps_value) * peak_delta
        else:
            epsilon = float(self.config.support_eps_value)
        active = np.flatnonzero(delta > epsilon)
        if active.size == 0:
            return int(protocol_start), int(protocol_end)
        start = int(protocol_start + active[0])
        end = int(protocol_start + active[-1] + 1)
        if anomaly_type != "extremum":
            min_label_length = int(
                max(1, self.config.min_effective_label_length_non_extremum)
            )
            if (end - start) < min_label_length:
                return self._expand_effective_support_to_min_length(
                    protocol_start=protocol_start,
                    protocol_end=protocol_end,
                    delta=delta,
                    min_label_length=min_label_length,
                )
        return start, end

    @staticmethod
    def _expand_effective_support_to_min_length(
        protocol_start: int,
        protocol_end: int,
        delta: np.ndarray,
        min_label_length: int,
    ) -> Tuple[int, int]:
        source_length = max(0, int(protocol_end - protocol_start))
        if source_length <= 0:
            return int(protocol_start), int(protocol_end)
        if source_length <= 1 or min_label_length <= 1:
            end = min(int(protocol_start) + 1, int(protocol_end))
            return int(protocol_start), int(end)

        window = min(int(min_label_length), source_length)
        if window <= 1:
            end = min(int(protocol_start) + 1, int(protocol_end))
            return int(protocol_start), int(end)

        if delta.shape[0] != source_length:
            resized = np.interp(
                np.linspace(0.0, 1.0, source_length, endpoint=True),
                np.linspace(0.0, 1.0, max(1, delta.shape[0]), endpoint=True),
                delta if delta.shape[0] > 0 else np.zeros(1, dtype=np.float64),
            ).astype(np.float64)
        else:
            resized = delta

        # Pick the contiguous support window with maximum total effect.
        scores = np.convolve(resized, np.ones(window, dtype=np.float64), mode="valid")
        best_start_local = int(np.argmax(scores)) if scores.size > 0 else 0
        start = int(protocol_start + best_start_local)
        end = int(start + window)
        return start, end

    def _normalize_subsequence_length(
        self, subsequence: np.ndarray, expected_length: int
    ) -> np.ndarray:
        if expected_length <= 0:
            return np.array([], dtype=np.float64)
        if subsequence.shape[0] == expected_length:
            return subsequence
        policy = self.config.length_normalization
        if policy == "none":
            raise ValueError(
                f"Subsequence length mismatch ({subsequence.shape[0]} != {expected_length}) "
                "and length_normalization='none'."
            )
        if policy == "crop":
            if subsequence.shape[0] < expected_length:
                raise ValueError(
                    f"Cannot crop subsequence of length {subsequence.shape[0]} "
                    f"to larger expected length {expected_length}."
                )
            return subsequence[:expected_length]
        if policy == "pad":
            if subsequence.shape[0] > expected_length:
                return subsequence[:expected_length]
            if subsequence.shape[0] == 0:
                return np.zeros(expected_length, dtype=np.float64)
            return np.pad(
                subsequence,
                (0, expected_length - subsequence.shape[0]),
                mode="edge",
            )
        # policy == "resample"
        if subsequence.shape[0] == 0:
            return np.zeros(expected_length, dtype=np.float64)
        if expected_length == 1:
            return np.array([float(subsequence[0])], dtype=np.float64)
        x_old = np.linspace(0.0, 1.0, subsequence.shape[0], endpoint=True)
        x_new = np.linspace(0.0, 1.0, expected_length, endpoint=True)
        return np.interp(x_new, x_old, subsequence).astype(np.float64)

    def _check_labels_and_events_consistency(
        self, labels: np.ndarray, events: List[Mapping[str, Any]]
    ) -> None:
        reconstructed = np.zeros_like(labels)
        for event in events:
            reconstructed[
                int(event["start"]) : int(event["end"]), int(event["channel"])
            ] = 1
        if not np.array_equal(labels, reconstructed):
            raise ValueError("labels_pointwise.csv is inconsistent with events.json")

    def _write_timeseries_csv(self, path: Path, values: np.ndarray) -> None:
        columns = [f"value-{i}" for i in range(self.config.channels)]
        df = pd.DataFrame(values, columns=columns)
        df.to_csv(path, index=False, float_format=self.config.csv_float_format)

    def _write_labels_csv(self, path: Path, labels: np.ndarray) -> None:
        columns = [f"label-{i}" for i in range(self.config.channels)]
        df = pd.DataFrame(labels.astype(np.int8), columns=columns)
        df.to_csv(path, index=False)

    def _write_instance_plots(
        self,
        instance_dir: Path,
        anomalous: np.ndarray,
        events: List[Mapping[str, Any]],
        zoom_seed: int,
    ) -> None:
        self._plot_window(
            output_path=instance_dir / "plot_full.png",
            anomalous=anomalous,
            events=events,
            window_start=0,
            window_end=self.config.length,
            title="Full instance view",
            selected_event=None,
        )

        selected_events = self._select_zoom_events(events, zoom_seed)
        for idx in range(self.config.zoom_count):
            output_path = instance_dir / f"zoom_{idx:02d}.png"
            event = selected_events[idx] if idx < len(selected_events) else None
            if event is None and self.config.zoom_fill_policy == "blank":
                self._write_blank_zoom(output_path, idx)
                continue
            if event is None and len(events) > 0:
                event = events[idx % len(events)]
            if event is None:
                self._write_blank_zoom(output_path, idx)
                continue
            start = int(event["start"])
            end = int(event["end"])
            margin = max(
                self.config.zoom_margin_min,
                self.config.zoom_margin,
                int(round(self.config.zoom_margin_alpha * (end - start))),
            )
            window_start = max(0, start - margin)
            window_end = min(self.config.length, end + margin)
            title = (
                f"Zoom {idx:02d} | type={event['anomaly_type']} | "
                f"channel={event['channel']} | ({start},{end})"
            )
            self._plot_window(
                output_path=output_path,
                anomalous=anomalous,
                events=events,
                window_start=window_start,
                window_end=window_end,
                title=title,
                selected_event=event,
            )

    def _select_zoom_events(
        self, events: List[Mapping[str, Any]], zoom_seed: int
    ) -> List[Mapping[str, Any]]:
        if len(events) == 0:
            return []
        rng = np.random.default_rng(zoom_seed)
        if len(events) <= self.config.zoom_count:
            selected = [events[i] for i in range(len(events))]
        else:
            indices = rng.choice(
                len(events), size=self.config.zoom_count, replace=False
            ).tolist()
            selected = [events[i] for i in indices]
            selected = sorted(
                selected, key=lambda e: (int(e["start"]), int(e["channel"]))
            )
        if (
            len(selected) < self.config.zoom_count
            and self.config.zoom_fill_policy == "repeat"
        ):
            repeated: List[Mapping[str, Any]] = []
            cursor = 0
            while len(selected) + len(repeated) < self.config.zoom_count:
                repeated.append(selected[cursor % len(selected)])
                cursor += 1
            selected = selected + repeated
        return selected[: self.config.zoom_count]

    def _plot_window(
        self,
        output_path: Path,
        anomalous: np.ndarray,
        events: List[Mapping[str, Any]],
        window_start: int,
        window_end: int,
        title: str,
        selected_event: Optional[Mapping[str, Any]],
    ) -> None:
        channels = self.config.channels
        figure_height = max(2 * channels, 8)
        fig, axes = plt.subplots(channels, 1, figsize=(16, figure_height), sharex=True)
        if channels == 1:
            axes = [axes]
        x_values = np.arange(window_start, window_end, dtype=int)

        selected_signature = None
        if selected_event is not None:
            selected_signature = (
                int(selected_event["start"]),
                int(selected_event["end"]),
                int(selected_event["channel"]),
            )
        for channel in range(channels):
            axis = axes[channel]
            axis.plot(
                x_values,
                anomalous[window_start:window_end, channel],
                linewidth=0.8,
                color="#1f77b4",
            )
            axis.set_ylabel(f"ch {channel}")
            has_point_anomaly = False
            has_selected_point = False
            for event in events:
                event_channel = int(event["channel"])
                if event_channel != channel:
                    continue
                start = int(event["start"])
                end = int(event["end"])
                if end <= window_start or start >= window_end:
                    continue
                span_start = max(start, window_start)
                span_end = min(end, window_end)
                signature = (start, end, event_channel)
                is_point_event = end - start <= 1
                if selected_signature is not None and signature == selected_signature:
                    if is_point_event:
                        has_selected_point = True
                        axis.axvline(
                            x=start,
                            color="#d62728",
                            alpha=0.85,
                            linestyle="--",
                            linewidth=1.5,
                        )
                    else:
                        axis.axvspan(
                            span_start,
                            span_end,
                            color="#d62728",
                            alpha=0.30,
                            label="selected anomaly",
                        )
                else:
                    if is_point_event:
                        has_point_anomaly = True
                        axis.axvline(
                            x=start,
                            color="#ff7f0e",
                            alpha=0.70,
                            linestyle="--",
                            linewidth=1.2,
                        )
                    else:
                        axis.axvspan(
                            span_start,
                            span_end,
                            color="#ff7f0e",
                            alpha=0.20,
                            label="anomaly",
                        )
            legend_handles: List[Any] = [
                Patch(color="#ff7f0e", alpha=0.20, label="anomaly"),
                Patch(color="#d62728", alpha=0.30, label="selected anomaly"),
            ]
            if has_point_anomaly:
                legend_handles.append(
                    Line2D(
                        [0, 1],
                        [0, 0],
                        color="#ff7f0e",
                        alpha=0.70,
                        linestyle="--",
                        linewidth=1.2,
                        label="point anomaly",
                    )
                )
            if has_selected_point:
                legend_handles.append(
                    Line2D(
                        [0, 1],
                        [0, 0],
                        color="#d62728",
                        alpha=0.85,
                        linestyle="--",
                        linewidth=1.5,
                        label="selected point anomaly",
                    )
                )
            axis.legend(handles=legend_handles, loc="upper right")
        axes[-1].set_xlabel("time")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=self.config.plot_dpi,
            metadata={
                "Software": "GutenTAG ts-dataset-generator",
                "Date": "1970-01-01",
            },
        )
        plt.close(fig)

    def _write_blank_zoom(self, output_path: Path, zoom_index: int) -> None:
        fig, axis = plt.subplots(1, 1, figsize=(8, 3))
        axis.text(
            0.5, 0.5, f"zoom_{zoom_index:02d}: no event", ha="center", va="center"
        )
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=self.config.plot_dpi,
            metadata={
                "Software": "GutenTAG ts-dataset-generator",
                "Date": "1970-01-01",
            },
        )
        plt.close(fig)


def _read_nested_dict(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if isinstance(value, Mapping):
        return value
    return {}


def _optional_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    raise ValueError(
        f"Expected list/tuple for configuration list field, got {type(raw)}"
    )


def _parse_pair(raw: Any, field_name: str) -> Tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"'{field_name}' must be a list/tuple with exactly two values."
        )
    return float(raw[0]), float(raw[1])


def _parse_pair_int(raw: Any, field_name: str) -> Tuple[int, int]:
    left, right = _parse_pair(raw, field_name)
    return int(left), int(right)


def _parse_pair_profiles(raw: Any) -> Dict[str, List[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("pair_profiles must be a mapping of pair_id -> [profile_ids]")
    parsed: Dict[str, List[str]] = {}
    for pair_id, profile_ids in raw.items():
        if not isinstance(profile_ids, (list, tuple)):
            raise ValueError(
                f"pair_profiles[{pair_id}] must be a list/tuple of profile ids."
            )
        parsed[str(pair_id)] = [str(profile_id) for profile_id in profile_ids]
    return parsed


def _parse_segment_planner(raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("segment_planner must be a mapping")
    parsed: Dict[str, Dict[str, Any]] = {}
    for planner_name, planner_cfg in raw.items():
        if not isinstance(planner_cfg, Mapping):
            raise ValueError(
                f"segment_planner[{planner_name}] must be a mapping of planner config"
            )
        parsed[str(planner_name)] = copy.deepcopy(dict(planner_cfg))
    return parsed


def _merge_dicts(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                nested = dict(merged[key])  # type: ignore[index]
                nested.update(copy.deepcopy(dict(value)))
                merged[key] = nested
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def _derive_seed(seed: int, *parts: str) -> int:
    payload = f"{seed}|" + "|".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _sample_between(lower: Any, upper: Any, rng: np.random.Generator) -> Any:
    if isinstance(lower, bool) or isinstance(upper, bool):
        return bool(lower)
    if isinstance(lower, int) and isinstance(upper, int):
        if lower > upper:
            lower, upper = upper, lower
        return int(rng.integers(lower, upper + 1))
    lower_f = float(lower)
    upper_f = float(upper)
    if lower_f > upper_f:
        lower_f, upper_f = upper_f, lower_f
    return float(rng.uniform(lower_f, upper_f))


def _unique_records(
    items: List[Dict[str, str]], key_fields: Tuple[str, ...]
) -> List[Dict[str, str]]:
    seen = set()
    unique: List[Dict[str, str]] = []
    for item in items:
        key = tuple(item.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _compute_split_statistics(
    split: str,
    instance_summaries: List[Mapping[str, Any]],
    length: int,
    channels: int,
) -> Dict[str, Any]:
    densities = np.array(
        [float(item["achieved_density"]) for item in instance_summaries], dtype=float
    )
    target_densities = np.array(
        [float(item["target_density"]) for item in instance_summaries], dtype=float
    )
    n_segments = np.array(
        [int(item["n_segments"]) for item in instance_summaries], dtype=int
    )
    per_channel_counts_per_instance: Dict[str, List[int]] = {
        str(channel): [] for channel in range(channels)
    }
    for item in instance_summaries:
        per_channel = item["per_channel_segment_counts"]
        for channel in range(channels):
            channel_key = str(channel)
            count = int(per_channel.get(channel_key, 0))
            per_channel_counts_per_instance[channel_key].append(count)

    per_channel_totals: Dict[str, int] = {}
    per_channel_stats: Dict[str, Dict[str, Any]] = {}
    for channel_key, counts in per_channel_counts_per_instance.items():
        values = np.array(counts, dtype=int)
        per_channel_totals[channel_key] = int(values.sum())
        per_channel_stats[channel_key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
            "total": int(values.sum()),
        }

    pooled_lengths = _collect_segment_lengths(instance_summaries)
    summary = {
        "split": split,
        "instances": len(instance_summaries),
        "length": length,
        "channels": channels,
        "target_density_mean": float(np.mean(target_densities)),
        "target_density_std": float(np.std(target_densities)),
        "target_density_min": float(np.min(target_densities)),
        "target_density_max": float(np.max(target_densities)),
        "achieved_density_mean": float(np.mean(densities)),
        "achieved_density_std": float(np.std(densities)),
        "achieved_density_min": float(np.min(densities)),
        "achieved_density_max": float(np.max(densities)),
        "n_segments_mean": float(np.mean(n_segments)),
        "n_segments_std": float(np.std(n_segments)),
        "n_segments_min": int(np.min(n_segments)),
        "n_segments_max": int(np.max(n_segments)),
        "segment_length_mean": float(np.mean(pooled_lengths)),
        "segment_length_std": float(np.std(pooled_lengths)),
        "segment_length_median": float(np.median(pooled_lengths)),
        "segment_length_min": int(np.min(pooled_lengths)),
        "segment_length_max": int(np.max(pooled_lengths)),
        "per_channel_segment_counts_total": per_channel_totals,
        "per_channel_segment_counts_stats": per_channel_stats,
    }
    return _to_builtin_types(summary)


def _compute_dataset_statistics(
    instance_summaries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    if len(instance_summaries) == 0:
        return {
            "instance_count": 0,
            "target_density_mean": None,
            "target_density_std": None,
            "target_density_min": None,
            "target_density_max": None,
            "achieved_density_mean": None,
            "achieved_density_std": None,
            "achieved_density_min": None,
            "achieved_density_max": None,
            "n_segments_mean": None,
            "n_segments_std": None,
            "n_segments_min": None,
            "n_segments_max": None,
            "segment_length_mean": None,
            "segment_length_std": None,
            "segment_length_median": None,
            "segment_length_min": None,
            "segment_length_max": None,
            "per_channel_segment_counts_total": {},
            "per_channel_segment_counts_stats": {},
        }

    densities = np.array(
        [float(item["achieved_density"]) for item in instance_summaries], dtype=float
    )
    target_densities = np.array(
        [float(item["target_density"]) for item in instance_summaries], dtype=float
    )
    n_segments = np.array(
        [int(item["n_segments"]) for item in instance_summaries], dtype=int
    )
    pooled_lengths = _collect_segment_lengths(instance_summaries)
    channel_keys = sorted(
        {
            str(channel)
            for item in instance_summaries
            for channel in item.get("per_channel_segment_counts", {}).keys()
        }
    )
    per_channel_counts_per_instance: Dict[str, List[int]] = {
        channel_key: [] for channel_key in channel_keys
    }
    for item in instance_summaries:
        per_channel = item.get("per_channel_segment_counts", {})
        for channel_key in channel_keys:
            per_channel_counts_per_instance[channel_key].append(
                int(per_channel.get(channel_key, 0))
            )
    per_channel_totals: Dict[str, int] = {}
    per_channel_stats: Dict[str, Dict[str, Any]] = {}
    for channel_key, counts in per_channel_counts_per_instance.items():
        values = np.array(counts, dtype=int)
        per_channel_totals[channel_key] = int(values.sum())
        per_channel_stats[channel_key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
            "total": int(values.sum()),
        }

    return _to_builtin_types(
        {
            "instance_count": len(instance_summaries),
            "target_density_mean": float(np.mean(target_densities)),
            "target_density_std": float(np.std(target_densities)),
            "target_density_min": float(np.min(target_densities)),
            "target_density_max": float(np.max(target_densities)),
            "achieved_density_mean": float(np.mean(densities)),
            "achieved_density_std": float(np.std(densities)),
            "achieved_density_min": float(np.min(densities)),
            "achieved_density_max": float(np.max(densities)),
            "n_segments_mean": float(np.mean(n_segments)),
            "n_segments_std": float(np.std(n_segments)),
            "n_segments_min": int(np.min(n_segments)),
            "n_segments_max": int(np.max(n_segments)),
            "segment_length_mean": float(np.mean(pooled_lengths)),
            "segment_length_std": float(np.std(pooled_lengths)),
            "segment_length_median": float(np.median(pooled_lengths)),
            "segment_length_min": int(np.min(pooled_lengths)),
            "segment_length_max": int(np.max(pooled_lengths)),
            "per_channel_segment_counts_total": per_channel_totals,
            "per_channel_segment_counts_stats": per_channel_stats,
        }
    )


def _collect_segment_lengths(instance_summaries: List[Mapping[str, Any]]) -> np.ndarray:
    lengths: List[float] = []
    for item in instance_summaries:
        segment_lengths = item.get("segment_lengths", [])
        lengths.extend(float(length) for length in segment_lengths)
    if len(lengths) == 0:
        return np.array([0.0])
    return np.array(lengths, dtype=np.float64)


def _to_builtin_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin_types(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_types(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin_types(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin_types(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _try_resolve_git_commit() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            return commit or None
    except Exception:
        return None
    return None


def generate_ts_dataset(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate a TS dataset directly from a raw configuration dictionary.

    Parameters
    ----------
    config : Mapping[str, Any]
        Raw YAML/JSON-compatible configuration.

    Returns
    -------
    Dict[str, Any]
        Dataset manifest dictionary.
    """
    return TSDatasetGenerator.from_dict(config).run()
