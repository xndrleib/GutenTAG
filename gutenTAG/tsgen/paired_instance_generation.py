"""Paired clean/anomalous instance generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..generator.anomaly_objects import build_anomalies
from ..generator.base_channels import apply_variations
from ..generator.base_generation import (
    generate_base_instance_series,
    prepare_base_instance,
)
from .config import TSGeneratorConfig
from .generation_runtime import (
    apply_runtime_anomalies,
    resolve_runtime_anomaly_parameters_for_segments,
)
from .io import write_json
from .output import build_paired_instance_summary, write_instance_artifacts
from .parameters import (
    annotate_segment_local_stats,
    realize_parameters,
    resolve_instance_anomaly_parameters,
    sanitize_anomaly_parameters,
)
from .planning import build_segment_plan_sampling_config, plan_instance_segments
from .seeding import derive_seed as _derive_seed
from .variants import VariantSpec
from .visualization import InstancePlotConfig, write_instance_plots


@dataclass(frozen=True)
class _PairedInstanceRequest:
    """Resolved inputs for generating one paired instance."""

    config: TSGeneratorConfig
    variant: VariantSpec
    split: str
    instance_dir: Path
    seeds: Mapping[str, int]
    base_parameter_template: Mapping[str, Any]
    base_channel_parameter_template: Mapping[str, Any]
    anomaly_parameter_template: Mapping[str, Any]
    fixed_base_parameters: Optional[Mapping[str, Any]]
    fixed_base_channel_parameters: Optional[Sequence[Mapping[str, Any]]]
    fixed_anomaly_parameters: Optional[Mapping[str, Any]]
    effective_base_channel_correlation: Mapping[str, Any]
    variant_anomaly_policy: Optional[Mapping[str, Any]]
    variant_segment_planner: Optional[Mapping[str, Any]]
    logger: logging.Logger


def generate_paired_instance(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    split: str,
    instance_dir: Path,
    seeds: Mapping[str, int],
    base_parameter_template: Mapping[str, Any],
    base_channel_parameter_template: Mapping[str, Any],
    anomaly_parameter_template: Mapping[str, Any],
    fixed_base_parameters: Optional[Mapping[str, Any]],
    fixed_base_channel_parameters: Optional[Sequence[Mapping[str, Any]]],
    fixed_anomaly_parameters: Optional[Mapping[str, Any]],
    effective_base_channel_correlation: Mapping[str, Any],
    variant_anomaly_policy: Optional[Mapping[str, Any]],
    variant_segment_planner: Optional[Mapping[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Generate one paired clean/anomalous instance and write its artifacts."""

    return _generate_paired_instance(_PairedInstanceRequest(**locals()))


def _generate_paired_instance(request: _PairedInstanceRequest) -> dict[str, Any]:
    """Generate one paired instance from an explicit request object."""

    anomaly_parameters_instance = _resolve_paired_anomaly_parameters_instance(
        config=request.config,
        seeds=request.seeds,
        anomaly_parameter_template=request.anomaly_parameter_template,
        fixed_anomaly_parameters=request.fixed_anomaly_parameters,
    )
    base_instance = _prepare_clean_base_instance(
        config=request.config,
        variant=request.variant,
        split=request.split,
        seeds=request.seeds,
        base_parameter_template=request.base_parameter_template,
        base_channel_parameter_template=request.base_channel_parameter_template,
        fixed_base_parameters=request.fixed_base_parameters,
        fixed_base_channel_parameters=request.fixed_base_channel_parameters,
        effective_base_channel_correlation=request.effective_base_channel_correlation,
    )
    clean_base = base_instance.series.base_values
    clean = base_instance.series.observed_values
    anomalous_series = _generate_anomalous_base_series(
        config=request.config,
        variant=request.variant,
        seeds=request.seeds,
        base_instance=base_instance,
        effective_base_channel_correlation=request.effective_base_channel_correlation,
    )
    anomaly_run = _plan_and_apply_instance_anomalies(
        config=request.config,
        variant=request.variant,
        split=request.split,
        seeds=request.seeds,
        clean_base=clean_base,
        anomalous_base=anomalous_series.base_values,
        anomalous_bos=anomalous_series.channel_bos,
        anomaly_parameter_template=request.anomaly_parameter_template,
        fixed_anomaly_parameters=request.fixed_anomaly_parameters,
        anomaly_parameters_instance=anomaly_parameters_instance,
        variant_anomaly_policy=request.variant_anomaly_policy,
        variant_segment_planner=request.variant_segment_planner,
        logger=request.logger,
    )
    anomalous = apply_variations(
        anomalous_series.base_values,
        anomalous_series.channel_bos,
    )
    _write_paired_instance_artifacts(
        instance_dir=request.instance_dir,
        clean=clean,
        anomalous=anomalous,
        anomaly_run=anomaly_run,
        config=request.config,
    )
    instance_summary = _write_paired_instance_summary(
        config=request.config,
        variant=request.variant,
        split=request.split,
        instance_dir=request.instance_dir,
        seeds=request.seeds,
        base_instance=base_instance,
        effective_base_channel_correlation=request.effective_base_channel_correlation,
        anomaly_parameters_instance=anomaly_parameters_instance,
        anomaly_run=anomaly_run,
    )
    _write_instance_plots_if_enabled(
        config=request.config,
        instance_dir=request.instance_dir,
        clean=clean,
        anomalous=anomalous,
        events=anomaly_run["events"],
        seeds=request.seeds,
    )
    return instance_summary


def _prepare_clean_base_instance(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    split: str,
    seeds: Mapping[str, int],
    base_parameter_template: Mapping[str, Any],
    base_channel_parameter_template: Mapping[str, Any],
    fixed_base_parameters: Optional[Mapping[str, Any]],
    fixed_base_channel_parameters: Optional[Sequence[Mapping[str, Any]]],
    effective_base_channel_correlation: Mapping[str, Any],
) -> Any:
    return prepare_base_instance(
        base_kind=variant.base_oscillation,
        split=split,
        seeds=seeds,
        base_parameter_template=base_parameter_template,
        base_channel_parameter_template=base_channel_parameter_template,
        fixed_base_parameters=fixed_base_parameters,
        fixed_base_channel_parameters=fixed_base_channel_parameters,
        base_parameter_policy=config.base_parameter_policy,
        base_channel_parameter_policy=config.base_channel_parameter_policy,
        channels=config.channels,
        length=config.length,
        split_phase_shift=config.split_phase_shift,
        base_channel_correlation=effective_base_channel_correlation,
        realize_parameters=realize_parameters,
    )


def _generate_anomalous_base_series(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    seeds: Mapping[str, int],
    base_instance: Any,
    effective_base_channel_correlation: Mapping[str, Any],
) -> Any:
    return generate_base_instance_series(
        base_kind=variant.base_oscillation,
        base_parameters_per_channel=base_instance.base_parameters_per_channel,
        seed=int(seeds["base_seed"]),
        shared_noise_seed=int(seeds["base_shared_noise_seed"]),
        base_channel_correlation=effective_base_channel_correlation,
        expected_length=config.length,
    )


def _write_paired_instance_artifacts(
    *,
    instance_dir: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    anomaly_run: Mapping[str, Any],
    config: TSGeneratorConfig,
) -> None:
    write_instance_artifacts(
        instance_dir=instance_dir,
        clean=clean,
        anomalous=anomalous,
        labels=anomaly_run["labels"],
        events=anomaly_run["events"],
        channels=config.channels,
        csv_float_format=config.csv_float_format,
    )


def _write_paired_instance_summary(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    split: str,
    instance_dir: Path,
    seeds: Mapping[str, int],
    base_instance: Any,
    effective_base_channel_correlation: Mapping[str, Any],
    anomaly_parameters_instance: Optional[Mapping[str, Any]],
    anomaly_run: Mapping[str, Any],
) -> dict[str, Any]:
    instance_summary = _build_paired_instance_summary(
        config=config,
        variant=variant,
        split=split,
        instance_dir=instance_dir,
        seeds=seeds,
        base_instance=base_instance,
        effective_base_channel_correlation=effective_base_channel_correlation,
        anomaly_parameters_instance=anomaly_parameters_instance,
        anomaly_run=anomaly_run,
    )
    write_json(
        instance_dir / "instance_summary.json",
        instance_summary,
        sort_keys=True,
        indent=2,
    )
    return instance_summary


def _resolve_paired_anomaly_parameters_instance(
    *,
    config: TSGeneratorConfig,
    seeds: Mapping[str, int],
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    anomaly_parameter_rng = np.random.default_rng(int(seeds["params_seed"]))
    return resolve_instance_anomaly_parameters(
        anomaly_parameter_policy=config.anomaly_parameter_policy,
        anomaly_parameter_template=anomaly_parameter_template,
        fixed_anomaly_parameters=fixed_anomaly_parameters,
        rng=anomaly_parameter_rng,
    )


def _plan_and_apply_instance_anomalies(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    split: str,
    seeds: Mapping[str, int],
    clean_base: np.ndarray,
    anomalous_base: np.ndarray,
    anomalous_bos: list[Any],
    anomaly_parameter_template: Mapping[str, Any],
    fixed_anomaly_parameters: Optional[Mapping[str, Any]],
    anomaly_parameters_instance: Optional[Mapping[str, Any]],
    variant_anomaly_policy: Optional[Mapping[str, Any]],
    variant_segment_planner: Optional[Mapping[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    planning = plan_instance_segments(
        rng=np.random.default_rng(int(seeds["plan_seed"])),
        anomaly_type=variant.anomaly_type,
        split=split,
        clean_values=clean_base,
        channel_bos=anomalous_bos,
        anomaly_parameter_template=anomaly_parameter_template,
        parameter_seed=int(seeds["params_seed"]),
        variant_anomaly_policy=variant_anomaly_policy,
        variant_segment_planner=variant_segment_planner,
        config=build_segment_plan_sampling_config(config),
        default_density_tolerance=config.density_tolerance,
        realize_parameters=realize_parameters,
        sanitize_parameters=sanitize_anomaly_parameters,
        derive_seed=_derive_seed,
        logger=logger,
    )
    segment_plan = planning.segment_plan
    annotate_segment_local_stats(segment_plan, clean_base)
    anomaly_params_per_segment = resolve_runtime_anomaly_parameters_for_segments(
        config=config,
        base_kind=variant.base_oscillation,
        anomaly_type=variant.anomaly_type,
        anomaly_parameter_template=anomaly_parameter_template,
        fixed_anomaly_parameters=fixed_anomaly_parameters,
        anomaly_parameters_instance=anomaly_parameters_instance,
        segment_plan=segment_plan,
        parameter_seed=int(seeds["params_seed"]),
        planner_cfg=planning.planner_cfg,
    )
    anomaly_objects = build_anomalies(
        anomaly_type=variant.anomaly_type,
        anomaly_parameters_per_segment=anomaly_params_per_segment,
        segment_plan=segment_plan,
    )
    labels, events = apply_runtime_anomalies(
        config=config,
        anomaly_objects=anomaly_objects,
        segment_plan=segment_plan,
        base=anomalous_base,
        channel_bos=anomalous_bos,
        anomaly_seed=int(seeds["anomaly_seed"]),
        anomaly_type=variant.anomaly_type,
        anomaly_parameters_per_segment=anomaly_params_per_segment,
    )
    return {
        "planning": planning,
        "segment_plan": segment_plan,
        "anomaly_params_per_segment": anomaly_params_per_segment,
        "labels": labels,
        "events": events,
    }


def _build_paired_instance_summary(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    split: str,
    instance_dir: Path,
    seeds: Mapping[str, int],
    base_instance: Any,
    effective_base_channel_correlation: Mapping[str, Any],
    anomaly_parameters_instance: Optional[Mapping[str, Any]],
    anomaly_run: Mapping[str, Any],
) -> dict[str, Any]:
    planning = anomaly_run["planning"]
    return build_paired_instance_summary(
        instance_id=instance_dir.name,
        split=split,
        variant_id=variant.variant_id,
        profile_id=variant.profile_id,
        base_oscillation=variant.base_oscillation,
        anomaly_type=variant.anomaly_type,
        target_density=planning.target_density,
        active_density_range=planning.active_density_range,
        active_density_tolerance=planning.active_density_tolerance,
        labels=anomaly_run["labels"],
        events=anomaly_run["events"],
        segment_plan=anomaly_run["segment_plan"],
        channels=config.channels,
        channel_policy=planning.active_channel_policy,
        overlap_policy=planning.active_overlap_policy,
        segment_planner=planning.planner_cfg,
        variant_anomaly_policy=planning.special_policy,
        length_normalization=config.length_normalization,
        support_label_mode=config.support_label_mode,
        support_eps_mode=config.support_eps_mode,
        support_eps_value=config.support_eps_value,
        base_parameter_policy=config.base_parameter_policy,
        base_channel_parameter_policy=config.base_channel_parameter_policy,
        anomaly_parameter_policy=config.anomaly_parameter_policy,
        base_parameters=base_instance.base_parameters,
        base_channel_parameters=base_instance.base_channel_parameters,
        base_parameters_per_channel=base_instance.base_parameters_per_channel,
        split_phase_shift=base_instance.split_phase_shift_info,
        base_channel_correlation=effective_base_channel_correlation,
        anomaly_parameters_instance=anomaly_parameters_instance,
        seeds=seeds,
    )


def _write_instance_plots_if_enabled(
    *,
    config: TSGeneratorConfig,
    instance_dir: Path,
    clean: np.ndarray,
    anomalous: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, int],
) -> None:
    if config.generate_plots:
        write_instance_plots(
            instance_dir=instance_dir,
            clean=clean,
            anomalous=anomalous,
            events=events,
            zoom_seed=int(seeds["zoom_seed"]),
            config=InstancePlotConfig(
                length=config.length,
                channels=config.channels,
                plot_dpi=config.plot_dpi,
                zoom_count=config.zoom_count,
                zoom_margin=config.zoom_margin,
                zoom_margin_min=config.zoom_margin_min,
                zoom_margin_alpha=config.zoom_margin_alpha,
                zoom_fill_policy=config.zoom_fill_policy,
            ),
        )


__all__ = [
    "generate_paired_instance",
]
