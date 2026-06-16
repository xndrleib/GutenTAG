"""Clean-only instance generation workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..generator.base_generation import prepare_base_instance
from .config import TSGeneratorConfig
from .io import write_json
from .output import build_clean_instance_summary, write_instance_artifacts
from .parameters import realize_parameters
from .variants import VariantSpec


def generate_clean_only_instance(
    *,
    config: TSGeneratorConfig,
    variant: VariantSpec,
    split: str,
    instance_dir: Path,
    seeds: Mapping[str, int],
    base_parameter_template: Mapping[str, Any],
    base_channel_parameter_template: Mapping[str, Any],
    fixed_base_parameters: Mapping[str, Any] | None,
    fixed_base_channel_parameters: Sequence[Mapping[str, Any]] | None,
    effective_base_channel_correlation: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate one clean-only instance and write its artifacts."""

    base_instance = prepare_base_instance(
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
    clean = base_instance.series.observed_values
    labels = np.zeros((config.length, config.channels), dtype=np.int8)
    events: list[dict[str, Any]] = []

    write_instance_artifacts(
        instance_dir=instance_dir,
        clean=clean,
        anomalous=clean,
        labels=labels,
        events=events,
        channels=config.channels,
        csv_float_format=config.csv_float_format,
    )

    instance_summary = build_clean_instance_summary(
        instance_id=instance_dir.name,
        split=split,
        variant_id=variant.variant_id,
        profile_id=variant.profile_id,
        base_oscillation=variant.base_oscillation,
        anomaly_type=variant.anomaly_type,
        channels=config.channels,
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
        seeds=seeds,
    )
    write_json(
        instance_dir / "instance_summary.json",
        instance_summary,
        sort_keys=True,
        indent=2,
    )
    return instance_summary


__all__ = ["generate_clean_only_instance"]
