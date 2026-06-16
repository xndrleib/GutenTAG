from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from .generator.base_generation import (
    realize_base_channel_parameters,
)
from .tsgen.config import (
    DEFAULT_ANOMALY_OVERRIDES,
    DEFAULT_BASE_OVERRIDES,
    DEFAULT_DENSITY_RANGE,
    DEFAULT_PROFILES_PER_PAIR,
    DEFAULT_SEGMENT_COUNT_RANGE,
    DEFAULT_SPLITS,
    TSGeneratorConfig,
    merge_dicts as _merge_dicts,
)
from .tsgen.clean_instance_generation import generate_clean_only_instance
from .tsgen.io import sanitize_json_value
from .tsgen.generation_run import (
    build_and_write_dataset_manifest,
    initialize_generation_run,
    record_variant_generation_failure,
    write_split_summary,
)
from .tsgen.planning import (
    resolve_segment_planner,
)
from .tsgen.output import (
    build_split_entry,
    build_variant_config,
    build_variant_manifest,
    write_variant_config,
)
from .tsgen.paired_instance_generation import generate_paired_instance
from .tsgen.parameters import (
    classify_base_family,
    prepare_variant_parameter_context,
    realize_parameters,
    resolve_base_channel_correlation,
    resolve_base_parameters,
)
from .tsgen.seeding import (
    derive_instance_seeds,
    derive_seed as _derive_seed,
    resolve_split_instance_specs,
)
from .tsgen.variants import VariantSpec, resolve_variants

__all__ = [
    "DEFAULT_ANOMALY_OVERRIDES",
    "DEFAULT_BASE_OVERRIDES",
    "DEFAULT_DENSITY_RANGE",
    "DEFAULT_PROFILES_PER_PAIR",
    "DEFAULT_SEGMENT_COUNT_RANGE",
    "DEFAULT_SPLITS",
    "TSGeneratorConfig",
    "TSDatasetGenerator",
    "_merge_dicts",
    "generate_ts_dataset",
]


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
        """Create a generator from either YAML or JSON config file."""
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
        variants_root = self._initialize_generation_run(output_root)
        variants, skipped_variants, disabled_anomaly_types = self._resolve_variants()
        generation_result = self._generate_all_variants(
            variants=variants,
            skipped_variants=skipped_variants,
            variants_root=variants_root,
        )
        manifest = build_and_write_dataset_manifest(
            output_root=output_root,
            config=self.config,
            generated_variant_entries=generation_result["generated_variant_entries"],
            skipped_variants=generation_result["skipped_variants"],
            disabled_anomaly_types=disabled_anomaly_types,
            all_instance_summaries=generation_result["instance_summaries"],
            all_seed_audit=generation_result["seed_audit"],
            repo_root=Path(__file__).resolve().parents[2],
        )

        self.logger.info(
            "Generation complete | variants=%d | skipped=%d | instances=%d",
            len(generation_result["generated_variant_entries"]),
            len(generation_result["skipped_variants"]),
            manifest["aggregated_statistics"]["instance_count"],
        )
        return manifest

    def _initialize_generation_run(self, output_root: Path) -> Path:
        return initialize_generation_run(
            output_root=output_root,
            overwrite_output=self.config.overwrite_output,
            log_level=self.config.log_level,
            master_seed=self.config.master_seed,
            logger=self.logger,
        )

    def _generate_all_variants(
        self,
        *,
        variants: Sequence[VariantSpec],
        skipped_variants: Sequence[Mapping[str, str]],
        variants_root: Path,
    ) -> Dict[str, Any]:
        generated_variant_entries: List[Dict[str, Any]] = []
        all_instance_summaries: List[Dict[str, Any]] = []
        all_seed_audit: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
        skipped = [dict(entry) for entry in skipped_variants]

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
                record_variant_generation_failure(
                    variant=variant,
                    variants_root=variants_root,
                    skipped_variants=skipped,
                    exc=exc,
                    logger=self.logger,
                )
                if self.config.on_variant_failure == "fail_fast":
                    raise

        return {
            "generated_variant_entries": sorted(
                generated_variant_entries,
                key=lambda entry: entry["variant_id"],
            ),
            "skipped_variants": sorted(skipped, key=lambda entry: entry["variant_id"]),
            "instance_summaries": all_instance_summaries,
            "seed_audit": all_seed_audit,
        }

    def _resolve_variants(
        self,
    ) -> Tuple[List[VariantSpec], List[Dict[str, str]], List[Dict[str, str]]]:
        return resolve_variants(
            base_oscillations=self.config.base_oscillations,
            skip_base_oscillations=self.config.skip_base_oscillations,
            anomaly_types=self.config.anomaly_types,
            skip_anomaly_types=self.config.skip_anomaly_types,
            disabled_anomaly_types=self.config.disabled_anomaly_types,
            pair_profiles=self.config.pair_profiles,
            profiles_per_pair=self.config.profiles_per_pair,
            compatibility_mode=self.config.compatibility_mode,
            special_anomaly_policies=self.config.special_anomaly_policies,
            skip_density_incompatible_variants=(
                self.config.skip_density_incompatible_variants
            ),
            resolve_segment_planner=self._resolve_segment_planner,
            resolve_base_parameters=self._resolve_base_parameters,
        )

    def _generate_variant(
        self, variant: VariantSpec, variants_root: Path
    ) -> Dict[str, Any]:
        variant_dir = variants_root / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_context = prepare_variant_parameter_context(
            variant=variant,
            config=self.config,
            realize_parameters=realize_parameters,
            realize_base_channel_parameters=self._realize_base_channel_parameters,
            derive_seed=_derive_seed,
        )

        self._write_variant_config(
            variant_dir=variant_dir,
            variant=variant,
            variant_context=variant_context,
        )

        split_results = [
            self._generate_split(
                variant=variant,
                variant_dir=variant_dir,
                split=split,
                variant_context=variant_context,
            )
            for split in self.config.splits
        ]
        variant_instance_summaries = [
            summary
            for split_result in split_results
            for summary in split_result["instance_summaries"]
        ]
        variant_manifest = self._build_variant_manifest(
            variant=variant,
            variant_context=variant_context,
            split_entries=[result["split_entry"] for result in split_results],
            instance_summaries=variant_instance_summaries,
        )
        return {
            "variant_manifest": variant_manifest,
            "instance_summaries": sanitize_json_value(variant_instance_summaries),
            "seed_audit": sanitize_json_value(
                {
                    split: result["seed_audit"]
                    for split, result in zip(self.config.splits, split_results)
                }
            ),
        }

    def _write_variant_config(
        self,
        *,
        variant_dir: Path,
        variant: VariantSpec,
        variant_context: Any,
    ) -> None:
        variant_config = build_variant_config(
            variant=variant,
            base_parameter_template=variant_context.base_parameter_template,
            fixed_base_parameters=variant_context.fixed_base_parameters,
            base_channel_parameter_template=(
                variant_context.base_channel_parameter_template
            ),
            fixed_base_channel_parameters=(
                variant_context.fixed_base_channel_parameters
            ),
            effective_base_channel_correlation=(
                variant_context.effective_base_channel_correlation
            ),
            anomaly_parameter_template=variant_context.anomaly_parameter_template,
            fixed_anomaly_parameters=variant_context.fixed_anomaly_parameters,
            length=self.config.length,
            channels=self.config.channels,
            splits=self.config.splits,
            instances_per_split=self.config.instances_per_split,
            split_instance_counts=self.config.split_instance_counts,
            density_range=self.config.density_range,
            density_tolerance=self.config.density_tolerance,
            segment_count_range=self.config.segment_count_range,
            placement_policy=self.config.placement_policy,
            channel_policy=self.config.channel_policy,
            effective_overlap_policy=variant_context.effective_overlap_policy,
            variant_segment_planner=variant_context.variant_segment_planner,
            variant_anomaly_policy=variant_context.variant_anomaly_policy,
            length_normalization=self.config.length_normalization,
            base_parameter_policy=self.config.base_parameter_policy,
            base_channel_parameter_policy=self.config.base_channel_parameter_policy,
            anomaly_parameter_policy=self.config.anomaly_parameter_policy,
            split_phase_shift=self.config.split_phase_shift,
        )
        write_variant_config(variant_dir / "variant_config.yaml", variant_config)

    def _build_variant_manifest(
        self,
        *,
        variant: VariantSpec,
        variant_context: Any,
        split_entries: Sequence[Mapping[str, Any]],
        instance_summaries: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return build_variant_manifest(
            variant=variant,
            carrier_family=classify_base_family(variant.base_oscillation),
            split_entries=split_entries,
            instance_summaries=instance_summaries,
            base_parameter_policy=self.config.base_parameter_policy,
            base_channel_parameter_policy=self.config.base_channel_parameter_policy,
            anomaly_parameter_policy=self.config.anomaly_parameter_policy,
            split_phase_shift=self.config.split_phase_shift,
            effective_overlap_policy=variant_context.effective_overlap_policy,
            variant_segment_planner=variant_context.variant_segment_planner,
            variant_anomaly_policy=variant_context.variant_anomaly_policy,
            length_normalization=self.config.length_normalization,
        )

    def _generate_split(
        self,
        *,
        variant: VariantSpec,
        variant_dir: Path,
        split: str,
        variant_context: Any,
    ) -> Dict[str, Any]:
        split_dir = variant_dir / split
        instances_dir = split_dir / "instances"
        instances_dir.mkdir(parents=True, exist_ok=True)
        split_seed_audit: Dict[str, Dict[str, int]] = {}
        split_instance_summaries: List[Dict[str, Any]] = []
        instance_specs = self._resolve_split_instance_specs(split)

        for instance_spec in tqdm(
            instance_specs,
            desc=f"{variant.variant_id}/{split}",
            leave=False,
        ):
            summary, seeds = self._generate_split_instance(
                variant=variant,
                split=split,
                instances_dir=instances_dir,
                instance_spec=instance_spec,
                variant_context=variant_context,
            )
            split_seed_audit[instance_spec.name] = seeds
            split_instance_summaries.append(summary)

        write_split_summary(
            split_dir=split_dir,
            split=split,
            split_instance_summaries=split_instance_summaries,
            length=self.config.length,
            channels=self.config.channels,
        )
        return {
            "instance_summaries": split_instance_summaries,
            "split_entry": build_split_entry(
                split=split,
                instances=len(instance_specs),
                paired_instances=sum(
                    1 for spec in instance_specs if spec.role == "paired"
                ),
                clean_only_instances=sum(
                    1 for spec in instance_specs if spec.role == "clean_only"
                ),
            ),
            "seed_audit": split_seed_audit,
        }

    def _resolve_split_instance_specs(self, split: str) -> List[Any]:
        split_counts = self.config.split_instance_counts.get(
            split,
            {
                "paired_instances_per_variant": self.config.instances_per_split,
                "clean_only_instances_per_variant": 0,
            },
        )
        return resolve_split_instance_specs(
            split_counts,
            default_paired_instances=self.config.instances_per_split,
        )

    def _generate_split_instance(
        self,
        *,
        variant: VariantSpec,
        split: str,
        instances_dir: Path,
        instance_spec: Any,
        variant_context: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        instance_dir = instances_dir / instance_spec.name
        instance_dir.mkdir(parents=True, exist_ok=True)
        seeds = derive_instance_seeds(
            master_seed=self.config.master_seed,
            variant_id=variant.variant_id,
            split=split,
            instance_index=instance_spec.seed_index,
        )
        if instance_spec.role == "clean_only":
            summary = self._generate_clean_split_instance(
                variant=variant,
                split=split,
                instance_dir=instance_dir,
                seeds=seeds,
                variant_context=variant_context,
            )
        else:
            summary = self._generate_paired_split_instance(
                variant=variant,
                split=split,
                instance_dir=instance_dir,
                seeds=seeds,
                variant_context=variant_context,
            )
        return summary, seeds

    def _generate_clean_split_instance(
        self,
        *,
        variant: VariantSpec,
        split: str,
        instance_dir: Path,
        seeds: Mapping[str, int],
        variant_context: Any,
    ) -> Dict[str, Any]:
        return self._generate_clean_only_instance(
            variant=variant,
            split=split,
            instance_dir=instance_dir,
            seeds=seeds,
            base_parameter_template=variant_context.base_parameter_template,
            base_channel_parameter_template=variant_context.base_channel_parameter_template,
            fixed_base_parameters=variant_context.fixed_base_parameters,
            fixed_base_channel_parameters=variant_context.fixed_base_channel_parameters,
        )

    def _generate_paired_split_instance(
        self,
        *,
        variant: VariantSpec,
        split: str,
        instance_dir: Path,
        seeds: Mapping[str, int],
        variant_context: Any,
    ) -> Dict[str, Any]:
        return generate_paired_instance(
            config=self.config,
            variant=variant,
            split=split,
            instance_dir=instance_dir,
            seeds=seeds,
            base_parameter_template=variant_context.base_parameter_template,
            base_channel_parameter_template=variant_context.base_channel_parameter_template,
            anomaly_parameter_template=variant_context.anomaly_parameter_template,
            fixed_base_parameters=variant_context.fixed_base_parameters,
            fixed_base_channel_parameters=variant_context.fixed_base_channel_parameters,
            fixed_anomaly_parameters=variant_context.fixed_anomaly_parameters,
            effective_base_channel_correlation=self._resolve_base_channel_correlation(
                variant
            ),
            variant_anomaly_policy=variant_context.variant_anomaly_policy,
            variant_segment_planner=variant_context.variant_segment_planner,
            logger=self.logger,
        )

    def _resolve_base_parameters(self, variant: VariantSpec) -> Dict[str, Any]:
        return resolve_base_parameters(
            variant,
            base_oscillation_overrides=self.config.base_oscillation_overrides,
            variant_overrides=self.config.variant_overrides,
            length=self.config.length,
        )

    def _generate_clean_only_instance(
        self,
        variant: VariantSpec,
        split: str,
        instance_dir: Path,
        seeds: Mapping[str, int],
        base_parameter_template: Mapping[str, Any],
        base_channel_parameter_template: Mapping[str, Any],
        fixed_base_parameters: Optional[Mapping[str, Any]],
        fixed_base_channel_parameters: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return generate_clean_only_instance(
            config=self.config,
            variant=variant,
            split=split,
            instance_dir=instance_dir,
            seeds=seeds,
            base_parameter_template=base_parameter_template,
            base_channel_parameter_template=base_channel_parameter_template,
            fixed_base_parameters=fixed_base_parameters,
            fixed_base_channel_parameters=fixed_base_channel_parameters,
            effective_base_channel_correlation=self._resolve_base_channel_correlation(
                variant
            ),
        )

    def _realize_base_channel_parameters(
        self, template: Mapping[str, Any], rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        return realize_base_channel_parameters(
            template=template,
            channels=self.config.channels,
            rng=rng,
            realize_parameters=realize_parameters,
        )

    def _resolve_base_channel_correlation(self, variant: VariantSpec) -> Dict[str, Any]:
        return resolve_base_channel_correlation(
            variant,
            base_channel_correlation=self.config.base_channel_correlation,
            variant_overrides=self.config.variant_overrides,
        )

    def _resolve_segment_planner(
        self,
        anomaly_type: str,
        planner_override: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return resolve_segment_planner(
            anomaly_type=anomaly_type,
            segment_planner=self.config.segment_planner,
            special_anomaly_policies=self.config.special_anomaly_policies,
            planner_override=planner_override,
        )


def generate_ts_dataset(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate a TS dataset directly from a raw configuration dictionary."""
    return TSDatasetGenerator.from_dict(config).run()
