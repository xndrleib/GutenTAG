"""Run-level orchestration helpers for dataset generation."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._version import __version__
from .config import TSGeneratorConfig
from .contracts import write_v12_metadata_registries
from .io import write_json
from .manifest import build_dataset_manifest
from .output import compute_dataset_statistics, compute_split_statistics
from .sidecars import write_v12_generator_sidecars
from .variants import VariantSpec


def initialize_generation_run(
    *,
    output_root: Path,
    overwrite_output: bool,
    log_level: str,
    master_seed: int,
    logger: logging.Logger,
) -> Path:
    """Prepare output directories and logging for one generation run."""

    prepare_output_directory(output_root, overwrite_output=overwrite_output)
    configure_generation_logging(logger, output_root=output_root, log_level=log_level)
    logger.info(
        "Starting TS dataset generation | output_root=%s | master_seed=%d",
        output_root,
        master_seed,
    )
    variants_root = output_root / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)
    return variants_root


def prepare_output_directory(
    output_root: Path,
    *,
    overwrite_output: bool,
) -> None:
    """Create or replace the dataset output root."""

    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output folder already exists and overwrite_output=false: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def configure_generation_logging(
    logger: logging.Logger,
    *,
    output_root: Path,
    log_level: str,
) -> None:
    """Configure stream and file logging for one generation run."""

    level = getattr(logging, log_level, logging.INFO)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_root / "generation.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def record_variant_generation_failure(
    *,
    variant: VariantSpec,
    variants_root: Path,
    skipped_variants: list[dict[str, str]],
    exc: Exception,
    logger: logging.Logger,
) -> None:
    """Record and clean up a failed variant generation attempt."""

    reason = f"{type(exc).__name__}: {exc}"
    variant_dir = variants_root / variant.variant_id
    if variant_dir.exists():
        shutil.rmtree(variant_dir, ignore_errors=True)
    skipped_variants.append({"variant_id": variant.variant_id, "reason": reason})
    logger.exception("Variant %s failed and was skipped", variant.variant_id)


def build_and_write_dataset_manifest(
    *,
    output_root: Path,
    config: TSGeneratorConfig,
    generated_variant_entries: Sequence[Mapping[str, Any]],
    skipped_variants: Sequence[Mapping[str, str]],
    disabled_anomaly_types: Sequence[Mapping[str, str]],
    all_instance_summaries: Sequence[Mapping[str, Any]],
    all_seed_audit: Mapping[str, Any],
    repo_root: Path,
    library_version: str = __version__,
) -> dict[str, Any]:
    """Build and write the dataset-level manifest."""

    dataset_stats = compute_dataset_statistics(all_instance_summaries)
    validate_non_empty_dataset(
        dataset_stats=dataset_stats,
        generated_variant_entries=generated_variant_entries,
        allow_empty_dataset=config.allow_empty_dataset,
    )
    manifest = build_dataset_manifest(
        output_root=output_root,
        repo_root=repo_root,
        dataset_version=config.dataset_version,
        library_version=library_version,
        config=config.to_dict(),
        generated_variant_entries=generated_variant_entries,
        skipped_variants=skipped_variants,
        disabled_anomaly_types=[
            str(item.get("anomaly_type", "")) for item in disabled_anomaly_types
        ],
        aggregated_statistics=dataset_stats,
        derived_seeds=all_seed_audit,
        metadata_registry=write_v12_metadata_registries(
            output_root,
            provenance="generated",
        ),
        generator_sidecars=write_v12_generator_sidecars(
            output_root,
            annotation_channels=config.annotation_channels,
            law_level_replicates=config.law_level_replicates,
        ),
    )
    write_json(
        output_root / "dataset_manifest.json",
        manifest,
        sort_keys=True,
        indent=2,
    )
    return manifest


def validate_non_empty_dataset(
    *,
    dataset_stats: Mapping[str, Any],
    generated_variant_entries: Sequence[Mapping[str, Any]],
    allow_empty_dataset: bool,
) -> None:
    """Reject empty generation output unless explicitly allowed."""

    if allow_empty_dataset:
        return
    if int(dataset_stats["instance_count"]) > 0 and len(generated_variant_entries) > 0:
        return
    raise ValueError(
        "Dataset generation produced no instances or no generated variants. "
        "Fix the config or set generator.allow_empty_dataset=true for debug-only runs."
    )


def write_split_summary(
    *,
    split_dir: Path,
    split: str,
    split_instance_summaries: Sequence[Mapping[str, Any]],
    length: int,
    channels: int,
) -> None:
    """Write ``split_summary.json`` for one generated split."""

    split_summary = compute_split_statistics(
        split,
        split_instance_summaries,
        length,
        channels,
    )
    write_json(
        split_dir / "split_summary.json",
        split_summary,
        sort_keys=True,
        indent=2,
    )


__all__ = [
    "build_and_write_dataset_manifest",
    "configure_generation_logging",
    "initialize_generation_run",
    "prepare_output_directory",
    "record_variant_generation_failure",
    "validate_non_empty_dataset",
    "write_split_summary",
]
