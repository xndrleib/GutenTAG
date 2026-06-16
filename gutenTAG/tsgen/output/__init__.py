"""Output helpers for TS dataset generation."""

from .files import (
    check_labels_and_events_consistency,
    write_instance_artifacts,
    write_labels_csv,
    write_timeseries_csv,
)
from .instance_density import compute_density_metrics
from .instance_summary import (
    build_clean_instance_summary,
    build_paired_instance_summary,
)
from .statistics import compute_dataset_statistics, compute_split_statistics
from .variant import (
    build_split_entry,
    build_variant_config,
    build_variant_manifest,
    write_variant_config,
)

__all__ = [
    "build_clean_instance_summary",
    "build_paired_instance_summary",
    "build_split_entry",
    "build_variant_config",
    "build_variant_manifest",
    "check_labels_and_events_consistency",
    "compute_density_metrics",
    "compute_dataset_statistics",
    "compute_split_statistics",
    "write_instance_artifacts",
    "write_labels_csv",
    "write_timeseries_csv",
    "write_variant_config",
]
