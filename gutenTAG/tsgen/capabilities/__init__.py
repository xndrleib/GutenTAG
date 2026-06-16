"""Theory-aligned capability analysis for generated TSAD datasets."""

from .cache import CacheFingerprint, CacheStore
from .output_store import (
    OutputStore,
    OutputTableManifest,
    OutputTableSpec,
    ParquetOutputError,
)
from .partitions import (
    PartitionResult,
    PartitionSpec,
    partition_sequence,
    run_partitions,
)
from .protocol import (
    CapabilityProtocol,
    CapabilityRunConfig,
    capability_run_config_from_yaml,
)
from .rolling import RollingStats


def run_capability_analysis(*args, **kwargs):
    """Run capability analysis without importing the runner at package import time."""

    from .runner import run_capability_analysis as _run_capability_analysis

    return _run_capability_analysis(*args, **kwargs)


__all__ = [
    "CapabilityProtocol",
    "CapabilityRunConfig",
    "CacheFingerprint",
    "CacheStore",
    "OutputStore",
    "OutputTableManifest",
    "OutputTableSpec",
    "ParquetOutputError",
    "PartitionResult",
    "PartitionSpec",
    "RollingStats",
    "capability_run_config_from_yaml",
    "partition_sequence",
    "run_capability_analysis",
    "run_partitions",
]
