"""Theory-aligned capability analysis for generated TSAD datasets."""

from .cache import CacheFingerprint, CacheStore
from .output_store import OutputStore, OutputTableManifest, OutputTableSpec, ParquetOutputError
from .partitions import PartitionResult, PartitionSpec, partition_sequence, run_partitions
from .protocol import CapabilityProtocol
from .rolling import RollingStats


def run_capability_analysis(*args, **kwargs):
    """Run capability analysis without importing the runner at package import time."""

    from .runner import run_capability_analysis as _run_capability_analysis

    return _run_capability_analysis(*args, **kwargs)

__all__ = [
    "CapabilityProtocol",
    "CacheFingerprint",
    "CacheStore",
    "OutputStore",
    "OutputTableManifest",
    "OutputTableSpec",
    "ParquetOutputError",
    "PartitionResult",
    "PartitionSpec",
    "RollingStats",
    "partition_sequence",
    "run_capability_analysis",
    "run_partitions",
]
