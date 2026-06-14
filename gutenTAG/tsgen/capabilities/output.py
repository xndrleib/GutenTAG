"""Backward-compatible output helpers for capability tables."""

from __future__ import annotations

from .output_store import OutputStore, OutputTableManifest, OutputTableSpec, ParquetOutputError


class OutputWriter(OutputStore):
    """Compatibility wrapper around :class:`OutputStore`."""


__all__ = [
    "OutputStore",
    "OutputTableManifest",
    "OutputTableSpec",
    "OutputWriter",
    "ParquetOutputError",
]
