"""Shared mode-boundary planning data contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeBoundaryGeometry:
    length: int
    n_channels: int


__all__ = ["ModeBoundaryGeometry"]
