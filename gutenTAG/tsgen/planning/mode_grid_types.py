"""Shared mode-grid planning data contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeGridGeometry:
    length: int
    n_channels: int
    block_size: int
    n_blocks: int


@dataclass(frozen=True)
class ModeGridOptions:
    min_blocks: int
    max_blocks: int
    min_gap_blocks: int
    min_gap_points: int


__all__ = ["ModeGridGeometry", "ModeGridOptions"]
