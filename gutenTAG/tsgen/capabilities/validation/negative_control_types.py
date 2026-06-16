"""Shared negative-control event contracts."""

from __future__ import annotations

from dataclasses import dataclass

MARGINAL_WITNESSES = ("mean_z", "variance_log_ratio", "local_energy_z")


@dataclass(frozen=True)
class NegativeControlEventContext:
    subsets: tuple[tuple[int, ...], ...]
    witnesses: tuple[str, ...]
    relation_like: bool
    clean_control_witnesses: tuple[str, ...]
    shifted_start: int
    shifted_end: int
    marginal_subsets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class NegativeControlScores:
    clean_score: float
    clean_witness: str
    clean_wrong_support_score: float
    wrong_support_score: float
    wrong_support_witness: str
    clean_marginal_score: float
    marginal_score: float
    marginal_witness: str
    boundary_ratio: float
    boundary_triggered: bool


__all__ = [
    "MARGINAL_WITNESSES",
    "NegativeControlEventContext",
    "NegativeControlScores",
]
