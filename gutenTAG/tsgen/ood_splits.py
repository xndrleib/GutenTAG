"""Deterministic OOD split contracts for v13 benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OODSplitContract:
    iid_splits: tuple[str, ...] = ("train", "validation", "test_iid")
    parameter_ood_split: str = "test_parameter_ood"
    family_ood_split: str = "test_family_ood"
    mechanism_ood_split: str = "test_mechanism_ood"
    relation_only_split: str = "test_relation_only"


def validate_disjoint_holdouts(
    *,
    train_families: Sequence[str],
    heldout_families: Sequence[str],
    train_mechanisms: Sequence[str],
    heldout_mechanisms: Sequence[str],
) -> None:
    """Reject OOD contracts whose held-out identities leak into training."""
    family_overlap = sorted(set(train_families) & set(heldout_families))
    mechanism_overlap = sorted(set(train_mechanisms) & set(heldout_mechanisms))
    if family_overlap:
        raise ValueError(f"family-OOD leakage: {family_overlap}")
    if mechanism_overlap:
        raise ValueError(f"mechanism-OOD leakage: {mechanism_overlap}")


def parameter_range_overlap(
    train_ranges: Mapping[str, tuple[float, float]],
    test_ranges: Mapping[str, tuple[float, float]],
) -> dict[str, bool]:
    """Return parameter-wise overlap flags for audit/certificate generation."""
    result: dict[str, bool] = {}
    for key in sorted(set(train_ranges) & set(test_ranges)):
        train_lo, train_hi = train_ranges[key]
        test_lo, test_hi = test_ranges[key]
        result[str(key)] = max(float(train_lo), float(test_lo)) <= min(
            float(train_hi), float(test_hi)
        )
    return result
