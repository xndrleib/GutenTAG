"""Repair operator dispatch."""

from __future__ import annotations

import numpy as np

from .correlation import match_correlation, match_covariance
from .lag import restore_lag_profile
from .mean import match_mean
from .mode import restore_mode_agreement
from .pattern import affine_channel_repair, match_pattern
from .spectral import match_spectral_profile
from .variance import match_variance


def repair_segment(
    clean_segment: np.ndarray,
    anomalous_segment: np.ndarray,
    *,
    constraint_tag: str,
    repair_operator: str | None = None,
) -> np.ndarray:
    """Apply the deterministic repair operator implied by a descriptor."""

    tag = str(constraint_tag or "unknown.generic")
    operator = str(repair_operator or "")
    if tag.startswith("location") or tag.startswith("point") or "mean" in operator:
        return match_mean(clean_segment, anomalous_segment)
    if tag.startswith("dependence.covariance") or tag.startswith("dependence.subspace") or "covariance" in operator:
        return match_covariance(clean_segment, anomalous_segment)
    if tag.startswith("dependence.correlation") or "correlation" in operator:
        return match_correlation(clean_segment, anomalous_segment)
    if tag.startswith("scale") or "variance" in operator or "rescale" in operator:
        return match_variance(clean_segment, anomalous_segment)
    if tag.startswith("dependence.lag") or "lag" in operator:
        return restore_lag_profile(clean_segment, anomalous_segment)
    if tag.startswith("regime") or "regime" in operator or "mode" in operator:
        return restore_mode_agreement(clean_segment, anomalous_segment)
    if tag.startswith("spectrum") or "spectral" in operator:
        return match_spectral_profile(clean_segment, anomalous_segment)
    if _is_local_template_repair(tag, operator):
        return match_pattern(clean_segment, anomalous_segment)
    if tag.startswith("trend") or tag.startswith("shape"):
        return affine_channel_repair(clean_segment, anomalous_segment)
    return np.asarray(anomalous_segment, dtype=np.float64)


def operator_family(constraint_tag: str) -> str:
    """Return the broad operator family for a constraint tag."""

    tag = str(constraint_tag or "unknown.generic")
    return tag.split(".", maxsplit=1)[0] if "." in tag else tag


def _is_local_template_repair(tag: str, operator: str) -> bool:
    return (
        tag.startswith("shape.local_template")
        or tag.startswith("shape.phase")
        or "template" in operator
        or "phase" in operator
    )
