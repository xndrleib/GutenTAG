"""Capability-layer ontologies for theory-aligned TSAD benchmark analysis.

The objects in this module intentionally do not depend on the historical QC
policy.  They provide a small, explicit vocabulary for mapping generator-level
anomaly names to statistical constraints and witness families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ConstraintDescriptor:
    """Statistical constraint descriptor attached to a generated event."""

    constraint_tag: str
    canonical_witnesses: tuple[str, ...]
    repair_operator: str
    semantic_scope: str


DEFAULT_DESCRIPTOR = ConstraintDescriptor(
    constraint_tag="unknown.generic",
    canonical_witnesses=("energy_delta", "mean_delta", "variance_delta"),
    repair_operator="identity_oracle_upper_bound",
    semantic_scope="unknown",
)

# The map is deliberately coarse.  Mechanism-level labels can be finer than
# observation-level labels; this map encodes the observation-level claim that the
# benchmark can test without relying on private implementation details.
CONSTRAINT_BY_ANOMALY: Mapping[str, ConstraintDescriptor] = {
    "mean": ConstraintDescriptor(
        "location.mean",
        ("mean_delta", "energy_delta"),
        "match_local_mean",
        "channel",
    ),
    "amplitude": ConstraintDescriptor(
        "scale.amplitude",
        ("energy_delta", "variance_delta", "shape_residual"),
        "affine_channel_rescale",
        "channel",
    ),
    "variance": ConstraintDescriptor(
        "scale.variance",
        ("variance_delta", "energy_delta"),
        "match_local_variance",
        "channel",
    ),
    "trend": ConstraintDescriptor(
        "trend.local_slope",
        ("shape_residual", "mean_delta", "energy_delta"),
        "match_local_linear_trend",
        "channel",
    ),
    "frequency": ConstraintDescriptor(
        "spectrum.frequency",
        ("spectral_delta", "shape_residual"),
        "match_spectral_profile",
        "channel",
    ),
    "extremum": ConstraintDescriptor(
        "point.extremum",
        ("energy_delta", "mean_delta"),
        "replace_point_by_local_baseline",
        "point",
    ),
    "pattern": ConstraintDescriptor(
        "shape.local_template",
        ("shape_residual", "energy_delta", "spectral_delta"),
        "match_nearest_local_template",
        "channel",
    ),
    "pattern-shift": ConstraintDescriptor(
        "shape.phase_or_motif_order",
        ("shape_residual", "spectral_delta", "energy_delta"),
        "align_local_template_phase",
        "channel",
    ),
    "platform": ConstraintDescriptor(
        "shape.platform",
        ("shape_residual", "mean_delta", "energy_delta"),
        "restore_transition_continuity",
        "channel",
    ),
    "correlation-flip": ConstraintDescriptor(
        "dependence.correlation",
        ("correlation_delta", "covariance_delta", "energy_delta"),
        "match_local_correlation",
        "relation",
    ),
    "covariance-change": ConstraintDescriptor(
        "dependence.covariance",
        ("covariance_delta", "correlation_delta", "energy_delta"),
        "match_local_covariance",
        "relation",
    ),
    "channel-rewiring": ConstraintDescriptor(
        "dependence.subspace",
        ("covariance_delta", "correlation_delta", "shape_residual"),
        "restore_channel_mixing_geometry",
        "relation",
    ),
    "lag-synchronization": ConstraintDescriptor(
        "dependence.lag_synchronization",
        ("lag_correlation_delta", "correlation_delta", "covariance_delta"),
        "restore_lag_profile",
        "relation",
    ),
    "mode-correlation": ConstraintDescriptor(
        "regime.mode_correlation",
        ("correlation_delta", "covariance_delta", "shape_residual"),
        "restore_regime_conditioned_dependence",
        "regime_relation",
    ),
}

ALL_WITNESSES: tuple[str, ...] = (
    "energy_delta",
    "mean_delta",
    "variance_delta",
    "shape_residual",
    "spectral_delta",
    "correlation_delta",
    "covariance_delta",
    "lag_correlation_delta",
)

DETECTION_WITNESSES: tuple[str, ...] = (
    "mean_z",
    "variance_log_ratio",
    "local_energy_z",
    "correlation_shift",
    "covariance_shift",
)


def descriptor_for_anomaly(anomaly_type: str | None) -> ConstraintDescriptor:
    """Return an observation-level descriptor for an anomaly type."""

    if anomaly_type is None:
        return DEFAULT_DESCRIPTOR
    return CONSTRAINT_BY_ANOMALY.get(str(anomaly_type), DEFAULT_DESCRIPTOR)


def canonical_witnesses_for_anomaly(anomaly_type: str | None) -> tuple[str, ...]:
    """Return canonical witness family names for an anomaly type."""

    return descriptor_for_anomaly(anomaly_type).canonical_witnesses


def constraint_tag_for_anomaly(anomaly_type: str | None) -> str:
    """Return the coarse statistical constraint tag for an anomaly type."""

    return descriptor_for_anomaly(anomaly_type).constraint_tag


def repair_operator_for_anomaly(anomaly_type: str | None) -> str:
    """Return the canonical repair operator name for an anomaly type."""

    return descriptor_for_anomaly(anomaly_type).repair_operator


def semantic_scope_for_anomaly(anomaly_type: str | None) -> str:
    """Return the coarse semantic scope for an anomaly type."""

    return descriptor_for_anomaly(anomaly_type).semantic_scope


def witness_requires_projection_size(witness: str) -> int:
    """Minimum channel projection size required by a witness."""

    if witness in {"correlation_delta", "covariance_delta", "lag_correlation_delta"}:
        return 2
    if witness in {"correlation_shift", "covariance_shift"}:
        return 2
    return 1


def available_witnesses_for_projection(
    projection_size: int,
    *,
    candidate_witnesses: Sequence[str] = ALL_WITNESSES,
) -> tuple[str, ...]:
    """Filter witness families by projection size."""

    size = int(projection_size)
    return tuple(
        witness
        for witness in candidate_witnesses
        if witness_requires_projection_size(witness) <= size
    )
