"""Witness/projection candidate policy for corrected detectability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .dataset import EventGroup, InstanceRecord
from .numerics import channel_subsets
from .ontology import witness_requires_projection_size
from .protocol import CapabilityProtocol

WITNESS_FAMILY: Mapping[str, str] = {
    "mean_z": "location.mean",
    "variance_log_ratio": "scale.variance",
    "local_energy_z": "local.energy",
    "correlation_shift": "dependence.correlation",
    "covariance_shift": "dependence.covariance",
}


@dataclass(frozen=True)
class CandidateSpec:
    """Single witness/projection candidate."""

    witness: str
    projection: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.witness}@{format_projection(self.projection)}"

    @property
    def family(self) -> str:
        return WITNESS_FAMILY.get(self.witness, "unknown")


def candidate_specs(
    instance: InstanceRecord,
    group: EventGroup,
    protocol: CapabilityProtocol,
) -> tuple[CandidateSpec, ...]:
    """Build witness/projection candidates for one event group."""

    channels = tuple(
        sorted(
            set(group.group_channels)
            | set(group.context_channels)
            | set(group.intervention_channels)
        )
    )
    if not channels:
        channels = tuple(range(instance.channels))
    specs: list[CandidateSpec] = []
    for subset in channel_subsets(channels, protocol.max_projection_size):
        for witness in protocol.detection_witnesses:
            if witness_requires_projection_size(witness) <= len(subset):
                specs.append(
                    CandidateSpec(witness=str(witness), projection=tuple(subset))
                )
    return tuple(specs)


def spec_by_key(candidates: Sequence[CandidateSpec], key: str) -> CandidateSpec | None:
    """Return a candidate by serialized key."""

    for candidate in candidates:
        if candidate.key == key:
            return candidate
    return None


def best_canonical_key(
    group: EventGroup,
    candidates: Sequence[CandidateSpec],
    candidate_p_values: Mapping[str, float],
) -> str | None:
    """Return the strongest calibrated canonical candidate for an event."""

    canonical: dict[str, float] = {}
    for candidate in candidates:
        if candidate.key not in candidate_p_values:
            continue
        if not candidate_is_canonical(group, candidate):
            continue
        value = float(candidate_p_values[candidate.key])
        if math.isfinite(value):
            canonical[candidate.key] = value
    if not canonical:
        return None
    return min(canonical, key=lambda key: canonical[key])


def candidate_is_canonical(group: EventGroup, candidate: CandidateSpec) -> bool:
    """Return whether a candidate matches the event's canonical detection policy."""

    canonical = {
        "mean": ("mean_z",),
        "variance": ("variance_log_ratio", "local_energy_z"),
        "amplitude": ("variance_log_ratio", "local_energy_z"),
        "platform": ("mean_z", "local_energy_z"),
        "pattern": ("local_energy_z",),
        "frequency": ("local_energy_z",),
        "correlation-flip": ("correlation_shift", "covariance_shift"),
        "covariance-change": ("covariance_shift", "correlation_shift"),
        "lag-synchronization": ("correlation_shift", "covariance_shift"),
        "mode-correlation": ("correlation_shift", "covariance_shift"),
    }.get(group.anomaly_type, ())
    if candidate.witness not in canonical:
        return False
    if "dependence" in group.constraint_tag or "relation" in group.semantic_scope:
        return len(candidate.projection) >= 2
    return True


def format_projection(channels: Sequence[int]) -> str:
    """Serialize a channel projection for capability output tables."""

    return "|".join(str(int(channel)) for channel in channels)


__all__ = [
    "CandidateSpec",
    "WITNESS_FAMILY",
    "best_canonical_key",
    "candidate_is_canonical",
    "candidate_specs",
    "format_projection",
    "spec_by_key",
]
