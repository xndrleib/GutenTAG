"""Problem genotype model for v12 benchmark contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ProblemGenotype:
    """Machine-readable description of a generated benchmark task."""

    version: str
    genotype_id: str
    observation: Mapping[str, Any]
    normal_process: Mapping[str, Any]
    alternative_process: Mapping[str, Any]
    constraints: Mapping[str, Any]
    support: Mapping[str, Any]
    target: Mapping[str, Any]
    annotation_channel: Mapping[str, Any]
    loss: Mapping[str, Any]
    provenance: Literal["generated", "legacy_inferred", "manual_override"]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable genotype mapping."""

        return asdict(self)
