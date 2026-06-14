"""Anomaly contract model used by v12 validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AnomalyContract:
    """Contract that maps anomaly implementations to validation obligations."""

    contract_id: str
    anomaly_type: str
    intended_constraints: tuple[str, ...]
    canonical_witnesses: tuple[str, ...]
    allowed_artifacts: Mapping[str, Any]
    forbidden_shortcuts: tuple[str, ...]
    support_policy: Mapping[str, Any]
    channel_role_policy: Mapping[str, Any]
    realized_effect_schema: tuple[str, ...]
    admission_policy: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AnomalyContract":
        """Build a contract from a YAML/JSON mapping."""

        required = {
            "contract_id",
            "anomaly_type",
            "intended_constraints",
            "canonical_witnesses",
            "allowed_artifacts",
            "forbidden_shortcuts",
            "support_policy",
            "channel_role_policy",
            "realized_effect_schema",
            "admission_policy",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Anomaly contract is missing required keys: {missing}")
        return cls(
            contract_id=str(payload["contract_id"]),
            anomaly_type=str(payload["anomaly_type"]),
            intended_constraints=tuple(map(str, payload.get("intended_constraints", ()))),
            canonical_witnesses=tuple(map(str, payload.get("canonical_witnesses", ()))),
            allowed_artifacts=dict(payload.get("allowed_artifacts", {})),
            forbidden_shortcuts=tuple(map(str, payload.get("forbidden_shortcuts", ()))),
            support_policy=dict(payload.get("support_policy", {})),
            channel_role_policy=dict(payload.get("channel_role_policy", {})),
            realized_effect_schema=tuple(map(str, payload.get("realized_effect_schema", ()))),
            admission_policy=dict(payload.get("admission_policy", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable contract mapping."""

        return asdict(self)
