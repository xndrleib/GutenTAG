"""Admission policy schema and YAML loading."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ADMISSION_POLICY_VERSION = "synthgen.admission.v12.1"
NUMERIC_GATE_NAMES = (
    "max_boundary_artifact_fail_share",
    "max_debug_event_share",
    "max_needs_repair_share",
    "min_canonical_observable_share",
    "min_oracle_annotation_pass_rate",
)


def default_status_rules() -> dict[str, Any]:
    return {
        "all": {
            "require_problem_genotype": True,
            "require_contract_id": True,
            "require_realized_effect_row": True,
            "require_support_integrity_row": True,
        },
        "non_boundary_anomalies": {
            "boundary_primary_detection_allowed": False,
        },
        "relation_anomalies": {
            "require_canonical_relation_witness": True,
            "allow_shortcut_but_downgrade_claim": True,
        },
    }


def default_numeric_gate_statuses() -> dict[str, str]:
    return {name: "provisional" for name in NUMERIC_GATE_NAMES}


@dataclass(frozen=True)
class AdmissionPolicy:
    """Status-driven admission policy with provisional numeric gates."""

    version: str = ADMISSION_POLICY_VERSION
    mode: str = "provisional"
    numeric_gate_enforcement: str = "provisional"
    promote_after_full_runs: int = 3
    max_boundary_artifact_fail_share: float = 0.0
    max_debug_event_share: float = 0.05
    max_needs_repair_share: float = 0.05
    min_canonical_observable_share: float = 0.90
    min_oracle_annotation_pass_rate: float = 0.95
    status_rules: Mapping[str, Any] = field(default_factory=default_status_rules)
    numeric_gate_statuses: Mapping[str, str] = field(
        default_factory=default_numeric_gate_statuses
    )
    extra_numeric_gates: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy payload."""

        numeric_gates: dict[str, Any] = {
            "enforcement": self.numeric_gate_enforcement,
            "promote_after_full_runs": self.promote_after_full_runs,
        }
        for gate_name in NUMERIC_GATE_NAMES:
            numeric_gates[gate_name] = {
                "value": getattr(self, gate_name),
                "status": str(self.numeric_gate_statuses.get(gate_name, "provisional")),
            }
        numeric_gates.update(dict(self.extra_numeric_gates))
        return {
            "admission_policy_version": self.version,
            "mode": self.mode,
            "status_rules": dict(self.status_rules),
            "numeric_gates": numeric_gates,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "AdmissionPolicy":
        """Construct an admission policy from a YAML-compatible mapping."""

        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Admission policy payload must contain a mapping")
        if "admission_policy" in payload:
            nested = payload["admission_policy"] or {}
            if not isinstance(nested, Mapping):
                raise ValueError("admission_policy must contain a mapping")
            payload = nested
        allowed = {
            "admission_policy_version",
            "version",
            "mode",
            "status_rules",
            "numeric_gates",
        }
        unknown = sorted(set(payload.keys()) - allowed)
        if unknown:
            raise ValueError(f"Unknown admission policy keys: {unknown}")
        numeric_gates = _mapping_or_empty(payload.get("numeric_gates"), "numeric_gates")
        gate_values: dict[str, float] = {}
        gate_statuses = default_numeric_gate_statuses()
        extra_numeric_gates: dict[str, Any] = {}
        for gate_name in NUMERIC_GATE_NAMES:
            if gate_name not in numeric_gates:
                continue
            gate_values[gate_name], gate_statuses[gate_name] = _parse_numeric_gate(
                gate_name,
                numeric_gates[gate_name],
            )
        for key, value in numeric_gates.items():
            if key in {*NUMERIC_GATE_NAMES, "enforcement", "promote_after_full_runs"}:
                continue
            extra_numeric_gates[str(key)] = value
        status_rules = payload.get("status_rules")
        if status_rules is None:
            active_status_rules = default_status_rules()
        elif isinstance(status_rules, Mapping):
            active_status_rules = dict(status_rules)
        else:
            raise ValueError("status_rules must contain a mapping")
        return cls(
            version=str(
                payload.get(
                    "admission_policy_version",
                    payload.get("version", ADMISSION_POLICY_VERSION),
                )
            ),
            mode=str(payload.get("mode", "provisional")),
            numeric_gate_enforcement=str(
                numeric_gates.get("enforcement", "provisional")
            ),
            promote_after_full_runs=int(
                numeric_gates.get("promote_after_full_runs", 3)
            ),
            status_rules=active_status_rules,
            numeric_gate_statuses=gate_statuses,
            extra_numeric_gates=extra_numeric_gates,
            **gate_values,
        )


def admission_policy_from_yaml(path: str | Path | None) -> AdmissionPolicy:
    """Load an admission policy from YAML if a path is supplied."""

    if path is None:
        return AdmissionPolicy()
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Admission policy YAML must contain a mapping")
    return AdmissionPolicy.from_mapping(payload)


def _mapping_or_empty(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must contain a mapping")
    return value


def _parse_numeric_gate(name: str, value: Any) -> tuple[float, str]:
    status = "provisional"
    raw_value = value
    if isinstance(value, Mapping):
        if "value" not in value:
            raise ValueError(f"numeric_gates.{name} must define a value")
        raw_value = value["value"]
        status = str(value.get("status", status))
    parsed = float(raw_value)
    if not math.isfinite(parsed):
        raise ValueError(f"numeric_gates.{name}.value must be finite")
    return parsed, status


__all__ = [
    "ADMISSION_POLICY_VERSION",
    "AdmissionPolicy",
    "NUMERIC_GATE_NAMES",
    "admission_policy_from_yaml",
    "default_numeric_gate_statuses",
    "default_status_rules",
]
