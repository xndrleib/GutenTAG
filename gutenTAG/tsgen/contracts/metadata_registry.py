"""Writers for v12 metadata registries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ..capabilities.dataset import (
    DatasetIndex,
    EventGroup,
    InstanceRecord,
    discover_dataset,
    event_uid,
)
from ..io import sanitize_json_value, write_json
from ..manifest import canonical_json_hash
from .anomaly_contract import AnomalyContract
from .problem_genotype import ProblemGenotype
from .registry import ContractRegistry
from .schema import (
    CONTRACT_REGISTRY_VERSION,
    METADATA_REGISTRY_VERSION,
    PROBLEM_GENOTYPE_VERSION,
)

Provenance = Literal["generated", "legacy_inferred", "manual_override"]


def build_event_registry_records(
    dataset: DatasetIndex,
    registry: ContractRegistry | None = None,
    *,
    provenance: Provenance = "generated",
) -> list[dict[str, Any]]:
    """Create compact event records for ``metadata/events.jsonl``."""

    active_registry = registry or ContractRegistry.from_resource_defaults()
    records: list[dict[str, Any]] = []
    for instance in dataset.instances:
        for group in instance.event_groups:
            genotype = _build_problem_genotype(instance, group, active_registry, provenance)
            records.append(
                {
                    "event_id": event_uid(instance, group),
                    "metadata_registry_version": METADATA_REGISTRY_VERSION,
                    "variant_id": instance.variant_id,
                    "split": instance.split,
                    "instance_id": instance.instance_id,
                    "group_id": group.group_id,
                    "genotype_id": genotype.genotype_id,
                    "contract_id": _contract_id_for_group(group, active_registry),
                    "requested_effect_id": _requested_effect_id(instance, group),
                    "anomaly_type": group.anomaly_type,
                    "base_oscillation": instance.base_oscillation,
                    "support_start": group.start,
                    "support_end": group.end,
                    "source_start": group.source_start,
                    "source_end": group.source_end,
                    "intervention_channels": group.intervention_channels,
                    "context_channels": group.context_channels,
                    "group_channels": group.group_channels,
                    "primary_channels": group.primary_channels,
                    "label_channel": "oracle",
                    "event_scope": group.event_scope,
                    "purity_hint": group.purity_hint,
                    "legacy_provenance": None if provenance == "generated" else provenance,
                }
            )
    return _sort_records(records, ("event_id",))


def build_problem_genotype_registry(
    dataset: DatasetIndex,
    registry: ContractRegistry | None = None,
    *,
    provenance: Provenance = "generated",
) -> list[dict[str, Any]]:
    """Create deduplicated genotype records for ``problem_genotypes.jsonl``."""

    active_registry = registry or ContractRegistry.from_resource_defaults()
    records_by_id: dict[str, dict[str, Any]] = {}
    for instance in dataset.instances:
        for group in instance.event_groups:
            genotype = _build_problem_genotype(instance, group, active_registry, provenance)
            payload = genotype.to_dict()
            existing = records_by_id.get(genotype.genotype_id)
            if existing is None:
                records_by_id[genotype.genotype_id] = payload
            elif existing != payload:
                raise ValueError(f"Conflicting genotype payload for {genotype.genotype_id}")
    return _sort_records(records_by_id.values(), ("genotype_id",))


def build_requested_effect_records(
    dataset: DatasetIndex,
    registry: ContractRegistry | None = None,
    *,
    provenance: Provenance = "generated",
) -> list[dict[str, Any]]:
    """Create requested-effect records separated from realized effects."""

    active_registry = registry or ContractRegistry.from_resource_defaults()
    records: list[dict[str, Any]] = []
    for instance in dataset.instances:
        for group in instance.event_groups:
            genotype = _build_problem_genotype(instance, group, active_registry, provenance)
            records.append(
                {
                    "requested_effect_id": _requested_effect_id(instance, group),
                    "metadata_registry_version": METADATA_REGISTRY_VERSION,
                    "event_id": event_uid(instance, group),
                    "genotype_id": genotype.genotype_id,
                    "contract_id": _contract_id_for_group(group, active_registry),
                    "variant_id": instance.variant_id,
                    "split": instance.split,
                    "instance_id": instance.instance_id,
                    "group_id": group.group_id,
                    "anomaly_type": group.anomaly_type,
                    "params": _group_params(group),
                    "intervention_channels": group.intervention_channels,
                    "context_channels": group.context_channels,
                    "group_channels": group.group_channels,
                }
            )
    return _sort_records(records, ("requested_effect_id",))


def write_v12_metadata_registries(
    dataset_root: Path,
    registry: ContractRegistry | None = None,
    *,
    provenance: Provenance = "generated",
) -> dict[str, Any]:
    """Write v12 metadata registries and return manifest additions."""

    root = Path(dataset_root).resolve()
    dataset = discover_dataset(root, prefer_metadata_registry=False)
    active_registry = registry or ContractRegistry.from_resource_defaults()
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    events = build_event_registry_records(dataset, active_registry, provenance=provenance)
    genotypes = build_problem_genotype_registry(dataset, active_registry, provenance=provenance)
    requested_effects = build_requested_effect_records(
        dataset,
        active_registry,
        provenance=provenance,
    )

    event_path = metadata_dir / "events.jsonl"
    genotype_path = metadata_dir / "problem_genotypes.jsonl"
    requested_effect_path = metadata_dir / "requested_effects.jsonl"
    contract_manifest_path = metadata_dir / "contract_registry_manifest.json"
    schema_manifest_path = metadata_dir / "schema_manifest.json"

    _write_jsonl(event_path, events)
    _write_jsonl(genotype_path, genotypes)
    _write_jsonl(requested_effect_path, requested_effects)

    contract_manifest = _contract_registry_manifest(active_registry)
    schema_manifest = {
        "metadata_registry_version": METADATA_REGISTRY_VERSION,
        "problem_genotype_version": PROBLEM_GENOTYPE_VERSION,
        "contract_registry_version": CONTRACT_REGISTRY_VERSION,
    }
    write_json(contract_manifest_path, contract_manifest, sort_keys=True, indent=2)
    write_json(schema_manifest_path, schema_manifest, sort_keys=True, indent=2)

    return {
        "metadata_registry_version": METADATA_REGISTRY_VERSION,
        "problem_genotype_version": PROBLEM_GENOTYPE_VERSION,
        "contract_registry_version": CONTRACT_REGISTRY_VERSION,
        "event_registry_path": _relative_path(event_path, root),
        "problem_genotype_registry_path": _relative_path(genotype_path, root),
        "requested_effects_path": _relative_path(requested_effect_path, root),
        "contract_registry_manifest_path": _relative_path(contract_manifest_path, root),
        "schema_manifest_path": _relative_path(schema_manifest_path, root),
        "event_registry_hash": _file_hash(event_path),
        "problem_genotype_registry_hash": _file_hash(genotype_path),
        "requested_effects_hash": _file_hash(requested_effect_path),
        "contract_registry_hash": contract_manifest["contract_registry_hash"],
        "contract_registry_manifest_hash": _file_hash(contract_manifest_path),
        "schema_manifest_hash": _file_hash(schema_manifest_path),
        "event_count": len(events),
        "problem_genotype_count": len(genotypes),
        "requested_effect_count": len(requested_effects),
    }


def _build_problem_genotype(
    instance: InstanceRecord,
    group: EventGroup,
    registry: ContractRegistry,
    provenance: Provenance,
) -> ProblemGenotype:
    contract = registry.get_by_anomaly(group.anomaly_type)
    warnings = _warnings_for_group(group, contract, provenance)
    has_required_roles = _has_required_channel_roles(group, contract)
    active_contract = contract if contract is not None and has_required_roles else None
    violated = (
        active_contract.intended_constraints
        if active_contract is not None
        else ("legacy.channel_event",)
    )
    canonical_witnesses = (
        active_contract.canonical_witnesses
        if active_contract is not None
        else ("energy_delta", "mean_delta")
    )
    return ProblemGenotype(
        version=PROBLEM_GENOTYPE_VERSION,
        genotype_id=_genotype_id(instance, active_contract),
        observation={
            "sampling": "regular",
            "length": instance.length,
            "channels": instance.channels,
            "preprocessing": "identity",
        },
        normal_process={
            "base_family": instance.base_oscillation,
        },
        alternative_process={
            "anomaly_type": group.anomaly_type,
            "operator_family": _operator_family(violated),
            "channel_role_policy": (
                active_contract.channel_role_policy if active_contract is not None else {}
            ),
        },
        constraints={
            "violated": violated,
            "forbidden_shortcuts": (
                active_contract.forbidden_shortcuts if active_contract is not None else ()
            ),
            "canonical_witnesses": canonical_witnesses,
        },
        support={
            "policy": (
                active_contract.support_policy
                if active_contract is not None
                else {"type": "localized_interval", "expected_primary_evidence": "unknown"}
            ),
            "event_support_in": "metadata/events.jsonl",
        },
        target={
            "tasks": ("event_detection", "constraint_diagnosis", "witness_description"),
        },
        annotation_channel={
            "type": "oracle_deterministic",
        },
        loss={
            "detection": "event_f1_with_latency",
        },
        provenance=provenance,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _contract_id_for_group(group: EventGroup, registry: ContractRegistry) -> str:
    contract = registry.get_by_anomaly(group.anomaly_type)
    if contract is None or not _has_required_channel_roles(group, contract):
        return "synthgen.contract.legacy_channel_event.v1"
    return contract.contract_id


def _genotype_id(instance: InstanceRecord, contract: AnomalyContract | None) -> str:
    contract_id = (
        contract.contract_id
        if contract is not None
        else "synthgen.contract.legacy_channel_event.v1"
    )
    return f"genotype:{instance.variant_id}:{contract_id}:v12"


def _warnings_for_group(
    group: EventGroup,
    contract: AnomalyContract | None,
    provenance: Provenance,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if provenance == "legacy_inferred":
        warnings.append("legacy_inferred")
    if contract is None:
        warnings.append("missing_contract")
    elif not _has_required_channel_roles(group, contract):
        warnings.extend(["missing_channel_roles", "relation_claim_not_inferred"])
    return tuple(warnings)


def _has_required_channel_roles(group: EventGroup, contract: AnomalyContract | None) -> bool:
    if contract is None:
        return False
    min_projection = int(contract.channel_role_policy.get("canonical_min_projection_size", 1))
    context_min = int(contract.channel_role_policy.get("context_min", 0))
    intervention_min = int(contract.channel_role_policy.get("intervention_min", 1))
    return (
        len(group.group_channels) >= min_projection
        and len(group.context_channels) >= context_min
        and len(group.intervention_channels) >= intervention_min
    )


def _operator_family(violated_constraints: Sequence[str]) -> str:
    if not violated_constraints:
        return "unknown"
    first = str(violated_constraints[0])
    return first.split(".", maxsplit=1)[0] if "." in first else first


def _group_params(group: EventGroup) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for event in group.raw_events:
        raw_params = event.get("params", {})
        params.append(dict(raw_params) if isinstance(raw_params, Mapping) else {})
    return params


def _requested_effect_id(instance: InstanceRecord, group: EventGroup) -> str:
    return f"req:{event_uid(instance, group)}"


def _contract_registry_manifest(registry: ContractRegistry) -> dict[str, Any]:
    contracts = [
        contract.to_dict()
        for contract in sorted(registry.contracts(), key=lambda item: item.contract_id)
    ]
    payload = {
        "contract_registry_version": CONTRACT_REGISTRY_VERSION,
        "contracts": contracts,
    }
    return {
        **payload,
        "contract_registry_hash": canonical_json_hash(payload),
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    sanitize_json_value(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sort_records(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in sorted(
            records,
            key=lambda record: tuple(str(record.get(key, "")) for key in keys),
        )
    ]
