"""Legacy genotype inference with conservative relation-claim handling."""

from __future__ import annotations

from .anomaly_contract import AnomalyContract
from .problem_genotype import ProblemGenotype
from .registry import ContractRegistry
from ..capabilities.dataset import EventGroup, InstanceRecord


def infer_legacy_genotype(
    instance: InstanceRecord,
    group: EventGroup,
    registry: ContractRegistry | None = None,
) -> ProblemGenotype:
    """Infer a limited genotype from legacy event metadata.

    Relation contracts are not inferred from the anomaly name alone. If the
    channel roles are insufficient for the contract's canonical projection size,
    the genotype is downgraded to a legacy channel event.
    """

    active_registry = registry or ContractRegistry.from_resource_defaults()
    contract = active_registry.get_by_anomaly(group.anomaly_type)
    warnings: list[str] = ["legacy_inferred"]
    if contract is None:
        warnings.append("missing_contract")
    has_required_roles = _has_required_channel_roles(group, contract)
    if contract is not None and not has_required_roles:
        warnings.extend(["missing_channel_roles", "relation_claim_not_inferred"])
    violated = (
        contract.intended_constraints
        if contract is not None and has_required_roles
        else ("legacy.channel_event",)
    )
    canonical_witnesses = (
        contract.canonical_witnesses
        if contract is not None and has_required_roles
        else ("energy_delta", "mean_delta")
    )
    contract_id = (
        contract.contract_id
        if contract is not None and has_required_roles
        else "synthgen.contract.legacy_channel_event.v1"
    )
    return ProblemGenotype(
        version="synthgen.problem_genotype.v1",
        genotype_id=_genotype_id(instance, group, contract_id),
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
            "intervention_channels": group.intervention_channels,
            "context_channels": group.context_channels,
            "group_channels": group.group_channels,
        },
        constraints={
            "violated": violated,
            "forbidden_shortcuts": (
                contract.forbidden_shortcuts if contract is not None else ()
            ),
            "canonical_witnesses": canonical_witnesses,
        },
        support={
            "temporal": (group.start, group.end),
            "source_temporal": (group.source_start, group.source_end),
            "channels": group.group_channels,
        },
        target={
            "tasks": ("event_detection", "witness_description"),
        },
        annotation_channel={
            "type": "legacy_oracle_pointwise",
        },
        loss={
            "detection": "event_f1_with_latency",
        },
        provenance="legacy_inferred",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _has_required_channel_roles(
    group: EventGroup, contract: AnomalyContract | None
) -> bool:
    if contract is None:
        return False
    min_projection = int(
        contract.channel_role_policy.get("canonical_min_projection_size", 1)
    )
    context_min = int(contract.channel_role_policy.get("context_min", 0))
    intervention_min = int(contract.channel_role_policy.get("intervention_min", 1))
    return (
        len(group.group_channels) >= min_projection
        and len(group.context_channels) >= context_min
        and len(group.intervention_channels) >= intervention_min
    )


def _operator_family(violated_constraints: tuple[str, ...]) -> str:
    if not violated_constraints:
        return "unknown"
    first = violated_constraints[0]
    return first.split(".", maxsplit=1)[0] if "." in first else first


def _genotype_id(instance: InstanceRecord, group: EventGroup, contract_id: str) -> str:
    return (
        f"{instance.variant_id}__{instance.split}__{instance.instance_id}"
        f"__g{group.group_id}__{contract_id}"
    )
