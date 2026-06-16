"""Anomaly contract registry loading."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from .anomaly_contract import AnomalyContract


class ContractRegistry:
    """Lookup registry for anomaly contracts."""

    def __init__(self, contracts: Iterable[AnomalyContract]) -> None:
        self._by_id = {contract.contract_id: contract for contract in contracts}
        self._by_anomaly = {contract.anomaly_type: contract for contract in contracts}

    @classmethod
    def from_resource_defaults(cls) -> "ContractRegistry":
        """Load bundled v12 anomaly contracts."""

        package = "gutenTAG.tsgen.contracts.resources.anomaly_contracts"
        root = resources.files(package)
        contracts = []
        for resource in sorted(root.iterdir(), key=lambda item: item.name):
            if not resource.name.endswith((".yaml", ".yml")):
                continue
            with resource.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"Contract resource must contain a mapping: {resource}"
                )
            contracts.append(AnomalyContract.from_mapping(payload))
        return cls(contracts)

    @classmethod
    def from_directory(cls, path: Path) -> "ContractRegistry":
        """Load anomaly contracts from a directory of YAML files."""

        contracts = []
        for resource in sorted(Path(path).glob("*.y*ml")):
            with resource.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if not isinstance(payload, Mapping):
                raise ValueError(f"Contract file must contain a mapping: {resource}")
            contracts.append(AnomalyContract.from_mapping(payload))
        return cls(contracts)

    def get(self, contract_id: str) -> AnomalyContract:
        """Return a contract by identifier."""

        return self._by_id[str(contract_id)]

    def get_by_anomaly(self, anomaly_type: str) -> AnomalyContract | None:
        """Return the default contract for an anomaly type, if known."""

        return self._by_anomaly.get(str(anomaly_type))

    def contracts(self) -> tuple[AnomalyContract, ...]:
        """Return all registered contracts."""

        return tuple(self._by_id.values())
