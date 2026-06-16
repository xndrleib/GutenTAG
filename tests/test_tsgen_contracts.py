import json
import tempfile
import unittest
from pathlib import Path

from gutenTAG.tsgen.capabilities.dataset import EventGroup, InstanceRecord
from gutenTAG.tsgen.contracts import (
    ContractRegistry,
    infer_legacy_genotype,
    write_v12_metadata_registries,
)


class TestTsgenContracts(unittest.TestCase):
    def test_default_contract_registry_loads_structural_contracts(self) -> None:
        registry = ContractRegistry.from_resource_defaults()

        mode = registry.get_by_anomaly("mode-correlation")
        corr = registry.get_by_anomaly("correlation-flip")

        self.assertIsNotNone(mode)
        self.assertIsNotNone(corr)
        self.assertIn("dependence.collective_factor", mode.intended_constraints)
        self.assertEqual(mode.channel_role_policy["canonical_min_projection_size"], 2)
        self.assertIn("dependence.correlation", corr.intended_constraints)

    def test_legacy_single_channel_mode_correlation_is_not_upgraded_to_relation(
        self,
    ) -> None:
        registry = ContractRegistry.from_resource_defaults()
        instance = _instance()
        group = _event_group(
            group_channels=(4,),
            intervention_channels=(4,),
            context_channels=(4,),
        )

        genotype = infer_legacy_genotype(instance, group, registry)

        self.assertEqual(genotype.provenance, "legacy_inferred")
        self.assertIn("missing_channel_roles", genotype.warnings)
        self.assertIn("relation_claim_not_inferred", genotype.warnings)
        self.assertEqual(genotype.constraints["violated"], ("legacy.channel_event",))
        self.assertEqual(
            genotype.constraints["canonical_witnesses"], ("energy_delta", "mean_delta")
        )

    def test_legacy_multichannel_mode_correlation_keeps_contract_claim(self) -> None:
        registry = ContractRegistry.from_resource_defaults()
        instance = _instance()
        group = _event_group(
            group_channels=(0, 1, 4),
            intervention_channels=(4,),
            context_channels=(0, 1),
        )

        genotype = infer_legacy_genotype(instance, group, registry)

        self.assertNotIn("relation_claim_not_inferred", genotype.warnings)
        self.assertIn("dependence.collective_factor", genotype.constraints["violated"])
        self.assertIn("regime.mode_alignment", genotype.constraints["violated"])
        self.assertEqual(genotype.alternative_process["context_channels"], (0, 1))

    def test_metadata_registry_writer_creates_compact_deduplicated_records(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset_instance(root, "instance_000", group_id=0)
            _write_dataset_instance(root, "instance_001", group_id=1)

            manifest = write_v12_metadata_registries(root, provenance="generated")

            self.assertEqual(manifest["event_count"], 2)
            self.assertEqual(manifest["problem_genotype_count"], 1)
            self.assertTrue(manifest["event_registry_hash"].startswith("sha256:"))
            self.assertTrue(manifest["contract_registry_hash"].startswith("sha256:"))

            events_path = root / manifest["event_registry_path"]
            genotypes_path = root / manifest["problem_genotype_registry_path"]
            events = _read_jsonl(events_path)
            genotypes = _read_jsonl(genotypes_path)

            self.assertEqual(len(events), 2)
            self.assertEqual(len(genotypes), 1)
            self.assertIn("genotype_id", events[0])
            self.assertIn("contract_id", events[0])
            self.assertNotIn("problem_genotype", events[0])
            self.assertEqual(genotypes[0]["provenance"], "generated")
            self.assertEqual(genotypes[0]["version"], "synthgen.problem_genotype.v1")


def _instance() -> InstanceRecord:
    root = Path("/tmp/synth-gen-test")
    return InstanceRecord(
        dataset_root=root,
        variant_id="random-mode-jump__mode-correlation__p00",
        split="train",
        instance_id="instance_001",
        instance_dir=root,
        clean_path=root / "clean.csv",
        anomalous_path=root / "anomalous.csv",
        events_path=root / "events.json",
        summary_path=root / "instance_summary.json",
        base_oscillation="random-mode-jump",
        anomaly_type="mode-correlation",
        channels=5,
        length=10000,
        event_groups=(),
    )


def _event_group(
    *,
    group_channels: tuple[int, ...],
    intervention_channels: tuple[int, ...],
    context_channels: tuple[int, ...],
) -> EventGroup:
    return EventGroup(
        group_id="39",
        start=9608,
        end=9613,
        source_start=9608,
        source_end=9613,
        anomaly_type="mode-correlation",
        constraint_tag="regime.mode_correlation",
        repair_operator="restore_regime_conditioned_dependence",
        semantic_scope="regime_relation",
        intervention_channels=intervention_channels,
        context_channels=context_channels,
        group_channels=group_channels,
        primary_channels=intervention_channels,
        event_scope="legacy",
        purity_hint="unknown",
        raw_events=(),
    )


def _write_dataset_instance(root: Path, instance_id: str, *, group_id: int) -> None:
    variant_id = "sine__mean__p00"
    instance_dir = root / "variants" / variant_id / "train" / "instances" / instance_id
    instance_dir.mkdir(parents=True)
    csv_payload = "c0,c1\n0.0,0.0\n0.1,0.1\n0.2,0.2\n0.3,0.3\n"
    (instance_dir / "clean.csv").write_text(csv_payload, encoding="utf-8")
    (instance_dir / "anomalous.csv").write_text(csv_payload, encoding="utf-8")
    events = [
        {
            "start": 1,
            "end": 3,
            "channel": 0,
            "anomaly_type": "mean",
            "group_id": group_id,
            "group_channels": [0],
            "intervention_channels": [0],
            "operator_target_channels": [0],
            "perturbed_channels": [0],
            "context_channels": [0],
            "event_scope": "channel",
            "params": {"offset": 1.0},
            "length": 2,
            "source_start": 1,
            "source_end": 3,
            "anomaly_object": "mean",
            "channel_visible": True,
            "purity_hint": "channel_visible",
        }
    ]
    summary = {
        "length": 4,
        "channels": 2,
        "variant_id": variant_id,
        "split": "train",
        "instance_id": instance_id,
        "anomaly_type": "mean",
        "base_oscillation": "sine",
    }
    (instance_dir / "events.json").write_text(json.dumps(events), encoding="utf-8")
    (instance_dir / "instance_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
