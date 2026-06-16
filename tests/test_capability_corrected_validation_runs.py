import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
import gutenTAG.tsgen.capabilities.corrected_detectability_blind_scan as corrected_blind_scan_module
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis
from gutenTAG.tsgen.capabilities.corrected_detectability import (
    compute_corrected_detectability_frontier,
)
from gutenTAG.tsgen.capabilities.dataset import (
    DatasetIndex,
    InstanceRecord,
    discover_dataset,
)
from gutenTAG.tsgen.contracts import write_v12_metadata_registries

from tests.capability_execution_fixtures import (
    _event_group,
    _small_generation_config,
)


class TestCapabilityExecutionProfiles(unittest.TestCase):
    def test_corrected_detectability_uses_clean_only_calibration_instances(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            config = _small_generation_config(dataset_root)
            config["dataset"]["length"] = 120
            config["dataset"]["splits"] = {
                "train": {"paired_instances_per_variant": 1},
                "calibration": {"clean_only_instances_per_variant": 3},
            }
            config["dataset"].pop("instances_per_split")
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            TSDatasetGenerator.from_dict(config).run()

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=4,
                    bootstrap_samples=0,
                ),
                profile_preset="calibration",
            )

            corrected = pd.read_csv(output_dir / "corrected_detectability_frontier.csv")
            self.assertEqual(set(corrected["split"]), {"train"})
            self.assertGreaterEqual(int(corrected["candidate_null_count"].max()), 12)

    def test_run_consumes_metadata_registry_without_legacy_events_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()
            write_v12_metadata_registries(dataset_root, provenance="generated")
            for events_path in dataset_root.glob(
                "variants/*/*/instances/*/events.json"
            ):
                events_path.unlink()

            dataset = discover_dataset(dataset_root)
            self.assertTrue(dataset.metadata_events)
            self.assertTrue(dataset.problem_genotypes)
            self.assertTrue(
                any(
                    group.genotype_id
                    for instance in dataset.instances
                    for group in instance.event_groups
                )
            )
            self.assertFalse(
                any(instance.events_path.exists() for instance in dataset.instances)
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=4,
                    bootstrap_samples=0,
                ),
                profile_preset="annotation",
                label_export="diagnostics",
            )

            alignment = pd.read_csv(output_dir / "annotation_alignment.csv")
            self.assertFalse(alignment.empty)

    def test_corrected_detectability_reuses_blind_scan_score_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            clean = np.zeros((80, 1), dtype=float)
            anomalous = clean.copy()
            anomalous[10:20, 0] = 3.0
            anomalous[30:40, 0] = 3.0
            pd.DataFrame(clean, columns=["value-0"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["value-0"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
            instance = InstanceRecord(
                dataset_root=root,
                variant_id="sine__mean__p00",
                split="train",
                instance_id="instance_000",
                instance_dir=instance_dir,
                clean_path=instance_dir / "clean.csv",
                anomalous_path=instance_dir / "anomalous.csv",
                events_path=instance_dir / "events.json",
                summary_path=instance_dir / "instance_summary.json",
                base_oscillation="sine",
                anomaly_type="mean",
                channels=1,
                length=80,
                event_groups=(
                    _event_group("0", 10, 20),
                    _event_group("1", 30, 40),
                ),
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(instance,))
            protocol = CapabilityProtocol(
                alpha_grid=(0.10,),
                detection_witnesses=("mean_z",),
                max_projection_size=1,
                max_scan_windows_per_length=6,
                bootstrap_samples=0,
            )

            with patch.object(
                corrected_blind_scan_module,
                "blind_scan_score_block",
                wraps=corrected_blind_scan_module.blind_scan_score_block,
            ) as score_block:
                result = compute_corrected_detectability_frontier(dataset, protocol)

            self.assertFalse(result.frontier.empty)
            self.assertEqual(score_block.call_count, 1)


if __name__ == "__main__":
    unittest.main()
