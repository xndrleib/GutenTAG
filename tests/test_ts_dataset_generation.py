import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.base_oscillations import BaseOscillation
from tests.ts_dataset_generation_fixtures import (
    TSDatasetGenerationConfigMixin,
    file_sha256,
)

LABEL_SIDECARS = (
    "labels_oracle_any",
    "labels_oracle_intervention",
    "labels_oracle_context",
    "labels_event_only",
    "labels_delayed",
    "labels_weak_point",
    "labels_visible_only",
    "labels_noisy_boundary",
    "labels_censored",
)

INSTANCE_ARTIFACTS = (
    "clean.csv",
    "anomalous.csv",
    "labels_pointwise.csv",
    "labels_any.csv",
    "labels_intervention.csv",
    "labels_context.csv",
    "events.json",
    "instance_summary.json",
    "plot_full.png",
)


def _assert_dataset_manifest_contract(
    testcase: unittest.TestCase,
    manifest: dict,
    output_root: Path,
) -> Path:
    testcase.assertIn("sine__mean__p00", manifest["generated_variants"])
    for key in (
        "config",
        "derived_seeds",
        "aggregated_statistics",
        "normalized_config_hash",
        "label_semantics",
        "artifacts",
        "annotation_channels",
        "law_level_replicates",
    ):
        testcase.assertIn(key, manifest)
    testcase.assertEqual(manifest["dataset_schema_version"], "synthgen.dataset.v1")
    variant_dir = output_root / "variants" / "sine__mean__p00"
    testcase.assertTrue((output_root / "dataset_manifest.json").exists())
    testcase.assertTrue((variant_dir / "variant_config.yaml").exists())
    return variant_dir


def _assert_label_sidecars(
    testcase: unittest.TestCase,
    manifest: dict,
    output_root: Path,
) -> None:
    labels_dir = output_root / "labels"
    for label_name in LABEL_SIDECARS:
        label_path = labels_dir / f"{label_name}.csv"
        testcase.assertTrue(label_path.exists(), label_path)
        testcase.assertEqual(
            manifest["annotation_channels"]["table_paths"][label_name],
            f"labels/{label_name}.csv",
        )
    testcase.assertTrue((labels_dir / "annotation_channel_manifest.json").exists())
    oracle_labels = pd.read_csv(labels_dir / "labels_oracle_any.csv")
    testcase.assertEqual(oracle_labels.shape[0], 600 * 2 * 2)


def _assert_law_sidecars(
    testcase: unittest.TestCase,
    manifest: dict,
    output_root: Path,
) -> None:
    law_manifest = manifest["law_level_replicates"]
    testcase.assertEqual(
        law_manifest["replicate_count"],
        manifest["metadata_registry"]["event_count"],
    )
    testcase.assertTrue(
        (output_root / law_manifest["replicate_registry_path"]).exists()
    )
    testcase.assertTrue((output_root / law_manifest["replicate_table_path"]).exists())
    law_table = pd.read_csv(output_root / law_manifest["replicate_table_path"])
    testcase.assertEqual(law_table.shape[0], law_manifest["replicate_count"])
    testcase.assertIn("genotype_id", law_table.columns)
    testcase.assertIn("paired_seed_policy", law_table.columns)


def _assert_variant_instances(
    testcase: unittest.TestCase,
    variant_dir: Path,
) -> None:
    for split in ["train", "val"]:
        split_dir = variant_dir / split
        testcase.assertTrue((split_dir / "split_summary.json").exists())
        for instance_idx in range(2):
            instance_dir = split_dir / "instances" / f"instance_{instance_idx:03d}"
            _assert_instance_artifacts(testcase, instance_dir)


def _assert_instance_artifacts(
    testcase: unittest.TestCase,
    instance_dir: Path,
) -> None:
    for filename in INSTANCE_ARTIFACTS:
        testcase.assertTrue((instance_dir / filename).exists())
    testcase.assertFalse((instance_dir / "plot.png").exists())
    for zoom_idx in range(5):
        testcase.assertTrue((instance_dir / f"zoom_{zoom_idx:02d}.png").exists())
    _assert_pointwise_labels_match_events(testcase, instance_dir)
    _assert_instance_summary_contract(testcase, instance_dir)


def _assert_pointwise_labels_match_events(
    testcase: unittest.TestCase,
    instance_dir: Path,
) -> None:
    labels = pd.read_csv(instance_dir / "labels_pointwise.csv").to_numpy()
    testcase.assertEqual(labels.shape, (600, 3))
    with (instance_dir / "events.json").open("r", encoding="utf-8") as f:
        events = json.load(f)

    reconstructed = np.zeros((600, 3), dtype=np.int8)
    for event in events:
        start = int(event["start"])
        end = int(event["end"])
        reconstructed[start:end, int(event["channel"])] = 1
        testcase.assertIn("params", event)
        testcase.assertIn("length", event)
        testcase.assertEqual(int(event["length"]), end - start)
    np.testing.assert_array_equal(labels, reconstructed)


def _assert_instance_summary_contract(
    testcase: unittest.TestCase,
    instance_dir: Path,
) -> None:
    with (instance_dir / "instance_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    testcase.assertEqual(summary["variant_id"], "sine__mean__p00")
    testcase.assertEqual(summary["profile_id"], "p00")
    testcase.assertEqual(summary["overlap_policy"], "global")
    testcase.assertEqual(summary["base_parameter_policy"], "fixed_per_variant")
    testcase.assertEqual(summary["anomaly_parameter_policy"], "fixed_per_variant")


class TestTSDatasetGeneration(TSDatasetGenerationConfigMixin, unittest.TestCase):
    def test_artifacts_manifest_and_label_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            manifest = TSDatasetGenerator.from_dict(config).run()

            variant_dir = _assert_dataset_manifest_contract(self, manifest, output_root)
            _assert_label_sidecars(self, manifest, output_root)
            _assert_law_sidecars(self, manifest, output_root)
            _assert_variant_instances(self, variant_dir)

    def test_v12_sidecar_config_controls_emitted_channels_and_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 120
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["segment_count_range"] = [2, 2]
            config["plot"]["enabled"] = False
            config["annotation_channels"] = {"emit": ["oracle", "event_only"]}
            config["law_level_replicates"] = {
                "enabled": True,
                "replicas_per_genotype": 1,
                "paired_seed_policy": "same_base_parameters",
                "output_split": "law_replicates_debug",
            }

            manifest = TSDatasetGenerator.from_dict(config).run()

            emitted = set(manifest["annotation_channels"]["emit"])
            self.assertEqual(
                emitted,
                {
                    "labels_oracle_any",
                    "labels_oracle_intervention",
                    "labels_oracle_context",
                    "labels_event_only",
                },
            )
            self.assertTrue((output_root / "labels" / "labels_event_only.csv").exists())
            self.assertFalse((output_root / "labels" / "labels_delayed.csv").exists())
            law_manifest = manifest["law_level_replicates"]
            self.assertEqual(law_manifest["output_split"], "law_replicates_debug")
            self.assertEqual(law_manifest["replicas_per_genotype_requested"], 1)
            self.assertEqual(
                law_manifest["replicate_count"], law_manifest["genotype_count"]
            )
            law_table = pd.read_csv(output_root / law_manifest["replicate_table_path"])
            self.assertEqual(
                law_table["paired_seed_policy"].unique().tolist(),
                ["same_base_parameters"],
            )

    def test_split_mapping_can_emit_clean_only_calibration_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 120
            config["dataset"]["splits"] = {
                "train": {"paired_instances_per_variant": 1},
                "calibration": {"clean_only_instances_per_variant": 2},
            }
            config["dataset"].pop("instances_per_split")
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()

            variant = output_root / "variants" / "sine__mean__p00"
            self.assertTrue((variant / "train" / "instances" / "instance_000").exists())
            for index in range(2):
                instance_dir = (
                    variant / "calibration" / "instances" / f"clean_only_{index:03d}"
                )
                self.assertTrue((instance_dir / "clean.csv").exists())
                self.assertTrue((instance_dir / "anomalous.csv").exists())
                with (instance_dir / "events.json").open(
                    "r", encoding="utf-8"
                ) as handle:
                    self.assertEqual(json.load(handle), [])
                with (instance_dir / "instance_summary.json").open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    summary = json.load(handle)
                self.assertEqual(summary["instance_role"], "clean_only")
                self.assertFalse(summary["has_anomaly"])

            split_entries = manifest["variant_manifests"][0]["splits"]
            by_split = {entry["split"]: entry for entry in split_entries}
            self.assertEqual(by_split["train"]["paired_instances"], 1)
            self.assertEqual(by_split["train"]["clean_only_instances"], 0)
            self.assertEqual(by_split["calibration"]["paired_instances"], 0)
            self.assertEqual(by_split["calibration"]["clean_only_instances"], 2)
            self.assertEqual(manifest["metadata_registry"]["event_count"], 1)

    def test_reproducible_outputs_for_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)

            generator = TSDatasetGenerator.from_dict(config)
            generator.run()
            first_hashes = {
                p.relative_to(output_root).as_posix(): file_sha256(p)
                for p in sorted(output_root.rglob("*"))
                if p.is_file() and p.name != "generation.log"
            }

            generator = TSDatasetGenerator.from_dict(config)
            generator.run()
            second_hashes = {
                p.relative_to(output_root).as_posix(): file_sha256(p)
                for p in sorted(output_root.rglob("*"))
                if p.is_file() and p.name != "generation.log"
            }
            self.assertEqual(first_hashes, second_hashes)

    def test_density_incompatible_variant_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["variants"]["anomaly_types"] = ["extremum"]
            with self.assertRaisesRegex(ValueError, "no instances"):
                TSDatasetGenerator.from_dict(config).run()

    def test_previously_failing_anomalies_do_not_crash_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 800
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.10]
            config["anomaly_policy"]["density_tolerance"] = 0.05
            config["anomaly_policy"]["segment_count_range"] = [20, 22]

            candidate_bases = ["sine", "cosine", "square", "sawtooth", "dirichlet"]
            bases = [
                kind for kind in candidate_bases if kind in BaseOscillation.key_mapping
            ]
            config["variants"]["base_oscillations"] = bases
            config["variants"]["anomaly_types"] = [
                "amplitude",
                "pattern",
                "pattern-shift",
                "trend",
            ]
            manifest = TSDatasetGenerator.from_dict(config).run()

            generated = set(manifest["generated_variants"])
            for base in bases:
                for anomaly in ["amplitude", "pattern", "pattern-shift", "trend"]:
                    variant_id = f"{base}__{anomaly}__p00"
                    self.assertIn(variant_id, generated)

    def test_profile_expansion_and_random_per_segment_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["variants"]["profiles_per_pair"] = 2
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["plot"]["enabled"] = False
            manifest = TSDatasetGenerator.from_dict(config).run()

            self.assertEqual(
                sorted(manifest["generated_variants"]),
                ["sine__mean__p00", "sine__mean__p01"],
            )
            events_path = (
                output_root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
                / "events.json"
            )
            with events_path.open("r", encoding="utf-8") as f:
                events = json.load(f)
            self.assertGreater(len(events), 0)
            self.assertTrue(all("params" in event for event in events))


if __name__ == "__main__":
    unittest.main()
