import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis
from gutenTAG.tsgen.capabilities.output_store import ParquetOutputError

from tests.capability_execution_fixtures import (
    _parquet_engine_available,
    _small_generation_config,
)


class TestCapabilityExecutionProfiles(unittest.TestCase):
    def test_run_can_materialize_arrays_and_write_observability_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            cache_dir = tmp_path / "cache"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profiles=("observability",),
                cache_dir=cache_dir,
                materialize_arrays=True,
            )

            self.assertEqual(certificate["profiles"], ["observability"])
            self.assertTrue((output_dir / "observability_profile.csv").exists())
            self.assertTrue(
                (output_dir / "tables_csv" / "observability_profile.csv").exists()
            )
            self.assertTrue((output_dir / "event_capability_summary.csv").exists())
            self.assertFalse((output_dir / "detectability_frontier.csv").exists())
            self.assertTrue(list((cache_dir / "arrays").glob("**/*.npy")))
            manifest = json.loads(
                (output_dir / "manifests" / "output_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["output_manifest_version"], "synthgen.capability.output.v2"
            )
            self.assertEqual(manifest["layout"]["release_csv_dir"], "tables_csv")
            table_names = {record["name"] for record in manifest["tables"]}
            self.assertIn("observability", table_names)
            self.assertIn("event_summary", table_names)
            observability_record = next(
                record
                for record in manifest["tables"]
                if record["name"] == "observability"
            )
            self.assertEqual(
                observability_record["canonical"]["path"],
                "tables_csv/observability_profile.csv",
            )
            self.assertEqual(
                observability_record["release_csv"]["path"],
                "tables_csv/observability_profile.csv",
            )
            self.assertEqual(
                observability_record["legacy_csv"]["path"], "observability_profile.csv"
            )
            hashes = json.loads(
                (output_dir / "manifests" / "table_hashes.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("tables_csv/observability_profile.csv", hashes["file_hashes"])
            self.assertIn("observability", hashes["table_content_hashes"])

    def test_run_parquet_output_fails_fast_without_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )
            if not _parquet_engine_available():
                with self.assertRaises(ParquetOutputError):
                    run_capability_analysis(
                        dataset_root=dataset_root,
                        output_dir=output_dir,
                        protocol=protocol,
                        profiles=("observability",),
                        output_format="parquet",
                    )
                return

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=protocol,
                profiles=("observability",),
                output_format="parquet",
            )

            self.assertEqual(certificate["profiles"], ["observability"])
            self.assertTrue(
                (output_dir / "tables_csv" / "observability_profile.csv").exists()
            )
            manifest = json.loads(
                (output_dir / "manifests" / "output_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            observability_record = next(
                record
                for record in manifest["tables"]
                if record["name"] == "observability"
            )
            canonical = observability_record["canonical"]
            self.assertEqual(canonical["format"], "parquet")
            self.assertTrue((output_dir / canonical["path"]).exists())
            self.assertTrue(canonical["path"].startswith("tables_parquet/"))

    def test_run_parquet_output_allows_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profiles=("observability",),
                output_format="parquet",
                allow_parquet_fallback=True,
            )

            self.assertEqual(certificate["profiles"], ["observability"])
            self.assertTrue((output_dir / "observability_profile.csv").exists())
            self.assertTrue(
                (output_dir / "tables_csv" / "observability_profile.csv").exists()
            )
            manifest = json.loads(
                (output_dir / "manifests" / "output_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["allow_parquet_fallback"])
            observability_record = next(
                record
                for record in manifest["tables"]
                if record["name"] == "observability"
            )
            canonical = observability_record["canonical"]
            self.assertIn(canonical["format"], {"parquet", "csv.gz"})
            self.assertTrue((output_dir / canonical["path"]).exists())
            if canonical["format"] == "parquet":
                self.assertTrue(canonical["path"].startswith("tables_parquet/"))
            else:
                self.assertTrue(canonical["path"].startswith("tables_csv/"))
                self.assertIn(
                    "parquet_fallback",
                    {warning["code"] for warning in manifest["warnings"]},
                )

    def test_run_can_write_model_zoo_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="model_zoo",
            )

            self.assertEqual(certificate["profiles"], ["model_zoo"])
            for filename in (
                "model_zoo_frontier.csv",
                "model_zoo_event_scores.csv",
                "model_zoo_model_manifest.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            frontier = pd.read_csv(output_dir / "model_zoo_frontier.csv")
            self.assertIn("model_id", frontier.columns)
            self.assertIn("scan_statistic", frontier.columns)
            self.assertIn("scan_level_p_value", frontier.columns)
            self.assertIn("model_metadata_hash", frontier.columns)
            self.assertFalse(frontier.empty)
            self.assertTrue(np.isfinite(frontier["scan_statistic"]).all())

    def test_run_can_write_law_observability_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 2
            TSDatasetGenerator.from_dict(config).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=8,
                ),
                profile_preset="law_observability",
            )

            self.assertEqual(certificate["profiles"], ["law_observability"])
            for filename in (
                "law_observability_profile.csv",
                "law_observability_summary.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            profile = pd.read_csv(output_dir / "law_observability_profile.csv")
            self.assertIn("energy_distance", profile.columns)
            self.assertIn("c2st_balanced_accuracy", profile.columns)
            self.assertIn("law_observability_status", profile.columns)
            self.assertFalse(profile.empty)

    def test_run_can_write_corrected_detectability_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="calibration",
            )

            self.assertEqual(
                certificate["profiles"], ["detectability", "corrected_detectability"]
            )
            for filename in (
                "corrected_detectability_frontier.csv",
                "oracle_window_diagnostic_frontier.csv",
                "blind_scan_events.csv",
                "calibration_resolution.csv",
                "candidate_nulls_manifest.json",
                "scan_nulls_manifest.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            corrected = pd.read_csv(output_dir / "corrected_detectability_frontier.csv")
            self.assertIn("candidate_level_p_value", corrected.columns)
            self.assertIn("scan_level_p_value", corrected.columns)
            self.assertIn("scan_statistic", corrected.columns)
            self.assertFalse(corrected.empty)

    def test_run_can_write_implementation_validity_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            output_dir = tmp_path / "analysis"
            TSDatasetGenerator.from_dict(_small_generation_config(dataset_root)).run()

            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    delta_grid=(0.20,),
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                profile_preset="implementation_validity",
            )

            self.assertEqual(
                certificate["profiles"],
                [
                    "observability",
                    "detectability",
                    "corrected_detectability",
                    "implementation_validity",
                ],
            )
            for filename in (
                "support_integrity.csv",
                "boundary_audit.csv",
                "shortcut_audit.csv",
                "realized_effects.csv",
                "detector_attribution.csv",
                "negative_controls.csv",
                "implementation_validity.csv",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            validity = pd.read_csv(output_dir / "implementation_validity.csv")
            self.assertIn("implementation_validity_status", validity.columns)
            self.assertFalse(validity.empty)
            attribution = pd.read_csv(output_dir / "detector_attribution.csv")
            self.assertIn("normalized_evidence", attribution.columns)
            self.assertIn("primary_detection_cause", attribution.columns)
            self.assertFalse(attribution.empty)
            controls = pd.read_csv(output_dir / "negative_controls.csv")
            self.assertIn("control_type", controls.columns)
            self.assertIn("control_status", controls.columns)
            self.assertFalse(controls.empty)


if __name__ == "__main__":
    unittest.main()
