import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis

from tests.capability_execution_fixtures import (
    _release_csv_paths,
    _small_generation_config,
    _sorted_frame,
)


class TestCapabilityExecutionProfiles(unittest.TestCase):
    def test_law_observability_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 4
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=8,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="law_observability",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="law_observability",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "law_observability_profile.csv",
                "law_observability_summary.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            metadata_paths = list(
                (
                    parallel_cache / "profile_partitions" / "law_observability_profile"
                ).glob("*.metadata.json")
            )
            self.assertGreaterEqual(len(metadata_paths), 1)

    def test_corrected_detectability_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 3
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="calibration",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="calibration",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "corrected_detectability_frontier.csv",
                "blind_scan_events.csv",
                "calibration_resolution.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            metadata_paths = list(
                (parallel_cache / "profile_partitions" / "blind_scan_events").glob(
                    "*.metadata.json"
                )
            )
            self.assertGreaterEqual(len(metadata_paths), 1)
            corrected_metadata_paths = list(
                (
                    parallel_cache / "profile_partitions" / "corrected_detectability"
                ).glob("*.metadata.json")
            )
            self.assertGreaterEqual(len(corrected_metadata_paths), 1)

    def test_implementation_validity_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 9
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="implementation_validity",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="implementation_validity",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "detector_attribution.csv",
                "negative_controls.csv",
                "implementation_validity.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            for profile_name in ("detector_attribution", "negative_controls"):
                metadata_paths = list(
                    (parallel_cache / "profile_partitions" / profile_name).glob(
                        "*.metadata.json"
                    )
                )
                self.assertGreaterEqual(len(metadata_paths), 1, profile_name)

    def test_annotation_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 9
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="annotation",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="annotation",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in (
                "labels/labels_oracle_any.csv",
                "labels/labels_weak_point.csv",
                "labels/labels_censored.csv",
                "annotation_alignment.csv",
                "annotation_robustness.csv",
            ):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            for profile_name in (
                "labels_oracle_any",
                "labels_weak_point",
                "labels_censored",
                "annotation_alignment",
            ):
                metadata_paths = list(
                    (parallel_cache / "profile_partitions" / profile_name).glob(
                        "*.metadata.json"
                    )
                )
                self.assertGreaterEqual(len(metadata_paths), 2, profile_name)

    def test_full_certificate_parallel_matches_serial_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 2
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            serial_certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="full_certificate",
                n_jobs=1,
            )
            parallel_certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="full_certificate",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            self.assertEqual(
                serial_certificate["profiles"], parallel_certificate["profiles"]
            )
            csv_paths = _release_csv_paths(serial_output)
            self.assertGreaterEqual(len(csv_paths), 30)
            self.assertEqual(csv_paths, _release_csv_paths(parallel_output))
            for relative_path in csv_paths:
                serial = pd.read_csv(serial_output / relative_path)
                parallel = pd.read_csv(parallel_output / relative_path)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            for profile_name in (
                "blind_scan_events",
                "detector_attribution",
                "law_observability_profile",
                "model_zoo_frontier",
                "negative_controls",
                "annotation_alignment",
                "labels_oracle_any",
            ):
                metadata_paths = list(
                    (parallel_cache / "profile_partitions" / profile_name).glob(
                        "*.metadata.json"
                    )
                )
                self.assertGreaterEqual(len(metadata_paths), 1, profile_name)


if __name__ == "__main__":
    unittest.main()
