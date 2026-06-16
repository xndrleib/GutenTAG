import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis
from gutenTAG.tsgen.capabilities.calibration import calibration_status
from gutenTAG.tsgen.capabilities.protocol import CapabilityProtocol as ProtocolModel
from gutenTAG.tsgen.capabilities.protocol import capability_run_config_from_yaml
from gutenTAG.tsgen.capabilities.protocol import protocol_from_yaml
from gutenTAG.tsgen.capabilities.protocol import window_length_bin


class TestCapabilityLayerV1(unittest.TestCase):
    def _config(self, output_root: Path) -> dict:
        return {
            "generator": {
                "output_root": str(output_root),
                "master_seed": 9001,
                "overwrite_output": True,
                "log_level": "WARNING",
                "on_variant_failure": "skip",
            },
            "dataset": {
                "length": 240,
                "channels": 3,
                "splits": ["train"],
                "instances_per_split": 2,
            },
            "anomaly_policy": {
                "density_range": [0.05, 0.08],
                "density_tolerance": 0.05,
                "segment_count_range": [2, 3],
                "placement_policy": "uniform",
                "channel_policy": "single-random",
                "overlap_policy": "global",
                "length_normalization": "resample",
            },
            "variants": {
                "base_oscillations": ["sine"],
                "anomaly_types": ["mean", "variance"],
                "profiles_per_pair": 1,
                "pair_profiles": {},
                "base_parameter_policy": "random_per_instance",
                "base_channel_parameter_policy": "random_per_instance",
                "anomaly_parameter_policy": "fixed_per_variant",
                "skip_base_oscillations": [],
                "skip_anomaly_types": [],
                "disabled_anomaly_types": [],
            },
            "plot": {"enabled": False},
        }

    def test_capability_analysis_writes_theory_aligned_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            output_dir = Path(tmp) / "capability"
            TSDatasetGenerator.from_dict(self._config(dataset_root)).run()
            certificate = run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=output_dir,
                protocol=CapabilityProtocol(
                    alpha_grid=(0.10, 0.05),
                    delta_grid=(0.20, 0.50),
                    max_scan_windows_per_length=32,
                    bootstrap_samples=0,
                ),
            )

            self.assertEqual(
                certificate["capability_certificate_version"], "synthgen.capability.v1"
            )
            self.assertGreater(certificate["summary"]["instance_count"], 0)
            self.assertGreater(certificate["summary"]["event_group_count"], 0)
            self.assertIn("protocol_hash", certificate)

            expected_files = [
                "observability_profile.csv",
                "arity_profile.csv",
                "event_capability_summary.csv",
                "detectability_frontier.csv",
                "identifiability_summary.csv",
                "description_profile.csv",
                "capability_certificate.json",
                "capability_report.md",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)

            observability = pd.read_csv(output_dir / "observability_profile.csv")
            arity = pd.read_csv(output_dir / "arity_profile.csv")
            frontier = pd.read_csv(output_dir / "detectability_frontier.csv")
            descriptions = pd.read_csv(output_dir / "description_profile.csv")
            self.assertFalse(observability.empty)
            self.assertFalse(arity.empty)
            self.assertFalse(frontier.empty)
            self.assertFalse(descriptions.empty)
            self.assertTrue((observability["distance_value"] >= 0).all())
            self.assertTrue(set(frontier["alpha"]).issubset({0.10, 0.05}))
            self.assertTrue(
                (
                    (descriptions["witness_sufficiency"] >= 0)
                    & (descriptions["witness_sufficiency"] <= 1.0 + 1e-9)
                ).all()
            )

            parsed = json.loads(
                (output_dir / "capability_certificate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parsed["summary"], certificate["summary"])

    def test_capability_protocol_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            ProtocolModel.from_mapping({"alpha_grid": [0.05], "silent_typo": True})

    def test_protocol_yaml_separates_protocol_and_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.yaml"
            path.write_text(
                "\n".join(
                    [
                        "profile_preset: full_certificate",
                        "profiles:",
                        "  - admission",
                        "alpha_grid: [0.1, 0.05]",
                        "output:",
                        "  internal_format: parquet",
                        "  release_csv: false",
                        "  allow_parquet_fallback: true",
                        "label_export: diagnostics",
                    ]
                ),
                encoding="utf-8",
            )

            protocol = protocol_from_yaml(str(path))
            run_config = capability_run_config_from_yaml(str(path))

            self.assertEqual(protocol.alpha_grid, (0.1, 0.05))
            self.assertEqual(run_config.protocol.alpha_grid, (0.1, 0.05))
            self.assertEqual(run_config.profiles, ("admission",))
            self.assertEqual(run_config.profile_preset, "full_certificate")
            self.assertEqual(run_config.output_format, "parquet")
            self.assertFalse(run_config.release_csv)
            self.assertTrue(run_config.allow_parquet_fallback)
            self.assertEqual(run_config.label_export, "diagnostics")

    def test_protocol_yaml_can_override_calibration_min_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.yaml"
            path.write_text(
                "\n".join(
                    [
                        "alpha_grid: [0.1]",
                        "calibration:",
                        "  split: calibration",
                        "  min_clean_scan_count_for_alpha:",
                        "    0.10: 2",
                        "    0.05: 3",
                        "legacy_detectability:",
                        "  max_clean_instances: 7",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            protocol = protocol_from_yaml(str(path))

            self.assertEqual(
                protocol.calibration_min_clean_scan_count_for_alpha,
                ((0.05, 3), (0.1, 2)),
            )
            self.assertEqual(protocol.calibration_split, "calibration")
            self.assertEqual(protocol.legacy_detectability_max_clean_instances, 7)
            self.assertEqual(
                calibration_status(
                    10,
                    0.1,
                    protocol.calibration_min_clean_scan_count_for_alpha,
                ),
                "calibration_ok",
            )

    def test_protocol_yaml_can_enable_window_length_binning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.yaml"
            path.write_text(
                "\n".join(
                    [
                        "window_length_policy:",
                        "  mode: binned",
                        "  bins: [8, 16, 32]",
                    ]
                ),
                encoding="utf-8",
            )

            protocol = protocol_from_yaml(str(path))

            self.assertEqual(protocol.window_length_policy_mode, "binned")
            self.assertEqual(protocol.window_length_bins, (8, 16, 32))
            self.assertEqual(window_length_bin(9, protocol), 16)
            self.assertEqual(window_length_bin(33, protocol), 33)


if __name__ == "__main__":
    unittest.main()
