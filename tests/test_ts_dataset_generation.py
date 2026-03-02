import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.base_oscillations import BaseOscillation


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestTSDatasetGeneration(unittest.TestCase):
    def _base_config(self, output_root: Path) -> Dict:
        return {
            "generator": {
                "output_root": str(output_root),
                "master_seed": 1234,
                "overwrite_output": True,
                "log_level": "INFO",
                "on_variant_failure": "skip",
            },
            "dataset": {
                "length": 600,
                "channels": 3,
                "splits": ["train", "val"],
                "instances_per_split": 2,
            },
            "anomaly_policy": {
                "density_range": [0.05, 0.06],
                "density_tolerance": 0.005,
                "segment_count_range": [4, 6],
                "placement_policy": "uniform",
                "channel_policy": "single-random",
                "overlap_policy": "global",
                "length_normalization": "resample",
            },
            "variants": {
                "base_oscillations": ["sine"],
                "anomaly_types": ["mean"],
                "disabled_anomaly_types": [],
                "profiles_per_pair": 1,
                "pair_profiles": {},
                "base_parameter_policy": "fixed_per_variant",
                "anomaly_parameter_policy": "fixed_per_variant",
                "skip_base_oscillations": [],
                "skip_anomaly_types": [],
            },
            "plot": {
                "enabled": True,
                "zoom_count": 5,
                "zoom_fill_policy": "repeat",
                "zoom_margin": 32,
                "zoom_margin_min": 16,
                "zoom_margin_alpha": 0.5,
            },
        }

    def test_artifacts_manifest_and_label_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            manifest = TSDatasetGenerator.from_dict(config).run()

            self.assertIn("sine__mean__p00", manifest["generated_variants"])
            self.assertIn("config", manifest)
            self.assertIn("derived_seeds", manifest)
            self.assertIn("aggregated_statistics", manifest)
            variant_dir = output_root / "variants" / "sine__mean__p00"
            self.assertTrue((output_root / "dataset_manifest.json").exists())
            self.assertTrue((variant_dir / "variant_config.yaml").exists())

            for split in ["train", "val"]:
                split_dir = variant_dir / split
                self.assertTrue((split_dir / "split_summary.json").exists())
                for instance_idx in range(2):
                    instance_dir = (
                        split_dir / "instances" / f"instance_{instance_idx:03d}"
                    )
                    self.assertTrue((instance_dir / "clean.csv").exists())
                    self.assertTrue((instance_dir / "anomalous.csv").exists())
                    self.assertTrue((instance_dir / "labels_pointwise.csv").exists())
                    self.assertTrue((instance_dir / "events.json").exists())
                    self.assertTrue((instance_dir / "instance_summary.json").exists())
                    self.assertTrue((instance_dir / "plot_full.png").exists())
                    self.assertFalse((instance_dir / "plot.png").exists())
                    for zoom_idx in range(5):
                        self.assertTrue(
                            (instance_dir / f"zoom_{zoom_idx:02d}.png").exists()
                        )

                    labels = pd.read_csv(
                        instance_dir / "labels_pointwise.csv"
                    ).to_numpy()
                    self.assertEqual(labels.shape, (600, 3))
                    with (instance_dir / "events.json").open(
                        "r", encoding="utf-8"
                    ) as f:
                        events = json.load(f)

                    reconstructed = np.zeros((600, 3), dtype=np.int8)
                    for event in events:
                        reconstructed[
                            int(event["start"]) : int(event["end"]),
                            int(event["channel"]),
                        ] = 1
                        self.assertIn("params", event)
                        self.assertIn("length", event)
                        self.assertEqual(
                            int(event["length"]),
                            int(event["end"]) - int(event["start"]),
                        )
                    np.testing.assert_array_equal(labels, reconstructed)

                    with (instance_dir / "instance_summary.json").open(
                        "r", encoding="utf-8"
                    ) as f:
                        summary = json.load(f)
                    self.assertEqual(summary["variant_id"], "sine__mean__p00")
                    self.assertEqual(summary["profile_id"], "p00")
                    self.assertEqual(summary["overlap_policy"], "global")
                    self.assertEqual(
                        summary["base_parameter_policy"], "fixed_per_variant"
                    )
                    self.assertEqual(
                        summary["anomaly_parameter_policy"], "fixed_per_variant"
                    )

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
            manifest = TSDatasetGenerator.from_dict(config).run()

            self.assertEqual(manifest["generated_variants"], [])
            skipped_ids = [item["variant_id"] for item in manifest["skipped_variants"]]
            self.assertIn("sine__extremum__p00", skipped_ids)

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

    def test_distribution_primitives_and_extremum_point_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 600
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.005
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "extremum": {
                    "planner": "point_events_from_density",
                    "density_range": [0.05, 0.10],
                    "unique_timestamps": True,
                },
            }
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_types"] = [
                "mean",
                "pattern-shift",
                "extremum",
            ]
            config["variants"]["disabled_anomaly_types"] = []
            config["plot"]["enabled"] = False
            config["variants"]["anomaly_overrides"] = {
                "mean": {
                    "offset": {
                        "distribution": "uniform",
                        "low": -1.5,
                        "high": 1.5,
                        "reject_if_abs_lt": 0.3,
                    }
                },
                "pattern-shift": {
                    "transition_window": {
                        "distribution": "int_uniform",
                        "low": 5,
                        "high": 25,
                    },
                    "shift_by": {"distribution": "int_uniform", "low": -25, "high": 25},
                },
                "extremum": {
                    "min": {"distribution": "bernoulli", "p": 0.5},
                    "local": {"distribution": "bernoulli", "p": 0.5},
                    "context_window": {
                        "distribution": "int_uniform",
                        "low": 20,
                        "high": 200,
                    },
                },
            }
            manifest = TSDatasetGenerator.from_dict(config).run()

            generated = set(manifest["generated_variants"])
            self.assertIn("sine__mean__p00", generated)
            self.assertIn("sine__pattern-shift__p00", generated)
            self.assertIn("sine__extremum__p00", generated)

            mean_events_path = (
                output_root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
                / "events.json"
            )
            with mean_events_path.open("r", encoding="utf-8") as f:
                mean_events = json.load(f)
            self.assertGreater(len(mean_events), 0)
            self.assertTrue(
                all(
                    abs(float(event["params"]["offset"])) >= 0.3
                    for event in mean_events
                )
            )

            ps_events_path = (
                output_root
                / "variants"
                / "sine__pattern-shift__p00"
                / "train"
                / "instances"
                / "instance_000"
                / "events.json"
            )
            with ps_events_path.open("r", encoding="utf-8") as f:
                ps_events = json.load(f)
            self.assertGreater(len(ps_events), 0)
            for event in ps_events:
                transition_window = int(event["params"]["transition_window"])
                shift_by = int(event["params"]["shift_by"])
                self.assertGreaterEqual(transition_window, 1)
                self.assertLessEqual(abs(shift_by), transition_window)

            extremum_events_path = (
                output_root
                / "variants"
                / "sine__extremum__p00"
                / "train"
                / "instances"
                / "instance_000"
                / "events.json"
            )
            with extremum_events_path.open("r", encoding="utf-8") as f:
                extremum_events = json.load(f)
            self.assertGreater(len(extremum_events), 0)
            self.assertTrue(all(int(event["length"]) == 1 for event in extremum_events))
            self.assertEqual(
                len({int(event["start"]) for event in extremum_events}),
                len(extremum_events),
            )

    def test_trend_random_walk_template_handles_short_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 300
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.10]
            config["anomaly_policy"]["density_tolerance"] = 0.05
            config["anomaly_policy"]["segment_count_range"] = [20, 22]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_types"] = ["trend"]
            config["plot"]["enabled"] = False
            config["variants"]["anomaly_overrides"] = {
                "trend": {
                    "oscillation": {
                        "distribution": "choice",
                        "values": [
                            {
                                "kind": "random-walk",
                                "smoothing": 0.005,
                                "amplitude": 0.5,
                            }
                        ],
                    }
                }
            }

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__trend__p00", manifest["generated_variants"])

    def test_frequency_period_locked_profile_generates_period_aligned_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 1200
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.01
            config["anomaly_policy"]["segment_count_range"] = [2, 6]
            config["variants"]["base_oscillations"] = ["sine"]
            config["variants"]["anomaly_types"] = ["frequency"]
            config["variants"]["profiles_per_pair"] = 1
            config["variants"]["pair_profiles"] = {"sine__frequency": ["p01"]}
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {"frequency": 5.0}
            }
            config["variants"]["anomaly_overrides"] = {"frequency": {"frequency_factor": 1.0}}
            config["variants"]["variant_overrides"] = {
                "sine__frequency": {
                    "anomaly_policy": {
                        "segment_count_range": [2, 6],
                        "segment_planner": {
                            "planner": "period_locked_frequency",
                            "periods_per_segment_range": [2, 4],
                            "align_to_period_start": True,
                            "period_ratio_offsets": [-1, 1, 2],
                            "frequency_factor_bounds": [0.7, 1.6],
                        },
                    }
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__frequency__p01", manifest["generated_variants"])

            events_path = (
                output_root
                / "variants"
                / "sine__frequency__p01"
                / "train"
                / "instances"
                / "instance_000"
                / "events.json"
            )
            with events_path.open("r", encoding="utf-8") as f:
                events = json.load(f)
            self.assertGreater(len(events), 0)
            instance_dir = (
                output_root
                / "variants"
                / "sine__frequency__p01"
                / "train"
                / "instances"
                / "instance_000"
            )

            period_size = (
                instance_dir / "instance_summary.json"
            )
            with period_size.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            base_frequency = float(summary["base_parameters"]["frequency"])
            period = int(100 / base_frequency)
            self.assertGreater(period, 1)

            for event in events:
                start = int(event["start"])
                length = int(event["length"])
                factor = float(event["params"]["frequency_factor"])
                self.assertEqual(start % period, 0)
                self.assertEqual(length % period, 0)
                periods = length // period
                m = int(round(factor * periods))
                self.assertNotEqual(m, periods)
                self.assertAlmostEqual(factor, m / periods, places=8)

            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            for event in events:
                start = int(event["start"])
                end = int(event["end"])
                channel = int(event["channel"])
                self.assertAlmostEqual(
                    float(anomalous[start, channel]),
                    float(clean[start, channel]),
                    places=8,
                )
                self.assertAlmostEqual(
                    float(anomalous[end - 1, channel]),
                    float(clean[end - 1, channel]),
                    places=8,
                )


if __name__ == "__main__":
    unittest.main()
