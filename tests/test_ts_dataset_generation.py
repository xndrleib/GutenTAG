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

    def test_trend_parameter_aware_planner_enforces_min_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 3000
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.07]
            config["anomaly_policy"]["density_tolerance"] = 0.03
            config["anomaly_policy"]["segment_count_range"] = [8, 10]
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "trend": {
                    "planner": "trend_parameter_aware_segments",
                    "sine_min_cycles": 0.30,
                    "random_walk_min_segment_length": 8,
                    "segment_count_range": [8, 10],
                },
            }
            config["variants"]["anomaly_types"] = ["trend"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "trend": {
                    "transition_length": 0,
                    "boundary_mode": "inside_window_zero_endpoints",
                    "envelope_kind": "sine2",
                    "oscillation": {
                        "distribution": "choice",
                        "values": [
                            {
                                "kind": "sine",
                                "frequency": {
                                    "distribution": "uniform",
                                    "low": 1.2,
                                    "high": 2.0,
                                },
                                "amplitude": {
                                    "distribution": "uniform",
                                    "low": 0.4,
                                    "high": 1.0,
                                },
                            }
                        ],
                    },
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__trend__p00", manifest["generated_variants"])
            events_path = (
                output_root
                / "variants"
                / "sine__trend__p00"
                / "train"
                / "instances"
                / "instance_000"
                / "events.json"
            )
            with events_path.open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                source_length = int(event["source_end"]) - int(event["source_start"])
                params = event.get("params", {})
                oscillation = params.get("oscillation", {})
                self.assertEqual(str(oscillation.get("kind")), "sine")
                frequency = float(oscillation.get("frequency"))
                min_required = int(np.ceil((100.0 * 0.30) / frequency))
                self.assertGreaterEqual(source_length, min_required)

    def test_trend_parameter_aware_planner_preserves_target_density(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 2000
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.05]
            config["anomaly_policy"]["density_tolerance"] = 0.01
            config["anomaly_policy"]["segment_count_range"] = [12, 12]
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "trend": {
                    "planner": "trend_parameter_aware_segments",
                    "sine_min_cycles": 0.30,
                    "random_walk_min_segment_length": 8,
                    "segment_count_range": [12, 12],
                },
            }
            config["variants"]["anomaly_types"] = ["trend"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "trend": {
                    "transition_length": 0,
                    "boundary_mode": "inside_window_zero_endpoints",
                    "envelope_kind": "sine2",
                    "oscillation": {
                        "distribution": "choice",
                        "values": [
                            {
                                "kind": "sine",
                                "frequency": 0.3,
                                "amplitude": 0.8,
                            }
                        ],
                    },
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__trend__p00", manifest["generated_variants"])
            self.assertNotIn("sine__trend__p00", [item["variant_id"] for item in manifest["skipped_variants"]])

            summary_path = (
                output_root
                / "variants"
                / "sine__trend__p00"
                / "train"
                / "instances"
                / "instance_000"
                / "instance_summary.json"
            )
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertLessEqual(
                abs(float(summary["achieved_density"]) - float(summary["target_density"])),
                float(summary["density_tolerance"]) + 1e-12,
            )
            self.assertGreaterEqual(int(summary["n_segments"]), 1)

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

    def test_frequency_variant_specific_density_tolerance_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 1200
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.10]
            config["anomaly_policy"]["density_tolerance"] = 1e-6
            config["anomaly_policy"]["segment_count_range"] = [2, 6]
            config["variants"]["base_oscillations"] = ["sine"]
            config["variants"]["anomaly_types"] = ["frequency"]
            config["variants"]["profiles_per_pair"] = 1
            config["variants"]["pair_profiles"] = {"sine__frequency": ["p01"]}
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {"frequency": 5.0}
            }
            config["variants"]["anomaly_overrides"] = {
                "frequency": {"frequency_factor": 1.0}
            }
            config["variants"]["variant_overrides"] = {
                "sine__frequency": {
                    "anomaly_policy": {
                        "density_tolerance": 0.02,
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
            self.assertEqual(len(manifest["skipped_variants"]), 0)

    def test_random_per_instance_base_parameters_are_split_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["splits"] = ["train", "val", "test"]
            config["dataset"]["instances_per_split"] = 2
            config["variants"]["base_parameter_policy"] = "random_per_instance"
            config["variants"]["base_channel_parameter_policy"] = "random_per_instance"
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {
                    "frequency": {"distribution": "uniform", "low": 7.5, "high": 8.5},
                    "amplitude": {"distribution": "uniform", "low": 0.8, "high": 1.2},
                    "variance": {"distribution": "uniform", "low": 0.02, "high": 0.05},
                }
            }
            config["variants"]["base_channel_overrides"] = {
                "sine": {
                    "phase": {
                        "distribution": "uniform",
                        "low": 0.0,
                        "high": float(2.0 * np.pi),
                    }
                }
            }
            config["plot"]["enabled"] = False
            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])

            def load_summary(split: str, index: int) -> Dict:
                path = (
                    output_root
                    / "variants"
                    / "sine__mean__p00"
                    / split
                    / "instances"
                    / f"instance_{index:03d}"
                    / "instance_summary.json"
                )
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            train_0 = load_summary("train", 0)
            val_0 = load_summary("val", 0)
            test_0 = load_summary("test", 0)
            self.assertEqual(train_0["base_parameters"], val_0["base_parameters"])
            self.assertEqual(train_0["base_parameters"], test_0["base_parameters"])
            self.assertEqual(
                train_0["base_parameters_per_channel"],
                val_0["base_parameters_per_channel"],
            )
            self.assertEqual(
                train_0["base_parameters_per_channel"],
                test_0["base_parameters_per_channel"],
            )

            train_1 = load_summary("train", 1)
            self.assertNotEqual(train_0["base_parameters"], train_1["base_parameters"])

    def test_split_phase_shift_applies_deterministic_offsets_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["splits"] = ["train", "val", "test"]
            config["dataset"]["instances_per_split"] = 1
            config["variants"]["base_parameter_policy"] = "random_per_instance"
            config["variants"]["base_channel_parameter_policy"] = "random_per_instance"
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {"frequency": 8.0, "amplitude": 1.0, "variance": 0.03}
            }
            config["variants"]["base_channel_overrides"] = {
                "sine": {"phase": 0.5, "amplitude": 1.0, "offset": 0.0, "variance": 0.03}
            }
            config["variants"]["split_phase_shift"] = {
                "enabled": True,
                "mode": "fixed_map",
                "phase_modulo": float(2.0 * np.pi),
                "values": {
                    "train": 0.0,
                    "val": float(2.0 * np.pi / 3.0),
                    "test": float(4.0 * np.pi / 3.0),
                },
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])

            def load_summary(split: str) -> Dict:
                path = (
                    output_root
                    / "variants"
                    / "sine__mean__p00"
                    / split
                    / "instances"
                    / "instance_000"
                    / "instance_summary.json"
                )
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

            train = load_summary("train")
            val = load_summary("val")
            test = load_summary("test")

            train_phase = float(train["base_parameters_per_channel"][0]["phase"])
            val_phase = float(val["base_parameters_per_channel"][0]["phase"])
            test_phase = float(test["base_parameters_per_channel"][0]["phase"])
            modulo = float(2.0 * np.pi)

            self.assertAlmostEqual(train_phase, 0.5, places=8)
            self.assertAlmostEqual(
                val_phase,
                float(np.mod(train_phase + (2.0 * np.pi / 3.0), modulo)),
                places=8,
            )
            self.assertAlmostEqual(
                test_phase,
                float(np.mod(train_phase + (4.0 * np.pi / 3.0), modulo)),
                places=8,
            )

            self.assertFalse(bool(train["split_phase_shift"]["phase_shift_applied"]))
            self.assertTrue(bool(val["split_phase_shift"]["phase_shift_applied"]))
            self.assertTrue(bool(test["split_phase_shift"]["phase_shift_applied"]))

    def test_split_phase_shift_requires_values_for_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["splits"] = ["train", "val", "test"]
            config["variants"]["split_phase_shift"] = {
                "enabled": True,
                "mode": "fixed_map",
                "phase_modulo": float(2.0 * np.pi),
                "values": {"train": 0.0, "val": float(np.pi)},
            }
            with self.assertRaises(ValueError):
                TSDatasetGenerator.from_dict(config)

    def test_shared_noise_correlation_increases_pairwise_noise_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["variants"]["base_channel_correlation"] = {"shared_noise_weight": 0.8}
            generator = TSDatasetGenerator.from_dict(config)

            class DummyBO:
                def __init__(self, noise: np.ndarray) -> None:
                    self.noise = noise

            rng = np.random.default_rng(777)
            bos = [DummyBO(rng.normal(0.0, 1.0, 4096)) for _ in range(4)]

            def mean_pairwise_corr(items) -> float:
                values = np.vstack([item.noise for item in items])
                corr = np.corrcoef(values)
                mask = ~np.eye(corr.shape[0], dtype=bool)
                return float(np.mean(corr[mask]))

            before = mean_pairwise_corr(bos)
            generator._apply_shared_noise_correlation(bos, seed=2026)
            after = mean_pairwise_corr(bos)
            self.assertGreater(after, before + 0.2)

    def test_amplitude_transition_length_zero_applies_nonzero_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 600
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.01
            config["anomaly_policy"]["segment_count_range"] = [8, 10]
            config["variants"]["anomaly_types"] = ["amplitude"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["anomaly_overrides"] = {
                "amplitude": {"amplitude_factor": 1.8, "transition_length": 0}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__amplitude__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__amplitude__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)

            for event in events:
                start = int(event["start"])
                end = int(event["end"])
                channel = int(event["channel"])
                delta = np.abs(anomalous[start:end, channel] - clean[start:end, channel])
                self.assertGreater(
                    float(delta.max()),
                    1e-8,
                    msg=f"Expected non-zero amplitude effect for event {event}",
                )

    def test_energy_aware_amplitude_segments_land_in_high_energy_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 600
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [4, 4]
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "amplitude": {
                    "planner": "energy_aware_segments",
                    "energy_metric": "rms",
                    "rms_quantile": 0.80,
                    "weighted_sampling": False,
                    "fallback": "error",
                    "segment_count_range": [4, 4],
                },
            }
            config["variants"]["anomaly_types"] = ["amplitude"]
            config["variants"]["base_oscillation_overrides"] = {"sine": {"frequency": 2.0}}
            config["variants"]["anomaly_overrides"] = {
                "amplitude": {"amplitude_factor": 1.8, "transition_length": 0}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__amplitude__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__amplitude__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)

            channel_values = clean[:, 0]
            sq_prefix = np.concatenate([[0.0], np.cumsum(np.square(channel_values))])
            for event in events:
                start = int(event["source_start"])
                end = int(event["source_end"])
                length = max(1, end - start)
                sums = sq_prefix[length:] - sq_prefix[:-length]
                rms_values = np.sqrt(np.maximum(sums / float(length), 0.0))
                threshold = float(np.quantile(rms_values, 0.80))
                event_rms = float(
                    np.sqrt(
                        max(
                            float(sq_prefix[end] - sq_prefix[start]) / float(length),
                            0.0,
                        )
                    )
                )
                self.assertGreaterEqual(event_rms + 1e-10, threshold)

    def test_bounded_trend_has_no_outside_label_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 700
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.08]
            config["anomaly_policy"]["density_tolerance"] = 0.05
            config["anomaly_policy"]["segment_count_range"] = [6, 8]
            config["variants"]["anomaly_types"] = ["trend"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "trend": {
                    "transition_length": 0,
                    "boundary_mode": "inside_window_zero_endpoints",
                    "envelope_kind": "sine2",
                    "oscillation": {
                        "distribution": "choice",
                        "values": [
                            {"kind": "sine", "frequency": 0.2, "amplitude": 0.8},
                            {"kind": "random-walk", "smoothing": 0.01, "amplitude": 0.6},
                        ],
                    },
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__trend__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__trend__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            labels = pd.read_csv(instance_dir / "labels_pointwise.csv").to_numpy(dtype=np.int8)
            delta = np.abs(anomalous - clean)
            outside_delta = delta[labels == 0]
            self.assertLess(float(np.max(outside_delta)), 1e-8)

    def test_trend_min_effect_delta_is_enforced_without_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 800
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "trend": {
                    "planner": "trend_parameter_aware_segments",
                    "segment_count_range": [5, 6],
                    "min_segment_length": 12,
                    "sine_min_cycles": 0.25,
                    "random_walk_min_segment_length": 12,
                    "adaptive_strength": True,
                    "min_effect_delta": 0.14,
                },
            }
            config["variants"]["anomaly_types"] = ["trend"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "trend": {
                    "transition_length": 0,
                    "boundary_mode": "inside_window_zero_endpoints",
                    "envelope_kind": "sine2",
                    "oscillation": {
                        "kind": "random-walk",
                        "smoothing": 0.02,
                        "amplitude": 0.05,
                    },
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__trend__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__trend__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)

            for event in events:
                start = int(event["source_start"])
                end = int(event["source_end"])
                channel = int(event["channel"])
                delta = np.abs(anomalous[start:end, channel] - clean[start:end, channel])
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.139,
                    msg=f"Trend effect floor was not met for event {event}",
                )

    def test_trend_effective_support_trims_zero_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 700
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.07]
            config["anomaly_policy"]["density_tolerance"] = 0.03
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["anomaly_policy"]["support_label_mode"] = "effective_support"
            config["anomaly_policy"]["support_eps_mode"] = "relative"
            config["anomaly_policy"]["support_eps_value"] = 0.05
            config["variants"]["anomaly_types"] = ["trend"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "trend": {
                    "transition_length": 0,
                    "boundary_mode": "inside_window_zero_endpoints",
                    "envelope_kind": "sine2",
                    "oscillation": {
                        "kind": "random-walk",
                        "smoothing": 0.02,
                        "amplitude": 0.5,
                    },
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__trend__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__trend__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                self.assertGreater(
                    int(event["start"]),
                    int(event["source_start"]),
                    msg=f"Expected left-edge shrink for trend event: {event}",
                )
                self.assertLess(
                    int(event["end"]),
                    int(event["source_end"]),
                    msg=f"Expected right-edge shrink for trend event: {event}",
                )

    def test_pattern_shift_min_effect_delta_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 800
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["anomaly_policy"]["support_label_mode"] = "effective_support"
            config["anomaly_policy"]["support_eps_mode"] = "relative"
            config["anomaly_policy"]["support_eps_value"] = 0.05
            config["variants"]["anomaly_types"] = ["pattern-shift"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "pattern-shift": {
                    "transition_window": 8,
                    "shift_by": 1,
                    "crossfade_mode": "cosine",
                    "min_effect_delta": 0.08,
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__pattern-shift__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__pattern-shift__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)

            for event in events:
                start = int(event["source_start"])
                end = int(event["source_end"])
                channel = int(event["channel"])
                delta = np.abs(anomalous[start:end, channel] - clean[start:end, channel])
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.079,
                    msg=f"Pattern-shift effect floor was not met for event {event}",
                )

    def test_pattern_min_effect_delta_is_enforced_for_ecg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 1200
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["variants"]["base_oscillations"] = ["ecg"]
            config["variants"]["anomaly_types"] = ["pattern"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "pattern": {
                    "sinusoid_k": 10.0,
                    "min_effect_delta": 0.08,
                    "adaptive_blend": True,
                    "blend_strength": 1.0,
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("ecg__pattern__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "ecg__pattern__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)

            for event in events:
                start = int(event["source_start"])
                end = int(event["source_end"])
                channel = int(event["channel"])
                delta = np.abs(anomalous[start:end, channel] - clean[start:end, channel])
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.079,
                    msg=f"Pattern effect floor was not met for ECG event {event}",
                )

    def test_effective_support_enforces_min_non_extremum_label_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 800
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["anomaly_policy"]["support_label_mode"] = "effective_support"
            config["anomaly_policy"]["support_eps_mode"] = "relative"
            config["anomaly_policy"]["support_eps_value"] = 0.05
            config["anomaly_policy"]["min_effective_label_length_non_extremum"] = 2
            config["variants"]["anomaly_types"] = ["pattern-shift"]
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["anomaly_overrides"] = {
                "pattern-shift": {
                    "transition_window": 6,
                    "shift_by": 1,
                    "crossfade_mode": "cosine",
                    "min_effect_delta": 0.08,
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__pattern-shift__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__pattern-shift__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                self.assertGreaterEqual(
                    int(event["length"]),
                    2,
                    msg=f"Expected non-extremum support length >=2: {event}",
                )

    def test_non_extremum_min_label_length_does_not_affect_extremum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 600
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.03
            config["anomaly_policy"]["support_label_mode"] = "effective_support"
            config["anomaly_policy"]["support_eps_mode"] = "relative"
            config["anomaly_policy"]["support_eps_value"] = 0.05
            config["anomaly_policy"]["min_effective_label_length_non_extremum"] = 3
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "extremum": {
                    "planner": "point_events_from_density",
                    "density_range": [0.05, 0.06],
                    "unique_timestamps": True,
                },
            }
            config["variants"]["anomaly_types"] = ["extremum"]
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__extremum__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__extremum__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                self.assertEqual(
                    int(event["length"]),
                    1,
                    msg=f"Expected extremum to remain point-wise: {event}",
                )

    def test_platform_min_effect_delta_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 800
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["variants"]["base_oscillations"] = ["square"]
            config["variants"]["anomaly_types"] = ["platform"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["anomaly_overrides"] = {
                "platform": {"value": 0.0, "min_effect_delta": 0.2}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("square__platform__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "square__platform__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                start = int(event["source_start"])
                end = int(event["source_end"])
                channel = int(event["channel"])
                delta = np.abs(anomalous[start:end, channel] - clean[start:end, channel])
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.199,
                    msg=f"Platform effect floor was not met for event {event}",
                )

    def test_variance_min_effect_delta_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 900
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [5, 6]
            config["anomaly_policy"]["min_segment_length_by_anomaly"] = {
                "variance": 8
            }
            config["variants"]["base_oscillations"] = ["cosine"]
            config["variants"]["anomaly_types"] = ["variance"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["anomaly_overrides"] = {
                "variance": {"variance": 0.05, "min_effect_delta": 0.12}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("cosine__variance__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "cosine__variance__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
            anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                dtype=np.float64
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                start = int(event["source_start"])
                end = int(event["source_end"])
                channel = int(event["channel"])
                delta = np.abs(anomalous[start:end, channel] - clean[start:end, channel])
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.119,
                    msg=f"Variance effect floor was not met for event {event}",
                )


if __name__ == "__main__":
    unittest.main()
