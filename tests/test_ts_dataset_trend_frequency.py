import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator

from tests.ts_dataset_generation_fixtures import (
    TSDatasetGenerationConfigMixin,
)


def _configure_period_locked_frequency_case(config: dict[str, Any]) -> None:
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
    config["variants"]["base_oscillation_overrides"] = {"sine": {"frequency": 5.0}}
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


def _frequency_instance_dir(output_root: Path) -> Path:
    return (
        output_root
        / "variants"
        / "sine__frequency__p01"
        / "train"
        / "instances"
        / "instance_000"
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_frequency_period_alignment(
    testcase: unittest.TestCase,
    events: list[dict[str, Any]],
    period: int,
) -> None:
    testcase.assertGreater(period, 1)
    for event in events:
        start = int(event["start"])
        length = int(event["length"])
        factor = float(event["params"]["frequency_factor"])
        testcase.assertEqual(start % period, 0)
        testcase.assertEqual(length % period, 0)
        periods = length // period
        ratio_numerator = int(round(factor * periods))
        testcase.assertNotEqual(ratio_numerator, periods)
        testcase.assertAlmostEqual(factor, ratio_numerator / periods, places=8)


def _assert_frequency_event_edges_are_continuous(
    testcase: unittest.TestCase,
    instance_dir: Path,
    events: list[dict[str, Any]],
) -> None:
    clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
    anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(dtype=np.float64)
    for event in events:
        start = int(event["start"])
        end = int(event["end"])
        channel = int(event["channel"])
        testcase.assertAlmostEqual(
            float(anomalous[start, channel]),
            float(clean[start, channel]),
            places=8,
        )
        testcase.assertAlmostEqual(
            float(anomalous[end - 1, channel]),
            float(clean[end - 1, channel]),
            places=8,
        )


class TestTSDatasetGenerationProfiles(
    TSDatasetGenerationConfigMixin, unittest.TestCase
):
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
            self.assertNotIn(
                "sine__trend__p00",
                [item["variant_id"] for item in manifest["skipped_variants"]],
            )

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
                abs(
                    float(summary["achieved_density"])
                    - float(summary["target_density"])
                ),
                float(summary["density_tolerance"]) + 1e-12,
            )
            self.assertGreaterEqual(int(summary["n_segments"]), 1)

    def test_frequency_period_locked_profile_generates_period_aligned_segments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            _configure_period_locked_frequency_case(config)

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__frequency__p01", manifest["generated_variants"])

            instance_dir = _frequency_instance_dir(output_root)
            events = _load_json(instance_dir / "events.json")
            self.assertGreater(len(events), 0)

            summary = _load_json(instance_dir / "instance_summary.json")
            base_frequency = float(summary["base_parameters"]["frequency"])
            period = int(100 / base_frequency)
            _assert_frequency_period_alignment(self, events, period)
            _assert_frequency_event_edges_are_continuous(self, instance_dir, events)

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

    def test_frequency_ecg_variant_override_enforces_long_slowdown_segments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 1600
            config["dataset"]["channels"] = 3
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.01
            config["anomaly_policy"]["segment_count_range"] = [8, 14]
            config["variants"]["base_oscillations"] = ["ecg"]
            config["variants"]["anomaly_types"] = ["frequency"]
            config["variants"]["profiles_per_pair"] = 1
            config["variants"]["pair_profiles"] = {"ecg__frequency": ["p01"]}
            config["variants"]["anomaly_parameter_policy"] = "random_per_segment"
            config["variants"]["base_oscillation_overrides"] = {
                "ecg": {"frequency": 8.0, "amplitude": 1.0, "variance": 0.02}
            }
            config["variants"]["variant_overrides"] = {
                "ecg__frequency": {
                    "anomaly_policy": {
                        "segment_count_range": [6, 10],
                        "min_segment_length": 72,
                        "segment_planner": {
                            "planner": "period_locked_frequency",
                            "min_segment_length": 72,
                            "periods_per_segment_range": [6, 8],
                            "align_to_period_start": True,
                            "period_ratio_offsets": [-2, -1],
                            "frequency_factor_bounds": [0.65, 0.95],
                        },
                    }
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("ecg__frequency__p01", manifest["generated_variants"])

            instance_dir = (
                output_root
                / "variants"
                / "ecg__frequency__p01"
                / "train"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            for event in events:
                self.assertGreaterEqual(int(event["length"]), 72)
                factor = float(event["params"]["frequency_factor"])
                self.assertGreaterEqual(factor, 0.65 - 1e-9)
                self.assertLessEqual(factor, 0.95 + 1e-9)

            with (instance_dir / "instance_summary.json").open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            self.assertGreaterEqual(int(summary["segment_length_min"]), 72)
            self.assertEqual(
                summary["variant_anomaly_policy"]["segment_planner"][
                    "frequency_factor_bounds"
                ],
                [0.65, 0.95],
            )

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
                            {
                                "kind": "random-walk",
                                "smoothing": 0.01,
                                "amplitude": 0.6,
                            },
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
            labels = pd.read_csv(instance_dir / "labels_pointwise.csv").to_numpy(
                dtype=np.int8
            )
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
                delta = np.abs(
                    anomalous[start:end, channel] - clean[start:end, channel]
                )
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


if __name__ == "__main__":
    unittest.main()
