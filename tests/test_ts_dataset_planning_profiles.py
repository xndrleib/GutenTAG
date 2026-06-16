import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.planning import (
    SegmentPlanSamplingConfig,
    sample_energy_aware_segments,
    sample_segment_plan,
)
from gutenTAG.tsgen.seeding import derive_seed

from tests.ts_dataset_generation_fixtures import (
    TSDatasetGenerationConfigMixin,
    _realize_parameters,
    _sanitize_parameters,
)


def _configure_distribution_primitives_case(config: dict[str, Any]) -> None:
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


def _load_profile_events(output_root: Path, variant_name: str) -> list[dict[str, Any]]:
    events_path = (
        output_root
        / "variants"
        / variant_name
        / "train"
        / "instances"
        / "instance_000"
        / "events.json"
    )
    with events_path.open("r", encoding="utf-8") as handle:
        return cast(list[dict[str, Any]], json.load(handle))


def _assert_distribution_variants(
    testcase: unittest.TestCase,
    manifest: dict[str, Any],
) -> None:
    generated = set(manifest["generated_variants"])
    testcase.assertIn("sine__mean__p00", generated)
    testcase.assertIn("sine__pattern-shift__p00", generated)
    testcase.assertIn("sine__extremum__p00", generated)


def _assert_mean_offsets_respect_rejection(
    testcase: unittest.TestCase,
    events: list[dict[str, Any]],
) -> None:
    testcase.assertGreater(len(events), 0)
    testcase.assertTrue(
        all(abs(float(event["params"]["offset"])) >= 0.3 for event in events)
    )


def _assert_pattern_shift_transition_bounds(
    testcase: unittest.TestCase,
    events: list[dict[str, Any]],
) -> None:
    testcase.assertGreater(len(events), 0)
    for event in events:
        transition_window = int(event["params"]["transition_window"])
        shift_by = int(event["params"]["shift_by"])
        testcase.assertGreaterEqual(transition_window, 1)
        testcase.assertLessEqual(abs(shift_by), transition_window)


def _assert_extremum_point_events(
    testcase: unittest.TestCase,
    events: list[dict[str, Any]],
) -> None:
    testcase.assertGreater(len(events), 0)
    testcase.assertTrue(all(int(event["length"]) == 1 for event in events))
    testcase.assertEqual(
        len({int(event["start"]) for event in events}),
        len(events),
    )


class TestTSDatasetGenerationProfiles(
    TSDatasetGenerationConfigMixin, unittest.TestCase
):
    def test_distribution_primitives_and_extremum_point_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            _configure_distribution_primitives_case(config)
            manifest = TSDatasetGenerator.from_dict(config).run()

            _assert_distribution_variants(self, manifest)
            _assert_mean_offsets_respect_rejection(
                self,
                _load_profile_events(output_root, "sine__mean__p00"),
            )
            _assert_pattern_shift_transition_bounds(
                self,
                _load_profile_events(output_root, "sine__pattern-shift__p00"),
            )
            _assert_extremum_point_events(
                self,
                _load_profile_events(output_root, "sine__extremum__p00"),
            )

    def test_extremum_point_planner_supports_count_based_event_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 600
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.005
            config["anomaly_policy"]["segment_planner"] = {
                "default": {"planner": "uniform_segments"},
                "extremum": {
                    "planner": "point_events_from_density",
                    "segment_count_range": [20, 25],
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
            with (instance_dir / "instance_summary.json").open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

            self.assertGreaterEqual(len(events), 20)
            self.assertLessEqual(len(events), 25)
            self.assertTrue(all(int(event["length"]) == 1 for event in events))
            self.assertAlmostEqual(
                float(summary["target_density"]),
                len(events) / int(config["dataset"]["length"]),
                places=9,
            )
            self.assertAlmostEqual(
                float(summary["achieved_density"]),
                len(events) / int(config["dataset"]["length"]),
                places=9,
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
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {"frequency": 2.0}
            }
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

    def test_amplitude_energy_planner_uses_residual_energy_for_min_effect(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 120
            config["dataset"]["channels"] = 1
            generator = TSDatasetGenerator.from_dict(config)

            clean_values = np.zeros((120, 1), dtype=np.float64)
            clean_values[:60, 0] = 10.0
            clean_values[60:, 0] = np.sin(np.linspace(0.0, 8.0 * np.pi, 60))

            segments = sample_energy_aware_segments(
                rng=np.random.default_rng(42),
                target_density=0.20,
                series_length=120,
                channels=1,
                max_placement_attempts=generator.config.max_placement_attempts,
                overlap_policy="global",
                planner_cfg={
                    "planner": "energy_aware_segments",
                    "energy_metric": "rms",
                    "rms_quantile": 0.75,
                    "weighted_sampling": False,
                    "fallback": "error",
                    "min_effect_delta": 0.12,
                    "min_residual_scale": 1e-6,
                    "segment_count_range": [2, 2],
                },
                anomaly_type="amplitude",
                clean_values=clean_values,
                segment_count_range=(2, 2),
                min_segment_length=12,
            )

            self.assertEqual(len(segments), 2)
            for segment in segments:
                self.assertGreater(float(segment.attrs["window_residual_scale"]), 1e-6)
                self.assertEqual(
                    segment.attrs["energy_reference"], "amplitude_residual"
                )
                self.assertGreater(
                    int(segment.end),
                    60,
                    msg="Planner selected a flat high-level plateau for amplitude.",
                )

    def test_fixed_first_onset_segments_from_split_write_onset_metadata_and_plots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"] = {
                "length": 128,
                "channels": 3,
                "splits": {
                    "ctx010": {"paired_instances_per_variant": 1},
                    "calibration": {"clean_only_instances_per_variant": 1},
                },
            }
            config["anomaly_policy"]["density_tolerance"] = 1e-9
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            config["anomaly_policy"]["segment_planner"] = {
                "default": {
                    "planner": "fixed_first_onset_segments",
                    "length": 16,
                    "alignment_strategy": "exact",
                }
            }
            config["plot"] = {
                "enabled": True,
                "zoom_count": 1,
                "zoom_fill_policy": "blank",
                "zoom_margin": 24,
                "zoom_margin_min": 12,
            }

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__mean__p00"
                / "ctx010"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(int(event["requested_pre_context"]), 10)
            self.assertEqual(str(event["onset_bucket"]), "ctx010")
            self.assertEqual(int(event["actual_source_start"]), 10)
            self.assertEqual(int(event["source_start"]), 10)
            self.assertEqual(str(event["alignment_strategy"]), "exact")
            self.assertEqual(int(event["alignment_error"]), 0)
            self.assertTrue((instance_dir / "plot_full.png").exists())
            self.assertTrue((instance_dir / "zoom_00.png").exists())
            self.assertFalse((instance_dir / "zoom_01.png").exists())

            clean_only_dir = (
                output_root
                / "variants"
                / "sine__mean__p00"
                / "calibration"
                / "instances"
                / "clean_only_000"
            )
            self.assertFalse((clean_only_dir / "zoom_00.png").exists())

    def test_fixed_first_onset_extremum_creates_one_point_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"] = {
                "length": 128,
                "channels": 3,
                "splits": {"ctx003": {"paired_instances_per_variant": 1}},
            }
            config["variants"]["anomaly_types"] = ["extremum"]
            config["anomaly_policy"]["density_tolerance"] = 1e-9
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            config["anomaly_policy"]["segment_planner"] = {
                "default": {
                    "planner": "fixed_first_onset_segments",
                    "length": 16,
                },
                "extremum": {
                    "planner": "fixed_first_onset_segments",
                    "length": 1,
                },
            }
            config["anomaly_policy"]["special_anomaly_policies"] = {"extremum": {}}
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__extremum__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__extremum__p00"
                / "ctx003"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(int(event["source_start"]), 3)
            self.assertEqual(int(event["source_end"]), 4)
            self.assertEqual(int(event["length"]), 1)
            with (instance_dir / "instance_summary.json").open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            self.assertEqual(int(summary["n_event_groups"]), 1)
            self.assertAlmostEqual(float(summary["target_density"]), 1.0 / 128.0)

    def test_fixed_first_onset_mode_grid_alignment_and_paired_channels(
        self,
    ) -> None:
        segments = sample_segment_plan(
            rng=np.random.default_rng(7),
            target_density=0.1,
            anomaly_type="mode-correlation",
            config=SegmentPlanSamplingConfig(
                series_length=128,
                channels=3,
                max_placement_attempts=2500,
                overlap_policy="global",
                channel_policy="single-random",
                density_range=(0.05, 0.10),
                segment_count_range=(20, 50),
                min_segment_length_by_anomaly={},
                segment_planner={},
                special_anomaly_policies={},
            ),
            anomaly_policy={"channel_policy": "paired-random"},
            planner_cfg={
                "planner": "fixed_first_onset_segments",
                "requested_pre_context": 10,
                "onset_bucket": "ctx010",
                "length": 16,
                "alignment_strategy": "mode_grid",
            },
            base_period_size=8,
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=derive_seed,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual({segment.start for segment in segments}, {16})
        self.assertEqual({segment.end for segment in segments}, {32})
        group_channels = segments[0].attrs["group_channels"]
        self.assertEqual(len(group_channels), 2)
        self.assertEqual(int(segments[0].attrs["requested_pre_context"]), 10)
        self.assertEqual(int(segments[0].attrs["alignment_error"]), 6)
        self.assertTrue(bool(segments[0].attrs["mode_grid_aligned"]))


if __name__ == "__main__":
    unittest.main()
