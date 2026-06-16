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

ONSET_METADATA_FIELDS = {
    "requested_pre_context",
    "actual_source_start",
    "actual_support_start",
    "onset_bucket",
    "alignment_strategy",
    "alignment_error",
}


def _mean_instance_dir(output_root: Path, split: str) -> Path:
    return (
        output_root
        / "variants"
        / "sine__mean__p00"
        / split
        / "instances"
        / "instance_000"
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_mean_summary(output_root: Path, split: str) -> dict[str, Any]:
    return _load_json(_mean_instance_dir(output_root, split) / "instance_summary.json")


def _configure_split_phase_shift_case(config: dict[str, Any]) -> None:
    config["dataset"]["splits"] = ["train", "val", "test"]
    config["dataset"]["instances_per_split"] = 1
    config["variants"]["base_parameter_policy"] = "random_per_instance"
    config["variants"]["base_channel_parameter_policy"] = "random_per_instance"
    config["variants"]["base_oscillation_overrides"] = {
        "sine": {"frequency": 8.0, "amplitude": 1.0, "variance": 0.03}
    }
    config["variants"]["base_channel_overrides"] = {
        "sine": {
            "phase": 0.5,
            "amplitude": 1.0,
            "offset": 0.0,
            "variance": 0.03,
        }
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


def _assert_split_phase_offsets(
    testcase: unittest.TestCase,
    summaries: dict[str, dict[str, Any]],
) -> None:
    train_phase = float(summaries["train"]["base_parameters_per_channel"][0]["phase"])
    val_phase = float(summaries["val"]["base_parameters_per_channel"][0]["phase"])
    test_phase = float(summaries["test"]["base_parameters_per_channel"][0]["phase"])
    modulo = float(2.0 * np.pi)

    testcase.assertAlmostEqual(train_phase, 0.5, places=8)
    testcase.assertAlmostEqual(
        val_phase,
        float(np.mod(train_phase + (2.0 * np.pi / 3.0), modulo)),
        places=8,
    )
    testcase.assertAlmostEqual(
        test_phase,
        float(np.mod(train_phase + (4.0 * np.pi / 3.0), modulo)),
        places=8,
    )


def _assert_split_clean_series_diverge(
    testcase: unittest.TestCase,
    output_root: Path,
) -> None:
    clean = {
        split: pd.read_csv(
            _mean_instance_dir(output_root, split) / "clean.csv"
        ).to_numpy(dtype=np.float64)
        for split in ("train", "val", "test")
    }
    testcase.assertGreater(float(np.max(np.abs(clean["train"] - clean["val"]))), 1e-3)
    testcase.assertGreater(float(np.max(np.abs(clean["train"] - clean["test"]))), 1e-3)


def _assert_phase_shift_metadata(
    testcase: unittest.TestCase,
    summaries: dict[str, dict[str, Any]],
) -> None:
    testcase.assertFalse(
        bool(summaries["train"]["split_phase_shift"]["phase_shift_applied"])
    )
    testcase.assertTrue(
        bool(summaries["val"]["split_phase_shift"]["phase_shift_applied"])
    )
    testcase.assertTrue(
        bool(summaries["test"]["split_phase_shift"]["phase_shift_applied"])
    )


def _configure_lockstep_base_channel_case(config: dict[str, Any]) -> None:
    config["dataset"]["length"] = 600
    config["dataset"]["channels"] = 4
    config["dataset"]["splits"] = ["train"]
    config["dataset"]["instances_per_split"] = 1
    config["variants"]["base_oscillations"] = ["sine"]
    config["variants"]["anomaly_types"] = ["mean"]
    config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
    config["variants"]["base_channel_overrides"] = {
        "sine": {
            "frequency": {
                "distribution": "uniform",
                "low": 5.0,
                "high": 9.0,
                "shared_across_channels": True,
            },
            "amplitude": {
                "distribution": "uniform",
                "low": 0.8,
                "high": 1.2,
                "shared_across_channels": True,
            },
            "variance": {
                "distribution": "uniform",
                "low": 0.02,
                "high": 0.05,
                "shared_across_channels": True,
            },
            "phase": {
                "distribution": "uniform",
                "low": 0.0,
                "high": 6.283185307179586,
            },
        }
    }
    config["plot"]["enabled"] = False


def _assert_lockstep_channel_sampling(
    testcase: unittest.TestCase,
    summary: dict[str, Any],
) -> None:
    channel_params = summary["base_channel_parameters"]
    values_by_parameter = {
        name: {round(float(params[name]), 10) for params in channel_params}
        for name in ("frequency", "amplitude", "variance", "phase")
    }

    testcase.assertEqual(len(values_by_parameter["frequency"]), 1)
    testcase.assertEqual(len(values_by_parameter["amplitude"]), 1)
    testcase.assertEqual(len(values_by_parameter["variance"]), 1)
    testcase.assertGreater(len(values_by_parameter["phase"]), 1)


class TestTSDatasetGenerationProfiles(
    TSDatasetGenerationConfigMixin, unittest.TestCase
):
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

            def load_summary(split: str, index: int) -> dict:
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

    def test_split_phase_shift_applies_deterministic_offsets_across_splits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            _configure_split_phase_shift_case(config)

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])

            summaries = {
                split: _load_mean_summary(output_root, split)
                for split in ("train", "val", "test")
            }
            _assert_split_phase_offsets(self, summaries)
            _assert_split_clean_series_diverge(self, output_root)
            _assert_phase_shift_metadata(self, summaries)

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

    def test_base_channel_shared_across_channels_lockstep_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            _configure_lockstep_base_channel_case(config)

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])
            self.assertEqual(
                set(manifest["onset_metadata"]["fields"]), ONSET_METADATA_FIELDS
            )
            summary = _load_mean_summary(output_root, "train")
            _assert_lockstep_channel_sampling(self, summary)

    def test_paired_random_channel_policy_creates_synchronous_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 600
            config["dataset"]["channels"] = 4
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [4, 4]
            config["anomaly_policy"]["channel_policy"] = "paired-random"
            config["variants"]["anomaly_types"] = ["mean"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["anomaly_overrides"] = {
                "mean": {"offset": 0.8, "transition_length": 4}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / "sine__mean__p00"
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

            self.assertEqual(summary["channel_policy"], "paired-random")
            grouped: dict[int, list] = {}
            for event in events:
                grouped.setdefault(int(event["group_id"]), []).append(event)
            self.assertEqual(len(grouped), int(summary["n_event_groups"]))
            self.assertGreater(len(grouped), 0)
            for group_id, group_events in grouped.items():
                self.assertEqual(
                    len(group_events),
                    2,
                    msg=f"group {group_id} must affect two channels",
                )
                channels = sorted(int(event["channel"]) for event in group_events)
                self.assertEqual(
                    channels,
                    sorted(int(ch) for ch in group_events[0]["group_channels"]),
                )
                starts = {int(event["source_start"]) for event in group_events}
                ends = {int(event["source_end"]) for event in group_events}
                self.assertEqual(len(starts), 1)
                self.assertEqual(len(ends), 1)


if __name__ == "__main__":
    unittest.main()
