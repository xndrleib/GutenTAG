import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.generator.base_channels import apply_shared_noise_correlation

from tests.ts_dataset_generation_fixtures import (
    TSDatasetGenerationConfigMixin,
)

PATTERN_EDGE_OVERRIDES = {
    "pattern": {
        "sinusoid_k": 8.0,
        "min_effect_delta": 0.10,
        "adaptive_blend": True,
        "transition_length": 6,
    },
    "pattern-shift": {
        "shift_by": 4,
        "transition_window": 6,
        "crossfade_mode": "cosine",
        "min_effect_delta": 0.10,
    },
}
PATTERN_EDGE_BASE_BY_TYPE = {"pattern": "square", "pattern-shift": "sawtooth"}


def _variant_instance_dir(output_root: Path, variant_id: str) -> Path:
    return (
        output_root / "variants" / variant_id / "train" / "instances" / "instance_000"
    )


def _configure_pattern_edge_case(config: dict[str, Any], anomaly_type: str) -> None:
    config["dataset"]["length"] = 700
    config["dataset"]["channels"] = 1
    config["dataset"]["splits"] = ["train"]
    config["dataset"]["instances_per_split"] = 1
    config["anomaly_policy"]["density_range"] = [0.05, 0.06]
    config["anomaly_policy"]["density_tolerance"] = 0.02
    config["anomaly_policy"]["segment_count_range"] = [4, 5]
    config["variants"]["base_oscillations"] = [PATTERN_EDGE_BASE_BY_TYPE[anomaly_type]]
    config["variants"]["anomaly_types"] = [anomaly_type]
    config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
    config["variants"]["anomaly_overrides"] = {
        anomaly_type: PATTERN_EDGE_OVERRIDES[anomaly_type]
    }
    config["plot"]["enabled"] = False


def _load_pattern_edge_artifacts(
    instance_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64)
    anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(dtype=np.float64)
    with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
        events = json.load(handle)
    return clean, anomalous, events


def _assert_source_edges_continuous(
    testcase: unittest.TestCase,
    anomaly_type: str,
    clean: np.ndarray,
    anomalous: np.ndarray,
    events: list[dict[str, Any]],
) -> None:
    testcase.assertGreater(len(events), 0)
    for event in events:
        source_start = int(event["source_start"])
        source_end = int(event["source_end"])
        channel = int(event["channel"])
        testcase.assertAlmostEqual(
            float(anomalous[source_start, channel]),
            float(clean[source_start, channel]),
            places=8,
            msg=f"Expected left source edge continuity for {anomaly_type}: {event}",
        )
        testcase.assertAlmostEqual(
            float(anomalous[source_end - 1, channel]),
            float(clean[source_end - 1, channel]),
            places=8,
            msg=f"Expected right source edge continuity for {anomaly_type}: {event}",
        )


class TestTSDatasetGenerationProfiles(
    TSDatasetGenerationConfigMixin, unittest.TestCase
):
    def test_shared_noise_correlation_increases_pairwise_noise_correlation(
        self,
    ) -> None:
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
        apply_shared_noise_correlation(
            channel_bos=bos,
            seed=2026,
            base_channel_correlation={"shared_noise_weight": 0.8},
        )
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
                delta = np.abs(
                    anomalous[start:end, channel] - clean[start:end, channel]
                )
                self.assertGreater(
                    float(delta.max()),
                    1e-8,
                    msg=f"Expected non-zero amplitude effect for event {event}",
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
                delta = np.abs(
                    anomalous[start:end, channel] - clean[start:end, channel]
                )
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
                delta = np.abs(
                    anomalous[start:end, channel] - clean[start:end, channel]
                )
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
                delta = np.abs(
                    anomalous[start:end, channel] - clean[start:end, channel]
                )
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.199,
                    msg=f"Platform effect floor was not met for event {event}",
                )

    def test_mean_platform_and_variance_keep_source_edges_continuous(self) -> None:
        anomaly_overrides = {
            "mean": {"offset": 0.8, "transition_length": 6},
            "platform": {
                "value": 0.25,
                "min_effect_delta": 0.15,
                "transition_length": 6,
            },
            "variance": {
                "variance": 0.10,
                "min_effect_delta": 0.12,
                "transition_length": 6,
            },
        }
        base_by_type = {"mean": "sine", "platform": "square", "variance": "cosine"}

        for anomaly_type in ["mean", "platform", "variance"]:
            with self.subTest(anomaly_type=anomaly_type):
                with tempfile.TemporaryDirectory() as tmp:
                    output_root = Path(tmp) / "dataset"
                    config = self._base_config(output_root)
                    config["dataset"]["length"] = 700
                    config["dataset"]["channels"] = 1
                    config["dataset"]["splits"] = ["train"]
                    config["dataset"]["instances_per_split"] = 1
                    config["anomaly_policy"]["density_range"] = [0.05, 0.06]
                    config["anomaly_policy"]["density_tolerance"] = 0.02
                    config["anomaly_policy"]["segment_count_range"] = [4, 5]
                    config["variants"]["base_oscillations"] = [
                        base_by_type[anomaly_type]
                    ]
                    config["variants"]["anomaly_types"] = [anomaly_type]
                    config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
                    config["variants"]["anomaly_overrides"] = {
                        anomaly_type: anomaly_overrides[anomaly_type]
                    }
                    config["plot"]["enabled"] = False

                    manifest = TSDatasetGenerator.from_dict(config).run()
                    variant_id = f"{base_by_type[anomaly_type]}__{anomaly_type}__p00"
                    self.assertIn(variant_id, manifest["generated_variants"])
                    instance_dir = (
                        output_root
                        / "variants"
                        / variant_id
                        / "train"
                        / "instances"
                        / "instance_000"
                    )
                    clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(
                        dtype=np.float64
                    )
                    anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                        dtype=np.float64
                    )
                    with (instance_dir / "events.json").open(
                        "r", encoding="utf-8"
                    ) as handle:
                        events = json.load(handle)
                    self.assertGreater(len(events), 0)
                    for event in events:
                        source_start = int(event["source_start"])
                        source_end = int(event["source_end"])
                        channel = int(event["channel"])
                        self.assertAlmostEqual(
                            float(anomalous[source_start, channel]),
                            float(clean[source_start, channel]),
                            places=8,
                            msg=f"Expected left source edge continuity for {anomaly_type}: {event}",
                        )
                        self.assertAlmostEqual(
                            float(anomalous[source_end - 1, channel]),
                            float(clean[source_end - 1, channel]),
                            places=8,
                            msg=(
                                "Expected right source edge continuity for "
                                f"{anomaly_type}: {event}"
                            ),
                        )

    def test_variance_zero_keeps_source_edges_continuous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 700
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [4, 5]
            config["variants"]["base_oscillations"] = ["cosine"]
            config["variants"]["anomaly_types"] = ["variance"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["anomaly_overrides"] = {
                "variance": {"variance": 0.0, "transition_length": 6}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            variant_id = "cosine__variance__p00"
            self.assertIn(variant_id, manifest["generated_variants"])
            instance_dir = (
                output_root
                / "variants"
                / variant_id
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
                source_start = int(event["source_start"])
                source_end = int(event["source_end"])
                channel = int(event["channel"])
                self.assertAlmostEqual(
                    float(anomalous[source_start, channel]),
                    float(clean[source_start, channel]),
                    places=8,
                    msg=f"Expected left source edge continuity for variance=0: {event}",
                )
                self.assertAlmostEqual(
                    float(anomalous[source_end - 1, channel]),
                    float(clean[source_end - 1, channel]),
                    places=8,
                    msg=f"Expected right source edge continuity for variance=0: {event}",
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
            config["anomaly_policy"]["min_segment_length_by_anomaly"] = {"variance": 8}
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
                delta = np.abs(
                    anomalous[start:end, channel] - clean[start:end, channel]
                )
                self.assertGreaterEqual(
                    float(delta.max()),
                    0.119,
                    msg=f"Variance effect floor was not met for event {event}",
                )

    def test_pattern_and_pattern_shift_keep_source_edges_continuous(self) -> None:
        for anomaly_type in ["pattern", "pattern-shift"]:
            with self.subTest(anomaly_type=anomaly_type):
                with tempfile.TemporaryDirectory() as tmp:
                    output_root = Path(tmp) / "dataset"
                    config = self._base_config(output_root)
                    _configure_pattern_edge_case(config, anomaly_type)

                    manifest = TSDatasetGenerator.from_dict(config).run()
                    variant_id = (
                        f"{PATTERN_EDGE_BASE_BY_TYPE[anomaly_type]}"
                        f"__{anomaly_type}__p00"
                    )
                    self.assertIn(variant_id, manifest["generated_variants"])
                    clean, anomalous, events = _load_pattern_edge_artifacts(
                        _variant_instance_dir(output_root, variant_id)
                    )
                    _assert_source_edges_continuous(
                        self,
                        anomaly_type,
                        clean,
                        anomalous,
                        events,
                    )

    def test_transition_policy_randomizes_default_transition_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 700
            config["dataset"]["channels"] = 1
            config["dataset"]["splits"] = ["train", "val"]
            config["dataset"]["instances_per_split"] = 2
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [4, 5]
            config["variants"]["base_oscillations"] = ["sine"]
            config["variants"]["anomaly_types"] = ["mean"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])
            transition_lengths = set()
            for split in ["train", "val"]:
                for instance_idx in range(2):
                    events_path = (
                        output_root
                        / "variants"
                        / "sine__mean__p00"
                        / split
                        / "instances"
                        / f"instance_{instance_idx:03d}"
                        / "events.json"
                    )
                    with events_path.open("r", encoding="utf-8") as handle:
                        events = json.load(handle)
                    for event in events:
                        if "transition_length" in event.get("params", {}):
                            transition_lengths.add(
                                int(event["params"]["transition_length"])
                            )
            self.assertGreater(len(transition_lengths), 1)
            self.assertTrue(all(length >= 0 for length in transition_lengths))


if __name__ == "__main__":
    unittest.main()
