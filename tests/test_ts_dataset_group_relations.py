import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator

from tests.ts_dataset_generation_fixtures import (
    TSDatasetGenerationConfigMixin,
    _pair_residual_corr,
)

MULTIVARIATE_METADATA_SETUPS = {
    "covariance-change": {
        "base": "sine",
        "override": {"coupling_strength": 0.95, "transition_length": 6},
    },
    "correlation-flip": {
        "base": "sine",
        "override": {"target_correlation": -0.85, "transition_length": 6},
    },
    "channel-rewiring": {
        "base": "sine",
        "override": {"rotation_degrees": 25, "transition_length": 6},
    },
    "lag-synchronization": {
        "base": "sawtooth",
        "override": {"lag_steps": 5, "transition_length": 6},
    },
    "shared-factor-break": {
        "base": "sine",
        "override": {"shared_factor_scale": 0.0, "transition_length": 6},
    },
}


def _variant_instance_dir(output_root: Path, variant_id: str) -> Path:
    return (
        output_root / "variants" / variant_id / "train" / "instances" / "instance_000"
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_events_and_summary(
    instance_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return (
        _load_json(instance_dir / "events.json"),
        _load_json(instance_dir / "instance_summary.json"),
    )


def _load_clean_and_anomalous(instance_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return (
        pd.read_csv(instance_dir / "clean.csv").to_numpy(dtype=np.float64),
        pd.read_csv(instance_dir / "anomalous.csv").to_numpy(dtype=np.float64),
    )


def _configure_mode_correlation_case(config: dict[str, Any]) -> None:
    config["dataset"]["length"] = 900
    config["dataset"]["channels"] = 4
    config["dataset"]["splits"] = ["train"]
    config["dataset"]["instances_per_split"] = 1
    config["anomaly_policy"]["density_range"] = [0.05, 0.06]
    config["anomaly_policy"]["density_tolerance"] = 0.02
    config["anomaly_policy"]["segment_count_range"] = [4, 4]
    config["anomaly_policy"]["channel_policy"] = "single-random"
    config["variants"]["base_oscillations"] = ["random-mode-jump"]
    config["variants"]["anomaly_types"] = ["mode-correlation"]
    config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
    config["plot"]["enabled"] = False


def _assert_mode_correlation_summary(
    testcase: unittest.TestCase,
    summary: Mapping[str, Any],
    events: list[dict[str, Any]],
) -> None:
    testcase.assertEqual(summary["channel_policy"], "paired-random")
    testcase.assertEqual(summary["segment_planner"]["planner"], "mode_grid_segments")
    testcase.assertGreater(len(events), 0)


def _assert_mode_correlation_event(
    testcase: unittest.TestCase,
    event: Mapping[str, Any],
    clean: np.ndarray,
    anomalous: np.ndarray,
) -> None:
    testcase.assertEqual(event["anomaly_object"], "relation_sign_flip")
    testcase.assertFalse(bool(event["mode_change_aligned"]))
    testcase.assertTrue(bool(event["mode_grid_aligned"]))
    testcase.assertTrue(bool(event["support_independent_of_realized_mode_state"]))
    testcase.assertTrue(bool(event["latent_mode_flip"]))
    testcase.assertEqual(len(event["group_channels"]), 2)
    testcase.assertIn(
        int(event["anchor_channel"]), [int(ch) for ch in event["group_channels"]]
    )
    testcase.assertIn(
        int(event["channel"]), [int(ch) for ch in event["flipped_channels"]]
    )
    source_start = int(event["source_start"])
    source_end = int(event["source_end"])
    block_size = int(event["mode_grid_block_size"])
    testcase.assertEqual(source_start % block_size, 0)
    testcase.assertEqual(source_end % block_size, 0)
    testcase.assertEqual(source_start, int(event["mode_grid_start_block"]) * block_size)
    testcase.assertEqual(source_end, int(event["mode_grid_end_block"]) * block_size)
    channel = int(event["channel"])
    testcase.assertNotAlmostEqual(
        float(anomalous[source_start, channel]),
        float(clean[source_start, channel]),
        places=6,
        msg=f"Mode-correlation should remain visible on the flipped channel: {event}",
    )


def _configure_multivariate_metadata_case(
    config: dict[str, Any],
    anomaly_type: str,
    setup: Mapping[str, Any],
) -> None:
    config["dataset"]["length"] = 900
    config["dataset"]["channels"] = 4
    config["dataset"]["splits"] = ["train"]
    config["dataset"]["instances_per_split"] = 1
    config["anomaly_policy"]["density_range"] = [0.05, 0.06]
    config["anomaly_policy"]["density_tolerance"] = 0.02
    config["anomaly_policy"]["segment_count_range"] = [4, 4]
    config["anomaly_policy"]["channel_policy"] = "single-random"
    config["variants"]["base_oscillations"] = [str(setup["base"])]
    config["variants"]["anomaly_types"] = [anomaly_type]
    config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
    config["variants"]["base_channel_correlation"] = {"shared_noise_weight": 0.35}
    config["variants"]["anomaly_overrides"] = {
        anomaly_type: dict(cast(Mapping[str, Any], setup["override"]))
    }
    config["plot"]["enabled"] = False


def _assert_multivariate_semantic_metadata(
    testcase: unittest.TestCase,
    summary: Mapping[str, Any],
    events: list[dict[str, Any]],
) -> None:
    testcase.assertEqual(summary["channel_policy"], "paired-random")
    testcase.assertGreater(len(events), 0)
    for event in events:
        for key in (
            "intervention_channels",
            "anomaly_object",
            "group_id",
            "group_channels",
            "channel_visible",
            "purity_hint",
        ):
            testcase.assertIn(key, event)
        intervention = [int(ch) for ch in event["intervention_channels"]]
        group_channels = [int(ch) for ch in event["group_channels"]]
        testcase.assertGreaterEqual(len(group_channels), 2)
        testcase.assertTrue(set(intervention).issubset(set(group_channels)))


class TestTSDatasetGenerationProfiles(
    TSDatasetGenerationConfigMixin, unittest.TestCase
):
    def test_mode_correlation_uses_paired_groups_and_relation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            _configure_mode_correlation_case(config)

            manifest = TSDatasetGenerator.from_dict(config).run()
            variant_id = "random-mode-jump__mode-correlation__p00"
            self.assertIn(variant_id, manifest["generated_variants"])
            instance_dir = _variant_instance_dir(output_root, variant_id)
            events, summary = _load_events_and_summary(instance_dir)
            clean, anomalous = _load_clean_and_anomalous(instance_dir)

            _assert_mode_correlation_summary(self, summary, events)
            for event in events:
                _assert_mode_correlation_event(self, event, clean, anomalous)

    def test_shared_noise_relation_anomalies_rewrite_observed_relation(self) -> None:
        for anomaly_type in ("correlation-flip", "covariance-change"):
            with self.subTest(anomaly_type=anomaly_type):
                with tempfile.TemporaryDirectory() as tmp:
                    output_root = Path(tmp) / "dataset"
                    config = self._base_config(output_root)
                    config["dataset"]["length"] = 900
                    config["dataset"]["channels"] = 4
                    config["dataset"]["splits"] = ["train"]
                    config["dataset"]["instances_per_split"] = 1
                    config["anomaly_policy"]["density_range"] = [0.05, 0.06]
                    config["anomaly_policy"]["density_tolerance"] = 0.02
                    config["anomaly_policy"]["segment_count_range"] = [2, 2]
                    config["anomaly_policy"]["channel_policy"] = "paired-random"
                    config["anomaly_policy"]["min_segment_length_by_anomaly"] = {
                        anomaly_type: 48
                    }
                    config["variants"]["base_oscillations"] = ["shared-noise-sine"]
                    config["variants"]["anomaly_types"] = [anomaly_type]
                    config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
                    config["plot"]["enabled"] = False

                    manifest = TSDatasetGenerator.from_dict(config).run()
                    variant_id = f"shared-noise-sine__{anomaly_type}__p00"
                    self.assertIn(variant_id, manifest["generated_variants"])
                    instance_dir = (
                        output_root
                        / "variants"
                        / variant_id
                        / "train"
                        / "instances"
                        / "instance_000"
                    )
                    events = json.loads(
                        (instance_dir / "events.json").read_text(encoding="utf-8")
                    )
                    clean = pd.read_csv(instance_dir / "clean.csv").to_numpy(
                        dtype=np.float64
                    )
                    anomalous = pd.read_csv(instance_dir / "anomalous.csv").to_numpy(
                        dtype=np.float64
                    )
                    event = events[0]
                    channels = [int(channel) for channel in event["group_channels"][:2]]
                    source_start = int(event["source_start"])
                    source_end = int(event["source_end"])

                    clean_corr = float(
                        np.corrcoef(
                            clean[source_start:source_end, channels], rowvar=False
                        )[0, 1]
                    )
                    anomalous_corr = float(
                        np.corrcoef(
                            anomalous[source_start:source_end, channels], rowvar=False
                        )[0, 1]
                    )

                    self.assertGreater(clean_corr, 0.5)
                    self.assertLess(anomalous_corr, -0.5)
                    self.assertEqual(event["injection_level"], "observed_window")

    def test_new_multivariate_anomalies_emit_semantic_metadata(self) -> None:
        for anomaly_type, setup in MULTIVARIATE_METADATA_SETUPS.items():
            with (
                self.subTest(anomaly_type=anomaly_type),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output_root = Path(tmp) / "dataset"
                config = self._base_config(output_root)
                _configure_multivariate_metadata_case(config, anomaly_type, setup)

                manifest = TSDatasetGenerator.from_dict(config).run()
                variant_id = f"{setup['base']}__{anomaly_type}__p00"
                self.assertIn(variant_id, manifest["generated_variants"])
                events, summary = _load_events_and_summary(
                    _variant_instance_dir(output_root, variant_id)
                )

                _assert_multivariate_semantic_metadata(self, summary, events)

    def test_covariance_change_increases_relation_shift_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 900
            config["dataset"]["channels"] = 4
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.08, 0.08]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [2, 2]
            config["variants"]["base_oscillations"] = ["sine"]
            config["variants"]["anomaly_types"] = ["covariance-change"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["base_channel_correlation"] = {
                "shared_noise_weight": 0.75
            }
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {"variance": 0.12}
            }
            config["variants"]["anomaly_overrides"] = {
                "covariance-change": {
                    "coupling_strength": -0.97,
                    "transition_length": 6,
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "sine__covariance-change__p00", manifest["generated_variants"]
            )
            instance_dir = (
                output_root
                / "variants"
                / "sine__covariance-change__p00"
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

            event = events[0]
            channels = [int(ch) for ch in event["group_channels"]]
            source_start = int(event["source_start"])
            source_end = int(event["source_end"])
            clean_corr = _pair_residual_corr(clean, channels, source_start, source_end)
            anom_corr = _pair_residual_corr(
                anomalous, channels, source_start, source_end
            )
            self.assertGreater(abs(float(anom_corr) - float(clean_corr)), 0.15)
            self.assertEqual(event["anomaly_object"], "shared_noise_coupling_change")
            self.assertEqual(event["purity_hint"], "operational_candidate")
            self.assertEqual(event["injection_level"], "noise")

    def test_correlation_flip_changes_local_correlation_sign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 900
            config["dataset"]["channels"] = 4
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.08, 0.08]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [2, 2]
            config["variants"]["base_oscillations"] = ["polynomial"]
            config["variants"]["anomaly_types"] = ["correlation-flip"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["base_channel_correlation"] = {
                "shared_noise_weight": 0.92
            }
            config["variants"]["base_oscillation_overrides"] = {
                "polynomial": {"polynomial": [0.015, 0.16], "variance": 0.12}
            }
            config["variants"]["anomaly_overrides"] = {
                "correlation-flip": {
                    "target_correlation": -1.0,
                    "transition_length": 10,
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "polynomial__correlation-flip__p00", manifest["generated_variants"]
            )
            instance_dir = (
                output_root
                / "variants"
                / "polynomial__correlation-flip__p00"
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

            self.assertEqual(len(events), 2)
            event = events[0]
            channels = [int(ch) for ch in event["group_channels"]]
            intervention = [int(ch) for ch in event["intervention_channels"]]
            self.assertEqual(len(channels), 2)
            self.assertEqual(len(intervention), 1)
            self.assertNotEqual(intervention[0], channels[0])
            source_start = int(event["source_start"])
            source_end = int(event["source_end"])
            clean_corr = _pair_residual_corr(clean, channels, source_start, source_end)
            anom_corr = _pair_residual_corr(
                anomalous, channels, source_start, source_end
            )
            self.assertGreater(clean_corr, 0.10)
            self.assertGreater(abs(float(anom_corr) - float(clean_corr)), 0.15)
            self.assertAlmostEqual(
                float(anomalous[source_start, intervention[0]]),
                float(clean[source_start, intervention[0]]),
                places=8,
            )
            self.assertAlmostEqual(
                float(anomalous[source_end - 1, intervention[0]]),
                float(clean[source_end - 1, intervention[0]]),
                places=8,
            )
            self.assertEqual(event["anomaly_object"], "pair_correlation_flip")
            self.assertEqual(event["purity_hint"], "operational_candidate")
            self.assertEqual(event["injection_level"], "noise")

    def test_shared_factor_break_reduces_local_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 900
            config["dataset"]["channels"] = 4
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.08, 0.08]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [2, 2]
            config["variants"]["base_oscillations"] = ["sine"]
            config["variants"]["anomaly_types"] = ["shared-factor-break"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["base_channel_correlation"] = {
                "shared_noise_weight": 0.75
            }
            config["variants"]["base_oscillation_overrides"] = {
                "sine": {"variance": 0.12}
            }
            config["variants"]["anomaly_overrides"] = {
                "shared-factor-break": {
                    "shared_factor_scale": 0.0,
                    "transition_length": 6,
                }
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "sine__shared-factor-break__p00", manifest["generated_variants"]
            )
            instance_dir = (
                output_root
                / "variants"
                / "sine__shared-factor-break__p00"
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

            event = events[0]
            channels = [int(ch) for ch in event["group_channels"]]
            intervention = [int(ch) for ch in event["intervention_channels"]]
            source_start = int(event["source_start"])
            source_end = int(event["source_end"])
            clean_corr = _pair_residual_corr(clean, channels, source_start, source_end)
            anom_corr = _pair_residual_corr(
                anomalous, channels, source_start, source_end
            )
            self.assertGreater(abs(float(clean_corr)) - abs(float(anom_corr)), 0.02)
            self.assertGreaterEqual(len(intervention), 1)
            self.assertEqual(event["anomaly_object"], "shared_factor_break")
            self.assertEqual(event["purity_hint"], "operational_candidate")
            self.assertEqual(event["injection_level"], "noise")

    def test_lag_synchronization_records_realized_lag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 900
            config["dataset"]["channels"] = 4
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [4, 4]
            config["variants"]["base_oscillations"] = ["sawtooth"]
            config["variants"]["anomaly_types"] = ["lag-synchronization"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["anomaly_overrides"] = {
                "lag-synchronization": {"lag_steps": 7, "transition_length": 6}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "sawtooth__lag-synchronization__p00", manifest["generated_variants"]
            )
            instance_dir = (
                output_root
                / "variants"
                / "sawtooth__lag-synchronization__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "events.json").open("r", encoding="utf-8") as handle:
                events = json.load(handle)
            self.assertGreater(len(events), 0)
            realized = [int(event["realized_lag_steps"]) for event in events]
            self.assertTrue(any(abs(value) > 0 for value in realized))
            self.assertTrue(
                all(event["purity_hint"] == "not_pure_local" for event in events)
            )

    def test_channel_rewiring_uses_noise_injection_on_structural_carrier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["length"] = 900
            config["dataset"]["channels"] = 4
            config["dataset"]["splits"] = ["train"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["density_range"] = [0.05, 0.06]
            config["anomaly_policy"]["density_tolerance"] = 0.02
            config["anomaly_policy"]["segment_count_range"] = [4, 4]
            config["variants"]["base_oscillations"] = ["shared-noise-sine"]
            config["variants"]["anomaly_types"] = ["channel-rewiring"]
            config["variants"]["anomaly_parameter_policy"] = "fixed_per_variant"
            config["variants"]["base_channel_correlation"] = {
                "shared_noise_weight": 0.75
            }
            config["anomaly_policy"]["special_anomaly_policies"] = {
                "channel-rewiring": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                }
            }
            config["anomaly_policy"]["min_segment_length_by_anomaly"] = {
                "channel-rewiring": 20
            }
            config["variants"]["anomaly_overrides"] = {
                "channel-rewiring": {"rotation_degrees": 18, "transition_length": 6}
            }
            config["plot"]["enabled"] = False

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "shared-noise-sine__channel-rewiring__p00",
                manifest["generated_variants"],
            )
            instance_dir = (
                output_root
                / "variants"
                / "shared-noise-sine__channel-rewiring__p00"
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
            self.assertTrue(
                all(event["injection_level"] == "noise" for event in events)
            )
            self.assertTrue(
                all(event["purity_hint"] == "operational_candidate" for event in events)
            )
            for event in events:
                source_start = int(event["source_start"])
                source_end = int(event["source_end"])
                channel = int(event["channel"])
                self.assertAlmostEqual(
                    float(anomalous[source_start, channel]),
                    float(clean[source_start, channel]),
                    places=8,
                )
                self.assertAlmostEqual(
                    float(anomalous[source_end - 1, channel]),
                    float(clean[source_end - 1, channel]),
                    places=8,
                )

    def test_recommended_compatibility_mode_skips_unvalidated_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["val"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            config["plot"]["enabled"] = False
            config["variants"]["base_oscillations"] = ["sine", "polynomial"]
            config["variants"]["anomaly_types"] = [
                "channel-rewiring",
                "covariance-change",
            ]
            config["variants"]["compatibility_mode"] = "recommended"
            config["anomaly_policy"]["special_anomaly_policies"] = {
                "channel-rewiring": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                },
                "covariance-change": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                },
            }
            config["anomaly_policy"]["min_segment_length_by_anomaly"] = {
                "channel-rewiring": 20,
                "covariance-change": 20,
            }
            config["variants"]["anomaly_overrides"] = {
                "channel-rewiring": {"rotation_degrees": 25, "transition_length": 8},
                "covariance-change": {
                    "coupling_strength": -0.95,
                    "transition_length": 8,
                },
            }

            manifest = TSDatasetGenerator.from_dict(config).run()

            self.assertIn(
                "polynomial__covariance-change__p00", manifest["generated_variants"]
            )
            self.assertNotIn(
                "sine__channel-rewiring__p00", manifest["generated_variants"]
            )
            skipped = {entry["variant_id"] for entry in manifest["skipped_variants"]}
            self.assertIn("sine__channel-rewiring__p00", skipped)

    def test_validated_compatibility_mode_admits_only_curated_structural_pairs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["val"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            config["plot"]["enabled"] = False
            config["variants"]["base_oscillations"] = [
                "sine",
                "cosine",
                "polynomial",
                "shared-noise-sine",
            ]
            config["variants"]["anomaly_types"] = [
                "correlation-flip",
                "covariance-change",
                "lag-synchronization",
                "shared-factor-break",
            ]
            config["variants"]["compatibility_mode"] = "validated"
            config["anomaly_policy"]["special_anomaly_policies"] = {
                "correlation-flip": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                },
                "covariance-change": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                },
                "lag-synchronization": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                },
                "shared-factor-break": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                },
            }
            config["anomaly_policy"]["min_segment_length_by_anomaly"] = {
                "correlation-flip": 20,
                "covariance-change": 20,
                "lag-synchronization": 20,
                "shared-factor-break": 20,
            }
            config["variants"]["anomaly_overrides"] = {
                "correlation-flip": {
                    "target_correlation": -0.95,
                    "transition_length": 8,
                },
                "covariance-change": {
                    "coupling_strength": -0.95,
                    "transition_length": 8,
                },
                "lag-synchronization": {"lag_steps": 6, "transition_length": 8},
                "shared-factor-break": {
                    "shared_factor_scale": 0.0,
                    "transition_length": 8,
                },
            }

            manifest = TSDatasetGenerator.from_dict(config).run()
            generated = set(manifest["generated_variants"])
            skipped = {entry["variant_id"] for entry in manifest["skipped_variants"]}

            self.assertIn("polynomial__covariance-change__p00", generated)
            self.assertIn("shared-noise-sine__covariance-change__p00", generated)
            self.assertIn("polynomial__correlation-flip__p00", generated)
            self.assertIn("shared-noise-sine__correlation-flip__p00", generated)
            self.assertIn("sine__lag-synchronization__p00", generated)
            self.assertIn("cosine__lag-synchronization__p00", generated)
            self.assertNotIn("cosine__correlation-flip__p00", generated)
            self.assertNotIn("shared-noise-sine__shared-factor-break__p00", generated)
            self.assertIn("cosine__correlation-flip__p00", skipped)
            self.assertIn("shared-noise-sine__shared-factor-break__p00", skipped)

    def test_pair_override_base_channel_correlation_reaches_instance_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = self._base_config(output_root)
            config["dataset"]["channels"] = 2
            config["dataset"]["splits"] = ["val"]
            config["dataset"]["instances_per_split"] = 1
            config["anomaly_policy"]["segment_count_range"] = [1, 1]
            config["plot"]["enabled"] = False
            config["variants"]["base_oscillations"] = ["polynomial"]
            config["variants"]["anomaly_types"] = ["covariance-change"]
            config["variants"]["base_channel_correlation"] = {
                "shared_noise_weight": 0.25
            }
            config["variants"]["variant_overrides"] = {
                "polynomial__covariance-change": {
                    "base_channel_correlation": {"shared_noise_weight": 0.93}
                }
            }
            config["anomaly_policy"]["special_anomaly_policies"] = {
                "covariance-change": {
                    "channel_policy": "paired-random",
                    "min_segment_length": 20,
                }
            }
            config["anomaly_policy"]["min_segment_length_by_anomaly"] = {
                "covariance-change": 20
            }
            config["variants"]["anomaly_overrides"] = {
                "covariance-change": {
                    "coupling_strength": -0.95,
                    "transition_length": 8,
                }
            }

            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "polynomial__covariance-change__p00", manifest["generated_variants"]
            )
            instance_dir = (
                output_root
                / "variants"
                / "polynomial__covariance-change__p00"
                / "val"
                / "instances"
                / "instance_000"
            )
            with (instance_dir / "instance_summary.json").open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            self.assertAlmostEqual(
                float(summary["base_channel_correlation"]["shared_noise_weight"]),
                0.93,
                places=8,
            )


if __name__ == "__main__":
    unittest.main()
