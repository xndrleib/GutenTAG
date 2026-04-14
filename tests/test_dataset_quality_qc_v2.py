import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gutenTAG import TSDatasetGenerator


def _load_qc_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validate_ts_dataset_quality.py"
    )
    spec = importlib.util.spec_from_file_location("ts_dataset_quality_qc", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load QC validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDatasetQualityQCV2(unittest.TestCase):
    def test_qc_v2_group_effect_center_metrics_keep_relative_and_offset_semantics(
        self,
    ) -> None:
        module = _load_qc_module()
        relative, offset = module.aggregate_group_effect_center_metrics(
            [
                {"effect_center_of_mass_relative": 0.05, "peak_delta": 1.0},
                {"effect_center_of_mass_relative": 0.95, "peak_delta": 2.0},
            ]
        )
        self.assertAlmostEqual(relative, 0.65, places=6)
        self.assertAlmostEqual(offset, 0.45, places=6)
        self.assertAlmostEqual(
            module.compute_effect_center_offset_from_midpoint(0.05),
            0.45,
            places=6,
        )

    def test_qc_v2_channel_role_distinguishes_primary_affected_and_context(self) -> None:
        module = _load_qc_module()
        self.assertEqual(
            module.classify_group_channel_role(
                channel=1,
                primary_channels={1},
                affected_channels={1, 2},
            ),
            "primary_event_channel",
        )
        self.assertEqual(
            module.classify_group_channel_role(
                channel=2,
                primary_channels={1},
                affected_channels={1, 2},
            ),
            "affected_channel",
        )
        self.assertEqual(
            module.classify_group_channel_role(
                channel=3,
                primary_channels={1},
                affected_channels={1, 2},
            ),
            "context_channel",
        )

    def test_qc_v2_semantic_family_classifies_boundary_anchored_structural(self) -> None:
        module = _load_qc_module()
        semantic_family = module.classify_semantic_family(
            anomaly_type="mode-correlation",
            group_size=2,
            channel_visible_hint=False,
            purity_hint="relation_change",
        )
        self.assertEqual(semantic_family, "boundary_anchored_structural")

    def test_qc_v2_relation_only_policy_uses_affected_channel_shortcut(self) -> None:
        module = _load_qc_module()
        thresholds = module.QCThresholds()
        passed, reason = module.evaluate_semantic_policy(
            semantic_family="relation_only_multivariate",
            review_bucket="relation_shift",
            zero_effect=False,
            detectability_pass=True,
            effect_inside_share_l1=0.95,
            far_field_out_ratio_l2=0.01,
            uni_detector_max=1.25,
            uni_detector_affected_max=1.25,
            uni_detector_context_max=0.65,
            multi_detector_max=0.70,
            detector_preference_margin=0.10,
            thresholds=thresholds,
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "uni_shortcut_affected")

        passed, reason = module.evaluate_semantic_policy(
            semantic_family="relation_only_multivariate",
            review_bucket="relation_shift",
            zero_effect=False,
            detectability_pass=True,
            effect_inside_share_l1=0.95,
            far_field_out_ratio_l2=0.01,
            uni_detector_max=1.25,
            uni_detector_affected_max=0.95,
            uni_detector_context_max=1.60,
            multi_detector_max=0.70,
            detector_preference_margin=0.10,
            thresholds=thresholds,
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "uni_shortcut_context")

        passed, reason = module.evaluate_semantic_policy(
            semantic_family="relation_only_multivariate",
            review_bucket="relation_shift",
            zero_effect=False,
            detectability_pass=True,
            effect_inside_share_l1=0.95,
            far_field_out_ratio_l2=0.01,
            uni_detector_max=1.15,
            uni_detector_affected_max=0.95,
            uni_detector_context_max=1.15,
            multi_detector_max=0.70,
            detector_preference_margin=0.10,
            thresholds=thresholds,
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "pass")
        warning = module.classify_semantic_policy_warning_reason(
            semantic_family="relation_only_multivariate",
            semantic_policy_pass=passed,
            uni_detector_context_max=1.15,
            thresholds=thresholds,
        )
        self.assertEqual(warning, "uni_shortcut_context_warn")

    def test_qc_v2_policy_handles_zero_effect_and_far_field_leakage(self) -> None:
        module = _load_qc_module()
        thresholds = module.QCThresholds()
        passed, reason = module.evaluate_semantic_policy(
            semantic_family="univariate",
            review_bucket="channel_visible",
            zero_effect=True,
            detectability_pass=False,
            effect_inside_share_l1=0.95,
            far_field_out_ratio_l2=0.01,
            uni_detector_max=0.10,
            uni_detector_affected_max=0.10,
            uni_detector_context_max=0.10,
            multi_detector_max=0.10,
            detector_preference_margin=0.0,
            thresholds=thresholds,
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "zero_effect")

        passed, reason = module.evaluate_semantic_policy(
            semantic_family="univariate",
            review_bucket="channel_visible",
            zero_effect=False,
            detectability_pass=True,
            effect_inside_share_l1=0.95,
            far_field_out_ratio_l2=0.40,
            uni_detector_max=0.50,
            uni_detector_affected_max=0.50,
            uni_detector_context_max=0.10,
            multi_detector_max=0.10,
            detector_preference_margin=0.0,
            thresholds=thresholds,
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "far_field_leakage")

    def test_qc_v2_review_bucket_uses_detector_geometry_not_peak_only(self) -> None:
        module = _load_qc_module()
        thresholds = module.QCThresholds()
        bucket = module.classify_review_bucket(
            semantic_family="relation_only_multivariate",
            boundary_jump_ratio=1.0,
            edge_effect_ratio=0.2,
            edge_peak_ratio=0.3,
            visual_artifact_score=0.1,
            uni_detector_max=0.72,
            uni_detector_affected_max=0.72,
            multi_detector_max=0.62,
            detector_preference_margin=-0.08,
            thresholds=thresholds,
        )
        self.assertEqual(bucket, "relation_shift")

    def test_qc_v2_invariant_policy_review_and_difficulty_fixtures(self) -> None:
        module = _load_qc_module()
        thresholds = module.QCThresholds()
        fixtures = [
            {
                "name": "clean_univariate_amplitude",
                "semantic_family": "univariate",
                "bucket_inputs": {
                    "semantic_family": "univariate",
                    "boundary_jump_ratio": 1.0,
                    "edge_effect_ratio": 0.12,
                    "edge_peak_ratio": 0.18,
                    "visual_artifact_score": 0.1,
                    "uni_detector_max": 0.90,
                    "uni_detector_affected_max": 0.90,
                    "multi_detector_max": 0.10,
                    "detector_preference_margin": -0.80,
                },
                "policy_inputs": {
                    "semantic_family": "univariate",
                    "zero_effect": False,
                    "detectability_pass": True,
                    "effect_inside_share_l1": 0.96,
                    "far_field_out_ratio_l2": 0.02,
                    "uni_detector_max": 0.90,
                    "uni_detector_affected_max": 0.90,
                    "uni_detector_context_max": 0.10,
                    "multi_detector_max": 0.10,
                    "detector_preference_margin": -0.80,
                },
                "peak_rz": 8.0,
                "expected_bucket": "channel_visible",
                "expected_pass": True,
                "expected_failure_reason": "pass",
                "expected_warning_reason": "none",
                "expected_difficulty": "easy",
            },
            {
                "name": "good_relation_only",
                "semantic_family": "relation_only_multivariate",
                "bucket_inputs": {
                    "semantic_family": "relation_only_multivariate",
                    "boundary_jump_ratio": 1.0,
                    "edge_effect_ratio": 0.12,
                    "edge_peak_ratio": 0.15,
                    "visual_artifact_score": 0.1,
                    "uni_detector_max": 0.40,
                    "uni_detector_affected_max": 0.40,
                    "multi_detector_max": 0.75,
                    "detector_preference_margin": 0.25,
                },
                "policy_inputs": {
                    "semantic_family": "relation_only_multivariate",
                    "zero_effect": False,
                    "detectability_pass": True,
                    "effect_inside_share_l1": 0.95,
                    "far_field_out_ratio_l2": 0.02,
                    "uni_detector_max": 0.40,
                    "uni_detector_affected_max": 0.40,
                    "uni_detector_context_max": 0.35,
                    "multi_detector_max": 0.75,
                    "detector_preference_margin": 0.25,
                },
                "peak_rz": 3.0,
                "expected_bucket": "multivariate_only",
                "expected_pass": True,
                "expected_failure_reason": "pass",
                "expected_warning_reason": "none",
                "expected_difficulty": "medium",
            },
            {
                "name": "relation_only_affected_shortcut",
                "semantic_family": "relation_only_multivariate",
                "bucket_inputs": {
                    "semantic_family": "relation_only_multivariate",
                    "boundary_jump_ratio": 1.0,
                    "edge_effect_ratio": 0.18,
                    "edge_peak_ratio": 0.22,
                    "visual_artifact_score": 0.1,
                    "uni_detector_max": 1.25,
                    "uni_detector_affected_max": 1.25,
                    "multi_detector_max": 0.72,
                    "detector_preference_margin": -0.20,
                },
                "policy_inputs": {
                    "semantic_family": "relation_only_multivariate",
                    "zero_effect": False,
                    "detectability_pass": True,
                    "effect_inside_share_l1": 0.94,
                    "far_field_out_ratio_l2": 0.02,
                    "uni_detector_max": 1.25,
                    "uni_detector_affected_max": 1.25,
                    "uni_detector_context_max": 0.45,
                    "multi_detector_max": 0.72,
                    "detector_preference_margin": -0.20,
                },
                "peak_rz": 4.0,
                "expected_bucket": "channel_visible",
                "expected_pass": False,
                "expected_failure_reason": "uni_shortcut_affected",
                "expected_warning_reason": "none",
                "expected_difficulty": "borderline",
            },
            {
                "name": "relation_only_context_shortcut_warn",
                "semantic_family": "relation_only_multivariate",
                "bucket_inputs": {
                    "semantic_family": "relation_only_multivariate",
                    "boundary_jump_ratio": 1.0,
                    "edge_effect_ratio": 0.18,
                    "edge_peak_ratio": 0.22,
                    "visual_artifact_score": 0.1,
                    "uni_detector_max": 1.05,
                    "uni_detector_affected_max": 0.85,
                    "multi_detector_max": 0.72,
                    "detector_preference_margin": 0.05,
                },
                "policy_inputs": {
                    "semantic_family": "relation_only_multivariate",
                    "zero_effect": False,
                    "detectability_pass": True,
                    "effect_inside_share_l1": 0.94,
                    "far_field_out_ratio_l2": 0.02,
                    "uni_detector_max": 1.05,
                    "uni_detector_affected_max": 0.85,
                    "uni_detector_context_max": 1.15,
                    "multi_detector_max": 0.72,
                    "detector_preference_margin": 0.05,
                },
                "peak_rz": 4.0,
                "expected_bucket": "relation_shift",
                "expected_pass": True,
                "expected_failure_reason": "pass",
                "expected_warning_reason": "uni_shortcut_context_warn",
                "expected_difficulty": "medium",
            },
            {
                "name": "boundary_dominated_artifact",
                "semantic_family": "univariate",
                "bucket_inputs": {
                    "semantic_family": "univariate",
                    "boundary_jump_ratio": 4.5,
                    "edge_effect_ratio": 0.90,
                    "edge_peak_ratio": 0.98,
                    "visual_artifact_score": 2.4,
                    "uni_detector_max": 0.80,
                    "uni_detector_affected_max": 0.80,
                    "multi_detector_max": 0.10,
                    "detector_preference_margin": -0.70,
                },
                "policy_inputs": {
                    "semantic_family": "univariate",
                    "zero_effect": False,
                    "detectability_pass": True,
                    "effect_inside_share_l1": 0.94,
                    "far_field_out_ratio_l2": 0.02,
                    "uni_detector_max": 0.80,
                    "uni_detector_affected_max": 0.80,
                    "uni_detector_context_max": 0.10,
                    "multi_detector_max": 0.10,
                    "detector_preference_margin": -0.70,
                },
                "peak_rz": 5.0,
                "expected_bucket": "boundary_dominated",
                "expected_pass": False,
                "expected_failure_reason": "boundary_dominated",
                "expected_warning_reason": "none",
                "expected_difficulty": "borderline",
            },
            {
                "name": "zero_effect_event",
                "semantic_family": "univariate",
                "bucket_inputs": {
                    "semantic_family": "univariate",
                    "boundary_jump_ratio": 1.0,
                    "edge_effect_ratio": 0.0,
                    "edge_peak_ratio": 0.0,
                    "visual_artifact_score": 0.0,
                    "uni_detector_max": 0.0,
                    "uni_detector_affected_max": 0.0,
                    "multi_detector_max": 0.0,
                    "detector_preference_margin": 0.0,
                },
                "policy_inputs": {
                    "semantic_family": "univariate",
                    "zero_effect": True,
                    "detectability_pass": False,
                    "effect_inside_share_l1": 1.0,
                    "far_field_out_ratio_l2": 0.0,
                    "uni_detector_max": 0.0,
                    "uni_detector_affected_max": 0.0,
                    "uni_detector_context_max": 0.0,
                    "multi_detector_max": 0.0,
                    "detector_preference_margin": 0.0,
                },
                "peak_rz": 0.0,
                "expected_bucket": "channel_visible",
                "expected_pass": False,
                "expected_failure_reason": "zero_effect",
                "expected_warning_reason": "none",
                "expected_difficulty": "borderline",
            },
            {
                "name": "far_field_leakage",
                "semantic_family": "univariate",
                "bucket_inputs": {
                    "semantic_family": "univariate",
                    "boundary_jump_ratio": 1.0,
                    "edge_effect_ratio": 0.15,
                    "edge_peak_ratio": 0.20,
                    "visual_artifact_score": 0.1,
                    "uni_detector_max": 0.50,
                    "uni_detector_affected_max": 0.50,
                    "multi_detector_max": 0.10,
                    "detector_preference_margin": -0.40,
                },
                "policy_inputs": {
                    "semantic_family": "univariate",
                    "zero_effect": False,
                    "detectability_pass": True,
                    "effect_inside_share_l1": 0.94,
                    "far_field_out_ratio_l2": 0.40,
                    "uni_detector_max": 0.50,
                    "uni_detector_affected_max": 0.50,
                    "uni_detector_context_max": 0.10,
                    "multi_detector_max": 0.10,
                    "detector_preference_margin": -0.40,
                },
                "peak_rz": 4.0,
                "expected_bucket": "channel_visible",
                "expected_pass": False,
                "expected_failure_reason": "far_field_leakage",
                "expected_warning_reason": "none",
                "expected_difficulty": "borderline",
            },
        ]

        for fixture in fixtures:
            with self.subTest(fixture=fixture["name"]):
                bucket = module.classify_review_bucket(
                    thresholds=thresholds,
                    **fixture["bucket_inputs"],
                )
                self.assertEqual(bucket, fixture["expected_bucket"])
                passed, reason = module.evaluate_semantic_policy(
                    thresholds=thresholds,
                    review_bucket=bucket,
                    **fixture["policy_inputs"],
                )
                self.assertEqual(passed, fixture["expected_pass"])
                self.assertEqual(reason, fixture["expected_failure_reason"])
                warning = module.classify_semantic_policy_warning_reason(
                    semantic_family=fixture["semantic_family"],
                    semantic_policy_pass=passed,
                    uni_detector_context_max=fixture["policy_inputs"][
                        "uni_detector_context_max"
                    ],
                    thresholds=thresholds,
                )
                self.assertEqual(warning, fixture["expected_warning_reason"])
                difficulty_signal = module.compute_difficulty_signal(
                    peak_rz=fixture["peak_rz"],
                    uni_detector_score=fixture["policy_inputs"]["uni_detector_max"],
                    multi_detector_score=fixture["policy_inputs"]["multi_detector_max"],
                    thresholds=thresholds,
                )
                difficulty = module.classify_difficulty_stratum(
                    zero_effect=fixture["policy_inputs"]["zero_effect"],
                    effect_inside_share_l1=fixture["policy_inputs"][
                        "effect_inside_share_l1"
                    ],
                    far_field_out_ratio_l2=fixture["policy_inputs"][
                        "far_field_out_ratio_l2"
                    ],
                    review_bucket=bucket,
                    difficulty_signal=difficulty_signal,
                    semantic_family=fixture["semantic_family"],
                    uni_detector_score=fixture["policy_inputs"]["uni_detector_max"],
                    detector_preference_margin=fixture["policy_inputs"][
                        "detector_preference_margin"
                    ],
                    thresholds=thresholds,
                )
                self.assertEqual(difficulty, fixture["expected_difficulty"])

    def test_qc_v2_emits_v11_profiles_and_probe_artifacts(self) -> None:
        module = _load_qc_module()
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            config = {
                "generator": {
                    "output_root": str(output_root),
                    "master_seed": 1234,
                    "overwrite_output": True,
                    "log_level": "INFO",
                    "on_variant_failure": "skip",
                },
                "dataset": {
                    "length": 700,
                    "channels": 2,
                    "splits": ["train"],
                    "instances_per_split": 1,
                },
                "anomaly_policy": {
                    "density_range": [0.06, 0.07],
                    "density_tolerance": 0.02,
                    "segment_count_range": [2, 2],
                    "placement_policy": "uniform",
                    "channel_policy": "single-random",
                    "overlap_policy": "global",
                    "length_normalization": "resample",
                    "special_anomaly_policies": {
                        "correlation-flip": {
                            "channel_policy": "paired-random",
                            "min_segment_length": 24,
                        }
                    },
                    "min_segment_length_by_anomaly": {"correlation-flip": 24},
                },
                "variants": {
                    "base_oscillations": ["polynomial"],
                    "anomaly_types": ["correlation-flip"],
                    "disabled_anomaly_types": [],
                    "profiles_per_pair": 1,
                    "pair_profiles": {},
                    "base_parameter_policy": "fixed_per_variant",
                    "anomaly_parameter_policy": "fixed_per_variant",
                    "skip_base_oscillations": [],
                    "skip_anomaly_types": [],
                    "compatibility_mode": "validated",
                    "base_channel_correlation": {"shared_noise_weight": 0.92},
                    "base_oscillation_overrides": {
                        "polynomial": {"polynomial": [0.015, 0.16], "variance": 0.12}
                    },
                    "anomaly_overrides": {
                        "correlation-flip": {
                            "target_correlation": -1.0,
                            "transition_length": 10,
                        }
                    },
                },
                "plot": {"enabled": False},
            }
            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertIn(
                "polynomial__correlation-flip__p00", manifest["generated_variants"]
            )

            qc_output = output_root / "analysis" / "qc_gate"
            gate_pass, summary = module.evaluate_dataset(
                dataset_root=output_root,
                output_dir=qc_output,
                event_eps=1e-8,
                generic_peak_rz_min=2.0,
                amplitude_peak_rz_min=3.0,
                thresholds=module.QCThresholds(),
                render_visual_review=False,
                review_top_k=8,
            )

            self.assertIn("advisory_checks", summary)
            self.assertIn("type_profiles", summary)
            self.assertIn("semantic_family_summary", summary)
            self.assertIn("carrier_family_summary", summary)
            self.assertIn("extrinsic_probe_summary", summary)
            self.assertIn("total_groups", summary)
            self.assertIn("total_group_channels", summary)
            self.assertIn("dataset_borderline_share", summary)
            self.assertIn("detector_calibration_summary", summary)
            self.assertIn("group_channel_summary", summary)
            self.assertGreater(int(summary["total_events"]), 0)
            self.assertGreater(int(summary["total_groups"]), 0)
            self.assertGreaterEqual(
                int(summary["total_group_channels"]), int(summary["total_groups"])
            )
            self.assertIsInstance(gate_pass, bool)

            event_metrics = pd.read_csv(qc_output / "event_metrics.csv")
            group_metrics = pd.read_csv(qc_output / "group_metrics.csv")
            group_channel_metrics = pd.read_csv(qc_output / "group_channel_metrics.csv")
            group_channel_summary = pd.read_csv(qc_output / "group_channel_summary.csv")
            detector_calibration_summary = pd.read_csv(
                qc_output / "detector_calibration_summary.csv"
            )
            self.assertIn("effect_inside_share_l1", event_metrics.columns)
            self.assertIn("effect_center_of_mass_relative", event_metrics.columns)
            self.assertIn("effect_center_offset_from_midpoint", event_metrics.columns)
            self.assertIn("near_field_out_ratio_l2", event_metrics.columns)
            self.assertIn("spectral_shift_score", event_metrics.columns)
            self.assertIn("covariance_shift_score", event_metrics.columns)
            self.assertIn("uni_probe_uplift_max", event_metrics.columns)
            self.assertIn("multi_probe_uplift_max", event_metrics.columns)
            self.assertIn("fitclean_uni_uplift_max", event_metrics.columns)
            self.assertIn("fitclean_multi_uplift_max", event_metrics.columns)
            self.assertIn("detector_preference_margin", event_metrics.columns)
            self.assertIn("semantic_family", event_metrics.columns)
            self.assertIn("carrier_family", event_metrics.columns)
            self.assertIn("semantic_policy_pass", event_metrics.columns)
            self.assertIn("semantic_policy_failure_reason", event_metrics.columns)
            self.assertIn("semantic_policy_warning_reason", event_metrics.columns)
            self.assertIn("difficulty_stratum", event_metrics.columns)
            self.assertIn("analysis_unit", group_metrics.columns)
            self.assertIn("group_key", group_metrics.columns)
            self.assertIn("uni_detector_raw_max", group_metrics.columns)
            self.assertIn("multi_detector_raw_max", group_metrics.columns)
            self.assertIn("uni_detector_primary_max", group_metrics.columns)
            self.assertIn("uni_detector_affected_max", group_metrics.columns)
            self.assertIn("uni_detector_context_max", group_metrics.columns)
            self.assertIn("effect_center_of_mass_relative", group_metrics.columns)
            self.assertIn(
                "effect_center_offset_from_midpoint", group_metrics.columns
            )
            self.assertIn("semantic_policy_pass", group_metrics.columns)
            self.assertIn("semantic_policy_warning_reason", group_metrics.columns)
            self.assertIn("channel_role", group_channel_metrics.columns)
            self.assertIn("uni_detector_max", group_channel_metrics.columns)
            self.assertIn("uni_visibility_sample_count", detector_calibration_summary.columns)
            self.assertIn("uses_dataset_default", detector_calibration_summary.columns)
            self.assertIn(
                "shortcut_share_ge_warn_threshold", group_channel_summary.columns
            )
            self.assertIn("uni_visibility_scale", detector_calibration_summary.columns)
            self.assertFalse(
                (group_metrics["semantic_policy_failure_reason"] == "pending_calibration").any()
            )
            self.assertFalse(
                (group_metrics["semantic_policy_warning_reason"] == "pending_calibration").any()
            )
            self.assertLessEqual(len(group_metrics), len(event_metrics))
            self.assertGreaterEqual(len(group_channel_metrics), len(group_metrics))
            self.assertTrue(
                (event_metrics["semantic_family"] == "relation_only_multivariate").any()
            )
            self.assertTrue(
                group_channel_metrics["channel_role"].isin(
                    ["primary_event_channel", "affected_channel", "context_channel"]
                ).all()
            )

            for name in [
                "type_profiles.csv",
                "semantic_family_summary.csv",
                "difficulty_summary.csv",
                "carrier_family_summary.csv",
                "generator_mode_summary.csv",
                "extrinsic_probe_summary.csv",
                "detector_behavior_summary.csv",
                "detector_calibration_summary.csv",
                "group_channel_summary.csv",
                "policy_summary.csv",
                "policy_reason_summary.csv",
                "column_guide.json",
            ]:
                self.assertTrue((qc_output / name).exists(), msg=name)


if __name__ == "__main__":
    unittest.main()
