import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

from gutenTAG.tsgen.output import (
    build_clean_instance_summary,
    build_paired_instance_summary,
    compute_density_metrics,
)


@dataclass
class Segment:
    attrs: dict[str, Any]


class TestOutputInstanceSummary(unittest.TestCase):
    def test_clean_instance_summary_has_stable_zero_anomaly_shape(self) -> None:
        summary = build_clean_instance_summary(
            instance_id="clean_only_000",
            split="train",
            variant_id="sine__mean__p00",
            profile_id="p00",
            base_oscillation="sine",
            anomaly_type="mean",
            channels=2,
            length_normalization="resample",
            support_label_mode="strict_segment",
            support_eps_mode="relative",
            support_eps_value=0.05,
            base_parameter_policy="fixed_per_variant",
            base_channel_parameter_policy="random_per_instance",
            anomaly_parameter_policy="fixed_per_variant",
            base_parameters={"frequency": 8.0},
            base_channel_parameters=[{"phase": 0.0}, {"phase": 1.0}],
            base_parameters_per_channel=[{"length": 16}, {"length": 16}],
            split_phase_shift={"enabled": False},
            base_channel_correlation={"shared_noise_weight": 0.0},
            seeds={"base_seed": 1},
        )

        self.assertFalse(summary["has_anomaly"])
        self.assertEqual(summary["density_validation_mode"], "clean_only")
        self.assertEqual(summary["per_channel_segment_counts"], {"0": 0, "1": 0})
        self.assertEqual(summary["segment_lengths"], [])
        self.assertEqual(summary["base_parameters"]["frequency"], 8.0)

    def test_compute_density_metrics_uses_source_support_for_effective_mode(
        self,
    ) -> None:
        labels = np.zeros((6, 2), dtype=np.int8)
        labels[2:4, 1] = 1

        metrics = compute_density_metrics(
            labels=labels,
            events=[
                {"start": 2, "end": 4, "source_start": 1, "source_end": 5, "channel": 1}
            ],
            support_label_mode="effective_support",
        )

        self.assertAlmostEqual(metrics["achieved_density_labeled"], 2 / 6)
        self.assertAlmostEqual(metrics["achieved_density_source"], 4 / 6)
        self.assertEqual(metrics["density_validation_mode"], "source_support")

    def test_paired_instance_summary_counts_groups_and_effective_support(
        self,
    ) -> None:
        labels = np.zeros((6, 2), dtype=np.int8)
        labels[2:4, 1] = 1
        events = [
            {
                "start": 2,
                "end": 4,
                "length": 2,
                "source_start": 1,
                "source_end": 5,
                "channel": 1,
                "group_id": 7,
                "requested_pre_context": 1,
                "onset_bucket": "ctx001",
                "actual_source_start": 1,
                "actual_support_start": 2,
                "alignment_strategy": "exact",
                "alignment_error": 0,
            }
        ]

        summary = build_paired_instance_summary(
            instance_id="instance_000",
            split="train",
            variant_id="sine__mean__p00",
            profile_id="p00",
            base_oscillation="sine",
            anomaly_type="mean",
            target_density=4 / 6,
            active_density_range=(0.60, 0.70),
            active_density_tolerance=0.02,
            labels=labels,
            events=events,
            segment_plan=[Segment(attrs={"energy_fallback": True})],
            channels=2,
            channel_policy="single-random",
            overlap_policy="global",
            segment_planner={"planner": "uniform_segments"},
            variant_anomaly_policy={"density_tolerance": 0.02},
            length_normalization="resample",
            support_label_mode="effective_support",
            support_eps_mode="relative",
            support_eps_value=0.05,
            base_parameter_policy="fixed_per_variant",
            base_channel_parameter_policy="random_per_instance",
            anomaly_parameter_policy="fixed_per_variant",
            base_parameters={"frequency": 8.0},
            base_channel_parameters=[{"phase": 0.0}, {"phase": 1.0}],
            base_parameters_per_channel=[{"length": 6}, {"length": 6}],
            split_phase_shift={"enabled": False},
            base_channel_correlation={"shared_noise_weight": 0.0},
            anomaly_parameters_instance={"offset": 1.0},
            seeds={"plan_seed": 2},
        )

        self.assertTrue(summary["has_anomaly"])
        self.assertAlmostEqual(summary["achieved_density_labeled"], 2 / 6)
        self.assertAlmostEqual(summary["achieved_density_source"], 4 / 6)
        self.assertEqual(summary["density_validation_mode"], "source_support")
        self.assertEqual(summary["source_segment_lengths"], [4])
        self.assertEqual(summary["effective_support_shrink_count"], 1)
        self.assertEqual(summary["energy_fallback_count"], 1)
        self.assertEqual(summary["n_event_groups"], 1)
        self.assertEqual(summary["per_channel_segment_counts"], {"0": 0, "1": 1})
        self.assertEqual(summary["requested_pre_context"], 1)
        self.assertEqual(summary["alignment_error"], 0)

    def test_paired_instance_summary_rejects_density_mismatch(self) -> None:
        labels = np.zeros((6, 1), dtype=np.int8)
        labels[0:1, 0] = 1

        with self.assertRaisesRegex(ValueError, "Density mismatch"):
            build_paired_instance_summary(
                instance_id="instance_000",
                split="train",
                variant_id="sine__mean__p00",
                profile_id="p00",
                base_oscillation="sine",
                anomaly_type="mean",
                target_density=0.50,
                active_density_range=(0.0, 1.0),
                active_density_tolerance=0.01,
                labels=labels,
                events=[{"start": 0, "end": 1, "length": 1, "channel": 0}],
                segment_plan=[Segment(attrs={})],
                channels=1,
                channel_policy="single-random",
                overlap_policy="global",
                segment_planner={},
                variant_anomaly_policy={},
                length_normalization="resample",
                support_label_mode="strict_segment",
                support_eps_mode="relative",
                support_eps_value=0.05,
                base_parameter_policy="fixed_per_variant",
                base_channel_parameter_policy="random_per_instance",
                anomaly_parameter_policy="fixed_per_variant",
                base_parameters={},
                base_channel_parameters=[],
                base_parameters_per_channel=[],
                split_phase_shift={},
                base_channel_correlation={},
                anomaly_parameters_instance=None,
                seeds={},
            )


if __name__ == "__main__":
    unittest.main()
