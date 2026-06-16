import unittest
from dataclasses import dataclass
from typing import Any

from gutenTAG.tsgen.parameters import (
    apply_amplitude_parameter_policy,
    apply_mean_parameter_policy,
    apply_trend_parameter_policy,
)


@dataclass
class Segment:
    attrs: dict[str, Any]


class TestAdaptiveEffectParameterPolicies(unittest.TestCase):
    def test_amplitude_adaptive_strength_uses_local_scale_floor(self) -> None:
        result = apply_amplitude_parameter_policy(
            segment_plan=[Segment(attrs={"window_residual_scale": 2.0})],
            segment_params=[{"amplitude_factor": 1.0}],
            planner_cfg={"adaptive_strength": True, "min_effect_delta": 1.0},
        )

        self.assertEqual(result[0]["amplitude_factor"], 0.5)
        self.assertEqual(result[0]["min_effect_delta"], 1.0)
        self.assertEqual(result[0]["center_mode"], "linear")

    def test_amplitude_deadzone_and_bounds_are_ordered(self) -> None:
        result = apply_amplitude_parameter_policy(
            segment_plan=[Segment(attrs={})],
            segment_params=[{"amplitude_factor": 1.02}],
            planner_cfg={
                "amplitude_factor_deadzone": [1.10, 0.90],
                "amplitude_factor_bounds": [1.05, 0.50],
            },
        )

        self.assertEqual(result[0]["amplitude_factor"], 1.05)

    def test_amplitude_policy_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "amplitude_factor_bounds"):
            apply_amplitude_parameter_policy(
                segment_plan=[Segment(attrs={})],
                segment_params=[{"amplitude_factor": 1.0}],
                planner_cfg={"amplitude_factor_bounds": [1.0]},
            )

    def test_mean_adaptive_strength_alternates_zero_offsets(self) -> None:
        result = apply_mean_parameter_policy(
            segment_params=[{"offset": 0.0}, {"offset": 0.0}],
            planner_cfg={"adaptive_strength": True, "min_effect_delta": 0.4},
        )

        self.assertEqual(result[0]["offset"], -0.4)
        self.assertEqual(result[1]["offset"], 0.4)
        self.assertEqual(result[0]["min_effect_delta"], 0.4)

    def test_mean_deadzone_pushes_offsets_to_deadzone_edge(self) -> None:
        result = apply_mean_parameter_policy(
            segment_params=[{"offset": 0.1}, {"offset": -0.1}],
            planner_cfg={"offset_deadzone": [0.25, -0.25]},
        )

        self.assertEqual(result[0]["offset"], 0.25)
        self.assertEqual(result[1]["offset"], -0.25)

    def test_trend_policy_attaches_local_stats_and_effect_floor(self) -> None:
        result = apply_trend_parameter_policy(
            segment_plan=[
                Segment(attrs={"window_rms": 0.8, "window_peak": 1.4}),
            ],
            segment_params=[{"min_effect_delta": 0.2}],
            planner_cfg={"adaptive_strength": True, "min_effect_delta": 0.5},
        )

        self.assertEqual(result[0]["min_effect_delta"], 0.5)
        self.assertEqual(result[0]["window_rms"], 0.8)
        self.assertEqual(result[0]["window_peak"], 1.4)


if __name__ == "__main__":
    unittest.main()
