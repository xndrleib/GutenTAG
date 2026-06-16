import unittest
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gutenTAG.tsgen.parameters import (
    apply_transition_policy_to_segments,
    sample_transition_span,
    transition_cap_for_anomaly,
)


@dataclass
class Segment:
    length: int


def derive_seed(seed: int, *parts: str) -> int:
    value = int(seed)
    for idx, part in enumerate(parts):
        value += (idx + 1) * sum(ord(char) for char in str(part))
    return value


def sanitize_parameters(
    anomaly_type: str, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = dict(parameters)
    resolved["sanitized_as"] = anomaly_type
    if "transition_length" in resolved:
        resolved["transition_length"] = max(0, int(resolved["transition_length"]))
    if "transition_window" in resolved:
        resolved["transition_window"] = max(1, int(resolved["transition_window"]))
    if "crossfade_mode" in resolved:
        resolved["crossfade_mode"] = str(resolved["crossfade_mode"]).lower()
    if "boundary_mode" in resolved:
        resolved["boundary_mode"] = str(resolved["boundary_mode"]).lower()
    if "envelope_kind" in resolved:
        resolved["envelope_kind"] = str(resolved["envelope_kind"]).lower()
    return resolved


class TestTransitionParameterPolicy(unittest.TestCase):
    def test_transition_cap_allows_full_trend_window(self) -> None:
        self.assertEqual(transition_cap_for_anomaly("trend", 9), 9)
        self.assertEqual(transition_cap_for_anomaly("mean", 9), 4)
        self.assertEqual(transition_cap_for_anomaly("mean", -1), 0)

    def test_sample_transition_span_respects_cap_and_extremum_policy(self) -> None:
        self.assertEqual(
            sample_transition_span(
                anomaly_type="extremum",
                base_family="smooth_periodic",
                segment_length=20,
                rng=np.random.default_rng(4),
            ),
            0,
        )

        span = sample_transition_span(
            anomaly_type="mean",
            base_family="smooth_periodic",
            segment_length=20,
            rng=np.random.default_rng(7),
        )

        self.assertGreaterEqual(span, 0)
        self.assertLessEqual(span, 10)

    def test_applies_transition_length_to_default_transition_anomalies(self) -> None:
        result = apply_transition_policy_to_segments(
            base_family="smooth_periodic",
            anomaly_type="mean",
            anomaly_parameter_template={"offset": 1.0},
            default_anomaly_overrides={"mean": {"offset": 1.0}},
            segment_plan=[Segment(length=30)],
            segment_params=[{"offset": 1.0}],
            parameter_seed=11,
            derive_seed=derive_seed,
            sanitize_parameters=sanitize_parameters,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sanitized_as"], "mean")
        self.assertIn("transition_length", result[0])
        self.assertGreaterEqual(result[0]["transition_length"], 0)
        self.assertLessEqual(result[0]["transition_length"], 15)

    def test_preserves_custom_transition_length_when_template_is_custom(self) -> None:
        result = apply_transition_policy_to_segments(
            base_family="smooth_periodic",
            anomaly_type="mean",
            anomaly_parameter_template={"offset": 1.0, "transition_length": 6},
            default_anomaly_overrides={"mean": {"transition_length": 2}},
            segment_plan=[Segment(length=30)],
            segment_params=[{"offset": 1.0, "transition_length": 6}],
            parameter_seed=11,
            derive_seed=derive_seed,
            sanitize_parameters=sanitize_parameters,
        )

        self.assertEqual(result[0]["transition_length"], 6)

    def test_applies_pattern_shift_window_and_crossfade_defaults(self) -> None:
        result = apply_transition_policy_to_segments(
            base_family="mode_switching",
            anomaly_type="pattern-shift",
            anomaly_parameter_template={"shift_by": 4, "transition_window": 10},
            default_anomaly_overrides={
                "pattern-shift": {"shift_by": 4, "transition_window": 10}
            },
            segment_plan=[Segment(length=40)],
            segment_params=[{"shift_by": 4, "transition_window": 10}],
            parameter_seed=12,
            derive_seed=derive_seed,
            sanitize_parameters=sanitize_parameters,
        )

        self.assertGreaterEqual(result[0]["transition_window"], 1)
        self.assertEqual(result[0]["crossfade_mode"], "cosine")

    def test_applies_pattern_adaptive_blend_for_discontinuous_bases(self) -> None:
        result = apply_transition_policy_to_segments(
            base_family="discontinuous_periodic",
            anomaly_type="pattern",
            anomaly_parameter_template={"sinusoid_k": 10.0},
            default_anomaly_overrides={"pattern": {"sinusoid_k": 10.0}},
            segment_plan=[Segment(length=40)],
            segment_params=[{"sinusoid_k": 10.0}],
            parameter_seed=13,
            derive_seed=derive_seed,
            sanitize_parameters=sanitize_parameters,
        )

        self.assertTrue(result[0]["adaptive_blend"])
        self.assertEqual(result[0]["blend_strength"], 1.0)

    def test_applies_trend_boundary_defaults_from_base_family(self) -> None:
        result = apply_transition_policy_to_segments(
            base_family="smooth_trend",
            anomaly_type="trend",
            anomaly_parameter_template={},
            default_anomaly_overrides={"trend": {}},
            segment_plan=[Segment(length=50)],
            segment_params=[{}],
            parameter_seed=14,
            derive_seed=derive_seed,
            sanitize_parameters=sanitize_parameters,
        )

        self.assertEqual(result[0]["boundary_mode"], "inside_window_zero_endpoints")
        self.assertEqual(result[0]["envelope_kind"], "transition")


if __name__ == "__main__":
    unittest.main()
