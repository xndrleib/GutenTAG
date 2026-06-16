import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

from gutenTAG.tsgen.parameters import (
    realize_parameters,
    resolve_anomaly_parameters_for_segments,
    resolve_instance_anomaly_parameters,
)


@dataclass
class Segment:
    length: int
    attrs: dict[str, Any]


def derive_seed(seed: int, *parts: str) -> int:
    value = int(seed)
    for idx, part in enumerate(parts):
        value += (idx + 1) * sum(ord(char) for char in str(part))
    return value


class TestParameterResolution(unittest.TestCase):
    def test_instance_fixed_per_variant_parameters_are_copied(self) -> None:
        fixed_parameters = {"offset": 1.5, "nested": {"scale": 2.0}}

        result = resolve_instance_anomaly_parameters(
            anomaly_parameter_policy="fixed_per_variant",
            anomaly_parameter_template={"offset": [1.0, 2.0]},
            fixed_anomaly_parameters=fixed_parameters,
            rng=np.random.default_rng(21),
        )

        self.assertEqual(result, fixed_parameters)
        self.assertIsNot(result, fixed_parameters)
        assert result is not None
        result["nested"]["scale"] = 99.0
        self.assertEqual(fixed_parameters["nested"]["scale"], 2.0)

    def test_instance_fixed_per_variant_realizes_missing_fixed_snapshot(self) -> None:
        template = {"offset": [1.0, 3.0]}

        result = resolve_instance_anomaly_parameters(
            anomaly_parameter_policy="fixed_per_variant",
            anomaly_parameter_template=template,
            fixed_anomaly_parameters=None,
            rng=np.random.default_rng(22),
        )
        expected = realize_parameters(template, np.random.default_rng(22))

        self.assertEqual(result, expected)

    def test_instance_random_per_instance_realizes_template(self) -> None:
        template = {"offset": {"distribution": "uniform", "low": -1.0, "high": 1.0}}

        result = resolve_instance_anomaly_parameters(
            anomaly_parameter_policy="random_per_instance",
            anomaly_parameter_template=template,
            fixed_anomaly_parameters={"offset": 5.0},
            rng=np.random.default_rng(23),
        )
        expected = realize_parameters(template, np.random.default_rng(23))

        self.assertEqual(result, expected)

    def test_instance_random_per_segment_has_no_instance_snapshot(self) -> None:
        result = resolve_instance_anomaly_parameters(
            anomaly_parameter_policy="random_per_segment",
            anomaly_parameter_template={"offset": [1.0, 2.0]},
            fixed_anomaly_parameters={"offset": 5.0},
            rng=np.random.default_rng(24),
        )

        self.assertIsNone(result)

    def test_fixed_per_variant_parameters_are_sanitized_and_copied(self) -> None:
        result = resolve_anomaly_parameters_for_segments(
            anomaly_parameter_policy="fixed_per_variant",
            base_family="smooth_periodic",
            anomaly_type="mean",
            anomaly_parameter_template={"offset": 1.0},
            fixed_anomaly_parameters={"offset": "1.5"},
            anomaly_parameters_instance=None,
            segment_plan=[Segment(length=20, attrs={}), Segment(length=20, attrs={})],
            parameter_seed=31,
            planner_cfg={},
            default_anomaly_overrides={"mean": {}},
            derive_seed=derive_seed,
        )

        self.assertEqual(result[0]["offset"], 1.5)
        self.assertEqual(result[1]["offset"], 1.5)
        result[0]["offset"] = 99.0
        self.assertEqual(result[1]["offset"], 1.5)

    def test_random_per_instance_reuses_instance_parameters(self) -> None:
        result = resolve_anomaly_parameters_for_segments(
            anomaly_parameter_policy="random_per_instance",
            base_family="smooth_periodic",
            anomaly_type="variance",
            anomaly_parameter_template={"variance": [1, 5]},
            fixed_anomaly_parameters=None,
            anomaly_parameters_instance={"variance": 3.0},
            segment_plan=[Segment(length=20, attrs={}), Segment(length=20, attrs={})],
            parameter_seed=32,
            planner_cfg={},
            default_anomaly_overrides={"variance": {}},
            derive_seed=derive_seed,
        )

        self.assertEqual(result[0]["variance"], 3.0)
        self.assertEqual(result[1]["variance"], 3.0)

    def test_trend_parameter_aware_segments_override_realized_template(self) -> None:
        result = resolve_anomaly_parameters_for_segments(
            anomaly_parameter_policy="random_per_segment",
            base_family="smooth_trend",
            anomaly_type="trend",
            anomaly_parameter_template={
                "oscillation": {"kind": "sine", "frequency": 2.0}
            },
            fixed_anomaly_parameters=None,
            anomaly_parameters_instance=None,
            segment_plan=[
                Segment(
                    length=40,
                    attrs={
                        "trend_params": {
                            "oscillation": {"kind": "sine", "frequency": 3.0},
                            "boundary_mode": "CUSTOM",
                        },
                        "window_rms": 0.8,
                    },
                )
            ],
            parameter_seed=33,
            planner_cfg={
                "planner": "trend_parameter_aware_segments",
                "adaptive_strength": True,
                "min_effect_delta": 0.4,
            },
            default_anomaly_overrides={"trend": {}},
            derive_seed=derive_seed,
        )

        self.assertEqual(result[0]["oscillation"]["frequency"], 3.0)
        self.assertEqual(result[0]["boundary_mode"], "custom")
        self.assertEqual(result[0]["envelope_kind"], "transition")
        self.assertEqual(result[0]["window_rms"], 0.8)
        self.assertEqual(result[0]["min_effect_delta"], 0.4)

    def test_routes_period_locked_frequency_policy(self) -> None:
        result = resolve_anomaly_parameters_for_segments(
            anomaly_parameter_policy="fixed_per_variant",
            base_family="smooth_periodic",
            anomaly_type="frequency",
            anomaly_parameter_template={"frequency_factor": 2.0},
            fixed_anomaly_parameters={"frequency_factor": 2.0},
            anomaly_parameters_instance=None,
            segment_plan=[Segment(length=20, attrs={"period_count": 4})],
            parameter_seed=34,
            planner_cfg={
                "planner": "period_locked_frequency",
                "period_ratio_offsets": [-1, 1],
            },
            default_anomaly_overrides={"frequency": {}},
            derive_seed=derive_seed,
        )

        self.assertIn(result[0]["frequency_factor"], {0.75, 1.25})


if __name__ == "__main__":
    unittest.main()
