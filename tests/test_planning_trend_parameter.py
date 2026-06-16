import unittest
from typing import Any, Mapping

import numpy as np

from gutenTAG.tsgen.planning import sample_trend_parameter_aware_segments


def _derive_seed(seed: int, *parts: str) -> int:
    payload = str(seed) + "|" + "|".join(parts)
    return abs(hash(payload)) % (2**32)


def _realize_parameters(
    template: Mapping[str, Any],
    _rng: np.random.Generator,
) -> dict[str, Any]:
    return dict(template)


def _sanitize_parameters(
    _anomaly_type: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(params)


class TestTrendParameterPlanning(unittest.TestCase):
    def test_sine_trend_min_cycles_controls_segment_lengths(self) -> None:
        template = {
            "oscillation": {
                "kind": "sine",
                "frequency": 2.0,
                "amplitude": 0.8,
            }
        }

        segments = sample_trend_parameter_aware_segments(
            rng=np.random.default_rng(22),
            target_density=0.40,
            series_length=200,
            channels=2,
            max_placement_attempts=30,
            overlap_policy="global",
            planner_cfg={
                "planner": "trend_parameter_aware_segments",
                "sine_min_cycles": 0.30,
            },
            anomaly_parameter_template=template,
            parameter_seed=123,
            clean_values=None,
            segment_count_range=(2, 2),
            base_min_segment_length=5,
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertGreaterEqual(segment.length, 15)
            self.assertEqual(segment.attrs["trend_oscillation_kind"], "sine")
            self.assertEqual(segment.attrs["trend_min_segment_length"], 15)
            self.assertEqual(
                segment.attrs["trend_params"]["oscillation"]["frequency"],
                2.0,
            )

    def test_attaches_clean_window_energy_when_clean_values_are_provided(
        self,
    ) -> None:
        clean = np.ones((80, 1), dtype=np.float64)

        segments = sample_trend_parameter_aware_segments(
            rng=np.random.default_rng(23),
            target_density=0.25,
            series_length=80,
            channels=1,
            max_placement_attempts=30,
            overlap_policy="global",
            planner_cfg={
                "planner": "trend_parameter_aware_segments",
                "sine_min_cycles": 0.30,
            },
            anomaly_parameter_template={"oscillation": {"kind": "random-walk"}},
            parameter_seed=456,
            clean_values=clean,
            segment_count_range=(1, 1),
            base_min_segment_length=5,
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(len(segments), 1)
        self.assertIn("window_rms", segments[0].attrs)
        self.assertIn("window_peak", segments[0].attrs)
        self.assertEqual(segments[0].attrs["trend_oscillation_kind"], "random-walk")

    def test_rejects_nonpositive_sine_min_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "sine_min_cycles"):
            sample_trend_parameter_aware_segments(
                rng=np.random.default_rng(24),
                target_density=0.20,
                series_length=80,
                channels=1,
                max_placement_attempts=30,
                overlap_policy="global",
                planner_cfg={
                    "planner": "trend_parameter_aware_segments",
                    "sine_min_cycles": 0.0,
                },
                anomaly_parameter_template={"oscillation": {"kind": "sine"}},
                parameter_seed=789,
                clean_values=None,
                segment_count_range=(1, 1),
                base_min_segment_length=5,
                realize_parameters=_realize_parameters,
                sanitize_parameters=_sanitize_parameters,
                derive_seed=_derive_seed,
            )


if __name__ == "__main__":
    unittest.main()
