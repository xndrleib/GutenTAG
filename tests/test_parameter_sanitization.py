import unittest

from gutenTAG.tsgen.parameters import sanitize_anomaly_parameters


class TestParameterSanitization(unittest.TestCase):
    def test_sanitizes_amplitude_parameters(self) -> None:
        result = sanitize_anomaly_parameters(
            "amplitude",
            {
                "amplitude_factor": "2.5",
                "center_mode": "LINEAR",
                "min_effect_delta": -1,
                "min_residual_scale": -2,
                "transition_length": -4,
            },
        )

        self.assertEqual(result["amplitude_factor"], 2.5)
        self.assertEqual(result["center_mode"], "linear")
        self.assertEqual(result["min_effect_delta"], 0.0)
        self.assertEqual(result["min_residual_scale"], 0.0)
        self.assertEqual(result["transition_length"], 0)

    def test_sanitizes_pattern_shift_parameters(self) -> None:
        result = sanitize_anomaly_parameters(
            "pattern-shift",
            {
                "transition_window": -3,
                "shift_by": 10,
                "crossfade_mode": "COSINE",
                "min_effect_delta": -1,
            },
        )

        self.assertEqual(result["transition_window"], 3)
        self.assertEqual(result["shift_by"], 3)
        self.assertEqual(result["crossfade_mode"], "cosine")
        self.assertEqual(result["min_effect_delta"], 0.0)

    def test_sanitizes_zero_pattern_shift_to_nonzero_shift(self) -> None:
        result = sanitize_anomaly_parameters(
            "pattern-shift", {"transition_window": 4, "shift_by": 0}
        )

        self.assertEqual(result["shift_by"], 1)

    def test_sanitizes_extremum_parameters(self) -> None:
        result = sanitize_anomaly_parameters(
            "extremum", {"context_window": -5, "min": 1, "local": 0}
        )

        self.assertEqual(result["context_window"], 5)
        self.assertTrue(result["min"])
        self.assertFalse(result["local"])

    def test_sanitizes_pattern_strength_parameters(self) -> None:
        result = sanitize_anomaly_parameters(
            "pattern",
            {
                "min_effect_delta": -1,
                "min_window_ptp": -2,
                "adaptive_blend": 1,
                "blend_strength": -3,
            },
        )

        self.assertEqual(result["min_effect_delta"], 0.0)
        self.assertEqual(result["min_window_ptp"], 0.0)
        self.assertTrue(result["adaptive_blend"])
        self.assertEqual(result["blend_strength"], 0.0)


if __name__ == "__main__":
    unittest.main()
