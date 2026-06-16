import unittest

import numpy as np

from gutenTAG.tsgen.labels import (
    expand_effective_support_to_min_length,
    normalize_subsequence_length,
    resolve_label_bounds_from_effect,
)


class TestEffectiveSupportHelpers(unittest.TestCase):
    def test_strict_segment_returns_protocol_bounds(self) -> None:
        bounds = resolve_label_bounds_from_effect(
            protocol_start=10,
            protocol_end=20,
            delta=np.array([0.0, 1.0]),
            anomaly_type="mean",
            support_label_mode="strict_segment",
            support_eps_mode="absolute",
            support_eps_value=0.1,
            min_effective_label_length_non_extremum=1,
        )

        self.assertEqual(bounds, (10, 20))

    def test_relative_threshold_resolves_active_effect_bounds(self) -> None:
        bounds = resolve_label_bounds_from_effect(
            protocol_start=10,
            protocol_end=14,
            delta=np.array([0.0, 0.1, 0.5, 0.05]),
            anomaly_type="extremum",
            support_label_mode="effective",
            support_eps_mode="relative",
            support_eps_value=0.2,
            min_effective_label_length_non_extremum=1,
        )

        self.assertEqual(bounds, (12, 13))

    def test_expands_non_extremum_to_minimum_effective_length(self) -> None:
        bounds = resolve_label_bounds_from_effect(
            protocol_start=10,
            protocol_end=14,
            delta=np.array([0.0, 1.0, 0.0, 0.0]),
            anomaly_type="mean",
            support_label_mode="effective",
            support_eps_mode="absolute",
            support_eps_value=0.2,
            min_effective_label_length_non_extremum=3,
        )

        self.assertEqual(bounds, (10, 13))

    def test_expand_effective_support_resizes_delta_to_source_length(self) -> None:
        bounds = expand_effective_support_to_min_length(
            protocol_start=5,
            protocol_end=10,
            delta=np.array([0.0, 2.0, 0.0]),
            min_label_length=2,
        )

        self.assertEqual(bounds[1] - bounds[0], 2)
        self.assertGreaterEqual(bounds[0], 5)
        self.assertLessEqual(bounds[1], 10)

    def test_normalize_subsequence_length_pads_crops_and_resamples(self) -> None:
        np.testing.assert_array_equal(
            normalize_subsequence_length(
                np.array([1.0, 2.0]), expected_length=4, policy="pad"
            ),
            np.array([1.0, 2.0, 2.0, 2.0]),
        )
        np.testing.assert_array_equal(
            normalize_subsequence_length(
                np.array([1.0, 2.0, 3.0]), expected_length=2, policy="crop"
            ),
            np.array([1.0, 2.0]),
        )
        resampled = normalize_subsequence_length(
            np.array([0.0, 10.0]), expected_length=3, policy="resample"
        )
        np.testing.assert_allclose(resampled, np.array([0.0, 5.0, 10.0]))

    def test_normalize_subsequence_length_rejects_unhandled_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "length_normalization='none'"):
            normalize_subsequence_length(
                np.array([1.0]), expected_length=2, policy="none"
            )


if __name__ == "__main__":
    unittest.main()
