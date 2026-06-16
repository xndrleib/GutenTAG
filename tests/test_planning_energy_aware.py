import unittest

import numpy as np

from gutenTAG.tsgen.planning import sample_energy_aware_segments


class TestEnergyAwarePlanning(unittest.TestCase):
    def test_samples_high_rms_windows(self) -> None:
        clean = np.zeros((80, 1), dtype=np.float64)
        clean[40:, 0] = 5.0

        segments = sample_energy_aware_segments(
            rng=np.random.default_rng(16),
            target_density=0.20,
            series_length=80,
            channels=1,
            max_placement_attempts=20,
            overlap_policy="global",
            planner_cfg={
                "planner": "energy_aware_segments",
                "energy_metric": "rms",
                "rms_quantile": 0.75,
                "weighted_sampling": False,
                "fallback": "error",
            },
            anomaly_type="mean",
            clean_values=clean,
            segment_count_range=(2, 2),
            min_segment_length=5,
        )

        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertGreaterEqual(segment.start, 40)
            self.assertEqual(segment.attrs["energy_metric"], "rms")
            self.assertFalse(segment.attrs["energy_fallback"])
            self.assertGreater(segment.attrs["window_rms"], 0.0)

    def test_amplitude_residual_energy_ignores_constant_offset(self) -> None:
        clean = np.zeros((120, 1), dtype=np.float64)
        clean[:60, 0] = 10.0
        clean[60:, 0] = np.sin(np.linspace(0.0, 8.0 * np.pi, 60))

        segments = sample_energy_aware_segments(
            rng=np.random.default_rng(42),
            target_density=0.20,
            series_length=120,
            channels=1,
            max_placement_attempts=20,
            overlap_policy="global",
            planner_cfg={
                "planner": "energy_aware_segments",
                "energy_metric": "rms",
                "rms_quantile": 0.75,
                "weighted_sampling": False,
                "fallback": "error",
                "min_effect_delta": 0.12,
                "min_residual_scale": 1e-6,
            },
            anomaly_type="amplitude",
            clean_values=clean,
            segment_count_range=(2, 2),
            min_segment_length=12,
        )

        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertGreater(float(segment.attrs["window_residual_scale"]), 1e-6)
            self.assertEqual(segment.attrs["energy_reference"], "amplitude_residual")

    def test_rejects_missing_or_wrong_shape_clean_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean_values"):
            sample_energy_aware_segments(
                rng=np.random.default_rng(17),
                target_density=0.20,
                series_length=20,
                channels=1,
                max_placement_attempts=20,
                overlap_policy="global",
                planner_cfg={"planner": "energy_aware_segments"},
                anomaly_type="mean",
                clean_values=None,
                segment_count_range=(1, 1),
                min_segment_length=4,
            )

        with self.assertRaisesRegex(ValueError, "expected clean_values shape"):
            sample_energy_aware_segments(
                rng=np.random.default_rng(18),
                target_density=0.20,
                series_length=20,
                channels=2,
                max_placement_attempts=20,
                overlap_policy="global",
                planner_cfg={"planner": "energy_aware_segments"},
                anomaly_type="mean",
                clean_values=np.zeros((20, 1), dtype=np.float64),
                segment_count_range=(1, 1),
                min_segment_length=4,
            )


if __name__ == "__main__":
    unittest.main()
