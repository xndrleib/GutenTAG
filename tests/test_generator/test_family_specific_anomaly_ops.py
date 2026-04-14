import unittest
from types import SimpleNamespace

import numpy as np
from scipy.signal import find_peaks

from gutenTAG.anomalies.types import AnomalyProtocol, LabelRange
from gutenTAG.anomalies.types.frequency import AnomalyFrequency
from gutenTAG.anomalies.types.pattern_shift import AnomalyPatternShift
from gutenTAG.anomalies.types.variance import AnomalyVariance
from gutenTAG.base_oscillations import ECG
from gutenTAG.base_oscillations.ecg import ecg
from gutenTAG.generator.correlation_geometry import correlation_flip_window
from gutenTAG.generator.multivariate_ops import (
    matched_coupling_window,
    residualized_correlation_flip_window,
    residualized_matched_coupling_window,
)


def _local_linear_trend(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, series.shape[0], dtype=np.float64)
    coeff = np.polyfit(x, series, deg=1)
    return np.polyval(coeff, x).astype(np.float64)


class TestFamilySpecificAnomalyOps(unittest.TestCase):
    def test_frequency_ecg_beat_aware_window_changes_peak_spacing(self) -> None:
        full = ecg(np.random.default_rng(42), length=320, frequency=8.0, amplitude=1.0)
        start, end = 96, 176
        reference = full[start:end]
        ecg_bo = ECG(length=320, frequency=8.0, amplitude=1.0)
        ecg_bo.timeseries = full
        warped = AnomalyFrequency._beat_aware_reference_window(
            ecg=ecg_bo,
            full_reference=full,
            start=start,
            end=end,
            frequency_factor=0.78,
        )
        blended = AnomalyFrequency.blend_with_reference(
            warped,
            reference,
            AnomalyFrequency._motif_transition_length(reference.shape[0]),
        )
        self.assertEqual(blended.shape, reference.shape)
        self.assertAlmostEqual(float(blended[0]), float(reference[0]), places=8)
        self.assertAlmostEqual(float(blended[-1]), float(reference[-1]), places=8)
        self.assertGreater(float(np.max(np.abs(blended - reference))), 0.05)
        self.assertLessEqual(
            float(np.max(np.abs(blended))),
            float(np.max(np.abs(reference))) + 1e-8,
        )
        delta = blended - reference
        self.assertLess(float(np.min(delta)), -0.01)
        self.assertGreater(float(np.max(delta)), 0.01)
        self.assertLess(
            float(abs(np.mean(delta))),
            0.4 * float(np.max(np.abs(delta))),
        )
        prominence = max(1e-6, 0.12 * float(np.ptp(reference)))
        base_peaks, _ = find_peaks(reference, prominence=prominence, distance=4)
        warped_peaks, _ = find_peaks(blended, prominence=prominence, distance=4)
        self.assertGreaterEqual(base_peaks.size, 4)
        self.assertGreaterEqual(warped_peaks.size, 3)
        overlap = min(base_peaks.size, warped_peaks.size)
        self.assertGreater(
            float(np.mean(np.abs(base_peaks[:overlap] - warped_peaks[:overlap]))), 0.5
        )
        np.testing.assert_allclose(
            np.quantile(blended, [0.1, 0.5, 0.9]),
            np.quantile(reference, [0.1, 0.5, 0.9]),
            atol=0.08,
        )

    def test_frequency_ecg_beat_aware_falls_back_without_detectable_peaks(self) -> None:
        full = np.linspace(-0.1, 0.1, 128, dtype=np.float64)
        ecg_bo = ECG(length=128, frequency=8.0, amplitude=1.0)
        ecg_bo.timeseries = full
        warped = AnomalyFrequency._beat_aware_reference_window(
            ecg=ecg_bo,
            full_reference=full,
            start=32,
            end=96,
            frequency_factor=0.8,
        )
        self.assertEqual(warped.shape[0], 64)
        self.assertTrue(np.all(np.isfinite(warped)))

    def test_frequency_ecg_generate_warps_visible_signal_not_only_raw_base(self) -> None:
        full = ecg(np.random.default_rng(7), length=320, frequency=8.0, amplitude=1.0)
        ecg_bo = ECG(length=320, frequency=8.0, amplitude=1.0)
        ecg_bo.timeseries = full
        ecg_bo.noise = 0.03 * np.sin(np.linspace(0.0, 6.0 * np.pi, 320, dtype=np.float64))
        ecg_bo.trend_series = np.linspace(-0.05, 0.05, 320, dtype=np.float64)
        ecg_bo.offset = 0.02
        start, end = 96, 176
        protocol = AnomalyProtocol(
            start=start,
            end=end,
            channel=0,
            ctx=SimpleNamespace(
                base_oscillation=ecg_bo,
                base_oscillation_kind=ECG.KIND,
                rng=np.random.default_rng(0),
            ),
            labels=LabelRange(start=start, length=end - start),
        )
        anomaly = AnomalyFrequency(
            AnomalyFrequency.get_parameter_class()(frequency_factor=0.78)
        )
        generated = anomaly.generate(protocol)
        self.assertEqual(len(generated.subsequences), 1)
        base_subsequence = generated.subsequences[0]
        variation = ecg_bo.noise[start:end] + ecg_bo.trend_series[start:end] + float(ecg_bo.offset)
        visible_reference = (
            ecg_bo.timeseries[start:end]
            + variation
        )
        visible_anomalous = base_subsequence + variation
        delta = visible_anomalous - visible_reference
        self.assertGreater(float(np.max(np.abs(delta))), 0.05)
        self.assertLess(float(np.min(delta)), -0.01)
        self.assertGreater(float(np.max(delta)), 0.01)
        self.assertLess(
            float(abs(np.mean(delta))),
            0.5 * float(np.max(np.abs(delta))),
        )

    def test_variance_reference_scale_is_boosted_for_smooth_trend(self) -> None:
        x = np.linspace(0.0, 1.0, 96, dtype=np.float64)
        clean_segment = 0.8 * x + 0.01 * np.sin(6.0 * np.pi * x)
        legacy_scale = max(float(np.std(clean_segment)), 0.25, 1e-6)
        boosted_scale = AnomalyVariance._estimate_reference_scale(
            clean_segment=clean_segment,
            base_kind="polynomial",
            amplitude=1.0,
        )
        self.assertGreater(boosted_scale, legacy_scale)
        self.assertGreaterEqual(boosted_scale, 0.45)

    def test_residualized_matched_coupling_preserves_linear_trend_better_than_raw(self) -> None:
        x = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
        reference = 0.6 * x + 0.15 * np.sin(5.0 * np.pi * x)
        anchor = -0.3 * x + 0.12 * np.cos(5.0 * np.pi * x)
        raw_candidate = matched_coupling_window(reference, anchor, coupling_strength=-0.95)
        residualized_candidate = residualized_matched_coupling_window(
            reference, anchor, coupling_strength=-0.95
        )
        reference_trend = _local_linear_trend(reference)
        raw_trend_error = float(
            np.mean(np.abs(_local_linear_trend(raw_candidate) - reference_trend))
        )
        residualized_trend_error = float(
            np.mean(np.abs(_local_linear_trend(residualized_candidate) - reference_trend))
        )
        self.assertLess(residualized_trend_error, raw_trend_error)
        self.assertGreater(float(np.max(np.abs(residualized_candidate - reference))), 0.01)

    def test_residualized_correlation_flip_preserves_linear_trend_better_than_raw(self) -> None:
        x = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
        reference = 0.4 * x + 0.10 * np.sin(7.0 * np.pi * x)
        anchor = -0.2 * x + 0.11 * np.cos(7.0 * np.pi * x)
        raw_candidate = correlation_flip_window(reference, anchor, target_correlation=-0.9)
        residualized_candidate = residualized_correlation_flip_window(
            reference, anchor, target_correlation=-0.9
        )
        reference_trend = _local_linear_trend(reference)
        raw_trend_error = float(
            np.mean(np.abs(_local_linear_trend(raw_candidate) - reference_trend))
        )
        residualized_trend_error = float(
            np.mean(np.abs(_local_linear_trend(residualized_candidate) - reference_trend))
        )
        self.assertLess(residualized_trend_error, raw_trend_error)
        self.assertGreater(float(np.max(np.abs(residualized_candidate - reference))), 0.01)

    def test_pattern_shift_finalize_candidate_locks_edges(self) -> None:
        baseline = np.r_[np.ones(24), -np.ones(24)].astype(np.float64)
        candidate = np.roll(baseline, 4)
        finalized = AnomalyPatternShift._finalize_candidate(
            candidate,
            baseline,
            transition_window=10,
            edge_lock_width=6,
        )
        np.testing.assert_allclose(finalized[:3], baseline[:3], atol=1e-12)
        np.testing.assert_allclose(finalized[-3:], baseline[-3:], atol=1e-12)
        self.assertGreater(float(np.max(np.abs(finalized[8:-8] - baseline[8:-8]))), 0.1)


if __name__ == "__main__":
    unittest.main()
