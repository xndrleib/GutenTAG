import numpy as np

from gutenTAG.tsgen.signal_energy import (
    peak_values_for_channel_length,
    residual_stats_values,
    rms_values_for_channel_length,
    window_residual_scale,
    window_rms_from_prefix,
)


def test_window_rms_from_prefix_matches_direct_rms() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(np.square(values))])

    assert window_rms_from_prefix(prefix, 1, 4) == np.sqrt((4.0 + 16.0 + 64.0) / 3.0)


def test_window_metric_series_match_direct_windows() -> None:
    values = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(np.square(values))])

    np.testing.assert_allclose(
        rms_values_for_channel_length(prefix, 2),
        np.asarray([np.sqrt(2.5), np.sqrt(6.5), 3.5355339059327378]),
    )
    np.testing.assert_allclose(
        peak_values_for_channel_length(np.abs(values), 2),
        np.asarray([2.0, 3.0, 4.0]),
    )


def test_residual_scale_ignores_linear_trend() -> None:
    linear = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    oscillating = np.asarray([1.0, 3.0, 1.0, 3.0], dtype=np.float64)

    assert window_residual_scale(np.array([], dtype=np.float64), "linear") == 0.0
    assert window_residual_scale(linear, "linear") < 1e-12
    assert window_residual_scale(oscillating, "mean") > 0.0


def test_residual_stats_return_scale_and_peak_per_window() -> None:
    values = np.asarray([1.0, 2.0, 1.0, 4.0], dtype=np.float64)

    scale, peak = residual_stats_values(values, length=3, center_mode="mean")

    assert scale.shape == (2,)
    assert peak.shape == (2,)
    assert np.all(scale > 0.0)
    assert np.all(peak > 0.0)
