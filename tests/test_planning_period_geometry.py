import numpy as np

from gutenTAG.tsgen.planning.period_geometry import (
    resolve_channel_boundaries,
    sanitize_period_boundaries,
)


def test_sanitize_period_boundaries_clips_sorts_and_adds_edges() -> None:
    boundaries = sanitize_period_boundaries([40, -5, 20, 20, 80], series_length=100)

    assert boundaries is not None
    np.testing.assert_array_equal(
        boundaries,
        np.asarray([0, 20, 40, 80, 100], dtype=int),
    )


def test_sanitize_period_boundaries_returns_none_for_missing_values() -> None:
    assert sanitize_period_boundaries(None, series_length=100) is None
    assert sanitize_period_boundaries([], series_length=100) is None


def test_resolve_channel_boundaries_prefers_channel_specific_boundaries() -> None:
    boundaries = resolve_channel_boundaries(
        series_length=100,
        channels=3,
        base_period_size=10,
        period_boundaries=[0, 50, 100],
        period_boundaries_by_channel={1: [0, 25, 50, 100]},
    )

    assert list(boundaries) == [1]
    np.testing.assert_array_equal(
        boundaries[1],
        np.asarray([0, 25, 50, 100], dtype=int),
    )


def test_resolve_channel_boundaries_uses_base_period_size_fallback() -> None:
    boundaries = resolve_channel_boundaries(
        series_length=95,
        channels=2,
        base_period_size=30,
        period_boundaries=None,
        period_boundaries_by_channel=None,
    )

    assert sorted(boundaries) == [0, 1]
    np.testing.assert_array_equal(
        boundaries[0],
        np.asarray([0, 30, 60, 90, 95], dtype=int),
    )
    np.testing.assert_array_equal(boundaries[1], boundaries[0])
