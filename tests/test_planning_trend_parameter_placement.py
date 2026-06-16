import numpy as np
import pytest

from gutenTAG.tsgen.planning.trend_parameter_placement import place_trend_segments


def test_place_trend_segments_uses_fallback_without_overlap() -> None:
    attrs = [
        {"trend_params": {"id": 1}},
        {"trend_params": {"id": 2}},
    ]

    segments = place_trend_segments(
        rng=np.random.default_rng(7),
        lengths=[5, 5],
        attrs=attrs,
        series_length=20,
        channels=1,
        max_placement_attempts=0,
        overlap_policy="global",
    )

    assert len(segments) == 2
    occupied = np.zeros(20, dtype=np.int8)
    for segment in segments:
        assert int(occupied[segment.start : segment.end].sum()) == 0
        occupied[segment.start : segment.end] = 1
    assert [segment.attrs["trend_params"]["id"] for segment in segments] == [1, 2]


def test_place_trend_segments_deep_copies_attrs() -> None:
    attrs = [{"trend_params": {"oscillation": {"kind": "sine"}}}]

    segments = place_trend_segments(
        rng=np.random.default_rng(8),
        lengths=[5],
        attrs=attrs,
        series_length=20,
        channels=1,
        max_placement_attempts=0,
        overlap_policy="global",
    )
    attrs[0]["trend_params"]["oscillation"]["kind"] = "changed"

    assert segments[0].attrs["trend_params"]["oscillation"]["kind"] == "sine"


def test_place_trend_segments_raises_when_slot_is_infeasible() -> None:
    with pytest.raises(ValueError, match="without overlap"):
        place_trend_segments(
            rng=np.random.default_rng(9),
            lengths=[11],
            attrs=[{"trend_params": {}}],
            series_length=10,
            channels=1,
            max_placement_attempts=0,
            overlap_policy="global",
        )
