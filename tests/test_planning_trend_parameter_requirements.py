import numpy as np

from gutenTAG.tsgen.planning.trend_parameter_requirements import (
    build_trend_segment_requirements,
    fit_trend_requirements_to_budget,
    sample_segment_lengths_with_minima,
)
from gutenTAG.tsgen.planning.trend_parameter_types import (
    TrendPlannerSettings,
    TrendSegmentRequirements,
)


def _derive_seed(seed: int, *parts: str) -> int:
    return int(seed) + int(parts[-1])


def _realize_parameters(template, _rng):
    return dict(template)


def _sanitize_parameters(_anomaly_type, params):
    return dict(params)


def test_build_trend_segment_requirements_derives_sine_minimum() -> None:
    requirements = build_trend_segment_requirements(
        n_segments=1,
        anomaly_parameter_template={
            "oscillation": {
                "kind": "sine",
                "frequency": 2.0,
            }
        },
        parameter_seed=123,
        base_min_segment_length=5,
        series_length=200,
        settings=TrendPlannerSettings(
            sine_min_cycles=0.30,
            random_walk_min_segment_length=8,
        ),
        realize_parameters=_realize_parameters,
        sanitize_parameters=_sanitize_parameters,
        derive_seed=_derive_seed,
    )

    assert requirements.min_lengths == [15]
    assert requirements.attrs[0]["trend_oscillation_kind"] == "sine"
    assert requirements.attrs[0]["trend_sine_frequency"] == 2.0
    assert requirements.attrs[0]["trend_min_segment_length"] == 15


def test_fit_trend_requirements_to_budget_drops_infeasible_minima() -> None:
    requirements = TrendSegmentRequirements(
        min_lengths=[80, 30, 20],
        attrs=[
            {"trend_oscillation_kind": "sine"},
            {"trend_oscillation_kind": "random-walk"},
            {"trend_oscillation_kind": "none"},
        ],
    )

    fit_trend_requirements_to_budget(
        requirements=requirements,
        series_length=100,
        target_points=90,
        logger=None,
    )

    assert requirements.min_lengths == [30, 20]
    assert [item["trend_oscillation_kind"] for item in requirements.attrs] == [
        "random-walk",
        "none",
    ]


def test_fit_trend_requirements_to_budget_relaxes_single_minimum() -> None:
    requirements = TrendSegmentRequirements(
        min_lengths=[80],
        attrs=[{"trend_oscillation_kind": "sine"}],
    )

    fit_trend_requirements_to_budget(
        requirements=requirements,
        series_length=100,
        target_points=30,
        logger=None,
    )

    assert requirements.min_lengths == [30]
    assert requirements.attrs[0]["trend_min_segment_length_original"] == 80
    assert requirements.attrs[0]["trend_min_segment_length_relaxed"] == 30


def test_sample_segment_lengths_with_minima_preserves_minima_and_budget() -> None:
    lengths = sample_segment_lengths_with_minima(
        rng=np.random.default_rng(42),
        target_points=20,
        minimum_lengths=[5, 10],
        series_length=100,
    )

    assert len(lengths) == 2
    assert sum(lengths) == 20
    assert lengths[0] >= 5
    assert lengths[1] >= 10
