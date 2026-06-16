from typing import Any, Mapping

import numpy as np

from gutenTAG.tsgen.planning.dispatcher_context import (
    configured_density_range,
    configured_min_segment_length,
    configured_segment_count_range,
    resolve_planner_dispatch_context,
)
from gutenTAG.tsgen.planning.dispatcher_types import (
    PlannerDispatchRequest,
    SegmentPlanSamplingConfig,
)
from gutenTAG.tsgen.planning.types import SegmentPlan


def _config(**overrides: Any) -> SegmentPlanSamplingConfig:
    values = {
        "series_length": 100,
        "channels": 2,
        "max_placement_attempts": 20,
        "overlap_policy": "global",
        "channel_policy": "single-random",
        "density_range": (0.10, 0.20),
        "segment_count_range": (1, 2),
        "min_segment_length_by_anomaly": {"mean": 6},
        "segment_planner": {},
        "special_anomaly_policies": {},
    }
    values.update(overrides)
    return SegmentPlanSamplingConfig(**values)


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


def _derive_seed(seed: int, *parts: str) -> int:
    return int(seed) + len(parts)


def _request(
    config: SegmentPlanSamplingConfig,
    *,
    anomaly_type: str = "mean",
) -> PlannerDispatchRequest:
    return PlannerDispatchRequest(
        rng=np.random.default_rng(1),
        target_density=0.20,
        anomaly_type=anomaly_type,
        config=config,
        clean_values=None,
        base_period_size=None,
        period_boundaries=None,
        period_boundaries_by_channel=None,
        anomaly_parameter_template=None,
        parameter_seed=None,
        realize_parameters=_realize_parameters,
        sanitize_parameters=_sanitize_parameters,
        derive_seed=_derive_seed,
        logger=None,
        segment_factory=SegmentPlan,
    )


def test_resolve_planner_dispatch_context_applies_policy_precedence() -> None:
    config = _config(
        overlap_policy="global",
        channel_policy="single-random",
        segment_planner={
            "mean": {
                "planner": "uniform_segments",
                "overlap_policy": "per-channel",
            }
        },
        special_anomaly_policies={
            "mean": {
                "overlap_policy": "none",
                "channel_policy": "all-channels",
            }
        },
    )

    context = resolve_planner_dispatch_context(
        anomaly_type="mean",
        config=config,
        anomaly_policy=None,
        planner_cfg=None,
    )

    assert context.planner_name == "uniform_segments"
    assert context.overlap_policy == "none"
    assert context.channel_policy == "all-channels"


def test_resolve_planner_dispatch_context_copies_explicit_inputs() -> None:
    planner_cfg = {
        "planner": "uniform_segments",
        "nested": {"value": 1},
    }
    anomaly_policy = {"channel_policy": "single-random"}

    context = resolve_planner_dispatch_context(
        anomaly_type="mean",
        config=_config(),
        anomaly_policy=anomaly_policy,
        planner_cfg=planner_cfg,
    )
    planner_cfg["nested"]["value"] = 2
    anomaly_policy["channel_policy"] = "all-channels"

    assert context.active_planner_cfg["nested"]["value"] == 1
    assert context.channel_policy == "single-random"


def test_configured_ranges_and_min_length_use_effective_overrides() -> None:
    config = _config(
        density_range=(0.01, 0.02),
        segment_count_range=(1, 1),
        min_segment_length_by_anomaly={"mean": 5},
    )
    context = resolve_planner_dispatch_context(
        anomaly_type="mean",
        config=config,
        anomaly_policy={
            "density_range": (0.30, 0.40),
            "segment_count_range": (4, 5),
            "min_segment_length": 9,
        },
        planner_cfg={
            "planner": "uniform_segments",
            "density_range": (0.20, 0.30),
            "segment_count_range": (2, 3),
            "min_segment_length": 7,
        },
    )
    request = _request(config)

    assert configured_density_range(request, context) == (0.30, 0.40)
    assert configured_segment_count_range(request, context) == (4, 5)
    assert configured_min_segment_length(request, context) == 9
