import unittest
from typing import Any, Mapping

import numpy as np

from gutenTAG.tsgen.planning import (
    SegmentPlanSamplingConfig,
    build_segment_plan_sampling_config,
    sample_segment_plan,
)


def _config(**overrides: Any) -> SegmentPlanSamplingConfig:
    values = {
        "series_length": 80,
        "channels": 3,
        "max_placement_attempts": 50,
        "overlap_policy": "global",
        "channel_policy": "single-random",
        "density_range": (0.10, 0.20),
        "segment_count_range": (1, 1),
        "min_segment_length_by_anomaly": {},
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


class TestSegmentPlanDispatcher(unittest.TestCase):
    def test_build_segment_plan_sampling_config_reads_runtime_fields(self) -> None:
        class RuntimeConfig:
            length = 81
            channels = 4
            max_placement_attempts = 99
            overlap_policy = "none"
            channel_policy = "all-channels"
            density_range = (0.05, 0.15)
            segment_count_range = (2, 3)
            min_segment_length_by_anomaly = {"mean": 7}
            segment_planner = {"mean": {"planner": "uniform_segments"}}
            special_anomaly_policies = {"mode-correlation": {}}

        config = build_segment_plan_sampling_config(RuntimeConfig())

        self.assertEqual(config.series_length, 81)
        self.assertEqual(config.channels, 4)
        self.assertEqual(config.max_placement_attempts, 99)
        self.assertEqual(config.channel_policy, "all-channels")
        self.assertEqual(config.min_segment_length_by_anomaly, {"mean": 7})

    def test_uniform_dispatch_applies_channel_policy(self) -> None:
        segments = sample_segment_plan(
            rng=np.random.default_rng(101),
            target_density=0.20,
            anomaly_type="mean",
            config=_config(channel_policy="all-channels"),
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(len(segments), 3)
        self.assertEqual([segment.channel for segment in segments], [0, 1, 2])
        for segment in segments:
            self.assertEqual(segment.attrs["group_channels"], [0, 1, 2])

    def test_point_event_dispatch_supports_count_based_policy(self) -> None:
        segments = sample_segment_plan(
            rng=np.random.default_rng(102),
            target_density=0.20,
            anomaly_type="extremum",
            config=_config(),
            planner_cfg={
                "planner": "point_events_from_density",
                "segment_count_range": (3, 3),
            },
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(len(segments), 3)
        self.assertTrue(all(segment.length == 1 for segment in segments))

    def test_trend_dispatch_requires_parameter_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires anomaly_parameter_template"):
            sample_segment_plan(
                rng=np.random.default_rng(103),
                target_density=0.20,
                anomaly_type="trend",
                config=_config(),
                planner_cfg={"planner": "trend_parameter_aware_segments"},
                realize_parameters=_realize_parameters,
                sanitize_parameters=_sanitize_parameters,
                derive_seed=_derive_seed,
            )


if __name__ == "__main__":
    unittest.main()
