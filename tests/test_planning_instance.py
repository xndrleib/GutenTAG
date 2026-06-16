import unittest
from typing import Any, Mapping

import numpy as np

from gutenTAG.tsgen.planning import SegmentPlanSamplingConfig, plan_instance_segments


class BaseChannel:
    frequency = None

    def get_period_size(self) -> None:
        return None


def _config(**overrides: Any) -> SegmentPlanSamplingConfig:
    values = {
        "series_length": 80,
        "channels": 2,
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


class TestInstanceSegmentPlanning(unittest.TestCase):
    def test_point_count_based_plan_normalizes_target_density(self) -> None:
        result = plan_instance_segments(
            rng=np.random.default_rng(201),
            anomaly_type="extremum",
            split="train",
            clean_values=np.zeros((80, 2)),
            channel_bos=[BaseChannel(), BaseChannel()],
            anomaly_parameter_template={},
            parameter_seed=12,
            variant_anomaly_policy={},
            variant_segment_planner={
                "planner": "point_events_from_density",
                "segment_count_range": (4, 4),
            },
            config=_config(),
            default_density_tolerance=0.05,
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(len(result.segment_plan), 4)
        self.assertEqual(result.target_density, 4.0 / 80.0)
        self.assertEqual(result.active_density_range, (4.0 / 80.0, 4.0 / 80.0))

    def test_fixed_onset_plan_uses_split_context_and_temporal_density(self) -> None:
        result = plan_instance_segments(
            rng=np.random.default_rng(202),
            anomaly_type="mean",
            split="ctx003",
            clean_values=np.zeros((80, 2)),
            channel_bos=[BaseChannel(), BaseChannel()],
            anomaly_parameter_template={},
            parameter_seed=13,
            variant_anomaly_policy={},
            variant_segment_planner={
                "planner": "fixed_first_onset_segments",
                "length": 8,
                "channel_policy": "all-channels",
            },
            config=_config(channel_policy="all-channels"),
            default_density_tolerance=0.05,
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(result.planner_cfg["requested_pre_context"], 3)
        self.assertEqual(result.target_density, 8.0 / 80.0)
        self.assertEqual(result.active_density_range, (8.0 / 80.0, 8.0 / 80.0))

    def test_variant_policy_overrides_special_policy_metadata(self) -> None:
        result = plan_instance_segments(
            rng=np.random.default_rng(203),
            anomaly_type="mode-correlation",
            split="train",
            clean_values=np.zeros((80, 2)),
            channel_bos=[BaseChannel(), BaseChannel()],
            anomaly_parameter_template={},
            parameter_seed=14,
            variant_anomaly_policy={
                "channel_policy": "all-channels",
                "density_tolerance": 0.01,
                "density_range": (0.05, 0.05),
            },
            variant_segment_planner={"planner": "uniform_segments"},
            config=_config(),
            default_density_tolerance=0.05,
            realize_parameters=_realize_parameters,
            sanitize_parameters=_sanitize_parameters,
            derive_seed=_derive_seed,
        )

        self.assertEqual(result.active_channel_policy, "all-channels")
        self.assertEqual(result.active_density_tolerance, 0.01)
        self.assertEqual(result.active_density_range, (0.05, 0.05))


if __name__ == "__main__":
    unittest.main()
