import unittest

from gutenTAG.tsgen.planning import (
    resolve_minimum_segment_length,
    resolve_segment_planner,
    resolve_special_anomaly_policy,
)


class TestPlanningPolicy(unittest.TestCase):
    def test_resolve_minimum_segment_length_prefers_overrides(self) -> None:
        self.assertEqual(
            resolve_minimum_segment_length("mean", {"mean": 11}),
            11,
        )
        self.assertEqual(resolve_minimum_segment_length("trend", {}), 5)
        self.assertEqual(resolve_minimum_segment_length("unknown", {}), 1)

    def test_special_policy_adds_group_defaults(self) -> None:
        mode_policy = resolve_special_anomaly_policy("mode-correlation", {})
        self.assertEqual(mode_policy["channel_policy"], "paired-random")
        self.assertEqual(
            mode_policy["segment_planner"]["planner"],
            "mode_grid_segments",
        )

        relation_policy = resolve_special_anomaly_policy("correlation-flip", {})
        self.assertEqual(relation_policy["channel_policy"], "paired-random")

    def test_special_policy_preserves_explicit_values(self) -> None:
        policy = resolve_special_anomaly_policy(
            "mode-correlation",
            {"mode-correlation": {"channel_policy": "all"}},
        )

        self.assertEqual(policy["channel_policy"], "all")
        self.assertEqual(policy["segment_planner"]["planner"], "mode_grid_segments")

    def test_resolve_segment_planner_merges_in_priority_order(self) -> None:
        planner = resolve_segment_planner(
            anomaly_type="mean",
            segment_planner={
                "default": {
                    "planner": "uniform_segments",
                    "nested": {"left": 1, "right": 1},
                },
                "mean": {
                    "planner": "POINT_EVENTS_FROM_DENSITY",
                    "nested": {"right": 2},
                },
            },
            special_anomaly_policies={},
            planner_override={"nested": {"override": 3}},
        )

        self.assertEqual(planner["planner"], "point_events_from_density")
        self.assertEqual(planner["nested"], {"left": 1, "right": 2, "override": 3})

    def test_resolve_segment_planner_applies_special_mode_grid_default(self) -> None:
        planner = resolve_segment_planner(
            anomaly_type="mode-correlation",
            segment_planner={"default": {"planner": "uniform_segments"}},
            special_anomaly_policies={},
        )

        self.assertEqual(planner["planner"], "mode_grid_segments")


if __name__ == "__main__":
    unittest.main()
