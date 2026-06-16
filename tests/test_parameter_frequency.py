import unittest
from dataclasses import dataclass
from typing import Any

from gutenTAG.tsgen.parameters import apply_period_locked_frequency_policy


@dataclass
class Segment:
    attrs: dict[str, Any]


def derive_seed(seed: int, *parts: str) -> int:
    value = int(seed)
    for idx, part in enumerate(parts):
        value += (idx + 1) * sum(ord(char) for char in str(part))
    return value


class TestPeriodLockedFrequencyPolicy(unittest.TestCase):
    def test_sets_frequency_factor_from_period_ratio_candidates(self) -> None:
        result = apply_period_locked_frequency_policy(
            segment_plan=[Segment(attrs={"period_count": 4})],
            segment_params=[{"frequency_factor": 2.0}],
            planner_cfg={"period_ratio_offsets": [-1, 1]},
            parameter_seed=21,
            derive_seed=derive_seed,
        )

        self.assertIn(result[0]["frequency_factor"], {0.75, 1.25})

    def test_filters_frequency_factor_candidates_by_bounds(self) -> None:
        result = apply_period_locked_frequency_policy(
            segment_plan=[Segment(attrs={"period_count": 4})],
            segment_params=[{"frequency_factor": 2.0}],
            planner_cfg={
                "period_ratio_offsets": [-1, 1],
                "frequency_factor_bounds": [1.3, 1.2],
            },
            parameter_seed=21,
            derive_seed=derive_seed,
        )

        self.assertEqual(result[0]["frequency_factor"], 1.25)

    def test_rejects_empty_effective_offsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero 'period_ratio_offsets'"):
            apply_period_locked_frequency_policy(
                segment_plan=[Segment(attrs={"period_count": 4})],
                segment_params=[{}],
                planner_cfg={"period_ratio_offsets": [0, 0]},
                parameter_seed=21,
                derive_seed=derive_seed,
            )

    def test_rejects_segments_without_period_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "attrs\\['period_count'\\]"):
            apply_period_locked_frequency_policy(
                segment_plan=[Segment(attrs={})],
                segment_params=[{}],
                planner_cfg={"period_ratio_offsets": [-1, 1]},
                parameter_seed=21,
                derive_seed=derive_seed,
            )

    def test_rejects_bounds_without_valid_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "No valid frequency_factor"):
            apply_period_locked_frequency_policy(
                segment_plan=[Segment(attrs={"period_count": 4})],
                segment_params=[{}],
                planner_cfg={
                    "period_ratio_offsets": [-1, 1],
                    "frequency_factor_bounds": [2.0, 3.0],
                },
                parameter_seed=21,
                derive_seed=derive_seed,
            )


if __name__ == "__main__":
    unittest.main()
