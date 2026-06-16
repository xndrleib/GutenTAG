import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gutenTAG.generator.base_generation import (
    apply_split_phase_shift,
    compose_base_parameters_per_channel,
    generate_base_instance_series,
    generate_base_channels,
    prepare_base_instance,
    realize_base_channel_parameters,
    sanitize_generated_base_channel,
    stack_channel_timeseries,
)
from gutenTAG.tsgen.parameters import realize_parameters


@dataclass
class BaseChannel:
    timeseries: Optional[np.ndarray]
    noise: Optional[np.ndarray] = None
    trend_series: Optional[np.ndarray] = None


class FakeBaseOscillation:
    def __init__(self, _base_kind: str, length: int, value: float = 0.0) -> None:
        self.length = int(length)
        self.value = float(value)
        self.timeseries: Optional[np.ndarray] = None
        self.noise: Optional[np.ndarray] = None
        self.trend_series: Optional[np.ndarray] = None
        self.offset: Optional[float] = None

    def generate_timeseries_and_variations(self, _ctx: object) -> None:
        self.timeseries = np.full(self.length, self.value, dtype=np.float64)


class TestBaseGenerationHelpers(unittest.TestCase):
    def test_realize_base_channel_parameters_honors_shared_specs(self) -> None:
        result = realize_base_channel_parameters(
            template={
                "shared": {
                    "shared_across_channels": True,
                    "min": 5,
                    "max": 5,
                },
                "per_channel": {"min": 1, "max": 1},
            },
            channels=3,
            rng=np.random.default_rng(1),
            realize_parameters=realize_parameters,
        )

        self.assertEqual(len(result), 3)
        self.assertTrue(all(params["shared"] == 5 for params in result))
        self.assertTrue(all(params["per_channel"] == 1 for params in result))

    def test_compose_base_parameters_per_channel_adds_length_and_overrides(
        self,
    ) -> None:
        result = compose_base_parameters_per_channel(
            base_parameters={"frequency": 2.0, "amplitude": 1.0},
            base_channel_parameters=[{"amplitude": 3.0}],
            channels=2,
            length=50,
        )

        self.assertEqual(result[0]["frequency"], 2.0)
        self.assertEqual(result[0]["amplitude"], 3.0)
        self.assertEqual(result[0]["length"], 50)
        self.assertEqual(result[1]["amplitude"], 1.0)
        self.assertEqual(result[1]["length"], 50)

    def test_apply_split_phase_shift_returns_shift_metadata(self) -> None:
        shifted, info = apply_split_phase_shift(
            base_parameters_per_channel=[{"phase": 5.5}, {"amplitude": 1.0}],
            split="test",
            split_phase_shift={
                "enabled": True,
                "values": {"test": 1.0},
                "phase_modulo": 6.0,
            },
        )

        self.assertAlmostEqual(shifted[0]["phase"], 0.5)
        self.assertEqual(info["channels_with_phase"], [0])
        self.assertTrue(info["phase_shift_applied"])

    def test_generate_base_channels_uses_factory_and_sanitizes_outputs(self) -> None:
        channels = generate_base_channels(
            base_kind="fake",
            base_parameters_per_channel=[
                {"length": 3, "value": 1.0},
                {"length": 3, "value": 2.0},
            ],
            seed=5,
            expected_length=3,
            base_factory=FakeBaseOscillation,
        )

        self.assertEqual(len(channels), 2)
        np.testing.assert_allclose(channels[0].timeseries, np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(channels[0].noise, np.zeros(3))
        np.testing.assert_allclose(channels[1].trend_series, np.zeros(3))

    def test_sanitize_generated_base_channel_rejects_missing_or_wrong_length(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "produced no timeseries"):
            sanitize_generated_base_channel(
                base_kind="fake",
                bo=BaseChannel(timeseries=None),
                expected_length=3,
            )

        with self.assertRaisesRegex(ValueError, "produced length 2"):
            sanitize_generated_base_channel(
                base_kind="fake",
                bo=BaseChannel(timeseries=np.array([1.0, 2.0])),
                expected_length=3,
            )

    def test_stack_channel_timeseries_returns_float_matrix(self) -> None:
        result = stack_channel_timeseries(
            [
                BaseChannel(timeseries=np.array([1, 2], dtype=np.int64)),
                BaseChannel(timeseries=np.array([3, 4], dtype=np.int64)),
            ]
        )

        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, np.array([[1.0, 3.0], [2.0, 4.0]]))

    def test_generate_base_instance_series_returns_base_and_observed_values(
        self,
    ) -> None:
        result = generate_base_instance_series(
            base_kind="fake",
            base_parameters_per_channel=[
                {"length": 3, "value": 1.0},
                {"length": 3, "value": 2.0},
            ],
            seed=7,
            shared_noise_seed=11,
            base_channel_correlation={},
            expected_length=3,
            base_factory=FakeBaseOscillation,
        )

        self.assertEqual(len(result.channel_bos), 2)
        np.testing.assert_allclose(
            result.base_values,
            np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
        )
        np.testing.assert_allclose(result.observed_values, result.base_values)

    def test_prepare_base_instance_resolves_params_and_generates_series(self) -> None:
        result = prepare_base_instance(
            base_kind="fake",
            split="train",
            seeds={
                "base_seed": 7,
                "base_shared_noise_seed": 11,
                "base_params_seed": 13,
                "base_channel_params_seed": 17,
            },
            base_parameter_template={},
            base_channel_parameter_template={},
            fixed_base_parameters={"value": 1.0},
            fixed_base_channel_parameters=[{"value": 2.0}, {"value": 3.0}],
            base_parameter_policy="fixed_per_variant",
            base_channel_parameter_policy="fixed_per_variant",
            channels=2,
            length=4,
            split_phase_shift={},
            base_channel_correlation={},
            realize_parameters=realize_parameters,
            base_factory=FakeBaseOscillation,
        )

        self.assertEqual(result.base_parameters, {"value": 1.0})
        self.assertEqual(
            result.base_channel_parameters, [{"value": 2.0}, {"value": 3.0}]
        )
        self.assertEqual(result.base_parameters_per_channel[0]["length"], 4)
        self.assertEqual(result.split_phase_shift_info["split"], "train")
        np.testing.assert_allclose(
            result.series.base_values,
            np.array(
                [
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
