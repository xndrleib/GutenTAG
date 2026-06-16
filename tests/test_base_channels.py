import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gutenTAG.generator.base_channels import (
    apply_shared_noise_correlation,
    apply_variations,
    compose_channel_noise_window,
    compose_channel_window_with_variations,
    replace_channel_noise_window,
    replace_channel_window_with_variations,
    shared_noise_weight,
)


@dataclass
class BaseChannel:
    noise: Optional[np.ndarray] = None
    trend_series: Optional[np.ndarray] = None
    offset: Optional[float] = None


class TestBaseChannelHelpers(unittest.TestCase):
    def test_shared_noise_weight_is_clipped(self) -> None:
        self.assertEqual(shared_noise_weight({"shared_noise_weight": -1.0}), 0.0)
        self.assertEqual(shared_noise_weight({"shared_noise_weight": 2.0}), 1.0)
        self.assertEqual(shared_noise_weight({"shared_noise_weight": 0.25}), 0.25)

    def test_apply_variations_adds_noise_trend_and_offset(self) -> None:
        base = np.zeros((3, 1), dtype=np.float64)
        bo = BaseChannel(
            noise=np.array([1.0, 2.0, 3.0]),
            trend_series=np.array([0.5, 0.5, 0.5]),
            offset=2.0,
        )

        result = apply_variations(base, [bo])

        np.testing.assert_allclose(result[:, 0], np.array([3.5, 4.5, 5.5]))

    def test_compose_and_replace_observed_window_with_clipping(self) -> None:
        base = np.zeros((5, 1), dtype=np.float64)
        bo = BaseChannel(noise=np.ones(5), trend_series=np.zeros(5), offset=1.0)

        window = compose_channel_window_with_variations(
            base=base,
            bo=bo,
            channel=0,
            start=-2,
            end=3,
            series_length=5,
        )
        np.testing.assert_allclose(window, np.array([2.0, 2.0, 2.0]))

        replace_channel_window_with_variations(
            base=base,
            bo=bo,
            channel=0,
            start=1,
            end=3,
            target_observed=np.array([10.0, 20.0]),
            series_length=5,
        )
        updated = compose_channel_window_with_variations(
            base=base,
            bo=bo,
            channel=0,
            start=1,
            end=3,
            series_length=5,
        )
        np.testing.assert_allclose(updated, np.array([10.0, 20.0]))

    def test_noise_window_helpers_create_and_replace_noise(self) -> None:
        bo = BaseChannel(noise=None)

        empty = compose_channel_noise_window(bo=bo, start=-2, end=3, series_length=5)
        np.testing.assert_allclose(empty, np.zeros(3, dtype=np.float64))

        replace_channel_noise_window(
            bo=bo,
            start=1,
            end=3,
            target_noise=np.array([4.0, 5.0]),
            series_length=5,
        )
        np.testing.assert_allclose(bo.noise, np.array([0.0, 4.0, 5.0, 0.0, 0.0]))

    def test_apply_shared_noise_correlation_sets_latent_components(self) -> None:
        bos = [
            BaseChannel(noise=np.array([0.0, 1.0, 2.0, 3.0])),
            BaseChannel(noise=np.array([3.0, 2.0, 1.0, 0.0])),
        ]

        apply_shared_noise_correlation(
            channel_bos=bos,
            seed=17,
            base_channel_correlation={"shared_noise_weight": 0.6},
        )

        for bo in bos:
            self.assertEqual(bo.noise.dtype, np.float64)
            self.assertEqual(bo._shared_noise_weight, 0.6)
            self.assertAlmostEqual(bo._residual_noise_weight, 0.8)
            self.assertTrue(hasattr(bo, "_idio_noise_component"))
            self.assertTrue(hasattr(bo, "_shared_noise_component"))

    def test_apply_shared_noise_correlation_rejects_mismatched_lengths(self) -> None:
        bos = [
            BaseChannel(noise=np.array([0.0, 1.0])),
            BaseChannel(noise=np.array([0.0, 1.0, 2.0])),
        ]

        with self.assertRaisesRegex(ValueError, "different lengths"):
            apply_shared_noise_correlation(
                channel_bos=bos,
                seed=17,
                base_channel_correlation={"shared_noise_weight": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
