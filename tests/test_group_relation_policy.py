import unittest

import numpy as np

from gutenTAG.generator.group_relation_policy import (
    effective_coupling_strength,
    effective_latent_transition_length,
    effective_relation_target,
    latent_shared_noise_attrs,
    prefer_observed_relation_rewrite,
)


class BaseChannel:
    def __init__(self, kind: str = "sine") -> None:
        self.kind = kind
        self.noise = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        self._idio_noise_component = np.array([1.0, 2.0, 3.0, 4.0])
        self._shared_noise_component = np.array([4.0, 3.0, 2.0, 1.0])
        self._noise_mean = 0.5
        self._shared_noise_weight = 0.6

    def get_base_oscillation_kind(self) -> str:
        return self.kind


class TestGroupRelationPolicy(unittest.TestCase):
    def test_latent_shared_noise_attrs_returns_window_and_residual_fallback(
        self,
    ) -> None:
        bo = BaseChannel()

        result = latent_shared_noise_attrs(bo, 1, 3)

        assert result is not None
        idio, shared, mean, shared_weight, residual_weight = result
        np.testing.assert_allclose(idio, np.array([2.0, 3.0]))
        np.testing.assert_allclose(shared, np.array([3.0, 2.0]))
        self.assertEqual(mean, 0.5)
        self.assertEqual(shared_weight, 0.6)
        self.assertAlmostEqual(residual_weight, 0.8)

    def test_latent_shared_noise_attrs_rejects_missing_components(self) -> None:
        bo = BaseChannel()
        bo._shared_noise_component = None

        self.assertIsNone(latent_shared_noise_attrs(bo, 1, 3))

    def test_shared_noise_sine_strengthens_relation_targets(self) -> None:
        bo = BaseChannel("shared-noise-sine")

        self.assertEqual(effective_latent_transition_length(bo, 20), 6)
        self.assertEqual(effective_latent_transition_length(bo, 0), 1)
        self.assertEqual(
            effective_relation_target(bo=bo, target_correlation=-0.2), -0.92
        )
        self.assertEqual(
            effective_coupling_strength(bo=bo, coupling_strength=0.3), 0.95
        )
        self.assertTrue(prefer_observed_relation_rewrite(bo))

    def test_default_base_keeps_relation_policy_values(self) -> None:
        bo = BaseChannel("sine")

        self.assertEqual(effective_latent_transition_length(bo, -3), 0)
        self.assertEqual(
            effective_relation_target(bo=bo, target_correlation=-0.2), -0.2
        )
        self.assertEqual(
            effective_relation_target(bo=bo, target_correlation=None), None
        )
        self.assertEqual(effective_coupling_strength(bo=bo, coupling_strength=0.3), 0.3)
        self.assertFalse(prefer_observed_relation_rewrite(bo))


if __name__ == "__main__":
    unittest.main()
