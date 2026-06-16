import unittest

import numpy as np

from gutenTAG.tsgen.parameters import realize_parameters, sample_between


class TestParameterSampling(unittest.TestCase):
    def test_sample_between_preserves_historical_scalar_semantics(self) -> None:
        rng = np.random.default_rng(1)

        self.assertFalse(sample_between(False, True, rng))
        self.assertIn(sample_between(3, 1, rng), {1, 2, 3})
        self.assertGreaterEqual(sample_between(2.0, 1.0, rng), 1.0)

    def test_realizes_nested_ranges_and_distributions(self) -> None:
        result = realize_parameters(
            {
                "factor": [1, 3],
                "nested": {"offset": {"min": -1.0, "max": -0.5}},
                "choice": {
                    "distribution": "choice",
                    "values": [{"mode": "a"}, {"mode": "b"}],
                },
                "enabled": {"distribution": "bernoulli", "p": 1.0},
            },
            np.random.default_rng(2),
        )

        self.assertIn(result["factor"], {1, 2, 3})
        self.assertLessEqual(result["nested"]["offset"], -0.5)
        self.assertIn(result["choice"]["mode"], {"a", "b"})
        self.assertTrue(result["enabled"])

    def test_reject_if_abs_lt_falls_back_to_threshold(self) -> None:
        result = realize_parameters(
            {
                "offset": {
                    "distribution": "reject_if_abs_lt",
                    "threshold": 0.25,
                    "base": 0.0,
                }
            },
            np.random.default_rng(3),
        )

        self.assertEqual(abs(result["offset"]), 0.25)

    def test_rejects_invalid_distribution_specs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty 'values'"):
            realize_parameters(
                {"choice": {"distribution": "choice", "values": []}},
                np.random.default_rng(4),
            )

        with self.assertRaisesRegex(ValueError, "Unsupported distribution"):
            realize_parameters(
                {"value": {"distribution": "unknown"}},
                np.random.default_rng(4),
            )


if __name__ == "__main__":
    unittest.main()
