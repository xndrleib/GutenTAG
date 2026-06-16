import unittest

import gutenTAG
from gutenTAG import ts_dataset_generation
from gutenTAG.tsgen.config import (
    ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY,
    BASES_REQUIRING_EXTRA_CONFIG,
    DEFAULT_ANOMALY_OVERRIDES,
    DEFAULT_BASE_OVERRIDES,
    DEFAULT_DENSITY_RANGE,
    DEFAULT_PROFILES_PER_PAIR,
    DEFAULT_SEGMENT_COUNT_RANGE,
    DEFAULT_SPLITS,
    TSGeneratorConfig,
)


class TestRuntimeConfigDefaults(unittest.TestCase):
    def test_config_defaults_are_available_from_config_owner(self) -> None:
        self.assertEqual(DEFAULT_SPLITS, ("train", "val", "test"))
        self.assertEqual(DEFAULT_DENSITY_RANGE, (0.05, 0.10))
        self.assertEqual(DEFAULT_SEGMENT_COUNT_RANGE, (20, 50))
        self.assertEqual(DEFAULT_PROFILES_PER_PAIR, 1)
        self.assertEqual(DEFAULT_BASE_OVERRIDES["random-mode-jump"]["frequency"], 250)
        self.assertEqual(
            DEFAULT_ANOMALY_OVERRIDES["frequency"]["frequency_factor"], 2.0
        )
        self.assertEqual(BASES_REQUIRING_EXTRA_CONFIG, ("custom-input", "formula"))
        self.assertEqual(ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY, ("extremum",))

    def test_legacy_ts_dataset_generation_exports_default_names(self) -> None:
        self.assertIs(ts_dataset_generation.DEFAULT_SPLITS, DEFAULT_SPLITS)
        self.assertIs(
            ts_dataset_generation.DEFAULT_DENSITY_RANGE,
            DEFAULT_DENSITY_RANGE,
        )
        self.assertIs(
            ts_dataset_generation.DEFAULT_SEGMENT_COUNT_RANGE,
            DEFAULT_SEGMENT_COUNT_RANGE,
        )
        self.assertIs(
            ts_dataset_generation.DEFAULT_BASE_OVERRIDES,
            DEFAULT_BASE_OVERRIDES,
        )
        self.assertIs(
            ts_dataset_generation.DEFAULT_ANOMALY_OVERRIDES,
            DEFAULT_ANOMALY_OVERRIDES,
        )

    def test_legacy_ts_dataset_generation_exports_runtime_config_class(self) -> None:
        self.assertIs(ts_dataset_generation.TSGeneratorConfig, TSGeneratorConfig)
        self.assertIs(gutenTAG.TSGeneratorConfig, TSGeneratorConfig)


if __name__ == "__main__":
    unittest.main()
