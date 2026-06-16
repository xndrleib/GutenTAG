from unittest import TestCase

from gutenTAG.config.parser import ConfigParser
from gutenTAG.utils.compatibility import Compatibility

VALIDATED_COMPATIBILITY_CASES = (
    ("extremum", "cosine", "validated", True),
    ("extremum", "ecg", "recommended", True),
    ("covariance-change", "polynomial", "validated", True),
    ("correlation-flip", "polynomial", "validated", True),
    ("covariance-change", "shared-noise-sine", "validated", True),
    ("correlation-flip", "cosine", "recommended", True),
    ("correlation-flip", "cosine", "validated", False),
    ("correlation-flip", "shared-noise-sine", "validated", True),
    ("shared-factor-break", "shared-noise-sine", "recommended", True),
    ("shared-factor-break", "shared-noise-sine", "validated", False),
    ("lag-synchronization", "sine", "validated", True),
    ("covariance-change", "polynomial", "recommended", True),
)


class TestCompatibility(TestCase):
    def setUp(self) -> None:
        self.config = {
            "timeseries": [
                {
                    "name": "ecg",
                    "length": 100,
                    "base-oscillations": [{"kind": "ecg"}],
                    "anomalies": [
                        {
                            "position": "middle",
                            "length": 40,
                            "channel": 0,
                            "kinds": [
                                {
                                    "kind": "pattern-shift",
                                    "shift_by": 5,
                                    "transition_window": 10,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        self.breaking_config = {
            "timeseries": [
                {
                    "name": "ecg",
                    "length": 100,
                    "base-oscillations": [{"kind": "ecg"}],
                    "anomalies": [
                        {
                            "position": "middle",
                            "length": 40,
                            "channel": 0,
                            "kinds": [
                                {
                                    "kind": "pattern-shift",
                                    "shift_by": 5,
                                    "transition_window": 10,
                                },
                                {
                                    "kind": "mode-correlation",
                                },
                            ],
                        }
                    ],
                }
            ]
        }

    def test_compatibility(self):
        ConfigParser().parse(self.config)

    def test_compatibility_breaks(self):
        with self.assertRaises(ValueError):
            ConfigParser().parse(self.breaking_config)

    def test_compatibility_breaks_and_ignores(self):
        ConfigParser(skip_errors=True).parse(self.breaking_config)

    def test_recommended_matrix_disables_unvalidated_pairs(self):
        self.assertFalse(
            Compatibility.check(
                anomaly="channel-rewiring",
                base_oscillation="sine",
                mode="recommended",
            )
        )

    def test_validated_matrix_keeps_only_admitted_structural_pairs(self):
        for anomaly, base_oscillation, mode, expected in VALIDATED_COMPATIBILITY_CASES:
            with self.subTest(
                anomaly=anomaly,
                base_oscillation=base_oscillation,
                mode=mode,
            ):
                self.assertEqual(
                    Compatibility.check(
                        anomaly=anomaly,
                        base_oscillation=base_oscillation,
                        mode=mode,
                    ),
                    expected,
                )
