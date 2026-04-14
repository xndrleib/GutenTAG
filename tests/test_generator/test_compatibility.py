from unittest import TestCase

from gutenTAG.config.parser import ConfigParser
from gutenTAG.utils.compatibility import Compatibility


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
        self.assertTrue(
            Compatibility.check(
                anomaly="extremum",
                base_oscillation="cosine",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="extremum",
                base_oscillation="ecg",
                mode="recommended",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="covariance-change",
                base_oscillation="polynomial",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="correlation-flip",
                base_oscillation="polynomial",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="covariance-change",
                base_oscillation="shared-noise-sine",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="correlation-flip",
                base_oscillation="cosine",
                mode="recommended",
            )
        )
        self.assertFalse(
            Compatibility.check(
                anomaly="correlation-flip",
                base_oscillation="cosine",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="correlation-flip",
                base_oscillation="shared-noise-sine",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="shared-factor-break",
                base_oscillation="shared-noise-sine",
                mode="recommended",
            )
        )
        self.assertFalse(
            Compatibility.check(
                anomaly="shared-factor-break",
                base_oscillation="shared-noise-sine",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="lag-synchronization",
                base_oscillation="sine",
                mode="validated",
            )
        )
        self.assertTrue(
            Compatibility.check(
                anomaly="covariance-change",
                base_oscillation="polynomial",
                mode="recommended",
            )
        )
