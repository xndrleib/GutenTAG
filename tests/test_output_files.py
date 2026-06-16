import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG.tsgen.output import (
    check_labels_and_events_consistency,
    write_instance_artifacts,
    write_labels_csv,
    write_timeseries_csv,
)


class TestOutputFileHelpers(unittest.TestCase):
    def test_check_labels_and_events_consistency_accepts_matching_labels(self) -> None:
        labels = np.zeros((5, 2), dtype=np.int8)
        labels[1:3, 0] = 1
        labels[2:5, 1] = 1

        check_labels_and_events_consistency(
            labels,
            [
                {"start": 1, "end": 3, "channel": 0},
                {"start": 2, "end": 5, "channel": 1},
            ],
        )

    def test_check_labels_and_events_consistency_rejects_mismatch(self) -> None:
        labels = np.zeros((5, 1), dtype=np.int8)
        labels[1:2, 0] = 1

        with self.assertRaisesRegex(ValueError, "labels_pointwise.csv"):
            check_labels_and_events_consistency(
                labels,
                [{"start": 1, "end": 3, "channel": 0}],
            )

    def test_write_timeseries_csv_uses_value_columns_and_float_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "series.csv"
            write_timeseries_csv(
                path,
                np.array([[1.23456, 2.0], [3.0, 4.5]], dtype=np.float64),
                channels=2,
                csv_float_format="%.2f",
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("value-0,value-1", text)
            self.assertIn("1.23,2.00", text)
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), ["value-0", "value-1"])

    def test_write_labels_csv_uses_label_columns_and_int_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.csv"
            write_labels_csv(
                path,
                np.array([[0, 1], [1, 0]], dtype=np.float64),
                channels=2,
            )

            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), ["label-0", "label-1"])
            self.assertEqual(
                frame.to_dict("records"),
                [
                    {"label-0": 0, "label-1": 1},
                    {"label-0": 1, "label-1": 0},
                ],
            )

    def test_write_instance_artifacts_writes_standard_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dir = Path(tmpdir)
            clean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            anomalous = clean.copy()
            anomalous[1, 0] = 9.0
            labels = np.zeros((2, 2), dtype=np.int8)
            labels[1:2, 0] = 1
            events = [{"start": 1, "end": 2, "channel": 0}]

            write_instance_artifacts(
                instance_dir=instance_dir,
                clean=clean,
                anomalous=anomalous,
                labels=labels,
                events=events,
                channels=2,
                csv_float_format="%.1f",
            )

            self.assertTrue((instance_dir / "clean.csv").exists())
            self.assertTrue((instance_dir / "anomalous.csv").exists())
            self.assertTrue((instance_dir / "labels_pointwise.csv").exists())
            self.assertTrue((instance_dir / "labels_any.csv").exists())
            self.assertTrue((instance_dir / "labels_intervention.csv").exists())
            self.assertTrue((instance_dir / "labels_context.csv").exists())
            self.assertTrue((instance_dir / "events.json").exists())
            self.assertEqual(
                json.loads((instance_dir / "events.json").read_text()),
                events,
            )


if __name__ == "__main__":
    unittest.main()
