import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.environment.backends import validate_release_backend_policy
from gutenTAG.tsgen.io import write_json
from gutenTAG.tsgen.labels import build_label_masks
from gutenTAG.tsgen.manifest import hash_generated_csv


class TestV11Contracts(unittest.TestCase):
    def _base_config(self, output_root: Path) -> dict:
        return {
            "generator": {
                "output_root": str(output_root),
                "master_seed": 1234,
                "overwrite_output": True,
                "log_level": "INFO",
                "on_variant_failure": "skip",
            },
            "dataset": {
                "length": 160,
                "channels": 2,
                "splits": ["train"],
                "instances_per_split": 1,
            },
            "anomaly_policy": {
                "density_range": [0.05, 0.08],
                "density_tolerance": 0.05,
                "segment_count_range": [2, 3],
                "placement_policy": "uniform",
                "channel_policy": "single-random",
                "overlap_policy": "global",
            },
            "variants": {
                "base_oscillations": ["sine"],
                "anomaly_types": ["mean"],
                "profiles_per_pair": 1,
                "base_parameter_policy": "fixed_per_variant",
                "anomaly_parameter_policy": "fixed_per_variant",
            },
            "plot": {"enabled": False},
        }

    def test_unknown_base_name_fails_with_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(Path(tmp) / "dataset")
            config["variants"]["base_oscillations"] = ["sien"]
            with self.assertRaisesRegex(ValueError, "Did you mean 'sine'"):
                TSDatasetGenerator.from_dict(config)

    def test_unknown_config_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(Path(tmp) / "dataset")
            config["generator"]["silent_typo"] = True
            with self.assertRaises(ValueError):
                TSDatasetGenerator.from_dict(config)

    def test_empty_dataset_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(Path(tmp) / "dataset")
            config["variants"]["anomaly_types"] = ["extremum"]
            with self.assertRaisesRegex(ValueError, "no instances"):
                TSDatasetGenerator.from_dict(config).run()

    def test_empty_dataset_can_be_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._base_config(Path(tmp) / "dataset")
            config["generator"]["allow_empty_dataset"] = True
            config["variants"]["anomaly_types"] = ["extremum"]
            manifest = TSDatasetGenerator.from_dict(config).run()
            self.assertEqual(manifest["aggregated_statistics"]["instance_count"], 0)

    def test_strict_json_writer_normalizes_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            write_json(path, {"value": float("inf"), "nan": np.float64("nan")})
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(parsed["value"])
            self.assertIsNone(parsed["nan"])

    def test_label_masks_include_context_without_forcing_primary_target(self) -> None:
        masks = build_label_masks(
            length=20,
            channels=4,
            events=[
                {
                    "start": 3,
                    "end": 8,
                    "channel": 3,
                    "operator_target_channels": [3],
                    "group_channels": [0, 3],
                    "context_channels": [0, 3],
                }
            ],
        )
        self.assertEqual(int(masks.labels_any[:, 0].sum()), 5)
        self.assertEqual(int(masks.labels_affected[:, 3].sum()), 5)
        self.assertEqual(int(masks.labels_affected[:, 0].sum()), 0)
        self.assertEqual(int(masks.labels_context[:, 0].sum()), 5)
        self.assertEqual(int(masks.labels_context[:, 3].sum()), 5)

    def test_generated_csv_hash_is_path_independent(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            root_a = Path(a)
            root_b = Path(b)
            for root in [root_a, root_b]:
                instance = root / "variants" / "v" / "train" / "instances" / "i"
                instance.mkdir(parents=True)
                (instance / "clean.csv").write_text("value-0\n1.0\n", encoding="utf-8")
                (instance / "anomalous.csv").write_text(
                    "value-0\n2.0\n", encoding="utf-8"
                )
            self.assertEqual(
                hash_generated_csv(root_a)["digest"],
                hash_generated_csv(root_b)["digest"],
            )

    def test_ecg_release_backend_rejects_legacy_neurokit2(self) -> None:
        with (
            patch(
                "gutenTAG.tsgen.environment.backends.find_spec",
                return_value=object(),
            ),
            patch(
                "gutenTAG.tsgen.environment.backends.version",
                return_value="0.1.2",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "neurokit2>=0.2.13"):
                validate_release_backend_policy(
                    base_oscillations=["ecg"],
                    all_base_oscillations=["sine", "ecg"],
                )


if __name__ == "__main__":
    unittest.main()
