import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from gutenTAG import TSDatasetGenerator


def _periodic_split_shift_config(output_root: Path) -> dict[str, Any]:
    return {
        "generator": {
            "output_root": str(output_root),
            "master_seed": 1234,
            "overwrite_output": True,
            "log_level": "INFO",
            "on_variant_failure": "skip",
        },
        "dataset": {
            "length": 600,
            "channels": 2,
            "splits": ["train", "val", "test"],
            "instances_per_split": 1,
        },
        "anomaly_policy": {
            "density_range": [0.05, 0.06],
            "density_tolerance": 0.005,
            "segment_count_range": [2, 3],
            "placement_policy": "uniform",
            "channel_policy": "single-random",
            "overlap_policy": "global",
            "length_normalization": "resample",
        },
        "variants": {
            "base_oscillations": ["sine"],
            "anomaly_types": ["mean"],
            "disabled_anomaly_types": [],
            "profiles_per_pair": 1,
            "pair_profiles": {},
            "base_parameter_policy": "random_per_instance",
            "base_channel_parameter_policy": "random_per_instance",
            "anomaly_parameter_policy": "fixed_per_variant",
            "skip_base_oscillations": [],
            "skip_anomaly_types": [],
            "split_phase_shift": {
                "enabled": True,
                "mode": "fixed_map",
                "phase_modulo": float(2.0 * np.pi),
                "values": {
                    "train": 0.0,
                    "val": float(2.0 * np.pi / 3.0),
                    "test": float(4.0 * np.pi / 3.0),
                },
            },
            "base_oscillation_overrides": {
                "sine": {"frequency": 8.0, "amplitude": 1.0, "variance": 0.02}
            },
            "base_channel_overrides": {
                "sine": {
                    "phase": 0.5,
                    "amplitude": 1.0,
                    "offset": 0.0,
                    "variance": 0.02,
                }
            },
        },
        "plot": {"enabled": False},
    }


def _run_split_diversity_analysis(module: Any, output_root: Path) -> dict[str, Any]:
    return module.analyze_split_diversity(
        module.SplitDiversityConfig(
            dataset_root=output_root,
            output_dir=output_root / "analysis" / "split_diversity",
            channel=0,
            max_points=600,
            min_observed_abs_lag=10,
            min_expected_abs_lag=10,
            fail_exit_nonzero=False,
        )
    )


def _load_split_diversity_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "check_ts_dataset_split_diversity.py"
    )
    spec = importlib.util.spec_from_file_location("split_diversity_check", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load split diversity checker from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSplitDiversityCheck(unittest.TestCase):
    def test_split_diversity_checker_detects_periodic_split_shift(self) -> None:
        module = _load_split_diversity_module()
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "dataset"
            manifest = TSDatasetGenerator.from_dict(
                _periodic_split_shift_config(output_root)
            ).run()
            self.assertIn("sine__mean__p00", manifest["generated_variants"])

            summary = _run_split_diversity_analysis(module, output_root)

            self.assertTrue(bool(summary["gate_pass"]))
            family_summary = summary["family_summary"]["sine"]
            self.assertTrue(bool(family_summary["metadata_pass"]))
            self.assertTrue(bool(family_summary["observed_pass"]))
            self.assertGreater(
                float(family_summary["median_observed_prefix_rms_relative"]), 0.25
            )


if __name__ == "__main__":
    unittest.main()
