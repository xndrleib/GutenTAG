import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import yaml

from gutenTAG.tsgen.output import (
    build_split_entry,
    build_variant_config,
    build_variant_manifest,
    write_variant_config,
)


@dataclass(frozen=True)
class Variant:
    base_oscillation: str = "sine"
    anomaly_type: str = "mean"
    profile_id: str = "p00"

    @property
    def pair_id(self) -> str:
        return f"{self.base_oscillation}__{self.anomaly_type}"

    @property
    def variant_id(self) -> str:
        return f"{self.base_oscillation}__{self.anomaly_type}__{self.profile_id}"


class TestOutputVariant(unittest.TestCase):
    def test_build_variant_config_keeps_public_metadata_shape(self) -> None:
        variant = Variant()

        config = build_variant_config(
            variant=variant,
            base_parameter_template={"frequency": 8.0},
            fixed_base_parameters={"frequency": 8.0},
            base_channel_parameter_template={"phase": [0.0, 1.0]},
            fixed_base_channel_parameters=[{"phase": 0.0}, {"phase": 1.0}],
            effective_base_channel_correlation={"shared_noise_weight": 0.0},
            anomaly_parameter_template={"offset": 1.0},
            fixed_anomaly_parameters={"offset": 1.0},
            length=32,
            channels=2,
            splits=("train",),
            instances_per_split=3,
            split_instance_counts={
                "train": {
                    "paired_instances_per_variant": 3,
                    "clean_only_instances_per_variant": 1,
                }
            },
            density_range=(0.1, 0.2),
            density_tolerance=0.01,
            segment_count_range=(1, 2),
            placement_policy="uniform",
            channel_policy="single-random",
            effective_overlap_policy="global",
            variant_segment_planner={"planner": "uniform_segments"},
            variant_anomaly_policy={"density_tolerance": 0.01},
            length_normalization="resample",
            base_parameter_policy="fixed_per_variant",
            base_channel_parameter_policy="fixed_per_variant",
            anomaly_parameter_policy="fixed_per_variant",
            split_phase_shift={"enabled": False},
        )

        self.assertEqual(config["variant_id"], variant.variant_id)
        self.assertEqual(config["base_oscillation"]["kind"], "sine")
        self.assertEqual(config["anomaly_policy"]["overlap_policy"], "global")
        self.assertEqual(
            config["parameter_policies"]["anomaly_parameter_policy"],
            "fixed_per_variant",
        )

    def test_write_variant_config_writes_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "variant_config.yaml"

            write_variant_config(path, {"variant_id": "sine__mean__p00"})

            self.assertEqual(
                yaml.safe_load(path.read_text(encoding="utf-8")),
                {"variant_id": "sine__mean__p00"},
            )

    def test_build_split_entry_uses_posix_summary_path(self) -> None:
        entry = build_split_entry(
            split="train",
            instances=4,
            paired_instances=3,
            clean_only_instances=1,
        )

        self.assertEqual(entry["summary_file"], "train/split_summary.json")
        self.assertEqual(entry["instances"], 4)

    def test_build_variant_manifest_attaches_statistics(self) -> None:
        variant = Variant()
        manifest = build_variant_manifest(
            variant=variant,
            carrier_family="smooth_periodic",
            split_entries=[
                build_split_entry(
                    split="train",
                    instances=1,
                    paired_instances=1,
                    clean_only_instances=0,
                )
            ],
            instance_summaries=[
                {
                    "achieved_density": 0.1,
                    "target_density": 0.1,
                    "n_segments": 1,
                    "segment_lengths": [8],
                    "per_channel_segment_counts": {"0": 1},
                }
            ],
            base_parameter_policy="fixed_per_variant",
            base_channel_parameter_policy="fixed_per_variant",
            anomaly_parameter_policy="fixed_per_variant",
            split_phase_shift={"enabled": False},
            effective_overlap_policy="global",
            variant_segment_planner={"planner": "uniform_segments"},
            variant_anomaly_policy={},
            length_normalization="resample",
        )

        self.assertEqual(manifest["variant_id"], variant.variant_id)
        self.assertEqual(manifest["carrier_family"], "smooth_periodic")
        self.assertEqual(manifest["statistics"]["instance_count"], 1)


if __name__ == "__main__":
    unittest.main()
