import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG import TSDatasetGenerator
from gutenTAG.tsgen.capabilities import CapabilityProtocol, run_capability_analysis
from gutenTAG.tsgen.capabilities.dataset import (
    DatasetIndex,
    InstanceRecord,
)
from gutenTAG.tsgen.capabilities.model_zoo import compute_model_zoo_frontier
from gutenTAG.tsgen.capabilities.models import (
    LowRankResidualModel,
    TargetRegressionResidualModel,
)

from tests.capability_execution_fixtures import (
    _relation_event_group,
    _small_generation_config,
    _sorted_frame,
)


class TestCapabilityExecutionProfiles(unittest.TestCase):
    def test_target_regression_scores_non_last_channel_break(self) -> None:
        base = np.linspace(-1.0, 1.0, 80)
        clean = np.column_stack([base, 2.0 * base + 0.1, -0.5 * base + 0.2])
        anomalous = clean.copy()
        anomalous[30:50, 1] *= -1.0

        model = TargetRegressionResidualModel()
        model.fit([clean])

        clean_score = float(
            model.score_windows(clean, np.asarray([[30, 50]], dtype=int))[0]
        )
        anomalous_score = float(
            model.score_windows(anomalous, np.asarray([[30, 50]], dtype=int))[0]
        )
        self.assertTrue(np.isfinite(clean_score))
        self.assertGreater(anomalous_score, clean_score * 1000.0)

    def test_lowrank_residual_keeps_residual_dimension_for_two_channel_projection(
        self,
    ) -> None:
        clean = np.column_stack(
            [
                np.linspace(-1.0, 1.0, 80),
                np.linspace(-1.0, 1.0, 80) * 0.5,
            ]
        )

        model = LowRankResidualModel(rank=2)
        model.fit([clean])

        self.assertEqual(model.metadata()["effective_rank"], 1)

    def test_model_zoo_uses_relation_group_channel_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "poly__relation__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            time = np.linspace(-1.0, 1.0, 120)
            clean = np.column_stack([time, 2.0 * time + 0.1, np.cos(time)])
            anomalous = clean.copy()
            anomalous[40:60, 1] *= -1.0
            pd.DataFrame(clean, columns=["ch_0", "ch_1", "ch_2"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(anomalous, columns=["ch_0", "ch_1", "ch_2"]).to_csv(
                instance_dir / "anomalous.csv",
                index=False,
            )
            group = _relation_event_group("0", 40, 60)
            record = InstanceRecord(
                dataset_root=root,
                variant_id="poly__relation__p00",
                split="train",
                instance_id="instance_000",
                instance_dir=instance_dir,
                clean_path=instance_dir / "clean.csv",
                anomalous_path=instance_dir / "anomalous.csv",
                events_path=instance_dir / "events.json",
                summary_path=instance_dir / "instance_summary.json",
                base_oscillation="polynomial",
                anomaly_type="correlation-flip",
                channels=3,
                length=120,
                event_groups=(group,),
            )
            dataset = DatasetIndex(root=root, manifest={}, instances=(record,))

            result = compute_model_zoo_frontier(
                dataset,
                CapabilityProtocol(
                    alpha_grid=(0.10,),
                    max_scan_windows_per_length=8,
                    bootstrap_samples=0,
                ),
                models=(LowRankResidualModel(rank=1), TargetRegressionResidualModel()),
            )

            self.assertEqual(set(result.frontier["model_projection"]), {"0|1"})
            self.assertEqual(set(result.event_scores["model_projection"]), {"0|1"})
            self.assertEqual(
                {row["model_projection"] for row in result.model_manifest["models"]},
                {"0|1"},
            )

    def test_model_zoo_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_root = tmp_path / "dataset"
            serial_output = tmp_path / "serial_analysis"
            parallel_output = tmp_path / "parallel_analysis"
            parallel_cache = tmp_path / "parallel_cache"
            config = _small_generation_config(dataset_root)
            config["dataset"]["instances_per_split"] = 2
            config["variants"]["anomaly_types"] = ["mean", "variance"]
            TSDatasetGenerator.from_dict(config).run()
            protocol = CapabilityProtocol(
                delta_grid=(0.20,),
                alpha_grid=(0.10,),
                max_scan_windows_per_length=8,
                bootstrap_samples=0,
            )

            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=serial_output,
                protocol=protocol,
                profile_preset="model_zoo",
                n_jobs=1,
            )
            run_capability_analysis(
                dataset_root=dataset_root,
                output_dir=parallel_output,
                protocol=protocol,
                profile_preset="model_zoo",
                n_jobs=2,
                cache_dir=parallel_cache,
            )

            for filename in ("model_zoo_frontier.csv", "model_zoo_event_scores.csv"):
                serial = pd.read_csv(serial_output / filename)
                parallel = pd.read_csv(parallel_output / filename)
                pd.testing.assert_frame_equal(
                    _sorted_frame(serial),
                    _sorted_frame(parallel),
                    check_dtype=False,
                )
            serial_manifest = json.loads(
                (serial_output / "model_zoo_model_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            parallel_manifest = json.loads(
                (parallel_output / "model_zoo_model_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                sorted(
                    serial_manifest["models"],
                    key=lambda item: (item["fit_variant_id"], item["model_id"]),
                ),
                sorted(
                    parallel_manifest["models"],
                    key=lambda item: (item["fit_variant_id"], item["model_id"]),
                ),
            )
            for profile_name in (
                "model_zoo_frontier",
                "model_zoo_event_scores",
                "model_zoo_manifest_rows",
            ):
                metadata_paths = list(
                    (parallel_cache / "profile_partitions" / profile_name).glob(
                        "*.metadata.json"
                    )
                )
                self.assertGreaterEqual(len(metadata_paths), 2, profile_name)


if __name__ == "__main__":
    unittest.main()
