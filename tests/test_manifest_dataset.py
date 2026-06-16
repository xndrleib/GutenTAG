import tempfile
import unittest
from pathlib import Path

from gutenTAG.tsgen.manifest import build_dataset_manifest, canonical_json_hash


class TestDatasetManifest(unittest.TestCase):
    def test_build_dataset_manifest_keeps_public_schema_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "clean.csv").write_text("value-0\n1.0\n", encoding="utf-8")
            config = {"dataset": {"length": 8}, "generator": {"master_seed": 1}}

            manifest = build_dataset_manifest(
                output_root=root,
                repo_root=root,
                dataset_version="ts_dataset_v12",
                library_version="1.2.3",
                config=config,
                generated_variant_entries=[
                    {"variant_id": "sine__mean__p00", "instance_count": 1}
                ],
                skipped_variants=[],
                disabled_anomaly_types=["extremum"],
                aggregated_statistics={"instance_count": 1},
                derived_seeds={"sine__mean__p00": {}},
                metadata_registry={"metadata_registry_version": "v1"},
                generator_sidecars={
                    "annotation_channels": {"enabled": False},
                    "law_level_replicates": {"enabled": False},
                },
            )

            self.assertEqual(
                manifest["dataset_schema_version"],
                "synthgen.dataset.v1",
            )
            self.assertEqual(manifest["generated_variants"], ["sine__mean__p00"])
            self.assertEqual(
                manifest["normalized_config_hash"],
                canonical_json_hash(config),
            )
            self.assertEqual(
                manifest["artifacts"]["data_content_hash"]["file_count"],
                1,
            )
            self.assertIn("label_semantics", manifest)
            self.assertIn("onset_metadata", manifest)


if __name__ == "__main__":
    unittest.main()
