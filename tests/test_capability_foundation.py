import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gutenTAG.tsgen.capabilities.array_store import ArrayStore
from gutenTAG.tsgen.capabilities.cache import CacheStore
from gutenTAG.tsgen.capabilities.dataset import InstanceRecord
from gutenTAG.tsgen.capabilities.hashes import (
    code_version_hash,
    file_sha256,
    table_content_hash,
)
from gutenTAG.tsgen.capabilities.partitions import (
    PartitionSpec,
    partition_sequence,
    run_partitions,
)
from gutenTAG.tsgen.capabilities.rolling import RollingStats


def _sum_partition(partition: PartitionSpec[int]) -> int:
    return int(sum(partition.items))


class TestCapabilityFoundation(unittest.TestCase):
    def test_hash_helpers_are_stable_and_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "module.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            first_code_hash = code_version_hash((path,))
            path.write_text("VALUE = 2\n", encoding="utf-8")
            second_code_hash = code_version_hash((path,))

        frame_a = pd.DataFrame({"id": [2, 1], "value": [20, 10]})
        frame_b = pd.DataFrame({"id": [1, 2], "value": [10, 20]})

        self.assertNotEqual(first_code_hash, second_code_hash)
        self.assertEqual(
            table_content_hash(frame_a, sort_by=("id",)),
            table_content_hash(frame_b, sort_by=("id",)),
        )
        self.assertNotEqual(table_content_hash(frame_a), table_content_hash(frame_b))

    def test_cache_store_reuses_and_invalidates_table_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheStore(
                cache_dir=Path(tmp),
                resume=True,
                run_fingerprint={
                    "dataset_hash": "sha256:dataset",
                    "protocol_hash": "sha256:protocol-a",
                    "code_hash": "sha256:code",
                },
            )
            calls = {"count": 0}

            def compute() -> pd.DataFrame:
                calls["count"] += 1
                return pd.DataFrame({"x": [calls["count"]]})

            fingerprint = cache.fingerprint(profile_name="demo", partition_id="p00000")
            first = cache.get_or_compute_table(
                profile_name="demo",
                partition_id="p00000",
                fingerprint=fingerprint,
                compute=compute,
            )
            second = cache.get_or_compute_table(
                profile_name="demo",
                partition_id="p00000",
                fingerprint=fingerprint,
                compute=compute,
            )
            changed = cache.fingerprint(
                profile_name="demo",
                partition_id="p00000",
                extra={"protocol_override": "changed"},
            )
            third = cache.get_or_compute_table(
                profile_name="demo",
                partition_id="p00000",
                fingerprint=changed,
                compute=compute,
            )
            metadata = json.loads(
                cache.metadata_path("demo", "p00000").read_text(encoding="utf-8")
            )

            self.assertEqual(calls["count"], 2)
            self.assertEqual(first.iloc[0]["x"], 1)
            self.assertEqual(second.iloc[0]["x"], 1)
            self.assertEqual(third.iloc[0]["x"], 2)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["fingerprint_hash"], changed.digest)

    def test_rolling_stats_match_naive_interval_computation(self) -> None:
        rng = np.random.default_rng(1729)
        series = rng.normal(size=(40, 3))
        starts = np.asarray([0, 3, 7, 19], dtype=np.int64)
        ends = np.asarray([5, 11, 22, 40], dtype=np.int64)
        stats = RollingStats(series)

        expected_mean = np.vstack(
            [
                series[start:end][:, [0, 2]].mean(axis=0)
                for start, end in zip(starts, ends)
            ]
        )
        expected_variance = np.vstack(
            [
                series[start:end][:, [0, 2]].var(axis=0)
                for start, end in zip(starts, ends)
            ]
        )
        expected_cov = np.asarray(
            [
                np.cov(series[start:end, 0], series[start:end, 1], bias=True)[0, 1]
                for start, end in zip(starts, ends)
            ]
        )
        expected_corr = np.asarray(
            [
                np.corrcoef(series[start:end, 0], series[start:end, 1])[0, 1]
                for start, end in zip(starts, ends)
            ]
        )
        expected_energy = np.asarray(
            [
                np.sum(series[start:end][:, [0, 2]] ** 2)
                for start, end in zip(starts, ends)
            ]
        )

        self.assertTrue(np.allclose(stats.mean(starts, ends, [0, 2]), expected_mean))
        self.assertTrue(
            np.allclose(stats.variance(starts, ends, [0, 2]), expected_variance)
        )
        self.assertTrue(np.allclose(stats.covariance(starts, ends, 0, 1), expected_cov))
        self.assertTrue(
            np.allclose(stats.correlation(starts, ends, 0, 1), expected_corr)
        )
        self.assertTrue(
            np.allclose(stats.energy(starts, ends, [0, 2]), expected_energy)
        )

    def test_partition_sequence_and_parallel_execution_are_deterministic(self) -> None:
        partitions = partition_sequence(tuple(range(10)), partition_size=3)
        serial = run_partitions(partitions, _sum_partition, n_jobs=1)
        parallel = run_partitions(partitions, _sum_partition, n_jobs=2)

        self.assertEqual(
            [item.partition_id for item in partitions],
            ["p00000", "p00001", "p00002", "p00003"],
        )
        self.assertEqual([item.value for item in serial], [3, 12, 21, 9])
        self.assertEqual(
            [item.value for item in parallel], [item.value for item in serial]
        )

    def test_array_store_metadata_records_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root
                / "variants"
                / "sine__mean__p00"
                / "train"
                / "instances"
                / "instance_000"
            )
            instance_dir.mkdir(parents=True)
            values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
            pd.DataFrame(values, columns=["ch_0", "ch_1"]).to_csv(
                instance_dir / "clean.csv", index=False
            )
            pd.DataFrame(values + 1.0, columns=["ch_0", "ch_1"]).to_csv(
                instance_dir / "anomalous.csv", index=False
            )
            record = InstanceRecord(
                dataset_root=root,
                variant_id="sine__mean__p00",
                split="train",
                instance_id="instance_000",
                instance_dir=instance_dir,
                clean_path=instance_dir / "clean.csv",
                anomalous_path=instance_dir / "anomalous.csv",
                events_path=instance_dir / "events.json",
                summary_path=instance_dir / "instance_summary.json",
                base_oscillation="sine",
                anomaly_type="mean",
                channels=2,
                length=2,
                event_groups=(),
            )

            store = ArrayStore(cache_dir=root / "cache", mmap=True)
            store.materialize((record,))
            metadata_path = (
                root
                / "cache"
                / "arrays"
                / "sine__mean__p00"
                / "train"
                / "instance_000"
                / "clean.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["array_store_version"], "synthgen.array_store.v2")
            self.assertEqual(
                metadata["source"]["sha256"], file_sha256(instance_dir / "clean.csv")
            )


if __name__ == "__main__":
    unittest.main()
