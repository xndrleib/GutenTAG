import unittest

from gutenTAG.tsgen.output import (
    compute_dataset_statistics,
    compute_split_statistics,
)


class TestOutputStatistics(unittest.TestCase):
    def test_dataset_statistics_empty_input_has_stable_shape(self) -> None:
        summary = compute_dataset_statistics([])

        self.assertEqual(summary["instance_count"], 0)
        self.assertIsNone(summary["target_density_mean"])
        self.assertIsNone(summary["achieved_density_mean"])
        self.assertEqual(summary["per_channel_segment_counts_total"], {})
        self.assertEqual(summary["per_channel_segment_counts_stats"], {})

    def test_split_statistics_aggregates_density_segments_and_channels(
        self,
    ) -> None:
        summaries = [
            {
                "target_density": 0.10,
                "achieved_density": 0.08,
                "n_segments": 2,
                "segment_lengths": [3, 5],
                "per_channel_segment_counts": {"0": 2, "1": 0},
            },
            {
                "target_density": 0.20,
                "achieved_density": 0.22,
                "n_segments": 3,
                "segment_lengths": [7],
                "per_channel_segment_counts": {"0": 1, "1": 2},
            },
        ]

        summary = compute_split_statistics(
            split="train",
            instance_summaries=summaries,
            length=128,
            channels=2,
        )

        self.assertEqual(summary["split"], "train")
        self.assertEqual(summary["instances"], 2)
        self.assertEqual(summary["length"], 128)
        self.assertEqual(summary["channels"], 2)
        self.assertAlmostEqual(summary["target_density_mean"], 0.15)
        self.assertAlmostEqual(summary["achieved_density_mean"], 0.15)
        self.assertAlmostEqual(summary["n_segments_mean"], 2.5)
        self.assertEqual(summary["segment_length_min"], 3)
        self.assertEqual(summary["segment_length_max"], 7)
        self.assertEqual(
            summary["per_channel_segment_counts_total"],
            {"0": 3, "1": 2},
        )
        self.assertEqual(
            summary["per_channel_segment_counts_stats"]["0"]["total"],
            3,
        )
        self.assertEqual(
            summary["per_channel_segment_counts_stats"]["1"]["max"],
            2,
        )

    def test_dataset_statistics_uses_observed_channel_keys(self) -> None:
        summaries = [
            {
                "target_density": 0.10,
                "achieved_density": 0.08,
                "n_segments": 1,
                "segment_lengths": [4],
                "per_channel_segment_counts": {"0": 1},
            },
            {
                "target_density": 0.20,
                "achieved_density": 0.18,
                "n_segments": 2,
                "segment_lengths": [6, 8],
                "per_channel_segment_counts": {"1": 2},
            },
        ]

        summary = compute_dataset_statistics(summaries)

        self.assertEqual(summary["instance_count"], 2)
        self.assertEqual(
            summary["per_channel_segment_counts_total"],
            {"0": 1, "1": 2},
        )
        self.assertEqual(
            summary["per_channel_segment_counts_stats"]["0"]["max"],
            1,
        )
        self.assertEqual(
            summary["per_channel_segment_counts_stats"]["1"]["min"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
