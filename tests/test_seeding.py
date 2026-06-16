import unittest

from gutenTAG.tsgen.seeding import (
    derive_instance_seeds,
    derive_seed,
    resolve_split_instance_specs,
    split_seed_index,
)


class TestDeterministicSeeding(unittest.TestCase):
    def test_derive_seed_preserves_sha256_based_contract(self) -> None:
        self.assertEqual(derive_seed(42, "variant", "train"), 1779071279)

    def test_split_seed_index_offsets_clean_only_instances(self) -> None:
        self.assertEqual(split_seed_index(role="paired", index=2, paired_count=5), 2)
        self.assertEqual(
            split_seed_index(role="clean_only", index=2, paired_count=5),
            7,
        )

    def test_resolve_split_instance_specs_preserves_order_names_and_seed_indices(
        self,
    ) -> None:
        specs = resolve_split_instance_specs(
            {
                "paired_instances_per_variant": 2,
                "clean_only_instances_per_variant": 2,
            },
            default_paired_instances=5,
        )

        self.assertEqual(
            [(spec.role, spec.index, spec.name, spec.seed_index) for spec in specs],
            [
                ("paired", 0, "instance_000", 0),
                ("paired", 1, "instance_001", 1),
                ("clean_only", 0, "clean_only_000", 2),
                ("clean_only", 1, "clean_only_001", 3),
            ],
        )

    def test_resolve_split_instance_specs_uses_default_paired_count(self) -> None:
        specs = resolve_split_instance_specs(None, default_paired_instances=2)

        self.assertEqual(
            [(spec.role, spec.name, spec.seed_index) for spec in specs],
            [
                ("paired", "instance_000", 0),
                ("paired", "instance_001", 1),
            ],
        )

    def test_derive_instance_seeds_preserves_named_seed_map(self) -> None:
        self.assertEqual(
            derive_instance_seeds(
                master_seed=42,
                variant_id="random-mode-jump__mode-correlation__p00",
                split="train",
                instance_index=3,
            ),
            {
                "instance_seed": 2474657182,
                "split_stable_instance_seed": 1468480211,
                "base_seed": 8462433,
                "base_shared_noise_seed": 602585646,
                "base_params_seed": 585254470,
                "base_channel_params_seed": 2208251264,
                "plan_seed": 3044163300,
                "anomaly_seed": 564291268,
                "params_seed": 1098239706,
                "zoom_seed": 1582876343,
            },
        )


if __name__ == "__main__":
    unittest.main()
