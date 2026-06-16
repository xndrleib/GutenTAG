import unittest

from gutenTAG.tsgen.variants import (
    VariantSpec,
    resolve_profile_ids,
    resolve_variants,
)


def _empty_planner(_anomaly_type: str) -> dict:
    return {}


def _empty_base_parameters(_variant: VariantSpec) -> dict:
    return {}


class TestVariantResolution(unittest.TestCase):
    def test_variant_spec_derives_pair_and_variant_ids(self) -> None:
        variant = VariantSpec("sine", "mean", profile_id="p02")

        self.assertEqual(variant.pair_id, "sine__mean")
        self.assertEqual(variant.variant_id, "sine__mean__p02")

    def test_resolve_profile_ids_uses_explicit_unique_sorted_ids(self) -> None:
        self.assertEqual(
            resolve_profile_ids(
                "sine__mean",
                pair_profiles={"sine__mean": ["p03", "p01", "p03"]},
                profiles_per_pair=2,
            ),
            ["p01", "p03"],
        )
        self.assertEqual(
            resolve_profile_ids(
                "sine__variance", pair_profiles={}, profiles_per_pair=2
            ),
            ["p00", "p01"],
        )

    def test_resolve_variants_returns_sorted_compatible_profiles(self) -> None:
        variants, skipped, disabled = resolve_variants(
            base_oscillations=["sine"],
            skip_base_oscillations=[],
            anomaly_types=["mean"],
            skip_anomaly_types=[],
            disabled_anomaly_types=[],
            pair_profiles={"sine__mean": ["p01", "p00"]},
            profiles_per_pair=1,
            compatibility_mode="hard",
            special_anomaly_policies={},
            skip_density_incompatible_variants=True,
            resolve_segment_planner=_empty_planner,
            resolve_base_parameters=_empty_base_parameters,
        )

        self.assertEqual(
            [variant.variant_id for variant in variants],
            ["sine__mean__p00", "sine__mean__p01"],
        )
        self.assertEqual(skipped, [])
        self.assertEqual(disabled, [])

    def test_resolve_variants_reports_disabled_anomaly_once(self) -> None:
        variants, skipped, disabled = resolve_variants(
            base_oscillations=["sine", "cosine"],
            skip_base_oscillations=[],
            anomaly_types=["mean"],
            skip_anomaly_types=[],
            disabled_anomaly_types=["mean"],
            pair_profiles={},
            profiles_per_pair=1,
            compatibility_mode="hard",
            special_anomaly_policies={},
            skip_density_incompatible_variants=True,
            resolve_segment_planner=_empty_planner,
            resolve_base_parameters=_empty_base_parameters,
        )

        self.assertEqual(variants, [])
        self.assertEqual(len(skipped), 2)
        self.assertEqual(
            disabled,
            [{"anomaly_type": "mean", "reason": "Disabled by configuration."}],
        )

    def test_resolve_variants_skips_density_incompatible_extremum_without_policy(
        self,
    ) -> None:
        variants, skipped, disabled = resolve_variants(
            base_oscillations=["sine"],
            skip_base_oscillations=[],
            anomaly_types=["extremum"],
            skip_anomaly_types=[],
            disabled_anomaly_types=[],
            pair_profiles={},
            profiles_per_pair=1,
            compatibility_mode="hard",
            special_anomaly_policies={},
            skip_density_incompatible_variants=True,
            resolve_segment_planner=_empty_planner,
            resolve_base_parameters=_empty_base_parameters,
        )

        self.assertEqual(variants, [])
        self.assertEqual(skipped[0]["variant_id"], "sine__extremum__p00")
        self.assertIn("requires special_anomaly_policies", skipped[0]["reason"])
        self.assertEqual(disabled[0]["anomaly_type"], "extremum")


if __name__ == "__main__":
    unittest.main()
