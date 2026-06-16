import unittest

from gutenTAG.tsgen.parameters import (
    classify_base_family,
    prepare_variant_parameter_context,
    realize_parameters,
    resolve_anomaly_parameters,
    resolve_base_channel_correlation,
    resolve_base_channel_parameters,
    resolve_base_parameters,
    resolve_variant_anomaly_policy,
)
from gutenTAG.tsgen.seeding import derive_seed
from gutenTAG.tsgen.variants import VariantSpec
from gutenTAG.utils.global_variables import PARAMETERS


def _realize_base_channel_parameters(template, rng):
    return [realize_parameters(template, rng) for _ in range(2)]


class VariantContextConfig:
    master_seed = 42
    length = 120
    overlap_policy = "global"
    base_parameter_policy = "fixed_per_variant"
    base_channel_parameter_policy = "fixed_per_variant"
    anomaly_parameter_policy = "fixed_per_variant"
    base_oscillation_overrides = {"sine": {"frequency": 8.0}}
    base_channel_overrides = {"sine": {"phase": 0.5}}
    anomaly_overrides = {"mean": {"offset": 2.0}}
    variant_overrides = {
        "sine__mean": {
            "anomaly_policy": {
                "overlap_policy": "none",
                "segment_planner": {"planner": "fixed_first_onset_segments"},
            },
            "base_channel_correlation": {"shared_noise_weight": 0.3},
        }
    }
    base_channel_correlation = {"shared_noise_weight": 0.0}
    segment_planner = {}
    special_anomaly_policies = {}


class TestVariantParameterTemplates(unittest.TestCase):
    def test_resolve_base_parameters_merges_defaults_and_overrides_in_order(
        self,
    ) -> None:
        variant = VariantSpec("sine", "mean", "p00")

        result = resolve_base_parameters(
            variant,
            base_oscillation_overrides={"sine": {"frequency": 2.0}},
            variant_overrides={
                "sine__mean": {"base_oscillation": {"variance": 0.2}},
                "sine__mean__p00": {"base_oscillation": {"frequency": 3.0}},
            },
            length=120,
            default_base_overrides={"sine": {"frequency": 1.0, "variance": 0.1}},
        )

        self.assertEqual(result["frequency"], 3.0)
        self.assertEqual(result["variance"], 0.2)
        self.assertEqual(result[PARAMETERS.LENGTH], 120)

    def test_resolve_base_channel_parameters_merges_pair_and_variant_overrides(
        self,
    ) -> None:
        variant = VariantSpec("sine", "mean", "p01")

        result = resolve_base_channel_parameters(
            variant,
            base_channel_overrides={"sine": {"amplitude": [0.5, 1.0]}},
            variant_overrides={
                "sine__mean": {"base_channel": {"phase": [0.0, 1.0]}},
                "sine__mean__p01": {"base_channel": {"amplitude": [0.7, 0.9]}},
            },
        )

        self.assertEqual(result["amplitude"], [0.7, 0.9])
        self.assertEqual(result["phase"], [0.0, 1.0])

    def test_resolve_anomaly_parameters_merges_defaults_and_overrides_in_order(
        self,
    ) -> None:
        variant = VariantSpec("sine", "mean", "p00")

        result = resolve_anomaly_parameters(
            variant,
            anomaly_overrides={"mean": {"offset": 2.0}},
            variant_overrides={
                "sine__mean": {"anomaly": {"transition_length": 4}},
                "sine__mean__p00": {"anomaly": {"offset": 3.0}},
            },
            default_anomaly_overrides={"mean": {"offset": 1.0}},
        )

        self.assertEqual(result, {"offset": 3.0, "transition_length": 4})

    def test_resolve_variant_anomaly_policy_uses_legacy_nested_merge(self) -> None:
        variant = VariantSpec("sine", "mean", "p00")

        result = resolve_variant_anomaly_policy(
            variant,
            variant_overrides={
                "sine__mean": {
                    "anomaly_policy": {
                        "density_range": [0.1, 0.2],
                        "nested": {"left": 1, "right": 1},
                    }
                },
                "sine__mean__p00": {
                    "anomaly_policy": {"nested": {"right": 2, "extra": 3}}
                },
            },
        )

        self.assertEqual(result["density_range"], [0.1, 0.2])
        self.assertEqual(result["nested"], {"left": 1, "right": 2, "extra": 3})

    def test_resolve_base_channel_correlation_merges_and_validates_weight(self) -> None:
        variant = VariantSpec("sine", "mean", "p00")

        result = resolve_base_channel_correlation(
            variant,
            base_channel_correlation={"shared_noise_weight": 0.2},
            variant_overrides={
                "sine__mean": {
                    "base_channel_correlation": {"shared_noise_weight": 0.4}
                },
                "sine__mean__p00": {"base_channel_correlation": {"other": 1}},
            },
        )

        self.assertEqual(result, {"shared_noise_weight": 0.4, "other": 1})

        with self.assertRaisesRegex(ValueError, "shared_noise_weight"):
            resolve_base_channel_correlation(
                variant,
                base_channel_correlation={"shared_noise_weight": 1.2},
                variant_overrides={},
            )

    def test_prepare_variant_parameter_context_resolves_fixed_variant_setup(
        self,
    ) -> None:
        variant = VariantSpec("sine", "mean", "p00")

        result = prepare_variant_parameter_context(
            variant=variant,
            config=VariantContextConfig(),
            realize_base_channel_parameters=_realize_base_channel_parameters,
            derive_seed=derive_seed,
        )

        self.assertEqual(result.base_parameter_template["frequency"], 8.0)
        self.assertEqual(result.base_parameter_template[PARAMETERS.LENGTH], 120)
        self.assertEqual(result.base_channel_parameter_template, {"phase": 0.5})
        self.assertEqual(result.anomaly_parameter_template["offset"], 2.0)
        self.assertEqual(
            result.variant_segment_planner["planner"],
            "fixed_first_onset_segments",
        )
        self.assertEqual(result.effective_overlap_policy, "none")
        self.assertEqual(
            result.effective_base_channel_correlation,
            {"shared_noise_weight": 0.3},
        )
        self.assertEqual(result.fixed_base_parameters["frequency"], 8.0)
        self.assertEqual(
            result.fixed_base_channel_parameters,
            [
                {"phase": 0.5},
                {"phase": 0.5},
            ],
        )
        self.assertEqual(result.fixed_anomaly_parameters, {"offset": 2.0})

    def test_classify_base_family_groups_known_base_kinds(self) -> None:
        self.assertEqual(classify_base_family("sine"), "smooth_periodic")
        self.assertEqual(classify_base_family("random-mode-jump"), "mode_switching")
        self.assertEqual(classify_base_family("unknown"), "other")


if __name__ == "__main__":
    unittest.main()
