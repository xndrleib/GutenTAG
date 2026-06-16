import unittest

from gutenTAG.tsgen.config import (
    merge_dicts,
    optional_str_list,
    parse_pair,
    parse_pair_int,
    parse_pair_profiles,
    parse_segment_planner,
    parse_split_instance_counts,
    read_nested_dict,
    validate_base_channel_correlation_mapping,
)


class TestRuntimeConfigParsing(unittest.TestCase):
    def test_read_nested_dict_returns_empty_for_missing_or_malformed_sections(
        self,
    ) -> None:
        self.assertEqual(read_nested_dict({}, "missing"), {})
        self.assertEqual(read_nested_dict({"section": 3}, "section"), {})
        self.assertEqual(
            read_nested_dict({"section": {"value": 1}}, "section"), {"value": 1}
        )

    def test_optional_str_list_parses_none_and_sequences(self) -> None:
        self.assertEqual(optional_str_list(None), [])
        self.assertEqual(optional_str_list(["a", 2]), ["a", "2"])
        with self.assertRaisesRegex(ValueError, "Expected list/tuple"):
            optional_str_list("abc")

    def test_parse_pair_and_int_pair_validate_shape(self) -> None:
        self.assertEqual(parse_pair([1, "2"], "field"), (1.0, 2.0))
        self.assertEqual(parse_pair_int([1.2, 2.9], "field"), (1, 2))
        with self.assertRaisesRegex(ValueError, "exactly two values"):
            parse_pair([1], "field")

    def test_parse_split_instance_counts_supports_mapping_and_sequence(self) -> None:
        mapped = parse_split_instance_counts(
            {
                "train": {
                    "instances_per_variant": 3,
                    "clean_only_instances_per_variant": 1,
                },
                "test": {"paired_instances_per_variant": 2},
            },
            default_instances_per_split=5,
        )

        self.assertEqual(
            mapped,
            {
                "train": {
                    "paired_instances_per_variant": 3,
                    "clean_only_instances_per_variant": 1,
                },
                "test": {
                    "paired_instances_per_variant": 2,
                    "clean_only_instances_per_variant": 0,
                },
            },
        )
        self.assertEqual(
            parse_split_instance_counts(["a", "b"], 4),
            {
                "a": {
                    "paired_instances_per_variant": 4,
                    "clean_only_instances_per_variant": 0,
                },
                "b": {
                    "paired_instances_per_variant": 4,
                    "clean_only_instances_per_variant": 0,
                },
            },
        )
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            parse_split_instance_counts({"train": 1}, 4)

    def test_parse_pair_profiles_and_segment_planner_validate_mappings(self) -> None:
        self.assertEqual(
            parse_pair_profiles({"base__anom": ["p1", 2]}),
            {"base__anom": ["p1", "2"]},
        )
        planner = parse_segment_planner({"mean": {"planner": "uniform"}})
        self.assertEqual(planner, {"mean": {"planner": "uniform"}})
        planner["mean"]["planner"] = "changed"
        self.assertEqual(
            parse_segment_planner({"mean": {"planner": "uniform"}}),
            {"mean": {"planner": "uniform"}},
        )
        with self.assertRaisesRegex(ValueError, "pair_profiles"):
            parse_pair_profiles(["not", "mapping"])
        with self.assertRaisesRegex(ValueError, "segment_planner"):
            parse_segment_planner(["not", "mapping"])

    def test_merge_dicts_preserves_legacy_one_level_nested_merge(self) -> None:
        result = merge_dicts(
            {"outer": {"left": 1, "nested": {"old": 1}}, "keep": 1},
            {"outer": {"right": 2, "nested": {"new": 2}}, "keep": 3},
        )

        self.assertEqual(
            result,
            {
                "outer": {"left": 1, "right": 2, "nested": {"new": 2}},
                "keep": 3,
            },
        )

    def test_validate_base_channel_correlation_mapping_checks_weight(self) -> None:
        validate_base_channel_correlation_mapping({})
        validate_base_channel_correlation_mapping({"shared_noise_weight": 1.0})

        with self.assertRaisesRegex(ValueError, "base_channel_correlation"):
            validate_base_channel_correlation_mapping(["not", "mapping"])
        with self.assertRaisesRegex(ValueError, "shared_noise_weight"):
            validate_base_channel_correlation_mapping({"shared_noise_weight": 1.1})


if __name__ == "__main__":
    unittest.main()
