"""Runtime config parsing helpers for TS dataset generation."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

DEFAULT_RUNTIME_SPLITS: tuple[str, ...] = ("train", "val", "test")


def read_nested_dict(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read a nested mapping section or return an empty mapping.

    Parameters
    ----------
    config : Mapping[str, Any]
        Raw config mapping.
    key : str
        Section key.

    Returns
    -------
    Mapping[str, Any]
        Nested mapping section, or ``{}`` when missing or malformed.
    """
    value = config.get(key, {})
    if isinstance(value, Mapping):
        return value
    return {}


def optional_str_list(raw: Any) -> list[str]:
    """Parse an optional list of strings.

    Parameters
    ----------
    raw : Any
        Raw value.

    Returns
    -------
    list[str]
        Parsed string list.

    Raises
    ------
    ValueError
        If ``raw`` is not ``None`` or a list/tuple.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    raise ValueError(
        f"Expected list/tuple for configuration list field, got {type(raw)}"
    )


def parse_pair(raw: Any, field_name: str) -> tuple[float, float]:
    """Parse a two-value numeric pair.

    Parameters
    ----------
    raw : Any
        Raw two-value list or tuple.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    tuple[float, float]
        Parsed numeric pair.

    Raises
    ------
    ValueError
        If ``raw`` is not a two-value list or tuple.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"'{field_name}' must be a list/tuple with exactly two values."
        )
    return float(raw[0]), float(raw[1])


def parse_pair_int(raw: Any, field_name: str) -> tuple[int, int]:
    """Parse a two-value numeric pair as integers.

    Parameters
    ----------
    raw : Any
        Raw two-value list or tuple.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    tuple[int, int]
        Parsed integer pair.
    """
    left, right = parse_pair(raw, field_name)
    return int(left), int(right)


def parse_split_instance_counts(
    raw_splits: Any,
    default_instances_per_split: int,
    default_splits: Sequence[str] = DEFAULT_RUNTIME_SPLITS,
) -> dict[str, dict[str, int]]:
    """Parse split names and per-split instance counts.

    Parameters
    ----------
    raw_splits : Any
        Raw split config, either a mapping or sequence.
    default_instances_per_split : int
        Legacy paired-instance count used for sequence split configs.
    default_splits : Sequence[str]
        Default split names used when ``raw_splits`` is empty.

    Returns
    -------
    dict[str, dict[str, int]]
        Split config with paired and clean-only instance counts.

    Raises
    ------
    ValueError
        If a mapping split entry is not a mapping.
    """
    if isinstance(raw_splits, Mapping):
        parsed: dict[str, dict[str, int]] = {}
        for split_name, raw_cfg in raw_splits.items():
            if not isinstance(raw_cfg, Mapping):
                raise ValueError(f"dataset.splits[{split_name}] must be a mapping")
            cfg = dict(raw_cfg)
            paired_count = int(
                cfg.get(
                    "paired_instances_per_variant",
                    cfg.get(
                        "instances_per_variant",
                        cfg.get("instances_per_split", 0),
                    ),
                )
            )
            clean_only_count = int(cfg.get("clean_only_instances_per_variant", 0))
            parsed[str(split_name)] = {
                "paired_instances_per_variant": paired_count,
                "clean_only_instances_per_variant": clean_only_count,
            }
        return parsed

    splits = tuple(str(split) for split in optional_str_list(raw_splits))
    if len(splits) == 0:
        splits = tuple(str(split) for split in default_splits)
    return {
        split: {
            "paired_instances_per_variant": int(default_instances_per_split),
            "clean_only_instances_per_variant": 0,
        }
        for split in splits
    }


def parse_pair_profiles(raw: Any) -> dict[str, list[str]]:
    """Parse pair-profile mapping.

    Parameters
    ----------
    raw : Any
        Raw ``pair_profiles`` config.

    Returns
    -------
    dict[str, list[str]]
        Parsed mapping of pair id to profile ids.

    Raises
    ------
    ValueError
        If the mapping or profile lists are malformed.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("pair_profiles must be a mapping of pair_id -> [profile_ids]")
    parsed: dict[str, list[str]] = {}
    for pair_id, profile_ids in raw.items():
        if not isinstance(profile_ids, (list, tuple)):
            raise ValueError(
                f"pair_profiles[{pair_id}] must be a list/tuple of profile ids."
            )
        parsed[str(pair_id)] = [str(profile_id) for profile_id in profile_ids]
    return parsed


def parse_segment_planner(raw: Any) -> dict[str, dict[str, Any]]:
    """Parse segment-planner mapping.

    Parameters
    ----------
    raw : Any
        Raw ``segment_planner`` config.

    Returns
    -------
    dict[str, dict[str, Any]]
        Parsed planner mapping.

    Raises
    ------
    ValueError
        If the mapping or planner entries are malformed.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("segment_planner must be a mapping")
    parsed: dict[str, dict[str, Any]] = {}
    for planner_name, planner_cfg in raw.items():
        if not isinstance(planner_cfg, Mapping):
            raise ValueError(
                f"segment_planner[{planner_name}] must be a mapping of planner config"
            )
        parsed[str(planner_name)] = copy.deepcopy(dict(planner_cfg))
    return parsed


def merge_dicts(*parts: Mapping[str, Any]) -> dict[str, Any]:
    """Merge config mappings with one-level nested mapping merge semantics.

    Parameters
    ----------
    *parts : Mapping[str, Any]
        Mappings to merge from lowest to highest priority.

    Returns
    -------
    dict[str, Any]
        Deep-copied merged config mapping.
    """
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                nested = dict(merged[key])
                nested.update(copy.deepcopy(dict(value)))
                merged[key] = nested
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def validate_base_channel_correlation_mapping(mapping: Mapping[str, Any]) -> None:
    """Validate base-channel correlation runtime configuration.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Raw base-channel correlation mapping.

    Raises
    ------
    ValueError
        If ``mapping`` is not a mapping or ``shared_noise_weight`` is outside
        ``[0, 1]``.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError("base_channel_correlation must be a mapping")
    shared_noise_weight = float(dict(mapping).get("shared_noise_weight", 0.0))
    if shared_noise_weight < 0.0 or shared_noise_weight > 1.0:
        raise ValueError(
            "base_channel_correlation.shared_noise_weight must be in [0, 1]"
        )
