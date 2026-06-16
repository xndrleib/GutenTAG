"""Variant-level parameter template resolution."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Protocol

from ...utils.global_variables import PARAMETERS
from ..config import (
    DEFAULT_ANOMALY_OVERRIDES,
    DEFAULT_BASE_OVERRIDES,
    merge_dicts,
    validate_base_channel_correlation_mapping,
)


class VariantLike(Protocol):
    """Variant fields required by parameter template resolution."""

    @property
    def base_oscillation(self) -> str: ...

    @property
    def anomaly_type(self) -> str: ...

    @property
    def pair_id(self) -> str: ...

    @property
    def variant_id(self) -> str: ...


def resolve_base_parameters(
    variant: VariantLike,
    *,
    base_oscillation_overrides: Mapping[str, Mapping[str, Any]],
    variant_overrides: Mapping[str, Any],
    length: int,
    default_base_overrides: Mapping[str, Mapping[str, Any]] = DEFAULT_BASE_OVERRIDES,
) -> dict[str, Any]:
    """Resolve effective base-oscillation parameters for one variant.

    Parameters
    ----------
    variant : VariantLike
        Variant identity.
    base_oscillation_overrides : Mapping[str, Mapping[str, Any]]
        Per-base configuration overrides.
    variant_overrides : Mapping[str, Any]
        Pair and variant-specific overrides.
    length : int
        Generated series length.
    default_base_overrides : Mapping[str, Mapping[str, Any]]
        Built-in base defaults.

    Returns
    -------
    dict[str, Any]
        Effective base parameters.
    """
    base_kind = variant.base_oscillation
    parameters: dict[str, Any] = {}
    parameters.update(copy.deepcopy(default_base_overrides.get(base_kind, {})))
    parameters.update(copy.deepcopy(base_oscillation_overrides.get(base_kind, {})))
    parameters.update(
        copy.deepcopy(
            _mapping_override(variant_overrides, variant.pair_id, "base_oscillation")
        )
    )
    parameters.update(
        copy.deepcopy(
            _mapping_override(variant_overrides, variant.variant_id, "base_oscillation")
        )
    )
    parameters[PARAMETERS.LENGTH] = int(length)
    return parameters


def resolve_base_channel_parameters(
    variant: VariantLike,
    *,
    base_channel_overrides: Mapping[str, Mapping[str, Any]],
    variant_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve effective per-channel base-parameter template for one variant."""
    base_kind = variant.base_oscillation
    parameters: dict[str, Any] = {}
    parameters.update(copy.deepcopy(base_channel_overrides.get(base_kind, {})))
    parameters.update(
        copy.deepcopy(
            _mapping_override(variant_overrides, variant.pair_id, "base_channel")
        )
    )
    parameters.update(
        copy.deepcopy(
            _mapping_override(variant_overrides, variant.variant_id, "base_channel")
        )
    )
    return parameters


def resolve_anomaly_parameters(
    variant: VariantLike,
    *,
    anomaly_overrides: Mapping[str, Mapping[str, Any]],
    variant_overrides: Mapping[str, Any],
    default_anomaly_overrides: Mapping[
        str, Mapping[str, Any]
    ] = DEFAULT_ANOMALY_OVERRIDES,
) -> dict[str, Any]:
    """Resolve effective anomaly parameter template for one variant."""
    anomaly_type = variant.anomaly_type
    parameters: dict[str, Any] = {}
    parameters.update(copy.deepcopy(default_anomaly_overrides.get(anomaly_type, {})))
    parameters.update(copy.deepcopy(anomaly_overrides.get(anomaly_type, {})))
    parameters.update(
        copy.deepcopy(_mapping_override(variant_overrides, variant.pair_id, "anomaly"))
    )
    parameters.update(
        copy.deepcopy(
            _mapping_override(variant_overrides, variant.variant_id, "anomaly")
        )
    )
    return parameters


def resolve_variant_anomaly_policy(
    variant: VariantLike,
    *,
    variant_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve pair and variant-specific anomaly policy overrides."""
    policy: dict[str, Any] = {}
    pair_policy = _mapping_override(
        variant_overrides, variant.pair_id, "anomaly_policy"
    )
    if pair_policy:
        policy = merge_dicts(policy, pair_policy)
    variant_policy = _mapping_override(
        variant_overrides,
        variant.variant_id,
        "anomaly_policy",
    )
    if variant_policy:
        policy = merge_dicts(policy, variant_policy)
    return policy


def resolve_base_channel_correlation(
    variant: VariantLike,
    *,
    base_channel_correlation: Mapping[str, Any],
    variant_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve effective base-channel correlation config for one variant."""
    correlation: dict[str, Any] = copy.deepcopy(dict(base_channel_correlation))
    pair_correlation = _mapping_override(
        variant_overrides,
        variant.pair_id,
        "base_channel_correlation",
    )
    if pair_correlation:
        correlation = merge_dicts(correlation, pair_correlation)
    variant_correlation = _mapping_override(
        variant_overrides,
        variant.variant_id,
        "base_channel_correlation",
    )
    if variant_correlation:
        correlation = merge_dicts(correlation, variant_correlation)
    validate_base_channel_correlation_mapping(correlation)
    return correlation


def classify_base_family(base_kind: str) -> str:
    """Classify base oscillations into broad parameter-policy families."""
    family_by_kind = {
        "sine": "smooth_periodic",
        "cosine": "smooth_periodic",
        "shared-noise-sine": "structural_periodic",
        "square": "discontinuous_periodic",
        "sawtooth": "discontinuous_periodic",
        "dirichlet": "discontinuous_periodic",
        "polynomial": "smooth_trend",
        "random-walk": "smooth_trend",
        "ecg": "motif_rich",
        "cylinder-bell-funnel": "motif_rich",
        "mls": "motif_rich",
        "random-mode-jump": "mode_switching",
    }
    return str(family_by_kind.get(str(base_kind), "other"))


def _mapping_override(
    variant_overrides: Mapping[str, Any],
    variant_key: str,
    section: str,
) -> dict[str, Any]:
    raw_override = variant_overrides.get(variant_key, {})
    if isinstance(raw_override, Mapping) and isinstance(
        raw_override.get(section),
        Mapping,
    ):
        return dict(raw_override[section])
    return {}
