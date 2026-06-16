"""Variant-level runtime parameter context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..planning import resolve_segment_planner
from .sampling import realize_parameters as default_realize_parameters
from .variant_templates import (
    VariantLike,
    resolve_anomaly_parameters,
    resolve_base_channel_correlation,
    resolve_base_channel_parameters,
    resolve_base_parameters,
    resolve_variant_anomaly_policy,
)

ParameterRealizer = Callable[[Mapping[str, Any], np.random.Generator], dict[str, Any]]
BaseChannelParameterRealizer = Callable[
    [Mapping[str, Any], np.random.Generator], list[dict[str, Any]]
]
SeedDeriver = Callable[..., int]


class VariantRuntimeConfig(Protocol):
    """Runtime config fields required for variant parameter context."""

    @property
    def master_seed(self) -> int: ...

    @property
    def length(self) -> int: ...

    @property
    def overlap_policy(self) -> str: ...

    @property
    def base_parameter_policy(self) -> str: ...

    @property
    def base_channel_parameter_policy(self) -> str: ...

    @property
    def anomaly_parameter_policy(self) -> str: ...

    @property
    def base_oscillation_overrides(self) -> Mapping[str, Mapping[str, Any]]: ...

    @property
    def base_channel_overrides(self) -> Mapping[str, Mapping[str, Any]]: ...

    @property
    def anomaly_overrides(self) -> Mapping[str, Mapping[str, Any]]: ...

    @property
    def variant_overrides(self) -> Mapping[str, Any]: ...

    @property
    def base_channel_correlation(self) -> Mapping[str, Any]: ...

    @property
    def segment_planner(self) -> Mapping[str, Any]: ...

    @property
    def special_anomaly_policies(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class VariantParameterContext:
    """Resolved variant-level parameter and planner context."""

    base_parameter_template: dict[str, Any]
    base_channel_parameter_template: dict[str, Any]
    anomaly_parameter_template: dict[str, Any]
    variant_anomaly_policy: dict[str, Any]
    variant_segment_planner: dict[str, Any]
    effective_overlap_policy: str
    fixed_base_parameters: dict[str, Any] | None
    fixed_base_channel_parameters: list[dict[str, Any]] | None
    fixed_anomaly_parameters: dict[str, Any] | None
    effective_base_channel_correlation: dict[str, Any]


@dataclass(frozen=True)
class _VariantParameterTemplates:
    base: dict[str, Any]
    base_channel: dict[str, Any]
    anomaly: dict[str, Any]


@dataclass(frozen=True)
class _VariantPolicyContext:
    anomaly_policy: dict[str, Any]
    segment_planner: dict[str, Any]
    overlap_policy: str


@dataclass(frozen=True)
class _FixedVariantParameters:
    base: dict[str, Any] | None
    base_channel: list[dict[str, Any]] | None
    anomaly: dict[str, Any] | None


def prepare_variant_parameter_context(
    *,
    variant: VariantLike,
    config: VariantRuntimeConfig,
    realize_parameters: ParameterRealizer = default_realize_parameters,
    realize_base_channel_parameters: BaseChannelParameterRealizer,
    derive_seed: SeedDeriver,
) -> VariantParameterContext:
    """Resolve variant-level templates, fixed parameters, and planner context."""

    templates = _resolve_variant_parameter_templates(variant=variant, config=config)
    policy_context = _resolve_variant_policy_context(variant=variant, config=config)
    fixed_parameters = _resolve_fixed_variant_parameters(
        templates=templates,
        config=config,
        variant=variant,
        realize_parameters=realize_parameters,
        realize_base_channel_parameters=realize_base_channel_parameters,
        derive_seed=derive_seed,
    )
    effective_base_channel_correlation = resolve_base_channel_correlation(
        variant,
        base_channel_correlation=config.base_channel_correlation,
        variant_overrides=config.variant_overrides,
    )
    return VariantParameterContext(
        base_parameter_template=templates.base,
        base_channel_parameter_template=templates.base_channel,
        anomaly_parameter_template=templates.anomaly,
        variant_anomaly_policy=policy_context.anomaly_policy,
        variant_segment_planner=policy_context.segment_planner,
        effective_overlap_policy=policy_context.overlap_policy,
        fixed_base_parameters=fixed_parameters.base,
        fixed_base_channel_parameters=fixed_parameters.base_channel,
        fixed_anomaly_parameters=fixed_parameters.anomaly,
        effective_base_channel_correlation=effective_base_channel_correlation,
    )


def _resolve_variant_parameter_templates(
    *,
    variant: VariantLike,
    config: VariantRuntimeConfig,
) -> _VariantParameterTemplates:
    return _VariantParameterTemplates(
        base=resolve_base_parameters(
            variant,
            base_oscillation_overrides=config.base_oscillation_overrides,
            variant_overrides=config.variant_overrides,
            length=config.length,
        ),
        base_channel=resolve_base_channel_parameters(
            variant,
            base_channel_overrides=config.base_channel_overrides,
            variant_overrides=config.variant_overrides,
        ),
        anomaly=resolve_anomaly_parameters(
            variant,
            anomaly_overrides=config.anomaly_overrides,
            variant_overrides=config.variant_overrides,
        ),
    )


def _resolve_variant_policy_context(
    *,
    variant: VariantLike,
    config: VariantRuntimeConfig,
) -> _VariantPolicyContext:
    anomaly_policy = resolve_variant_anomaly_policy(
        variant,
        variant_overrides=config.variant_overrides,
    )
    segment_planner = resolve_segment_planner(
        anomaly_type=variant.anomaly_type,
        segment_planner=config.segment_planner,
        special_anomaly_policies=config.special_anomaly_policies,
        planner_override=_segment_planner_override(anomaly_policy),
    )
    overlap_policy = str(
        anomaly_policy.get(
            "overlap_policy",
            segment_planner.get("overlap_policy", config.overlap_policy),
        )
    )
    return _VariantPolicyContext(
        anomaly_policy=anomaly_policy,
        segment_planner=segment_planner,
        overlap_policy=overlap_policy,
    )


def _segment_planner_override(anomaly_policy: Mapping[str, Any]) -> Mapping[str, Any]:
    override = anomaly_policy.get("segment_planner", {})
    if isinstance(override, Mapping):
        return override
    return {}


def _resolve_fixed_variant_parameters(
    *,
    templates: _VariantParameterTemplates,
    config: VariantRuntimeConfig,
    variant: VariantLike,
    realize_parameters: ParameterRealizer,
    realize_base_channel_parameters: BaseChannelParameterRealizer,
    derive_seed: SeedDeriver,
) -> _FixedVariantParameters:
    variant_param_rng = np.random.default_rng(
        derive_seed(config.master_seed, variant.variant_id, "variant-params")
    )
    return _FixedVariantParameters(
        base=_resolve_fixed_base_parameters(
            policy=config.base_parameter_policy,
            template=templates.base,
            rng=variant_param_rng,
            realize_parameters=realize_parameters,
        ),
        base_channel=_resolve_fixed_base_channel_parameters(
            policy=config.base_channel_parameter_policy,
            template=templates.base_channel,
            rng=variant_param_rng,
            realize_base_channel_parameters=realize_base_channel_parameters,
        ),
        anomaly=_resolve_fixed_anomaly_parameters(
            policy=config.anomaly_parameter_policy,
            template=templates.anomaly,
            rng=variant_param_rng,
            realize_parameters=realize_parameters,
        ),
    )


def _resolve_fixed_base_parameters(
    *,
    policy: str,
    template: Mapping[str, Any],
    rng: np.random.Generator,
    realize_parameters: ParameterRealizer,
) -> dict[str, Any] | None:
    if policy != "fixed_per_variant":
        return None
    return realize_parameters(template, rng)


def _resolve_fixed_base_channel_parameters(
    *,
    policy: str,
    template: Mapping[str, Any],
    rng: np.random.Generator,
    realize_base_channel_parameters: BaseChannelParameterRealizer,
) -> list[dict[str, Any]] | None:
    if policy != "fixed_per_variant":
        return None
    return realize_base_channel_parameters(template, rng)


def _resolve_fixed_anomaly_parameters(
    *,
    policy: str,
    template: Mapping[str, Any],
    rng: np.random.Generator,
    realize_parameters: ParameterRealizer,
) -> dict[str, Any] | None:
    if policy != "fixed_per_variant":
        return None
    return realize_parameters(template, rng)
