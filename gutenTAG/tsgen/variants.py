"""Variant matrix resolution for TS dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..utils.compatibility import Compatibility
from ..utils.global_variables import PARAMETERS
from .config import (
    ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY,
    BASES_REQUIRING_EXTRA_CONFIG,
    resolve_anomaly_names,
    resolve_base_names,
)


@dataclass
class VariantSpec:
    """Single base-oscillation/anomaly/profile combination."""

    base_oscillation: str
    anomaly_type: str
    profile_id: str = "p00"

    @property
    def pair_id(self) -> str:
        """Return base/anomaly pair identifier."""
        return f"{self.base_oscillation}__{self.anomaly_type}"

    @property
    def variant_id(self) -> str:
        """Return the deterministic and human-readable identifier."""
        return f"{self.base_oscillation}__{self.anomaly_type}__{self.profile_id}"


@dataclass(frozen=True)
class _VariantResolutionRequest:
    base_oscillations: list[str] | None
    skip_base_oscillations: list[str]
    anomaly_types: list[str] | None
    skip_anomaly_types: list[str]
    disabled_anomaly_types: list[str]
    pair_profiles: Mapping[str, list[str]]
    profiles_per_pair: int
    compatibility_mode: str
    special_anomaly_policies: Mapping[str, Mapping[str, Any]]
    skip_density_incompatible_variants: bool
    resolve_segment_planner: Callable[[str], Mapping[str, Any]]
    resolve_base_parameters: Callable[[VariantSpec], Mapping[str, Any]]


@dataclass
class _VariantResolutionState:
    variants: list[VariantSpec]
    skipped: list[dict[str, str]]
    disabled: list[dict[str, str]]


def resolve_profile_ids(
    pair_id: str,
    *,
    pair_profiles: Mapping[str, list[str]],
    profiles_per_pair: int,
) -> list[str]:
    """Resolve profile ids for one base/anomaly pair.

    Parameters
    ----------
    pair_id : str
        Base/anomaly pair id in ``base__anomaly`` form.
    pair_profiles : Mapping[str, list[str]]
        Explicit per-pair profile ids.
    profiles_per_pair : int
        Default number of profiles per pair.

    Returns
    -------
    list[str]
        Sorted profile ids.
    """
    explicit_ids = pair_profiles.get(pair_id)
    if explicit_ids:
        return sorted({str(profile_id) for profile_id in explicit_ids})
    return [f"p{index:02d}" for index in range(profiles_per_pair)]


def resolve_variants(
    *,
    base_oscillations: list[str] | None,
    skip_base_oscillations: list[str],
    anomaly_types: list[str] | None,
    skip_anomaly_types: list[str],
    disabled_anomaly_types: list[str],
    pair_profiles: Mapping[str, list[str]],
    profiles_per_pair: int,
    compatibility_mode: str,
    special_anomaly_policies: Mapping[str, Mapping[str, Any]],
    skip_density_incompatible_variants: bool,
    resolve_segment_planner: Callable[[str], Mapping[str, Any]],
    resolve_base_parameters: Callable[[VariantSpec], Mapping[str, Any]],
) -> tuple[list[VariantSpec], list[dict[str, str]], list[dict[str, str]]]:
    """Resolve runnable variant specs and skipped-variant audit records.

    Parameters
    ----------
    base_oscillations, anomaly_types
        Optional explicit include lists. ``None`` means all registered names.
    skip_base_oscillations, skip_anomaly_types
        Exclude lists.
    disabled_anomaly_types
        Anomaly names disabled by configuration.
    pair_profiles : Mapping[str, list[str]]
        Explicit per-pair profile ids.
    profiles_per_pair : int
        Default number of profiles per pair.
    compatibility_mode : str
        Compatibility matrix mode.
    special_anomaly_policies : Mapping[str, Mapping[str, Any]]
        Per-anomaly policies used to decide density compatibility.
    skip_density_incompatible_variants : bool
        Whether variants without a density-compatible policy should be skipped.
    resolve_segment_planner : Callable[[str], Mapping[str, Any]]
        Callback returning effective planner config for an anomaly type.
    resolve_base_parameters : Callable[[VariantSpec], Mapping[str, Any]]
        Callback returning effective base parameters for a variant.

    Returns
    -------
    tuple[list[VariantSpec], list[dict[str, str]], list[dict[str, str]]]
        Runnable variants, skipped variant records, and unique disabled-anomaly
        records.
    """
    return _resolve_variants_from_request(_VariantResolutionRequest(**locals()))


def _resolve_variants_from_request(
    request: _VariantResolutionRequest,
) -> tuple[list[VariantSpec], list[dict[str, str]], list[dict[str, str]]]:
    base_kinds = resolve_base_names(
        requested=request.base_oscillations,
        excluded=request.skip_base_oscillations,
    )
    anomaly_kinds = resolve_anomaly_names(
        requested=request.anomaly_types,
        excluded=request.skip_anomaly_types,
    )
    disabled_anomaly_set = {
        str(anomaly_type) for anomaly_type in request.disabled_anomaly_types
    }
    state = _VariantResolutionState(variants=[], skipped=[], disabled=[])

    for base_kind in base_kinds:
        for anomaly_kind in anomaly_kinds:
            for variant in _iter_profile_variants(base_kind, anomaly_kind, request):
                _resolve_one_variant(
                    variant,
                    request=request,
                    state=state,
                    disabled_anomaly_set=disabled_anomaly_set,
                )

    variants = sorted(state.variants, key=lambda variant: variant.variant_id)
    skipped = sorted(state.skipped, key=lambda entry: entry["variant_id"])
    return (
        variants,
        skipped,
        _unique_records(
            state.disabled,
            key_fields=("anomaly_type", "reason"),
        ),
    )


def _iter_profile_variants(
    base_kind: str,
    anomaly_kind: str,
    request: _VariantResolutionRequest,
) -> list[VariantSpec]:
    pair_variant = VariantSpec(base_kind, anomaly_kind, profile_id="p00")
    profile_ids = resolve_profile_ids(
        pair_variant.pair_id,
        pair_profiles=request.pair_profiles,
        profiles_per_pair=request.profiles_per_pair,
    )
    return [
        VariantSpec(base_kind, anomaly_kind, profile_id=profile_id)
        for profile_id in profile_ids
    ]


def _resolve_one_variant(
    variant: VariantSpec,
    *,
    request: _VariantResolutionRequest,
    state: _VariantResolutionState,
    disabled_anomaly_set: set[str],
) -> None:
    if _record_disabled_variant_if_needed(variant, state, disabled_anomaly_set):
        return
    skip_reason = _variant_skip_reason(variant, request)
    if skip_reason is not None:
        _record_skipped_variant(state, variant, skip_reason)
        if _is_density_policy_skip(variant.anomaly_type, skip_reason):
            state.disabled.append(
                {"anomaly_type": variant.anomaly_type, "reason": skip_reason}
            )
        return
    state.variants.append(variant)


def _record_disabled_variant_if_needed(
    variant: VariantSpec,
    state: _VariantResolutionState,
    disabled_anomaly_set: set[str],
) -> bool:
    if variant.anomaly_type not in disabled_anomaly_set:
        return False
    reason = "Disabled by configuration."
    _record_skipped_variant(state, variant, reason)
    state.disabled.append({"anomaly_type": variant.anomaly_type, "reason": reason})
    return True


def _variant_skip_reason(
    variant: VariantSpec,
    request: _VariantResolutionRequest,
) -> str | None:
    if not Compatibility.check(
        variant.anomaly_type,
        variant.base_oscillation,
        mode=request.compatibility_mode,
    ):
        return (
            "Incompatible (base_oscillation, anomaly_type) "
            f"pair under compatibility_mode={request.compatibility_mode}."
        )
    density_reason = _density_policy_skip_reason(variant.anomaly_type, request)
    if density_reason is not None:
        return density_reason
    return _extra_config_skip_reason(variant, request)


def _density_policy_skip_reason(
    anomaly_kind: str,
    request: _VariantResolutionRequest,
) -> str | None:
    if not request.skip_density_incompatible_variants:
        return None
    if anomaly_kind not in ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY:
        return None
    if anomaly_kind in request.special_anomaly_policies:
        return None
    planner_cfg = request.resolve_segment_planner(anomaly_kind)
    planner_name = str(planner_cfg.get("planner", "uniform_segments")).lower()
    if planner_name in ("point_events_from_density", "fixed_first_onset_segments"):
        return None
    return (
        f"Anomaly '{anomaly_kind}' requires special_anomaly_policies "
        "or a dedicated segment_planner under current density "
        "constraints."
    )


def _extra_config_skip_reason(
    variant: VariantSpec,
    request: _VariantResolutionRequest,
) -> str | None:
    if variant.base_oscillation not in BASES_REQUIRING_EXTRA_CONFIG:
        return None
    base_params = request.resolve_base_parameters(variant)
    if variant.base_oscillation == "formula" and PARAMETERS.FORMULA not in base_params:
        return "Base oscillation 'formula' requires explicit 'formula' configuration."
    if (
        variant.base_oscillation == "custom-input"
        and PARAMETERS.INPUT_TIMESERIES_PATH_TEST not in base_params
    ):
        return (
            "Base oscillation 'custom-input' requires " "'input-timeseries-path-test'."
        )
    return None


def _record_skipped_variant(
    state: _VariantResolutionState,
    variant: VariantSpec,
    reason: str,
) -> None:
    state.skipped.append({"variant_id": variant.variant_id, "reason": reason})


def _is_density_policy_skip(anomaly_kind: str, reason: str) -> bool:
    return (
        anomaly_kind in ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY
        and "requires special_anomaly_policies" in reason
    )


def _unique_records(
    items: list[dict[str, str]],
    key_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    seen = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = tuple(item.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
