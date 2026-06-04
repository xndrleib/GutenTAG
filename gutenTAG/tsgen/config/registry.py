"""Registry-backed name validation for TS dataset configs."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable, Mapping, Optional, Sequence

from ...anomalies import AnomalyKind
from ...base_oscillations import BaseOscillation


def available_base_names() -> list[str]:
    """Return registered base oscillation names in deterministic order."""
    return sorted(str(name) for name in BaseOscillation.key_mapping.keys())


def available_anomaly_names() -> list[str]:
    """Return registered anomaly names in deterministic order."""
    return sorted(str(kind.value) for kind in AnomalyKind)


def resolve_base_names(
    requested: Optional[Sequence[str]],
    excluded: Sequence[str],
) -> list[str]:
    """Resolve requested/excluded base names with strict validation."""
    return _resolve_names(
        requested=requested,
        excluded=excluded,
        available=available_base_names(),
        label="base_oscillations",
        excluded_label="skip_base_oscillations",
    )


def resolve_anomaly_names(
    requested: Optional[Sequence[str]],
    excluded: Sequence[str],
) -> list[str]:
    """Resolve requested/excluded anomaly names with strict validation."""
    return _resolve_names(
        requested=requested,
        excluded=excluded,
        available=available_anomaly_names(),
        label="anomaly_types",
        excluded_label="skip_anomaly_types",
    )


def validate_name_selections(
    *,
    base_oscillations: Optional[Sequence[str]],
    anomaly_types: Optional[Sequence[str]],
    skip_base_oscillations: Sequence[str],
    skip_anomaly_types: Sequence[str],
    disabled_anomaly_types: Sequence[str],
    pair_profiles: Mapping[str, Sequence[str]],
) -> None:
    """Validate all registry-backed name references in a TS config.

    Parameters
    ----------
    base_oscillations, anomaly_types
        Optional explicit include lists. ``None`` means all registered names.
    skip_base_oscillations, skip_anomaly_types
        Exclude lists that must also reference known registry names.
    disabled_anomaly_types
        Anomaly names disabled by configuration.
    pair_profiles
        Mapping keyed by ``base__anomaly`` pair ids.
    """
    base_names = available_base_names()
    anomaly_names = available_anomaly_names()
    _validate_known(
        values=base_oscillations or [],
        available=base_names,
        label="variants.base_oscillations",
    )
    _validate_known(
        values=skip_base_oscillations,
        available=base_names,
        label="variants.skip_base_oscillations",
    )
    _validate_known(
        values=anomaly_types or [],
        available=anomaly_names,
        label="variants.anomaly_types",
    )
    _validate_known(
        values=skip_anomaly_types,
        available=anomaly_names,
        label="variants.skip_anomaly_types",
    )
    _validate_known(
        values=disabled_anomaly_types,
        available=anomaly_names,
        label="variants.disabled_anomaly_types",
    )
    for pair_id in pair_profiles:
        if "__" not in str(pair_id):
            raise ValueError(
                f"variants.pair_profiles key '{pair_id}' must have form "
                "'<base_oscillation>__<anomaly_type>'."
            )
        base_name, anomaly_name = str(pair_id).split("__", 1)
        _validate_known(
            values=[base_name],
            available=base_names,
            label=f"variants.pair_profiles[{pair_id}].base",
        )
        _validate_known(
            values=[anomaly_name],
            available=anomaly_names,
            label=f"variants.pair_profiles[{pair_id}].anomaly",
        )


def _resolve_names(
    *,
    requested: Optional[Sequence[str]],
    excluded: Sequence[str],
    available: Sequence[str],
    label: str,
    excluded_label: str,
) -> list[str]:
    _validate_known(values=requested or [], available=available, label=label)
    _validate_known(values=excluded, available=available, label=excluded_label)
    requested_names = (
        sorted(set(str(name) for name in requested)) if requested else list(available)
    )
    excluded_names = set(str(name) for name in excluded)
    selected = [name for name in requested_names if name not in excluded_names]
    if len(selected) == 0:
        raise ValueError(
            f"{label} resolves to an empty selection after applying {excluded_label}."
        )
    return sorted(selected)


def _validate_known(
    *,
    values: Iterable[str],
    available: Sequence[str],
    label: str,
) -> None:
    allowed = set(str(name) for name in available)
    for value in values:
        normalized = str(value)
        if normalized in allowed:
            continue
        message = f"Unknown {label} value: '{normalized}'."
        suggestion = _suggest(normalized, available)
        if suggestion is not None:
            message += f" Did you mean '{suggestion}'?"
        message += f" Allowed values: {', '.join(sorted(allowed))}."
        raise ValueError(message)


def _suggest(value: str, available: Sequence[str]) -> Optional[str]:
    matches = get_close_matches(str(value), [str(name) for name in available], n=1)
    return matches[0] if matches else None
