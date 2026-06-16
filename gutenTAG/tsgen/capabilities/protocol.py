"""Protocol definitions for capability analysis.

The protocol is the place where unavoidable operating-point choices live.  The
analysis still publishes full curves; these values define the default grid and
runtime bounds rather than hard-coded QC pass/fail thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence, cast


@dataclass(frozen=True)
class CapabilityProtocol:
    """Configuration for theory-aligned capability analysis."""

    protocol_version: str = "synthgen.capability.v1"
    delta_grid: tuple[float, ...] = (0.20, 0.50, 1.00, 2.00)
    alpha_grid: tuple[float, ...] = (0.10, 0.05, 0.01)
    max_projection_size: int = 2
    max_scan_windows_per_length: int = 96
    window_length_policy_mode: str = "exact"
    window_length_bins: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
    calibration_split: str | None = None
    legacy_detectability_max_clean_instances: int = 0
    clean_window_stride_fraction: float = 0.25
    context_window_multiplier: int = 4
    min_context_points: int = 8
    bootstrap_samples: int = 200
    random_seed: int = 1729
    include_pairwise_identifiability: bool = True
    calibration_min_clean_scan_count_for_alpha: tuple[tuple[float, int], ...] = (
        (0.10, 50),
        (0.05, 100),
        (0.01, 300),
    )
    witness_families: tuple[str, ...] = (
        "energy_delta",
        "mean_delta",
        "variance_delta",
        "shape_residual",
        "spectral_delta",
        "correlation_delta",
        "covariance_delta",
        "lag_correlation_delta",
    )
    detection_witnesses: tuple[str, ...] = (
        "mean_z",
        "variance_log_ratio",
        "local_energy_z",
        "correlation_shift",
        "covariance_shift",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CapabilityProtocol":
        """Construct a protocol from a mapping with conservative validation."""

        if payload is None:
            return cls()
        allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        unknown = sorted(set(payload.keys()) - allowed)
        if unknown:
            raise ValueError(f"Unknown capability protocol keys: {unknown}")
        normalized: dict[str, Any] = dict(payload)
        for key in [
            "delta_grid",
            "alpha_grid",
            "witness_families",
            "detection_witnesses",
            "window_length_bins",
        ]:
            if key in normalized and not isinstance(normalized[key], tuple):
                normalized[key] = tuple(normalized[key])
        if "calibration_min_clean_scan_count_for_alpha" in normalized:
            normalized["calibration_min_clean_scan_count_for_alpha"] = (
                _normalize_min_count_table(
                    normalized["calibration_min_clean_scan_count_for_alpha"]
                )
            )
        if "window_length_policy_mode" in normalized:
            mode = str(normalized["window_length_policy_mode"]).lower()
            if mode not in {"exact", "binned"}:
                raise ValueError(
                    "window_length_policy_mode must be 'exact' or 'binned'"
                )
            normalized["window_length_policy_mode"] = mode
        if "window_length_bins" in normalized:
            bins = tuple(
                sorted({int(item) for item in normalized["window_length_bins"]})
            )
            if any(item <= 0 for item in bins):
                raise ValueError("window_length_bins must contain positive integers")
            normalized["window_length_bins"] = bins
        if (
            "calibration_split" in normalized
            and normalized["calibration_split"] is not None
        ):
            normalized["calibration_split"] = str(normalized["calibration_split"])
        if "legacy_detectability_max_clean_instances" in normalized:
            limit = int(normalized["legacy_detectability_max_clean_instances"])
            if limit < 0:
                raise ValueError(
                    "legacy_detectability_max_clean_instances must be >= 0"
                )
            normalized["legacy_detectability_max_clean_instances"] = limit
        return cls(**normalized)


@dataclass(frozen=True)
class CapabilityRunConfig:
    """Run-level settings loaded from a capability YAML file."""

    protocol: CapabilityProtocol
    profiles: tuple[str, ...] | None = None
    profile_preset: str | None = None
    output_format: Literal["csv", "parquet", "both"] | None = None
    release_csv: bool | None = None
    allow_parquet_fallback: bool | None = None
    label_export: Literal["full", "diagnostics"] | None = None


def protocol_from_yaml(path: str | None) -> CapabilityProtocol:
    """Load a protocol from YAML if a path is supplied."""

    if path is None:
        return CapabilityProtocol()
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Capability protocol YAML must contain a mapping")
    return _protocol_from_payload(payload)


def capability_run_config_from_yaml(path: str | None) -> CapabilityRunConfig:
    """Load protocol and run-level settings from YAML.

    The numeric capability protocol remains separate from execution settings.
    CLI entrypoints use this object with the precedence ``CLI > YAML > default``.
    """

    if path is None:
        return CapabilityRunConfig(protocol=CapabilityProtocol())
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Capability protocol YAML must contain a mapping")
    return CapabilityRunConfig(
        protocol=_protocol_from_payload(payload),
        profiles=_normalize_profiles(payload.get("profiles")),
        profile_preset=_optional_string(payload.get("profile_preset")),
        output_format=_output_format_from_payload(payload),
        release_csv=_optional_bool(_nested_get(payload, ("output", "release_csv"))),
        allow_parquet_fallback=_optional_bool(
            _nested_get(payload, ("output", "allow_parquet_fallback"))
        ),
        label_export=_label_export_from_payload(payload),
    )


def _normalize_min_count_table(value: Any) -> tuple[tuple[float, int], ...]:
    if isinstance(value, Mapping):
        pairs = [(float(alpha), int(count)) for alpha, count in value.items()]
    else:
        pairs = []
        for item in value:
            if isinstance(item, Mapping):
                alpha = item.get("alpha")
                count = item.get("count")
            else:
                alpha, count = item
            pairs.append((float(cast(Any, alpha)), int(cast(Any, count))))
    for alpha, count in pairs:
        if alpha <= 0:
            raise ValueError("calibration minimum-count alpha keys must be positive")
        if count < 0:
            raise ValueError("calibration minimum counts must be non-negative")
    return tuple(sorted(pairs, key=lambda item: item[0]))


def _protocol_from_payload(payload: Mapping[str, Any]) -> CapabilityProtocol:
    if "capability_protocol" in payload:
        nested = payload["capability_protocol"] or {}
        if not isinstance(nested, Mapping):
            raise ValueError("capability_protocol must contain a mapping")
        return CapabilityProtocol.from_mapping(nested)
    allowed = set(CapabilityProtocol.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    protocol_payload = {key: value for key, value in payload.items() if key in allowed}
    calibration = payload.get("calibration")
    if (
        isinstance(calibration, Mapping)
        and "min_clean_scan_count_for_alpha" in calibration
    ):
        protocol_payload["calibration_min_clean_scan_count_for_alpha"] = calibration[
            "min_clean_scan_count_for_alpha"
        ]
    if isinstance(calibration, Mapping) and "split" in calibration:
        protocol_payload["calibration_split"] = calibration["split"]
    legacy_detectability = payload.get("legacy_detectability")
    if (
        isinstance(legacy_detectability, Mapping)
        and "max_clean_instances" in legacy_detectability
    ):
        protocol_payload["legacy_detectability_max_clean_instances"] = (
            legacy_detectability["max_clean_instances"]
        )
    _apply_window_length_policy(protocol_payload, payload)
    return CapabilityProtocol.from_mapping(protocol_payload)


def _normalize_profiles(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items: Sequence[Any] = value.replace(",", " ").split()
    elif isinstance(value, Sequence):
        items = value
    else:
        raise ValueError("profiles must be a string or sequence")
    profiles: list[str] = []
    for item in items:
        for part in str(item).split(","):
            name = part.strip()
            if name:
                profiles.append(name)
    return tuple(profiles) if profiles else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _nested_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _output_format_from_payload(
    payload: Mapping[str, Any],
) -> Literal["csv", "parquet", "both"] | None:
    output = payload.get("output")
    if not isinstance(output, Mapping):
        return None
    raw = output.get(
        "output_format", output.get("internal_format", output.get("format"))
    )
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value not in {"csv", "parquet", "both"}:
        raise ValueError("output format must be one of {'csv', 'parquet', 'both'}")
    return value  # type: ignore[return-value]


def _label_export_from_payload(
    payload: Mapping[str, Any],
) -> Literal["full", "diagnostics"] | None:
    raw = payload.get("label_export", _nested_get(payload, ("output", "label_export")))
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value not in {"full", "diagnostics"}:
        raise ValueError("label_export must be 'full' or 'diagnostics'")
    return value  # type: ignore[return-value]


def _apply_window_length_policy(
    protocol_payload: dict[str, Any], payload: Mapping[str, Any]
) -> None:
    raw_policy = payload.get("window_length_policy")
    if raw_policy is None:
        blind_scan = payload.get("blind_scan")
        if isinstance(blind_scan, Mapping):
            raw_policy = blind_scan.get("window_length_policy")
    if raw_policy is None:
        return
    if isinstance(raw_policy, str):
        protocol_payload["window_length_policy_mode"] = raw_policy
        return
    if not isinstance(raw_policy, Mapping):
        raise ValueError("window_length_policy must be a string or mapping")
    if "mode" in raw_policy:
        protocol_payload["window_length_policy_mode"] = raw_policy["mode"]
    if "bins" in raw_policy:
        protocol_payload["window_length_bins"] = raw_policy["bins"]


def window_length_bin(length: int, protocol: CapabilityProtocol) -> int:
    """Return the protocol scan-window length for an event support length."""

    value = max(1, int(length))
    if str(protocol.window_length_policy_mode).lower() != "binned":
        return value
    bins = tuple(int(item) for item in protocol.window_length_bins if int(item) > 0)
    if not bins:
        return value
    for candidate in sorted(bins):
        if value <= candidate:
            return int(candidate)
    return value
