"""Segment-planning policy resolution."""

from __future__ import annotations

import copy
from typing import Any, Mapping

DEFAULT_MIN_SEGMENT_LENGTH_BY_ANOMALY: dict[str, int] = {
    "amplitude": 5,
    "channel-rewiring": 8,
    "correlation-flip": 8,
    "covariance-change": 8,
    "shared-factor-break": 8,
    "mean": 5,
    "variance": 5,
    "platform": 5,
    "pattern": 5,
    "pattern-shift": 5,
    "trend": 5,
    "lag-synchronization": 10,
    "mode-correlation": 5,
}
GROUP_RELATION_ANOMALY_TYPES = frozenset(
    {
        "correlation-flip",
        "covariance-change",
        "channel-rewiring",
        "lag-synchronization",
        "shared-factor-break",
    }
)


def resolve_minimum_segment_length(
    anomaly_type: str,
    min_segment_length_by_anomaly: Mapping[str, Any],
) -> int:
    """Resolve minimum segment length for an anomaly type.

    Parameters
    ----------
    anomaly_type : str
        Canonical anomaly type.
    min_segment_length_by_anomaly : Mapping[str, Any]
        User-configured anomaly-specific overrides.

    Returns
    -------
    int
        Minimum segment length.
    """
    if anomaly_type in min_segment_length_by_anomaly:
        return int(min_segment_length_by_anomaly[anomaly_type])
    return int(DEFAULT_MIN_SEGMENT_LENGTH_BY_ANOMALY.get(anomaly_type, 1))


def resolve_special_anomaly_policy(
    anomaly_type: str,
    special_anomaly_policies: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve implicit special policy defaults for an anomaly type.

    Parameters
    ----------
    anomaly_type : str
        Canonical anomaly type.
    special_anomaly_policies : Mapping[str, Any]
        User-configured special policy mapping.

    Returns
    -------
    dict[str, Any]
        Effective special policy.
    """
    raw = special_anomaly_policies.get(anomaly_type, {})
    if isinstance(raw, Mapping):
        special = copy.deepcopy(dict(raw))
    else:
        special = {}
    if anomaly_type == "mode-correlation":
        special.setdefault("channel_policy", "paired-random")
        special.setdefault("segment_planner", {"planner": "mode_grid_segments"})
    elif anomaly_type in GROUP_RELATION_ANOMALY_TYPES:
        special.setdefault("channel_policy", "paired-random")
    return special


def resolve_segment_planner(
    *,
    anomaly_type: str,
    segment_planner: Mapping[str, Any],
    special_anomaly_policies: Mapping[str, Any],
    planner_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve effective segment planner configuration.

    Parameters
    ----------
    anomaly_type : str
        Canonical anomaly type.
    segment_planner : Mapping[str, Any]
        Configured segment planner mapping.
    special_anomaly_policies : Mapping[str, Any]
        Configured special anomaly policies.
    planner_override : Mapping[str, Any] or None
        Optional runtime planner override.

    Returns
    -------
    dict[str, Any]
        Effective planner config with normalized planner name.
    """
    default_cfg = {"planner": "uniform_segments"}
    configured_default = segment_planner.get("default", {})
    planner_cfg = _merge_dicts(default_cfg, _mapping_or_empty(configured_default))

    if anomaly_type in segment_planner:
        planner_cfg = _merge_dicts(
            planner_cfg, _mapping_or_empty(segment_planner[anomaly_type])
        )
    elif anomaly_type == "extremum" and "extremum" in segment_planner:
        planner_cfg = _merge_dicts(
            planner_cfg, _mapping_or_empty(segment_planner["extremum"])
        )

    special = resolve_special_anomaly_policy(anomaly_type, special_anomaly_policies)
    special_planner = special.get("segment_planner", {})
    if isinstance(special_planner, Mapping):
        planner_cfg = _merge_dicts(planner_cfg, special_planner)
    if isinstance(planner_override, Mapping):
        planner_cfg = _merge_dicts(planner_cfg, planner_override)

    planner_name = str(planner_cfg.get("planner", "uniform_segments")).lower()
    planner_cfg["planner"] = planner_name
    return planner_cfg


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _merge_dicts(*parts: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in dict(part).items():
            if (
                key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged
