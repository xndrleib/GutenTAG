"""Runtime anomaly object builders."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ..anomalies import Anomaly, AnomalyKind, Position
from ..config.parser import decode_trend_obj
from ..utils.global_variables import PARAMETERS

GROUP_LEVEL_ANOMALY_TYPES = frozenset(
    {
        "mode-correlation",
        "correlation-flip",
        "covariance-change",
        "channel-rewiring",
        "lag-synchronization",
        "shared-factor-break",
    }
)


class AnomalySegment(Protocol):
    """Minimal segment contract required to build runtime anomalies."""

    start: int
    length: int
    channel: int


def build_anomalies(
    *,
    anomaly_type: str,
    anomaly_parameters_per_segment: Sequence[Mapping[str, Any]],
    segment_plan: Iterable[AnomalySegment],
    group_level_anomaly_types: frozenset[str] = GROUP_LEVEL_ANOMALY_TYPES,
) -> list[Anomaly]:
    """Build runtime anomaly objects for planned segments.

    Parameters
    ----------
    anomaly_type : str
        Canonical anomaly type.
    anomaly_parameters_per_segment : Sequence[Mapping[str, Any]]
        Per-segment anomaly parameters.
    segment_plan : Iterable[AnomalySegment]
        Planned anomaly segments.
    group_level_anomaly_types : frozenset[str]
        Anomaly types applied by group-level operators rather than built-in
        single-channel anomaly objects.

    Returns
    -------
    list[Anomaly]
        Runtime anomaly objects.

    Raises
    ------
    ValueError
        If the segment and parameter counts differ.
    """
    segments = list(segment_plan)
    if len(segments) != len(anomaly_parameters_per_segment):
        raise ValueError(
            "segment_plan and anomaly_parameters_per_segment must have equal length"
        )

    anomalies: list[Anomaly] = []
    for segment, anomaly_parameters in zip(segments, anomaly_parameters_per_segment):
        anomaly = Anomaly(
            position=Position.Middle,
            exact_position=segment.start,
            anomaly_length=segment.length,
            channel=segment.channel,
            creeping_length=0,
        )
        if anomaly_type not in group_level_anomaly_types:
            anomaly_kind_object = build_single_anomaly_kind(
                anomaly_type=anomaly_type,
                parameters=anomaly_parameters,
                anomaly_length=segment.length,
            )
            anomaly.set_anomaly(anomaly_kind_object)
        anomalies.append(anomaly)
    return anomalies


def build_single_anomaly_kind(
    anomaly_type: str, parameters: Mapping[str, Any], anomaly_length: int
) -> Any:
    """Build a built-in anomaly kind object from resolved parameters.

    Parameters
    ----------
    anomaly_type : str
        Canonical anomaly type.
    parameters : Mapping[str, Any]
        Resolved anomaly parameters.
    anomaly_length : int
        Segment length used to decode trend objects.

    Returns
    -------
    Any
        Runtime anomaly kind object created by ``AnomalyKind``.
    """
    if anomaly_type == "trend":
        raw = copy.deepcopy(dict(parameters))
        oscillation = raw.get(
            PARAMETERS.OSCILLATION,
            {"kind": "sine", "frequency": 2.0, "amplitude": 1.0},
        )
        trend = decode_trend_obj(copy.deepcopy(oscillation), anomaly_length)
        trend_parameters: dict[str, Any] = {PARAMETERS.TREND: trend}
        if "transition_length" in raw:
            trend_parameters["transition_length"] = int(raw["transition_length"])
        if "boundary_mode" in raw:
            trend_parameters["boundary_mode"] = str(raw["boundary_mode"])
        if "envelope_kind" in raw:
            trend_parameters["envelope_kind"] = str(raw["envelope_kind"])
        if "min_effect_delta" in raw:
            trend_parameters["min_effect_delta"] = float(raw["min_effect_delta"])
        return AnomalyKind(anomaly_type).create(trend_parameters)
    return AnomalyKind(anomaly_type).create(copy.deepcopy(dict(parameters)))
