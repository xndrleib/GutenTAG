from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .group_channel_rewiring import (
    apply_channel_rewiring_group as _apply_channel_rewiring_group,
)
from .group_correlation_flip import (
    apply_correlation_flip_group as _apply_correlation_flip_group,
)
from .group_covariance_change import (
    apply_covariance_change_group as _apply_covariance_change_group,
)
from .group_lag_synchronization import (
    apply_lag_synchronization_group as _apply_lag_synchronization_group,
)
from .group_mode_correlation import (
    apply_mode_correlation_group as _apply_mode_correlation_group,
)
from .group_shared_factor_break import (
    apply_shared_factor_break_group as _apply_shared_factor_break_group,
)


@dataclass(frozen=True)
class GroupAnomalyRuntime:
    compose_window: Callable[..., np.ndarray]
    replace_window: Callable[..., None]
    compose_noise: Callable[..., np.ndarray]
    replace_noise: Callable[..., None]
    resolve_label_bounds: Callable[..., Tuple[int, int]]
    to_builtin: Callable[[Any], Any]


def apply_group_anomaly(
    *,
    anomaly_type: str,
    group_indices: List[int],
    segment_plan: List[Any],
    base: np.ndarray,
    channel_bos: List[Any],
    labels: np.ndarray,
    used_positions: Dict[int, List[Tuple[int, int]]],
    anomaly_parameters_per_segment: List[Dict[str, Any]],
    runtime: GroupAnomalyRuntime,
) -> List[Dict[str, Any]]:
    handler = _GROUP_ANOMALY_HANDLERS.get(anomaly_type)
    if handler is None:
        raise ValueError(f"Unsupported group-level anomaly type: {anomaly_type}")
    return handler(
        group_indices=group_indices,
        segment_plan=segment_plan,
        base=base,
        channel_bos=channel_bos,
        labels=labels,
        used_positions=used_positions,
        anomaly_type=anomaly_type,
        anomaly_parameters_per_segment=anomaly_parameters_per_segment,
        runtime=runtime,
    )


_GROUP_ANOMALY_HANDLERS = {
    "mode-correlation": _apply_mode_correlation_group,
    "covariance-change": _apply_covariance_change_group,
    "correlation-flip": _apply_correlation_flip_group,
    "channel-rewiring": _apply_channel_rewiring_group,
    "lag-synchronization": _apply_lag_synchronization_group,
    "shared-factor-break": _apply_shared_factor_break_group,
}
