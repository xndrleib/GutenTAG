"""Per-instance density metrics and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def compute_density_metrics(
    *,
    labels: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    support_label_mode: str,
) -> dict[str, Any]:
    """Compute labeled/source density metrics for one instance."""

    achieved_density_labeled = float(labels.max(axis=1).mean())
    achieved_density_source = achieved_density_labeled
    if support_label_mode == "effective_support":
        source_labels = np.zeros_like(labels, dtype=np.int8)
        for event in events:
            source_start = int(event.get("source_start", event["start"]))
            source_end = int(event.get("source_end", event["end"]))
            source_channel = int(event["channel"])
            if source_end > source_start:
                source_labels[source_start:source_end, source_channel] = 1
        achieved_density_source = float(source_labels.max(axis=1).mean())
        density_for_validation = achieved_density_source
        density_validation_mode = "source_support"
    else:
        density_for_validation = achieved_density_labeled
        density_validation_mode = "labeled_support"
    return {
        "achieved_density_labeled": achieved_density_labeled,
        "achieved_density_source": achieved_density_source,
        "density_for_validation": density_for_validation,
        "density_validation_mode": density_validation_mode,
    }


def validate_density(
    *,
    target_density: float,
    active_density_range: tuple[float, float],
    active_density_tolerance: float,
    achieved_density_labeled: float,
    achieved_density_source: float,
    density_for_validation: float,
    density_validation_mode: str,
) -> None:
    """Raise if achieved density violates configured range or target tolerance."""

    if active_density_tolerance < 0:
        raise ValueError("density_tolerance must be >= 0")
    lower_with_tolerance = active_density_range[0] - active_density_tolerance
    upper_with_tolerance = active_density_range[1] + active_density_tolerance
    if not (lower_with_tolerance <= density_for_validation <= upper_with_tolerance):
        raise ValueError(
            f"Achieved density {density_for_validation:.6f} outside target range "
            f"{active_density_range} | mode={density_validation_mode} "
            f"| tolerance={active_density_tolerance:.6f}"
            f"| labeled={achieved_density_labeled:.6f} | source={achieved_density_source:.6f}."
        )
    if abs(density_for_validation - target_density) > active_density_tolerance:
        raise ValueError(
            f"Density mismatch | target={target_density:.6f} | achieved={density_for_validation:.6f} "
            f"| tolerance={active_density_tolerance:.6f} | mode={density_validation_mode} "
            f"| labeled={achieved_density_labeled:.6f} | source={achieved_density_source:.6f}"
        )


__all__ = ["compute_density_metrics", "validate_density"]
