"""Fixed-onset segment planning.

This module owns the planner that places the first anomaly support from
split-level onset metadata such as ``ctx003``. The generator facade supplies the
runtime shape and segment factory; this module owns the onset interpretation and
alignment rules.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ONSET_EVENT_EXTRA_KEYS = (
    "requested_pre_context",
    "onset_bucket",
    "alignment_strategy",
    "alignment_error",
    "mode_grid_aligned",
    "mode_change_aligned",
    "support_independent_of_realized_mode_state",
    "mode_grid_block_size",
    "mode_grid_start_block",
    "mode_grid_end_block",
    "mode_grid_block_length",
    "mode_grid_min_gap_blocks",
)


@dataclass(frozen=True)
class _FixedOnsetRequest:
    requested_start: int
    planned_length: int
    alignment_strategy: str
    onset_bucket: str


@dataclass(frozen=True)
class _FixedOnsetPlacement:
    start: int
    length: int
    attrs: dict[str, Any]


def requested_pre_context_from_split(split: str) -> int | None:
    """Return fixed-onset context encoded in split names like ``ctx003``."""

    split_name = str(split)
    if not split_name.startswith("ctx"):
        return None
    raw_value = split_name[3:]
    if not raw_value.isdigit():
        return None
    return int(raw_value)


def inject_fixed_onset_split_context(
    planner_cfg: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """Inject split-encoded onset metadata into fixed-onset planner config."""

    resolved = copy.deepcopy(dict(planner_cfg))
    if str(resolved.get("planner", "uniform_segments")).lower() != (
        "fixed_first_onset_segments"
    ):
        return resolved
    if "requested_pre_context" not in resolved:
        requested = requested_pre_context_from_split(split)
        if requested is None:
            raise ValueError(
                "fixed_first_onset_segments requires split names like 'ctx003' "
                "or an explicit planner requested_pre_context."
            )
        resolved["requested_pre_context"] = int(requested)
    resolved.setdefault("onset_bucket", str(split))
    return resolved


def temporal_union_density(segment_plan: Sequence[Any], series_length: int) -> float:
    """Compute temporal source density without double-counting group channels."""

    if len(segment_plan) == 0 or int(series_length) <= 0:
        return 0.0
    length = int(series_length)
    mask = np.zeros(length, dtype=np.int8)
    for segment in segment_plan:
        start = max(0, min(int(segment.start), length))
        end = max(0, min(int(segment.end), length))
        if end > start:
            mask[start:end] = 1
    return float(mask.mean())


def event_extra_from_segment_attrs(
    attrs: Mapping[str, Any],
    source_start: int,
    support_start: int,
    sanitize: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Return event metadata derived from fixed-onset segment attrs."""

    extra: dict[str, Any] = {
        key: attrs[key] for key in ONSET_EVENT_EXTRA_KEYS if key in attrs
    }
    if "requested_pre_context" in attrs:
        extra["actual_source_start"] = int(source_start)
        extra["actual_support_start"] = int(support_start)
    if sanitize is not None:
        return sanitize(extra)
    return extra


def sample_fixed_first_onset_segments(
    *,
    rng: np.random.Generator,
    anomaly_type: str,
    planner_cfg: Mapping[str, Any],
    series_length: int,
    channels: int,
    segment_factory: Callable[..., Any],
    base_period_size: int | None = None,
) -> list[Any]:
    """Sample one segment whose first onset is controlled by split metadata."""

    request = _fixed_onset_request(
        anomaly_type=anomaly_type,
        planner_cfg=planner_cfg,
        base_period_size=base_period_size,
    )
    placement = _fixed_onset_placement(
        request=request,
        anomaly_type=anomaly_type,
        planner_cfg=planner_cfg,
        series_length=series_length,
        base_period_size=base_period_size,
    )
    channel = int(rng.integers(0, int(channels)))
    return [
        segment_factory(
            start=placement.start,
            end=placement.start + placement.length,
            length=placement.length,
            channel=channel,
            attrs=placement.attrs,
        )
    ]


def _fixed_onset_request(
    *,
    anomaly_type: str,
    planner_cfg: Mapping[str, Any],
    base_period_size: int | None,
) -> _FixedOnsetRequest:
    if "requested_pre_context" not in planner_cfg:
        raise ValueError("fixed_first_onset_segments requires 'requested_pre_context'.")
    requested_start = int(planner_cfg["requested_pre_context"])
    if requested_start < 0:
        raise ValueError("requested_pre_context must be >= 0")

    default_length = 1 if anomaly_type == "extremum" else 64
    raw_length = planner_cfg.get(
        "length",
        planner_cfg.get("segment_length", default_length),
    )
    planned_length = int(raw_length)
    if anomaly_type == "extremum":
        planned_length = 1
    if planned_length <= 0:
        raise ValueError("fixed_first_onset_segments length must be > 0")

    return _FixedOnsetRequest(
        requested_start=requested_start,
        planned_length=planned_length,
        alignment_strategy=_fixed_onset_alignment_strategy(
            anomaly_type=anomaly_type,
            planner_cfg=planner_cfg,
            base_period_size=base_period_size,
        ),
        onset_bucket=str(planner_cfg.get("onset_bucket", f"ctx{requested_start:03d}")),
    )


def _fixed_onset_alignment_strategy(
    *,
    anomaly_type: str,
    planner_cfg: Mapping[str, Any],
    base_period_size: int | None,
) -> str:
    strategy = str(
        planner_cfg.get("alignment_strategy", planner_cfg.get("align_to", "exact"))
    ).lower()
    if strategy != "auto":
        return strategy
    if anomaly_type == "mode-correlation" and base_period_size is not None:
        return "mode_grid"
    return "exact"


def _fixed_onset_placement(
    *,
    request: _FixedOnsetRequest,
    anomaly_type: str,
    planner_cfg: Mapping[str, Any],
    series_length: int,
    base_period_size: int | None,
) -> _FixedOnsetPlacement:
    actual_start = int(request.requested_start)
    planned_length = int(request.planned_length)
    attrs: dict[str, Any] = {
        "requested_pre_context": int(request.requested_start),
        "onset_bucket": request.onset_bucket,
        "alignment_strategy": request.alignment_strategy,
    }

    if request.alignment_strategy == "mode_grid":
        actual_start, planned_length, grid_attrs = _mode_grid_aligned_onset(
            request=request,
            planner_cfg=planner_cfg,
            series_length=series_length,
            base_period_size=base_period_size,
        )
        attrs.update(grid_attrs)
    elif request.alignment_strategy != "exact":
        raise ValueError(
            "fixed_first_onset_segments alignment_strategy must be one of "
            "{'exact','mode_grid','auto'}"
        )

    max_start = int(series_length) - planned_length
    if max_start < 0:
        raise ValueError(
            f"fixed_first_onset_segments length={planned_length} exceeds "
            f"series length={int(series_length)}."
        )
    actual_start = int(max(0, min(actual_start, max_start)))
    attrs["alignment_error"] = int(actual_start - request.requested_start)
    attrs["actual_source_start"] = int(actual_start)
    return _FixedOnsetPlacement(
        start=actual_start,
        length=planned_length,
        attrs=attrs,
    )


def _mode_grid_aligned_onset(
    *,
    request: _FixedOnsetRequest,
    planner_cfg: Mapping[str, Any],
    series_length: int,
    base_period_size: int | None,
) -> tuple[int, int, dict[str, Any]]:
    if base_period_size is None or int(base_period_size) <= 0:
        raise ValueError(
            "fixed_first_onset_segments alignment_strategy='mode_grid' "
            "requires a positive base_period_size."
        )
    block_size = int(base_period_size)
    actual_start = _ceil_to_block(request.requested_start, block_size)
    planned_length = int(request.planned_length)
    if bool(planner_cfg.get("align_length_to_grid", True)):
        planned_length = _ceil_to_block(planned_length, block_size)
    max_start = max(0, int(series_length) - planned_length)
    if actual_start > max_start:
        actual_start = int(max_start // block_size) * block_size
    return (
        actual_start,
        planned_length,
        _mode_grid_attrs(
            actual_start=actual_start,
            planned_length=planned_length,
            block_size=block_size,
        ),
    )


def _ceil_to_block(value: int, block_size: int) -> int:
    return int(np.ceil(float(value) / float(block_size)) * block_size)


def _mode_grid_attrs(
    *,
    actual_start: int,
    planned_length: int,
    block_size: int,
) -> dict[str, Any]:
    return {
        "mode_grid_aligned": True,
        "mode_change_aligned": False,
        "support_independent_of_realized_mode_state": True,
        "mode_grid_block_size": int(block_size),
        "mode_grid_start_block": int(actual_start // block_size),
        "mode_grid_end_block": int((actual_start + planned_length) // block_size),
        "mode_grid_block_length": int(planned_length // block_size),
    }
