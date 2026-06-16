"""Frequency anomaly parameter policies."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .common import ordered_numeric_pair


class PeriodLockedSegment(Protocol):
    """Minimal segment contract for period-locked frequency policy."""

    @property
    def attrs(self) -> Mapping[str, Any]: ...


SeedDeriver = Callable[..., int]


def apply_period_locked_frequency_policy(
    *,
    segment_plan: Sequence[PeriodLockedSegment],
    segment_params: Sequence[Mapping[str, Any]],
    planner_cfg: Mapping[str, Any],
    parameter_seed: int,
    derive_seed: SeedDeriver,
) -> list[dict[str, Any]]:
    """Set frequency factors from period-locked segment metadata.

    Parameters
    ----------
    segment_plan : Sequence[PeriodLockedSegment]
        Planned frequency segments with ``period_count`` in segment attrs.
    segment_params : Sequence[Mapping[str, Any]]
        Realized per-segment anomaly parameters.
    planner_cfg : Mapping[str, Any]
        Planner options containing period-ratio offsets and optional bounds.
    parameter_seed : int
        Seed used to derive deterministic per-segment choices.
    derive_seed : Callable[..., int]
        Repository seed-derivation helper.

    Returns
    -------
    list[dict[str, Any]]
        Parameter dictionaries with ``frequency_factor`` set per segment.

    Raises
    ------
    ValueError
        If offsets, period metadata, or bounds make the policy unsatisfiable.
    """
    offsets = _resolve_period_ratio_offsets(planner_cfg)
    factor_bounds = _resolve_frequency_factor_bounds(planner_cfg)

    resolved: list[dict[str, Any]] = []
    for idx, segment in enumerate(segment_plan):
        period_count = int(segment.attrs.get("period_count", 0))
        if period_count <= 0:
            raise ValueError(
                "period_locked_frequency requires segment attrs['period_count']."
            )

        candidate_factors = [
            float(period_count + offset) / float(period_count)
            for offset in offsets
            if period_count + offset > 0
        ]
        if factor_bounds is not None:
            candidate_factors = [
                factor
                for factor in candidate_factors
                if factor_bounds[0] <= factor <= factor_bounds[1]
            ]
        if len(candidate_factors) == 0:
            raise ValueError(
                "No valid frequency_factor candidates for period-locked "
                f"segment with period_count={period_count} and offsets={offsets}."
            )

        segment_rng = np.random.default_rng(
            derive_seed(parameter_seed, "period-locked-frequency", str(idx))
        )
        chosen_idx = int(segment_rng.integers(0, len(candidate_factors)))
        params = copy.deepcopy(dict(segment_params[idx]))
        params["frequency_factor"] = float(candidate_factors[chosen_idx])
        resolved.append(params)
    return resolved


def _resolve_period_ratio_offsets(planner_cfg: Mapping[str, Any]) -> list[int]:
    offsets_raw = planner_cfg.get("period_ratio_offsets", [-1, 1])
    if not isinstance(offsets_raw, (list, tuple)):
        offsets_raw = [-1, 1]
    offsets = sorted(
        {
            int(offset)
            for offset in offsets_raw
            if isinstance(offset, (int, float)) and int(offset) != 0
        }
    )
    if len(offsets) == 0:
        raise ValueError(
            "period_locked_frequency requires non-zero 'period_ratio_offsets'."
        )
    return offsets


def _resolve_frequency_factor_bounds(
    planner_cfg: Mapping[str, Any],
) -> tuple[float, float] | None:
    factor_bounds_raw = planner_cfg.get("frequency_factor_bounds")
    if factor_bounds_raw is None:
        return None
    return ordered_numeric_pair(
        factor_bounds_raw, "segment_planner.frequency_factor_bounds"
    )
