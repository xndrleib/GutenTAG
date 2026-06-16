"""Shared helpers for anomaly parameter policies."""

from __future__ import annotations

from typing import Any


def parse_numeric_pair(raw: Any, field_name: str) -> tuple[float, float]:
    """Parse a two-value numeric configuration field.

    Parameters
    ----------
    raw : Any
        Raw configuration value.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    tuple[float, float]
        Parsed numeric pair.

    Raises
    ------
    ValueError
        If ``raw`` is not a two-item list or tuple.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"'{field_name}' must be a list/tuple with exactly two values."
        )
    return float(raw[0]), float(raw[1])


def ordered_numeric_pair(raw: Any, field_name: str) -> tuple[float, float]:
    """Parse a two-value numeric field and return it in ascending order.

    Parameters
    ----------
    raw : Any
        Raw configuration value.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    tuple[float, float]
        Parsed numeric pair with the smaller value first.
    """
    left, right = parse_numeric_pair(raw, field_name)
    if left > right:
        return right, left
    return left, right
