"""Small pandas typing helpers for capability tables."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy


def as_frame(value: object) -> pd.DataFrame:
    """Return ``value`` as a DataFrame at a known pandas boundary."""
    return cast(pd.DataFrame, value)


def as_series(value: object) -> pd.Series:
    """Return ``value`` as a Series at a known pandas boundary."""
    return cast(pd.Series, value)


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one DataFrame column as a typed Series."""
    return as_series(frame[name])


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one DataFrame column converted to numeric values."""
    return as_series(pd.to_numeric(column(frame, name), errors="coerce"))


def frame_groupby(
    frame: pd.DataFrame, by: str, *, dropna: bool = True
) -> DataFrameGroupBy:
    """Return a typed DataFrame groupby object."""
    return cast(DataFrameGroupBy, frame.groupby(by, dropna=dropna))


def sorted_frame(frame: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Return a typed DataFrame after sorting by column names."""
    return as_frame(frame.sort_values(by))


def row_mapping(row: object) -> dict[str, object]:
    """Return a pandas row as a plain mapping."""
    return dict(cast(Any, row))
