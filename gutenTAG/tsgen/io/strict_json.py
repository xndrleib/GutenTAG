"""Strict JSON serialization helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sanitize_json_value(value: Any) -> Any:
    """Convert Python/NumPy values into strict JSON-compatible values.

    Non-finite floats are represented as ``None`` so that writers can use
    ``allow_nan=False`` and produce standards-compliant JSON.
    """
    if isinstance(value, Mapping):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    indent: int = 2,
) -> None:
    """Write strict JSON to disk."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json_value(payload),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=sort_keys,
        )
        handle.write("\n")
