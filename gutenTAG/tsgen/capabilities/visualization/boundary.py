"""Boundary-focused visual audit helpers."""

from __future__ import annotations

from typing import Any, Mapping, cast


def boundary_annotation(event: Mapping[str, object]) -> str:
    """Return a compact boundary diagnostic label for an event."""

    status = str(event.get("boundary_status", "unknown"))
    share = event.get("boundary_energy_share")
    if share is None:
        return status
    return f"{status} (boundary_energy_share={float(cast(Any, share)):.3g})"
