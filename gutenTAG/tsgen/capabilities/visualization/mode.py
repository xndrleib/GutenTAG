"""Mode-correlation visual audit helpers."""

from __future__ import annotations

from typing import Mapping


def is_mode_event(event: Mapping[str, object]) -> bool:
    """Return whether an event belongs to a mode/regime relation family."""

    return "mode" in str(event.get("anomaly_type", "")) or "regime" in str(
        event.get("constraint_tag", "")
    )
