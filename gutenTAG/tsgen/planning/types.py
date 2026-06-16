"""Planning domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SegmentPlan:
    """Specification of a single anomaly segment."""

    start: int
    end: int
    length: int
    channel: int
    attrs: dict[str, Any] = field(default_factory=dict)
