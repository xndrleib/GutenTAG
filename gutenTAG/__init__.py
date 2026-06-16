"""Public package interface for GutenTAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .gutenTAG import GutenTAG
    from .timeseries import (
        INDEX_COLUMN_NAME,
        LABEL_COLUMN_NAME,
        TimeSeries,
        TrainingType,
    )
    from .ts_dataset_generation import (
        TSDatasetGenerator,
        TSGeneratorConfig,
        generate_ts_dataset,
    )

__all__ = [
    "__version__",
    "GutenTAG",
    "TimeSeries",
    "TrainingType",
    "LABEL_COLUMN_NAME",
    "INDEX_COLUMN_NAME",
    "TSDatasetGenerator",
    "TSGeneratorConfig",
    "generate_ts_dataset",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy-loading shim
    if name == "GutenTAG":
        from .gutenTAG import GutenTAG

        return GutenTAG
    if name in ("TimeSeries", "TrainingType", "LABEL_COLUMN_NAME", "INDEX_COLUMN_NAME"):
        from .timeseries import (
            INDEX_COLUMN_NAME,
            LABEL_COLUMN_NAME,
            TimeSeries,
            TrainingType,
        )

        mapping = {
            "TimeSeries": TimeSeries,
            "TrainingType": TrainingType,
            "LABEL_COLUMN_NAME": LABEL_COLUMN_NAME,
            "INDEX_COLUMN_NAME": INDEX_COLUMN_NAME,
        }
        return mapping[name]
    if name in ("TSDatasetGenerator", "TSGeneratorConfig", "generate_ts_dataset"):
        from .ts_dataset_generation import (
            TSGeneratorConfig,
            TSDatasetGenerator,
            generate_ts_dataset,
        )

        mapping = {
            "TSDatasetGenerator": TSDatasetGenerator,
            "TSGeneratorConfig": TSGeneratorConfig,
            "generate_ts_dataset": generate_ts_dataset,
        }
        return mapping[name]
    raise AttributeError(f"module 'gutenTAG' has no attribute '{name}'")
