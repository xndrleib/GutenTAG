"""Structural model registry."""

from __future__ import annotations

from .base import CapabilityModel
from .lowrank import LowRankResidualModel
from .pca import PCAResidualModel
from .regression import TargetRegressionResidualModel
from .var import VARResidualModel


def default_model_registry() -> tuple[CapabilityModel, ...]:
    """Return the default deterministic structural model zoo."""

    return (
        PCAResidualModel(rank=1),
        LowRankResidualModel(rank=2),
        TargetRegressionResidualModel(),
        VARResidualModel(),
    )
