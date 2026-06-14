"""Interpretable structural models for capability analysis."""

from .base import CapabilityModel
from .lowrank import LowRankResidualModel
from .pca import PCAResidualModel
from .registry import default_model_registry
from .regression import TargetRegressionResidualModel
from .var import VARResidualModel

__all__ = [
    "CapabilityModel",
    "LowRankResidualModel",
    "PCAResidualModel",
    "TargetRegressionResidualModel",
    "VARResidualModel",
    "default_model_registry",
]
