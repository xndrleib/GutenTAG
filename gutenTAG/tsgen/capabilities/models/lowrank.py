"""Low-rank/factor residual model."""

from __future__ import annotations

from .pca import PCAResidualModel


class LowRankResidualModel(PCAResidualModel):
    """Factor-style residual using a slightly wider subspace than PCA v1."""

    model_id = "lowrank_factor_residual_v1"
    family = "structural.lowrank_factor_residual"

    def __init__(self, rank: int = 2) -> None:
        super().__init__(rank=rank)
