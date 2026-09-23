"""Stochastic Gaussian-process carrier for v13 synthetic benchmarks."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import BaseOscillation
from .interface import BaseOscillationInterface
from ..utils.types import BOGenerationContext


class GaussianProcessMixture(BaseOscillationInterface):
    """Sample a zero-mean GP from a randomized compositional kernel.

    The implementation intentionally avoids a dense T x T covariance matrix.
    Stationary components are sampled through random Fourier features, which
    keeps generation practical for the 10k-point benchmark regime.
    """

    KIND = "gp-mixture"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gp_components = int(kwargs.get("gp_components", 4))
        self.gp_features = int(kwargs.get("gp_features", 48))
        self.gp_length_scale_min = float(kwargs.get("gp_length_scale_min", 0.01))
        self.gp_length_scale_max = float(kwargs.get("gp_length_scale_max", 0.35))
        self.gp_period_min = float(kwargs.get("gp_period_min", 0.03))
        self.gp_period_max = float(kwargs.get("gp_period_max", 0.5))
        self.gp_linear_weight = float(kwargs.get("gp_linear_weight", 0.2))

    def get_base_oscillation_kind(self) -> str:
        return self.KIND

    def get_timeseries_periods(self) -> Optional[int]:
        return None

    def generate_only_base(
        self, ctx: BOGenerationContext, length: Optional[int] = None, *args, **kwargs
    ) -> np.ndarray:
        n = int(length if length is not None else self.length)
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        rng = ctx.rng
        signal = np.zeros(n, dtype=np.float64)
        component_count = max(1, self.gp_components)
        features = max(8, self.gp_features)
        for _ in range(component_count):
            kernel = str(rng.choice(["rbf", "matern32", "periodic", "rq"]))
            weight = float(rng.lognormal(mean=0.0, sigma=0.45))
            length_scale = float(
                np.exp(
                    rng.uniform(
                        np.log(max(self.gp_length_scale_min, 1e-4)),
                        np.log(max(self.gp_length_scale_max, self.gp_length_scale_min + 1e-4)),
                    )
                )
            )
            if kernel == "periodic":
                period = float(rng.uniform(self.gp_period_min, self.gp_period_max))
                frequencies = (
                    rng.integers(1, 8, size=features).astype(np.float64) / max(period, 1e-4)
                )
            elif kernel == "matern32":
                frequencies = rng.standard_t(df=3.0, size=features) / length_scale
            elif kernel == "rq":
                frequencies = rng.standard_t(df=5.0, size=features) / length_scale
            else:
                frequencies = rng.normal(0.0, 1.0 / length_scale, size=features)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=features)
            coeff = rng.normal(0.0, 1.0, size=features)
            basis = np.cos(2.0 * np.pi * t[:, None] * frequencies[None, :] + phases)
            component = np.sqrt(2.0 / features) * (basis @ coeff)
            signal += weight * component
        if self.gp_linear_weight > 0.0:
            slope = float(rng.normal(0.0, self.gp_linear_weight))
            curvature = float(rng.normal(0.0, 0.5 * self.gp_linear_weight))
            centered_t = t - 0.5
            signal += slope * centered_t + curvature * centered_t**2
        signal -= float(np.mean(signal))
        scale = float(np.std(signal))
        if scale > 1e-12:
            signal /= scale
        return (float(self.offset) + float(self.amplitude) * signal).astype(np.float64)


BaseOscillation.register(GaussianProcessMixture.KIND, GaussianProcessMixture)
