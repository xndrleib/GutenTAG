"""Gaussian feature-mixture carrier with explicitly specified target kernels."""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base_oscillation import BaseOscillation
from .interface import BaseOscillationInterface
from ..utils.types import BOGenerationContext
from ..tsgen.processes.kernels import KernelSpec, KERNELS, feature_sample


class GaussianProcessMixture(BaseOscillationInterface):
    """Sample a finite-feature approximation without sample normalization.

    Offset is applied by ``apply_variations``, never by the carrier. The legacy
    carrier API samples spectra per call; reproducible law-level replicates use
    ``ProcessLaw`` in ``tsgen.processes.laws`` instead.
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
        if self.gp_components < 1 or self.gp_features < 1:
            raise ValueError("GP components and features must be positive")
        for lo, hi in ((self.gp_length_scale_min, self.gp_length_scale_max),
                       (self.gp_period_min, self.gp_period_max)):
            if not np.isfinite([lo, hi]).all() or not 0 < lo <= hi:
                raise ValueError("GP ranges must be finite, positive and ordered")
        if not np.isfinite(self.gp_linear_weight) or self.gp_linear_weight < 0:
            raise ValueError("gp_linear_weight must be nonnegative and finite")

    def get_base_oscillation_kind(self) -> str:
        return self.KIND

    def get_timeseries_periods(self) -> Optional[int]:
        return None

    def generate_only_base(self, ctx: BOGenerationContext,
                           length: Optional[int] = None, *args, **kwargs) -> np.ndarray:
        n = int(self.length if length is None else length)
        if n < 1:
            raise ValueError("GP length must be positive")
        rng = ctx.rng
        t = np.linspace(0.0, 1.0, n)
        weights = rng.lognormal(0.0, 0.45, self.gp_components)
        weights /= np.linalg.norm(weights)
        result = np.zeros(n)
        self.kernel_metadata = []
        for weight in weights:
            spec = KernelSpec(
                kind=str(rng.choice(KERNELS)),
                length_scale=float(np.exp(rng.uniform(np.log(self.gp_length_scale_min),
                                                     np.log(self.gp_length_scale_max)))),
                period=float(rng.uniform(self.gp_period_min, self.gp_period_max)),
                alpha=float(np.exp(rng.uniform(-1.0, 1.0))),
            )
            omega = spec.spectrum(rng, self.gp_features)
            result += weight * feature_sample(omega, t, rng)
            self.kernel_metadata.append({**vars(spec), "weight": float(weight)})
        if self.gp_linear_weight:
            slope, curvature = rng.normal(size=2) * self.gp_linear_weight
            result += slope * (t - 0.5) + 0.5 * curvature * (t - 0.5)**2
        return float(self.amplitude) * result


BaseOscillation.register(GaussianProcessMixture.KIND, GaussianProcessMixture)
