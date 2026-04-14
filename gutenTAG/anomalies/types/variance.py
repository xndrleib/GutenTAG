from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import CylinderBellFunnel, RandomModeJump


@dataclass
class AnomalyVarianceParameters:
    variance: float = 0.0
    min_effect_delta: float = 0.0
    transition_length: Optional[int] = None


class AnomalyVariance(BaseAnomaly):
    def __init__(self, parameters: AnomalyVarianceParameters):
        super().__init__()
        self.variance = parameters.variance
        self.min_effect_delta = max(0.0, float(parameters.min_effect_delta))
        self.transition_length = parameters.transition_length

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        base = anomaly_protocol.base_oscillation
        base_kind = str(base.get_base_oscillation_kind())
        if anomaly_protocol.base_oscillation_kind == RandomModeJump.KIND:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )

        elif anomaly_protocol.base_oscillation_kind == CylinderBellFunnel.KIND:
            subsequence = base.generate_only_base(
                anomaly_protocol.ctx.to_bo(), variance=self.variance
            )[anomaly_protocol.start : anomaly_protocol.end]
            anomaly_protocol.subsequences.append(subsequence)

        else:
            length = anomaly_protocol.end - anomaly_protocol.start
            if length <= 0:
                return anomaly_protocol

            clean_segment = base.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            amplitude = abs(float(getattr(base, "amplitude", 1.0)))
            reference_scale = self._estimate_reference_scale(
                clean_segment=clean_segment,
                base_kind=base_kind,
                amplitude=amplitude,
            )

            target_std = max(0.0, float(self.variance) * reference_scale)

            if target_std <= 0:
                original_noise = np.array(
                    base.noise[anomaly_protocol.start : anomaly_protocol.end], copy=True
                )
                envelope = self.build_symmetric_envelope(length, self.transition_length)
                candidate_noise = original_noise * (1.0 - envelope)
                if candidate_noise.size > 0:
                    candidate_noise[0] = original_noise[0]
                    candidate_noise[-1] = original_noise[-1]
                base.noise[anomaly_protocol.start : anomaly_protocol.end] = candidate_noise
                return anomaly_protocol

            original_noise = np.array(
                base.noise[anomaly_protocol.start : anomaly_protocol.end], copy=True
            )
            subsequence_noise = base.generate_noise(
                anomaly_protocol.ctx.to_bo(), target_std, length
            )
            envelope = self.build_symmetric_envelope(length, self.transition_length)
            if base_kind in {"polynomial", "random-walk"}:
                # Smooth-trend carriers need an additive residual burst; simple
                # replacement tends to stay too close to the original low-noise
                # regime and becomes visually weak under the QC detector stack.
                candidate_noise = original_noise + envelope * subsequence_noise
            else:
                candidate_noise = original_noise + envelope * (
                    subsequence_noise - original_noise
                )
            effective_min_effect = float(self.min_effect_delta)
            if base_kind in {"polynomial", "random-walk"}:
                effective_min_effect = max(effective_min_effect, 0.75 * reference_scale)
            if effective_min_effect > 0.0:
                candidate_noise = self._enforce_min_effect_delta(
                    candidate_noise=candidate_noise,
                    original_noise=original_noise,
                    min_effect_delta=effective_min_effect,
                    rng=anomaly_protocol.rng,
                    preserve_edges=False,
                )
            if candidate_noise.size > 0:
                candidate_noise[0] = original_noise[0]
                candidate_noise[-1] = original_noise[-1]
            if effective_min_effect > 0.0:
                candidate_noise = self._enforce_min_effect_delta(
                    candidate_noise=candidate_noise,
                    original_noise=original_noise,
                    min_effect_delta=effective_min_effect,
                    rng=anomaly_protocol.rng,
                    preserve_edges=True,
                )
            base.noise[anomaly_protocol.start : anomaly_protocol.end] = candidate_noise
        return anomaly_protocol

    @staticmethod
    def _linear_detrend(values: np.ndarray) -> np.ndarray:
        series = np.asarray(values, dtype=np.float64)
        n = int(series.shape[0])
        if n <= 2:
            return series - float(np.mean(series))
        x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        degree = 2 if n >= 5 else 1
        coeff = np.polyfit(x, series, deg=degree)
        trend = np.polyval(coeff, x)
        return (series - trend).astype(np.float64)

    @classmethod
    def _estimate_reference_scale(
        cls,
        *,
        clean_segment: np.ndarray,
        base_kind: str,
        amplitude: float,
    ) -> float:
        local = np.asarray(clean_segment, dtype=np.float64)
        if local.size == 0:
            return max(0.25 * abs(float(amplitude)), 1e-6)
        local_scale = float(np.std(local))
        base_scale = max(local_scale, 0.25 * abs(float(amplitude)), 1e-6)
        if str(base_kind) not in {"polynomial", "random-walk"}:
            return base_scale
        residual = cls._linear_detrend(local)
        residual_scale = float(np.std(residual))
        window_ptp = float(np.ptp(local))
        trend_floor = 0.75 * max(window_ptp, abs(float(amplitude)), 1e-6)
        return max(base_scale, residual_scale * 3.0, trend_floor, 1e-6)

    @staticmethod
    def _enforce_min_effect_delta(
        *,
        candidate_noise: np.ndarray,
        original_noise: np.ndarray,
        min_effect_delta: float,
        rng,
        preserve_edges: bool,
    ) -> np.ndarray:
        """Ensure a minimum visible perturbation inside the source window."""
        candidate = np.asarray(candidate_noise, dtype=np.float64).copy()
        original = np.asarray(original_noise, dtype=np.float64)
        if candidate.size == 0 or min_effect_delta <= 0.0:
            return candidate

        left = 1 if preserve_edges and candidate.size > 2 else 0
        right = candidate.size - 1 if preserve_edges and candidate.size > 2 else candidate.size
        if right <= left:
            return candidate

        delta = candidate - original
        active_delta = delta[left:right]
        if active_delta.size == 0:
            return candidate

        max_abs_delta = float(np.max(np.abs(active_delta)))
        if max_abs_delta >= min_effect_delta:
            return candidate

        if max_abs_delta > 1e-12:
            scale = float(min_effect_delta / max_abs_delta)
            delta[left:right] = active_delta * scale
            return original + delta

        center = left + active_delta.size // 2
        sign = -1.0 if rng.random() < 0.5 else 1.0
        delta[center] = sign * min_effect_delta
        return original + delta

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyVarianceParameters]:
        return AnomalyVarianceParameters
