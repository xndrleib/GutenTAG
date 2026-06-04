from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from . import BaseAnomaly, AnomalyProtocol
from ...base_oscillations import Polynomial, Formula, RandomModeJump
from ...tsgen.signal_ops import (
    enforce_min_effect_delta,
    local_centerline,
    match_boundary_value_and_slope,
    symmetric_envelope,
)


@dataclass
class AnomalyAmplitudeParameters:
    amplitude_factor: float = 1.0
    transition_length: Optional[int] = None
    center_mode: str = "linear"
    min_effect_delta: float = 0.0
    min_residual_scale: float = 1e-6


class AnomalyAmplitude(BaseAnomaly):
    def __init__(self, parameters: AnomalyAmplitudeParameters):
        super().__init__()
        self.amplitude_factor = float(parameters.amplitude_factor)
        self.transition_length = parameters.transition_length
        self.center_mode = str(parameters.center_mode or "linear")
        self.min_effect_delta = max(0.0, float(parameters.min_effect_delta))
        self.min_residual_scale = max(0.0, float(parameters.min_residual_scale))

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind in [
            Polynomial.KIND,
            Formula.KIND,
            RandomModeJump.KIND,
        ]:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
            return anomaly_protocol

        length = anomaly_protocol.end - anomaly_protocol.start
        if length <= 0:
            return anomaly_protocol
        clean_segment = anomaly_protocol.base_oscillation.timeseries[
            anomaly_protocol.start : anomaly_protocol.end
        ]
        if clean_segment.size == 0:
            return anomaly_protocol
        if np.isclose(self.amplitude_factor, 1.0) and self.min_effect_delta <= 0.0:
            anomaly_protocol.subsequences.append(
                clean_segment.astype(np.float64, copy=True)
            )
            return anomaly_protocol

        centerline = local_centerline(clean_segment, self.center_mode)
        residual = np.asarray(clean_segment, dtype=np.float64) - centerline
        residual_scale = max(float(np.std(residual)), 0.25 * float(np.ptp(residual)))
        if self.min_effect_delta > 0.0 and residual_scale < self.min_residual_scale:
            raise ValueError(
                "Amplitude anomaly cannot satisfy min_effect_delta on a flat or "
                "near-flat residual window without becoming additive."
            )

        factor_profile = self._factor_profile(anomaly_protocol, length)
        subsequence = centerline + residual * factor_profile
        subsequence = match_boundary_value_and_slope(subsequence, clean_segment)
        if self.min_effect_delta > 0.0:
            subsequence = enforce_min_effect_delta(
                baseline=clean_segment,
                candidate=subsequence,
                target_effect=self.min_effect_delta,
                transition_length=None,
                allow_zero_delta_fallback=False,
            )
        anomaly_protocol.subsequences.append(subsequence.astype(np.float64, copy=False))
        return anomaly_protocol

    def _factor_profile(
        self, anomaly_protocol: AnomalyProtocol, length: int
    ) -> np.ndarray:
        if anomaly_protocol.creeping_length == 0:
            envelope = symmetric_envelope(length, self.transition_length)
        else:
            creep_len = min(max(0, int(anomaly_protocol.creeping_length)), length)
            anomaly_len = max(0, length - creep_len)
            creep = (
                np.linspace(0.0, 1.0, creep_len, endpoint=False, dtype=np.float64)
                if creep_len > 0
                else np.array([], dtype=np.float64)
            )
            plateau_len = max(0, int(round(anomaly_len * 0.8)))
            end_transition_len = max(0, anomaly_len - plateau_len)
            plateau = np.ones(plateau_len, dtype=np.float64)
            if end_transition_len > 0:
                phase = np.linspace(0.0, np.pi, end_transition_len, dtype=np.float64)
                end_transition = 0.5 * (1.0 + np.cos(phase))
            else:
                end_transition = np.array([], dtype=np.float64)
            envelope = self._match_length(
                np.concatenate([creep, plateau, end_transition]).astype(np.float64),
                length,
            )
            if envelope.size > 0:
                envelope[0] = 0.0
                envelope[-1] = 0.0
        return (1.0 + envelope * (self.amplitude_factor - 1.0)).astype(np.float64)

    @staticmethod
    def _match_length(values: np.ndarray, expected_length: int) -> np.ndarray:
        if values.shape[0] == expected_length:
            return values
        if values.shape[0] > expected_length:
            return values[:expected_length]
        if values.shape[0] == 0:
            return np.ones(expected_length, dtype=np.float64)
        pad_count = expected_length - values.shape[0]
        return np.pad(values, (0, pad_count), mode="edge")

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyAmplitudeParameters]:
        return AnomalyAmplitudeParameters
