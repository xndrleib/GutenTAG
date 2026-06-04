from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import RandomModeJump
from ...tsgen.signal_ops import enforce_min_effect_delta


@dataclass
class AnomalyMeanParameters:
    offset: float = 0.0
    transition_length: Optional[int] = None
    min_effect_delta: float = 0.0


class AnomalyMean(BaseAnomaly):
    def __init__(self, parameters: AnomalyMeanParameters):
        super().__init__()
        self.offset = float(parameters.offset)
        self.transition_length = parameters.transition_length
        self.min_effect_delta = max(0.0, float(parameters.min_effect_delta))

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind == RandomModeJump.KIND:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
            return anomaly_protocol

        base = anomaly_protocol.base_oscillation
        ts: np.ndarray = base.timeseries
        clean_segment = ts[anomaly_protocol.start : anomaly_protocol.end]
        if clean_segment.size == 0:
            return anomaly_protocol
        offset = self._effective_offset(anomaly_protocol)
        envelope = self.build_symmetric_envelope(
            clean_segment.shape[0], self.transition_length
        )
        subsequence = clean_segment + offset * envelope
        if self.min_effect_delta > 0.0:
            subsequence = enforce_min_effect_delta(
                baseline=clean_segment,
                candidate=subsequence,
                target_effect=self.min_effect_delta,
                transition_length=self.transition_length,
                allow_zero_delta_fallback=True,
            )
        anomaly_protocol.subsequences.append(subsequence.astype(np.float64, copy=False))
        return anomaly_protocol

    def _effective_offset(self, anomaly_protocol: AnomalyProtocol) -> float:
        if self.min_effect_delta <= 0.0 or abs(self.offset) >= self.min_effect_delta:
            return float(self.offset)
        if abs(self.offset) > 1e-12:
            sign = 1.0 if self.offset > 0.0 else -1.0
        else:
            sign = -1.0 if anomaly_protocol.rng.random() < 0.5 else 1.0
        return float(sign * self.min_effect_delta)

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyMeanParameters]:
        return AnomalyMeanParameters
