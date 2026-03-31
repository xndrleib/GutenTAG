from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import RandomModeJump


@dataclass
class AnomalyMeanParameters:
    offset: float = 0.0
    transition_length: Optional[int] = None


class AnomalyMean(BaseAnomaly):
    def __init__(self, parameters: AnomalyMeanParameters):
        super().__init__()
        self.offset = parameters.offset
        self.transition_length = parameters.transition_length

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind == RandomModeJump.KIND:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
            return anomaly_protocol

        base = anomaly_protocol.base_oscillation
        ts: np.ndarray = base.timeseries
        clean_segment = ts[anomaly_protocol.start : anomaly_protocol.end]
        envelope = self.build_symmetric_envelope(
            clean_segment.shape[0], self.transition_length
        )
        subsequence = clean_segment + float(self.offset) * envelope
        anomaly_protocol.subsequences.append(subsequence)
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyMeanParameters]:
        return AnomalyMeanParameters
