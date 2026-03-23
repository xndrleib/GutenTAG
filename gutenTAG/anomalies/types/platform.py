from dataclasses import dataclass
from typing import Type

import numpy as np

from . import BaseAnomaly, AnomalyProtocol
from ...base_oscillations import RandomModeJump


@dataclass
class AnomalyPlatformParameters:
    value: float = 0.0
    min_effect_delta: float = 0.0


class AnomalyPlatform(BaseAnomaly):
    def __init__(self, parameters: AnomalyPlatformParameters):
        super().__init__()
        self.value = parameters.value
        self.min_effect_delta = max(0.0, float(parameters.min_effect_delta))

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind == RandomModeJump.KIND:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
            return anomaly_protocol

        length = anomaly_protocol.end - anomaly_protocol.start
        if length <= 0:
            return anomaly_protocol

        target_value = float(self.value)
        if self.min_effect_delta > 0.0:
            clean_segment = anomaly_protocol.base_oscillation.timeseries[
                anomaly_protocol.start : anomaly_protocol.end
            ]
            if clean_segment.size > 0:
                current_peak = float(np.max(np.abs(target_value - clean_segment)))
                if current_peak < self.min_effect_delta:
                    center = float(np.median(clean_segment))
                    direction = 1.0 if (target_value - center) >= 0.0 else -1.0
                    target_value = target_value + direction * (
                        self.min_effect_delta - current_peak
                    )

        values = np.zeros(length) + target_value
        anomaly_protocol.subsequences.append(values)
        self.turn_off_trend(anomaly_protocol)
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyPlatformParameters]:
        return AnomalyPlatformParameters
