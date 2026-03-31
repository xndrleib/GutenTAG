from dataclasses import dataclass
from typing import Optional, Type

from . import BaseAnomaly
from .. import AnomalyProtocol


@dataclass
class AnomalyCorrelationFlipParameters:
    target_correlation: Optional[float] = -0.85
    transition_length: int = 8


class AnomalyCorrelationFlip(BaseAnomaly):
    def __init__(self, parameters: AnomalyCorrelationFlipParameters):
        super().__init__()
        self.parameters = parameters

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyCorrelationFlipParameters]:
        return AnomalyCorrelationFlipParameters
