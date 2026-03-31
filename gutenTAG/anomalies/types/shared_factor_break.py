from dataclasses import dataclass
from typing import Type

from . import BaseAnomaly
from .. import AnomalyProtocol


@dataclass
class AnomalySharedFactorBreakParameters:
    shared_factor_scale: float = 0.0
    transition_length: int = 8


class AnomalySharedFactorBreak(BaseAnomaly):
    def __init__(self, parameters: AnomalySharedFactorBreakParameters):
        super().__init__()
        self.parameters = parameters

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalySharedFactorBreakParameters]:
        return AnomalySharedFactorBreakParameters
