from dataclasses import dataclass
from typing import Type

from . import BaseAnomaly
from .. import AnomalyProtocol


@dataclass
class AnomalyCovarianceChangeParameters:
    coupling_strength: float = -0.9
    transition_length: int = 8


class AnomalyCovarianceChange(BaseAnomaly):
    def __init__(self, parameters: AnomalyCovarianceChangeParameters):
        super().__init__()
        self.parameters = parameters

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyCovarianceChangeParameters]:
        return AnomalyCovarianceChangeParameters
