from dataclasses import dataclass
from typing import Type

from . import BaseAnomaly
from .. import AnomalyProtocol


@dataclass
class AnomalyChannelRewiringParameters:
    transition_length: int = 8


class AnomalyChannelRewiring(BaseAnomaly):
    def __init__(self, parameters: AnomalyChannelRewiringParameters):
        super().__init__()
        self.parameters = parameters

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyChannelRewiringParameters]:
        return AnomalyChannelRewiringParameters
