from dataclasses import dataclass
from typing import Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol


@dataclass
class AnomalyPatternShiftParameters:
    shift_by: int = 5
    transition_window: int = 10


class AnomalyPatternShift(BaseAnomaly):
    def __init__(self, parameters: AnomalyPatternShiftParameters):
        super().__init__()
        self.shift_by = parameters.shift_by
        self.transition_window = parameters.transition_window

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation.is_periodic():
            assert (
                abs(self.shift_by) <= self.transition_window
            ), "The parameter 'shift_by' must not be larger than 'transition_window' in absolute terms! Guten Tag!"

            base = anomaly_protocol.base_oscillation

            subsequence = base.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            length = subsequence.shape[0]
            if length <= 1:
                anomaly_protocol.subsequences.append(subsequence)
                return anomaly_protocol

            transition_window = min(self.transition_window, max(1, length // 2))
            shift_by = int(
                np.clip(self.shift_by, -transition_window, transition_window)
            )

            if transition_window <= 1 or length <= 2 * transition_window:
                anomaly_protocol.subsequences.append(np.roll(subsequence, shift_by))
                return anomaly_protocol

            transition_start_num = max(1, transition_window + shift_by)
            transition_end_num = max(1, transition_window - shift_by)

            transition_start = np.interp(
                np.linspace(0, transition_window, transition_start_num),
                np.arange(transition_window),
                subsequence[:transition_window],
            )
            shifted = subsequence[transition_window:-transition_window]
            transition_end = np.interp(
                np.linspace(0, transition_window, transition_end_num),
                np.arange(transition_window),
                subsequence[-transition_window:],
            )

            subsequence = np.concatenate([transition_start, shifted, transition_end])
            if subsequence.shape[0] > length:
                subsequence = subsequence[:length]
            elif subsequence.shape[0] < length:
                subsequence = np.pad(
                    subsequence, (0, length - subsequence.shape[0]), mode="edge"
                )

            anomaly_protocol.subsequences.append(subsequence)
        else:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return True

    @staticmethod
    def get_parameter_class() -> Type[AnomalyPatternShiftParameters]:
        return AnomalyPatternShiftParameters
