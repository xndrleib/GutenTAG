from dataclasses import dataclass
from typing import Type

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import CylinderBellFunnel, ECG, Square, Sawtooth, MLS


@dataclass
class AnomalyPatternParameters:
    sinusoid_k: float = 10.0
    cbf_pattern_factor: float = 2.0
    square_duty: float = 1.0
    sawtooth_width: float = 0.5


class AnomalyPattern(BaseAnomaly):
    def __init__(self, parameters: AnomalyPatternParameters):
        super().__init__()
        self.sinusoid_k = parameters.sinusoid_k
        self.cbf_pattern_factor = parameters.cbf_pattern_factor
        self.square_duty = parameters.square_duty
        self.sawtooth_width = parameters.sawtooth_width

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind == CylinderBellFunnel.KIND:
            cbf = anomaly_protocol.base_oscillation
            subsequence = cbf.generate_only_base(
                anomaly_protocol.ctx.to_bo(),
                variance_pattern_length=cbf.variance_pattern_length
                * self.cbf_pattern_factor,
            )[anomaly_protocol.start : anomaly_protocol.end]
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == ECG.KIND:
            ecg = anomaly_protocol.base_oscillation
            length = anomaly_protocol.end - anomaly_protocol.start
            window = max(1, int(length * 0.05))

            for slide in range(-3, 3):
                start_idx = max(0, anomaly_protocol.start + slide)
                end_idx = min(start_idx + window, ecg.timeseries.shape[0])
                start = ecg.timeseries[start_idx:end_idx]
                if start.size == 0:
                    continue
                if np.argmax(start) == 0:
                    break
            else:
                slide = 0

            start_idx = anomaly_protocol.start + slide
            start_idx = max(0, min(start_idx, ecg.timeseries.shape[0] - length))
            end_idx = start_idx + length
            subsequence = ecg.timeseries[start_idx:end_idx][::-1]
            if subsequence.shape[0] != length:
                if subsequence.shape[0] > length:
                    subsequence = subsequence[:length]
                elif subsequence.shape[0] > 0:
                    subsequence = np.pad(
                        subsequence,
                        (0, length - subsequence.shape[0]),
                        mode="edge",
                    )
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == Sawtooth.KIND:
            subsequence = anomaly_protocol.base_oscillation.generate_only_base(
                anomaly_protocol.ctx.to_bo(), width=self.sawtooth_width
            )[anomaly_protocol.start : anomaly_protocol.end]
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == Square.KIND:
            subsequence = anomaly_protocol.base_oscillation.generate_only_base(
                anomaly_protocol.ctx.to_bo(), duty=self.square_duty
            )[anomaly_protocol.start : anomaly_protocol.end]
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == MLS.KIND:
            transition_window = int(0.1 * anomaly_protocol.length)
            transition_window = transition_window - transition_window % 2
            subsequence = anomaly_protocol.base_oscillation.timeseries[
                anomaly_protocol.start : anomaly_protocol.end
            ]
            reversed = subsequence[::-1]

            if (
                transition_window < 2
                or subsequence.shape[0] < 4
                or 2 * transition_window >= subsequence.shape[0]
            ):
                subsequence = reversed
            else:
                transition_start = np.interp(
                    np.linspace(0, transition_window * 2, transition_window),
                    np.arange(transition_window * 2),
                    np.r_[
                        subsequence[:transition_window], reversed[:transition_window]
                    ],
                )
                transition_end = np.interp(
                    np.linspace(0, transition_window * 2, transition_window),
                    np.arange(transition_window * 2),
                    np.r_[
                        reversed[-transition_window:], subsequence[-transition_window:]
                    ],
                )

                subsequence = np.concatenate(
                    [
                        transition_start,
                        reversed[transition_window:-transition_window],
                        transition_end,
                    ]
                )
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation.is_periodic():

            def sinusoid(
                t: np.ndarray, k: float, a_min: float, a_max: float
            ) -> np.ndarray:
                if t.size == 0:
                    return t
                if np.isclose(a_min, a_max):
                    return np.zeros_like(t) + a_min
                pattern = np.arctan(k * t) / np.arctan(k)
                scaled = (
                    MinMaxScaler(feature_range=(a_min, a_max))
                    .fit_transform(pattern.reshape(-1, 1))
                    .reshape(-1)
                )
                return scaled

            bo = anomaly_protocol.base_oscillation
            snippet = bo.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            if snippet.size == 0:
                anomaly_protocol.subsequences.append(snippet)
                return anomaly_protocol
            subsequence = sinusoid(
                snippet, self.sinusoid_k, snippet.min(), snippet.max()
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
    def get_parameter_class() -> Type[AnomalyPatternParameters]:
        return AnomalyPatternParameters
