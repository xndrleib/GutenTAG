from dataclasses import dataclass
from typing import Type

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from . import BaseAnomaly, AnomalyProtocol
from ...base_oscillations import Polynomial, Formula, RandomModeJump


@dataclass
class AnomalyAmplitudeParameters:
    amplitude_factor: float = 1.0


class AnomalyAmplitude(BaseAnomaly):
    def __init__(self, parameters: AnomalyAmplitudeParameters):
        super().__init__()
        self.amplitude_factor = parameters.amplitude_factor

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
        if self.amplitude_factor == 1.0:
            anomaly_protocol.subsequences.append(
                anomaly_protocol.base_oscillation.timeseries[
                    anomaly_protocol.start : anomaly_protocol.end
                ]
            )
            return anomaly_protocol

        if anomaly_protocol.creeping_length == 0:
            if length < 3:
                amplitude_bell = np.ones(length, dtype=np.float64)
            else:
                transition_length = int(round(length * 0.2))
                transition_length = max(1, min(transition_length, length // 2))
                plateau_length = max(0, length - 2 * transition_length)
                plateau = np.ones(plateau_length)
                start_transition = self._normalize_safe(
                    norm.pdf(np.linspace(-3, 0, transition_length), scale=1.05)
                )
                end_transition = self._normalize_safe(
                    norm.pdf(np.linspace(0, 3, transition_length), scale=1.05)
                )
                amplitude_bell = np.concatenate(
                    [start_transition, plateau, end_transition]
                )
        else:
            anomaly_length = max(0, length - anomaly_protocol.creeping_length)
            creeping_length = int(round(anomaly_length * 0.8))
            creeping = self.generate_creeping(
                anomaly_protocol, custom_anomaly_length=max(creeping_length, 1)
            )
            end_transition_length = max(0, anomaly_length - creeping_length)
            if end_transition_length > 0:
                end_transition = self._normalize_safe(
                    norm.pdf(np.linspace(0, 3, end_transition_length), scale=1.05)
                )
            else:
                end_transition = np.array([], dtype=np.float64)
            amplitude_bell = np.concatenate([creeping, end_transition])

        amplitude_bell = self._match_length(amplitude_bell, length)

        if self.amplitude_factor < 1.0:
            amplitude_bell = (
                MinMaxScaler(feature_range=(1.0, 2.0 - self.amplitude_factor))
                .fit_transform(amplitude_bell.reshape(-1, 1))
                .reshape(-1)
            )
            amplitude_bell = amplitude_bell * -1 + 2
        else:
            amplitude_bell = (
                MinMaxScaler(feature_range=(1.0, self.amplitude_factor))
                .fit_transform(amplitude_bell.reshape(-1, 1))
                .reshape(-1)
            )

        subsequence = (
            anomaly_protocol.base_oscillation.timeseries[
                anomaly_protocol.start : anomaly_protocol.end
            ]
            * amplitude_bell
        )
        anomaly_protocol.subsequences.append(subsequence)
        return anomaly_protocol

    @staticmethod
    def _normalize_safe(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        vmax = values.max()
        if vmax == 0:
            return np.ones_like(values)
        return values / vmax

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
