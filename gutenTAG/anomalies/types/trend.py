from dataclasses import dataclass
from typing import Type

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from . import BaseAnomaly, AnomalyProtocol
from ...base_oscillations import RandomModeJump


@dataclass
class AnomalyTrendParameters:
    trend: "BaseOscillationInterface"  # type: ignore # noqa: F821 # otherwise we have a circular import


class AnomalyTrend(BaseAnomaly):
    def __init__(self, parameters: AnomalyTrendParameters):
        super().__init__()
        self.trend = parameters.trend

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind == RandomModeJump.KIND:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
            return anomaly_protocol

        length = anomaly_protocol.end - anomaly_protocol.start
        if length <= 0:
            return anomaly_protocol

        transition_length = int(round(length * 0.2))
        transition_length = max(1, min(transition_length, length))
        plateau_length = max(0, length - transition_length)

        start_transition = norm.pdf(np.linspace(-3, 0, transition_length), scale=1.05)
        if start_transition.size == 0:
            amplitude_bell = np.ones(length, dtype=np.float64)
        else:
            start_max = start_transition.max()
            if start_max == 0:
                start_transition = np.ones_like(start_transition)
            else:
                start_transition = start_transition / start_max
            amplitude_bell = np.concatenate([start_transition, np.ones(plateau_length)])
            if amplitude_bell.shape[0] > length:
                amplitude_bell = amplitude_bell[:length]
            elif amplitude_bell.shape[0] < length:
                amplitude_bell = np.pad(
                    amplitude_bell,
                    (0, length - amplitude_bell.shape[0]),
                    mode="edge",
                )
            amplitude_bell = (
                MinMaxScaler(feature_range=(0, 1))
                .fit_transform(amplitude_bell.reshape(-1, 1))
                .reshape(-1)
            )

        self.trend.length = length
        self.trend.generate_timeseries_and_variations(anomaly_protocol.ctx.to_bo())
        timeseries = self.trend.timeseries
        if timeseries is None:
            return anomaly_protocol
        if timeseries.shape[0] > length:
            timeseries = timeseries[:length]
        elif timeseries.shape[0] < length:
            if timeseries.shape[0] == 0:
                return anomaly_protocol
            timeseries = np.pad(
                timeseries, (0, length - timeseries.shape[0]), mode="edge"
            )

        timeseries *= amplitude_bell
        end_point = timeseries[-1]

        anomaly_protocol.base_oscillation.trend_series[
            anomaly_protocol.start : anomaly_protocol.end
        ] += timeseries
        anomaly_protocol.base_oscillation.trend_series[
            anomaly_protocol.end :
        ] += end_point

        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyTrendParameters]:
        return AnomalyTrendParameters
