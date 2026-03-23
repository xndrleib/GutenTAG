from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from . import BaseAnomaly, AnomalyProtocol
from ...base_oscillations import RandomModeJump


@dataclass
class AnomalyTrendParameters:
    trend: "BaseOscillationInterface"  # type: ignore # noqa: F821 # otherwise we have a circular import
    transition_length: Optional[int] = None
    boundary_mode: str = "inside_window_zero_endpoints"
    envelope_kind: str = "sine2"
    min_effect_delta: float = 0.0


class AnomalyTrend(BaseAnomaly):
    def __init__(self, parameters: AnomalyTrendParameters):
        super().__init__()
        self.trend = parameters.trend
        self.transition_length = parameters.transition_length
        self.boundary_mode = str(parameters.boundary_mode).lower()
        self.envelope_kind = str(parameters.envelope_kind).lower()
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

        if self.transition_length is None:
            transition_length = int(round(length * 0.2))
            transition_length = max(1, min(transition_length, length))
        else:
            transition_length = max(0, min(int(self.transition_length), length))
        if transition_length == 0:
            amplitude_bell = np.ones(length, dtype=np.float64)
        else:
            plateau_length = max(0, length - transition_length)
            start_transition = norm.pdf(
                np.linspace(-3, 0, transition_length), scale=1.05
            )
            if start_transition.size == 0:
                amplitude_bell = np.ones(length, dtype=np.float64)
            else:
                start_max = start_transition.max()
                if start_max == 0:
                    start_transition = np.ones_like(start_transition)
                else:
                    start_transition = start_transition / start_max
                amplitude_bell = np.concatenate(
                    [start_transition, np.ones(plateau_length)]
                )
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
        if self.boundary_mode == "legacy_carry_over":
            timeseries *= amplitude_bell
            end_point = timeseries[-1]
            anomaly_protocol.base_oscillation.trend_series[
                anomaly_protocol.start : anomaly_protocol.end
            ] += timeseries
            anomaly_protocol.base_oscillation.trend_series[
                anomaly_protocol.end :
            ] += end_point
            return anomaly_protocol

        local = np.asarray(timeseries, dtype=np.float64).copy()
        local = self._anchor_zero_endpoints(local)
        if self.envelope_kind in ("sine2", "sin2") and local.shape[0] > 1:
            phase = np.linspace(0.0, np.pi, local.shape[0], dtype=np.float64)
            local *= np.sin(phase) ** 2
        elif self.envelope_kind == "transition":
            local *= amplitude_bell
        local = self._enforce_min_effect(local, self.min_effect_delta)

        anomaly_protocol.base_oscillation.trend_series[
            anomaly_protocol.start : anomaly_protocol.end
        ] += local

        return anomaly_protocol

    @staticmethod
    def _anchor_zero_endpoints(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        if values.size == 1:
            values[0] = 0.0
            return values
        left = float(values[0])
        right = float(values[-1])
        alpha = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
        baseline = (1.0 - alpha) * left + alpha * right
        return values - baseline

    @staticmethod
    def _enforce_min_effect(values: np.ndarray, min_effect_delta: float) -> np.ndarray:
        if values.size == 0 or min_effect_delta <= 0.0:
            return values
        peak = float(np.max(np.abs(values)))
        if peak >= min_effect_delta:
            return values
        if peak > 1e-12:
            return values * (min_effect_delta / peak)
        if values.size == 1:
            return np.array([min_effect_delta], dtype=np.float64)
        bump = np.sin(np.linspace(0.0, np.pi, values.size, dtype=np.float64))
        bump_peak = float(np.max(np.abs(bump)))
        if bump_peak <= 1e-12:
            return values
        return (min_effect_delta / bump_peak) * bump

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyTrendParameters]:
        return AnomalyTrendParameters
