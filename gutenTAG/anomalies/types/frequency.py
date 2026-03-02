from dataclasses import dataclass
from typing import Type

import numpy as np

from . import BaseAnomaly, AnomalyProtocol
from ...base_oscillations import ECG
from ...base_oscillations.utils.math_func_support import prepare_base_signal


@dataclass
class AnomalyFrequencyParameters:
    frequency_factor: float = 1.0


class AnomalyFrequency(BaseAnomaly):
    def __init__(self, parameters: AnomalyFrequencyParameters):
        super().__init__()
        self.frequency_factor = parameters.frequency_factor

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        if anomaly_protocol.base_oscillation_kind == ECG.KIND:
            ecg = anomaly_protocol.base_oscillation
            subsequence = ecg.generate_only_base(
                anomaly_protocol.ctx.to_bo(),
                frequency=ecg.frequency * self.frequency_factor,
            )[anomaly_protocol.start : anomaly_protocol.end]
            if ecg.timeseries is not None:
                reference = ecg.timeseries[anomaly_protocol.start : anomaly_protocol.end]
                subsequence = self._anchor_subsequence_to_reference(
                    subsequence, reference
                )
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation.is_periodic():
            bo = anomaly_protocol.base_oscillation
            full_length = anomaly_protocol.ctx.base_oscillation.length
            base_original = prepare_base_signal(full_length, bo.frequency)
            base_anomalous = prepare_base_signal(
                full_length, bo.frequency * self.frequency_factor
            )
            start_idx = anomaly_protocol.start
            phase_shift = float(base_original[start_idx] - base_anomalous[start_idx])
            subsequence = bo.generate_only_base(
                anomaly_protocol.ctx.to_bo(),
                frequency=bo.frequency * self.frequency_factor,
                freq_mod=bo.freq_mod,
                phase=phase_shift,
            )[anomaly_protocol.start : anomaly_protocol.end]
            if bo.timeseries is not None:
                reference = bo.timeseries[anomaly_protocol.start : anomaly_protocol.end]
                subsequence = self._anchor_subsequence_to_reference(
                    subsequence, reference
                )
            anomaly_protocol.subsequences.append(subsequence)

        else:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )

        return anomaly_protocol

    @staticmethod
    def _anchor_subsequence_to_reference(
        subsequence: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Match the first and last points to the clean reference segment.

        This removes visible edge jumps on the discrete [start, end) write interval
        while keeping the interior frequency pattern intact.
        """
        if subsequence.size == 0 or reference.size == 0:
            return subsequence
        n = min(subsequence.shape[0], reference.shape[0])
        anchored = np.asarray(subsequence[:n], dtype=np.float64).copy()
        ref = np.asarray(reference[:n], dtype=np.float64)
        if n == 1:
            anchored[0] = ref[0]
            return anchored

        left_delta = float(ref[0] - anchored[0])
        right_delta = float(ref[-1] - anchored[-1])
        alpha = np.linspace(0.0, 1.0, n, dtype=np.float64)
        anchored += (1.0 - alpha) * left_delta + alpha * right_delta
        return anchored

    @property
    def requires_period_start_position(self) -> bool:
        return True

    @staticmethod
    def get_parameter_class() -> Type[AnomalyFrequencyParameters]:
        return AnomalyFrequencyParameters
