from dataclasses import dataclass
from typing import Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import CylinderBellFunnel, ECG, Square, Sawtooth, MLS


@dataclass
class AnomalyPatternParameters:
    sinusoid_k: float = 10.0
    cbf_pattern_factor: float = 2.0
    square_duty: float = 1.0
    sawtooth_width: float = 0.5
    min_effect_delta: float = 0.0
    min_window_ptp: float = 0.0
    adaptive_blend: bool = False
    blend_strength: float = 1.0


class AnomalyPattern(BaseAnomaly):
    def __init__(self, parameters: AnomalyPatternParameters):
        super().__init__()
        self.sinusoid_k = parameters.sinusoid_k
        self.cbf_pattern_factor = parameters.cbf_pattern_factor
        self.square_duty = parameters.square_duty
        self.sawtooth_width = parameters.sawtooth_width
        self.min_effect_delta = max(0.0, float(parameters.min_effect_delta))
        self.min_window_ptp = max(0.0, float(parameters.min_window_ptp))
        self.adaptive_blend = bool(parameters.adaptive_blend)
        self.blend_strength = max(0.0, float(parameters.blend_strength))

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        def sinusoid_template(length: int, k: float) -> np.ndarray:
            if length <= 0:
                return np.array([], dtype=np.float64)
            if length == 1:
                return np.zeros(1, dtype=np.float64)
            t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
            if np.isclose(k, 0.0):
                return t
            return np.arctan(k * t) / np.arctan(k)

        def enforce_min_effect(
            candidate: np.ndarray, reference: np.ndarray
        ) -> np.ndarray:
            if reference.size == 0:
                return candidate
            if candidate.shape[0] != reference.shape[0]:
                return candidate
            delta = candidate - reference
            max_delta = float(np.max(np.abs(delta)))
            min_required = max(self.min_effect_delta, 1e-10)
            if max_delta >= min_required:
                return candidate
            template = sinusoid_template(int(reference.size), float(self.sinusoid_k))
            window_ptp = float(np.ptp(reference))
            scale = max(window_ptp / 2.0, self.min_window_ptp / 2.0, 1e-6)
            center = float(np.mean(reference))
            target = center + scale * template
            target_delta = target - reference
            target_max_delta = float(np.max(np.abs(target_delta)))
            if target_max_delta <= 1e-12:
                corrected = candidate.copy()
                corrected[0] = corrected[0] + min_required
                return corrected
            blend = float(self.blend_strength)
            if self.adaptive_blend and self.min_effect_delta > 0.0:
                blend = max(blend, self.min_effect_delta / target_max_delta)
            return reference + blend * target_delta

        if anomaly_protocol.base_oscillation_kind == CylinderBellFunnel.KIND:
            cbf = anomaly_protocol.base_oscillation
            reference = cbf.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            subsequence = cbf.generate_only_base(
                anomaly_protocol.ctx.to_bo(),
                variance_pattern_length=cbf.variance_pattern_length
                * self.cbf_pattern_factor,
            )[anomaly_protocol.start : anomaly_protocol.end]
            subsequence = enforce_min_effect(subsequence, reference)
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
            reference = ecg.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            subsequence = enforce_min_effect(subsequence, reference)
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == Sawtooth.KIND:
            reference = anomaly_protocol.base_oscillation.timeseries[
                anomaly_protocol.start : anomaly_protocol.end
            ]
            subsequence = anomaly_protocol.base_oscillation.generate_only_base(
                anomaly_protocol.ctx.to_bo(), width=self.sawtooth_width
            )[anomaly_protocol.start : anomaly_protocol.end]
            subsequence = enforce_min_effect(subsequence, reference)
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == Square.KIND:
            reference = anomaly_protocol.base_oscillation.timeseries[
                anomaly_protocol.start : anomaly_protocol.end
            ]
            subsequence = anomaly_protocol.base_oscillation.generate_only_base(
                anomaly_protocol.ctx.to_bo(), duty=self.square_duty
            )[anomaly_protocol.start : anomaly_protocol.end]
            subsequence = enforce_min_effect(subsequence, reference)
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation_kind == MLS.KIND:
            transition_window = int(0.1 * anomaly_protocol.length)
            transition_window = transition_window - transition_window % 2
            reference = anomaly_protocol.base_oscillation.timeseries[
                anomaly_protocol.start : anomaly_protocol.end
            ]
            subsequence = reference
            reversed = reference[::-1]

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
            subsequence = enforce_min_effect(subsequence, reference)
            anomaly_protocol.subsequences.append(subsequence)

        elif anomaly_protocol.base_oscillation.is_periodic():
            bo = anomaly_protocol.base_oscillation
            snippet = bo.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            if snippet.size == 0:
                anomaly_protocol.subsequences.append(snippet)
                return anomaly_protocol
            template = sinusoid_template(int(snippet.size), float(self.sinusoid_k))
            window_ptp = float(np.ptp(snippet))
            scale = max(window_ptp / 2.0, self.min_window_ptp / 2.0, 1e-6)
            center = float(np.mean(snippet))
            target = center + scale * template
            delta = target - snippet
            blend = float(self.blend_strength)
            if self.adaptive_blend and self.min_effect_delta > 0.0:
                max_delta = float(np.max(np.abs(delta)))
                if max_delta > 1e-12:
                    blend = max(blend, self.min_effect_delta / max_delta)
            subsequence = snippet + blend * delta
            subsequence = enforce_min_effect(subsequence, snippet)
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
