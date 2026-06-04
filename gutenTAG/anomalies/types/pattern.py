from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import CylinderBellFunnel, ECG, Square, Sawtooth, MLS
from ...tsgen.signal_ops import match_boundary_value_and_slope


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
    transition_length: Optional[int] = None


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
        self.transition_length = parameters.transition_length

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        base_kind = anomaly_protocol.base_oscillation_kind

        def effective_transition_length(length: int) -> Optional[int]:
            requested = (
                0 if self.transition_length is None else int(self.transition_length)
            )
            if base_kind in {Sawtooth.KIND, Square.KIND}:
                return int(max(1, min(max(2, requested), max(2, length // 10))))
            if base_kind in {
                ECG.KIND,
                CylinderBellFunnel.KIND,
                MLS.KIND,
            }:
                return int(
                    max(
                        requested,
                        min(
                            max(2, int(round(0.18 * float(length)))),
                            max(0, length // 3),
                        ),
                    )
                )
            return self.transition_length

        def edge_lock_width(length: int) -> int:
            if base_kind in {Sawtooth.KIND, Square.KIND}:
                return int(max(2, min(length // 6, round(0.10 * float(length)))))
            if base_kind in {
                ECG.KIND,
                CylinderBellFunnel.KIND,
                MLS.KIND,
            }:
                return int(max(1, min(length // 8, round(0.06 * float(length)))))
            return 0

        def finalize_candidate(
            candidate: np.ndarray, reference: np.ndarray
        ) -> np.ndarray:
            if base_kind in {Sawtooth.KIND, Square.KIND}:
                return self._replace_core_keep_edges(
                    candidate,
                    reference,
                    edge_lock_width(reference.shape[0]),
                )
            candidate = self.blend_with_reference(
                candidate,
                reference,
                effective_transition_length(reference.shape[0]),
            )
            candidate = self._lock_reference_edges(
                candidate,
                reference,
                edge_lock_width(reference.shape[0]),
            )
            return match_boundary_value_and_slope(candidate, reference)

        def sinusoid_template(length: int, k: float) -> np.ndarray:
            if length <= 0:
                return np.array([], dtype=np.float64)
            if length == 1:
                return np.zeros(1, dtype=np.float64)
            t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
            if np.isclose(k, 0.0):
                return t
            return np.arctan(k * t) / np.arctan(k)

        def restore_effect_floor(
            candidate: np.ndarray, reference: np.ndarray, min_required: float
        ) -> np.ndarray:
            if min_required <= 0.0 or candidate.shape[0] != reference.shape[0]:
                return candidate
            delta = candidate - reference
            max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
            if max_delta <= 1e-12 or max_delta >= min_required:
                return candidate
            return reference + delta * (min_required / max_delta)

        def enforce_min_effect(
            candidate: np.ndarray, reference: np.ndarray
        ) -> np.ndarray:
            if reference.size == 0:
                return candidate
            if candidate.shape[0] != reference.shape[0]:
                return candidate
            min_required = max(self.min_effect_delta, 1e-10)
            candidate = restore_effect_floor(
                finalize_candidate(candidate, reference), reference, min_required
            )
            delta = candidate - reference
            max_delta = float(np.max(np.abs(delta)))
            if max_delta >= min_required:
                return candidate
            template = sinusoid_template(int(reference.size), float(self.sinusoid_k))
            window_ptp = float(np.ptp(reference))
            scale = max(window_ptp / 2.0, self.min_window_ptp / 2.0, 1e-6)
            center = float(np.mean(reference))
            target = center + scale * template
            target = finalize_candidate(target, reference)
            target_delta = target - reference
            target_max_delta = float(np.max(np.abs(target_delta)))
            if target_max_delta <= 1e-12:
                corrected = reference.astype(np.float64, copy=True)
                if corrected.size == 1:
                    corrected[0] = corrected[0] + min_required
                    return corrected
                center_idx = int(corrected.size // 2)
                corrected[center_idx] = corrected[center_idx] + min_required
                return restore_effect_floor(
                    finalize_candidate(corrected, reference), reference, min_required
                )
            blend = float(self.blend_strength)
            if self.adaptive_blend and self.min_effect_delta > 0.0:
                blend = max(blend, self.min_effect_delta / target_max_delta)
            blended = reference + blend * target_delta
            return restore_effect_floor(
                finalize_candidate(blended, reference), reference, min_required
            )

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

    @staticmethod
    def _lock_reference_edges(
        candidate: np.ndarray,
        reference: np.ndarray,
        edge_width: int,
    ) -> np.ndarray:
        if edge_width <= 0:
            return np.asarray(candidate, dtype=np.float64)
        ref = np.asarray(reference, dtype=np.float64)
        cand = np.asarray(candidate, dtype=np.float64).copy()
        n = min(ref.shape[0], cand.shape[0])
        if n <= 2:
            return cand[:n]
        edge = min(int(edge_width), max(1, n // 4))
        if edge <= 0:
            return cand[:n]
        alpha = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge, dtype=np.float64)))
        cand = cand[:n]
        ref = ref[:n]
        cand[:edge] = (alpha * cand[:edge]) + ((1.0 - alpha) * ref[:edge])
        cand[-edge:] = alpha[::-1] * cand[-edge:] + (1.0 - alpha[::-1]) * ref[-edge:]
        lock = min(edge, max(1, edge // 2))
        cand[:lock] = ref[:lock]
        cand[-lock:] = ref[-lock:]
        cand[0] = ref[0]
        cand[-1] = ref[-1]
        return cand.astype(np.float64, copy=False)

    @staticmethod
    def _replace_core_keep_edges(
        candidate: np.ndarray,
        reference: np.ndarray,
        edge_width: int,
    ) -> np.ndarray:
        ref = np.asarray(reference, dtype=np.float64)
        cand = np.asarray(candidate, dtype=np.float64)
        n = min(ref.shape[0], cand.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        edge = min(int(max(0, edge_width)), max(0, n // 3))
        if edge <= 0 or 2 * edge >= n:
            result = np.array(cand[:n], dtype=np.float64, copy=True)
            result[0] = ref[0]
            result[-1] = ref[-1]
            return result
        result = np.array(ref[:n], dtype=np.float64, copy=True)
        result[edge : n - edge] = cand[edge : n - edge]
        result[0] = ref[0]
        result[-1] = ref[-1]
        return result

    @property
    def requires_period_start_position(self) -> bool:
        return True

    @staticmethod
    def get_parameter_class() -> Type[AnomalyPatternParameters]:
        return AnomalyPatternParameters
