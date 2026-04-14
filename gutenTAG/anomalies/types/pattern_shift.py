from dataclasses import dataclass
from typing import Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol


@dataclass
class AnomalyPatternShiftParameters:
    shift_by: int = 5
    transition_window: int = 10
    crossfade_mode: str = "cosine"
    min_effect_delta: float = 0.0


class AnomalyPatternShift(BaseAnomaly):
    def __init__(self, parameters: AnomalyPatternShiftParameters):
        super().__init__()
        self.shift_by = parameters.shift_by
        self.transition_window = parameters.transition_window
        self.crossfade_mode = str(parameters.crossfade_mode).lower()
        self.min_effect_delta = max(0.0, float(parameters.min_effect_delta))

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        base_kind = anomaly_protocol.base_oscillation_kind

        def effective_transition_window(length: int) -> int:
            requested = int(max(0, self.transition_window))
            if base_kind in {"square", "sawtooth", "dirichlet"}:
                return int(max(1, min(max(2, requested), max(2, length // 10))))
            return requested

        def edge_lock_width(length: int) -> int:
            if base_kind in {"square", "sawtooth", "dirichlet"}:
                return int(max(2, min(length // 6, round(0.10 * float(length)))))
            return 0

        def edge_penalty(candidate: np.ndarray, reference: np.ndarray) -> float:
            width = edge_lock_width(reference.shape[0])
            if width <= 0:
                return 0.0
            width = min(width, reference.shape[0] // 2)
            if width <= 0:
                return 0.0
            left = float(np.mean(np.abs(candidate[:width] - reference[:width])))
            right = float(np.mean(np.abs(candidate[-width:] - reference[-width:])))
            return left + right

        def build_shifted_subsequence(
            subsequence: np.ndarray, shift_value: int, transition_window: int
        ) -> np.ndarray:
            length = subsequence.shape[0]
            if transition_window <= 1 or length <= 2 * transition_window or shift_value == 0:
                shifted_simple = np.roll(subsequence, shift_value).astype(
                    np.float64, copy=True
                )
                if shifted_simple.size > 0:
                    shifted_simple[0] = subsequence[0]
                    shifted_simple[-1] = subsequence[-1]
                return shifted_simple

            shifted = np.roll(subsequence, shift_value)
            local_window = min(transition_window, length // 2)
            if self.crossfade_mode == "linear":
                alpha = np.linspace(0.0, 1.0, local_window, dtype=np.float64)
            else:
                alpha = 0.5 * (
                    1.0
                    - np.cos(
                        np.linspace(0.0, np.pi, local_window, dtype=np.float64)
                    )
                )
            blended = shifted.astype(np.float64, copy=True)
            blended[:local_window] = (
                (1.0 - alpha) * subsequence[:local_window]
                + alpha * shifted[:local_window]
            )
            blended[-local_window:] = (
                (1.0 - alpha[::-1]) * subsequence[-local_window:]
                + alpha[::-1] * shifted[-local_window:]
            )
            blended[0] = subsequence[0]
            blended[-1] = subsequence[-1]
            return blended

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

            transition_window = min(
                effective_transition_window(length),
                max(1, length // 2),
            )
            shift_by = int(
                np.clip(self.shift_by, -transition_window, transition_window)
            )
            if shift_by == 0 and transition_window > 0 and length > 1:
                shift_by = 1

            blended = build_shifted_subsequence(subsequence, shift_by, transition_window)
            blended = self._finalize_candidate(
                blended,
                subsequence,
                transition_window=transition_window,
                edge_lock_width=edge_lock_width(length),
            )

            # Avoid degenerate cases where phase-shift produces an identical window.
            effect = float(np.max(np.abs(blended - subsequence)))
            if effect <= 1e-10 and length > 1:
                candidate_shifts = [1, -1, 2, -2, transition_window, -transition_window]
                for candidate in candidate_shifts:
                    if candidate == 0:
                        continue
                    if abs(candidate) >= length:
                        continue
                    if candidate == shift_by:
                        continue
                    alt = build_shifted_subsequence(subsequence, candidate, transition_window)
                    alt = self._finalize_candidate(
                        alt,
                        subsequence,
                        transition_window=transition_window,
                        edge_lock_width=edge_lock_width(length),
                    )
                    alt_effect = float(np.max(np.abs(alt - subsequence)))
                    if alt_effect > 1e-10:
                        blended = alt
                        effect = alt_effect
                        break

            target_effect = float(self.min_effect_delta)
            if target_effect > 0.0 and effect < target_effect and length > 1:
                max_shift = int(min(length - 1, max(1, transition_window)))
                best = blended
                best_effect = effect
                for candidate in range(-max_shift, max_shift + 1):
                    if candidate == 0:
                        continue
                    alt = build_shifted_subsequence(
                        subsequence, candidate, transition_window
                    )
                    alt = self._finalize_candidate(
                        alt,
                        subsequence,
                        transition_window=transition_window,
                        edge_lock_width=edge_lock_width(length),
                    )
                    alt_effect = float(np.max(np.abs(alt - subsequence)))
                    score = alt_effect - 0.75 * edge_penalty(alt, subsequence)
                    best_score = best_effect - 0.75 * edge_penalty(best, subsequence)
                    if score > best_score:
                        best = alt
                        best_effect = alt_effect
                blended = best
                effect = best_effect

            if target_effect > 0.0 and effect < target_effect:
                blended = self._enforce_min_effect(
                    baseline=subsequence,
                    candidate=blended,
                    target_effect=target_effect,
                )
                blended = self._finalize_candidate(
                    blended,
                    subsequence,
                    transition_window=transition_window,
                    edge_lock_width=edge_lock_width(length),
                )
            elif effect <= 1e-10 and blended.size > 0:
                force_idx = int(min(max(1, blended.size // 2), blended.size - 1))
                blended[force_idx] = blended[force_idx] + 1e-3
                blended = self._finalize_candidate(
                    blended,
                    subsequence,
                    transition_window=transition_window,
                    edge_lock_width=edge_lock_width(length),
                )
            anomaly_protocol.subsequences.append(blended)
        else:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )
        return anomaly_protocol

    @classmethod
    def _finalize_candidate(
        cls,
        candidate: np.ndarray,
        baseline: np.ndarray,
        *,
        transition_window: int,
        edge_lock_width: int,
    ) -> np.ndarray:
        if edge_lock_width > 0:
            return cls._replace_core_keep_edges(
                candidate, baseline, int(max(0, edge_lock_width))
            )
        blended = cls.blend_with_reference(candidate, baseline, transition_window)
        return cls._lock_reference_edges(blended, baseline, 0)

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
        alpha = 0.5 * (
            1.0 - np.cos(np.linspace(0.0, np.pi, edge, dtype=np.float64))
        )
        cand = cand[:n]
        ref = ref[:n]
        cand[:edge] = alpha * cand[:edge] + (1.0 - alpha) * ref[:edge]
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

    @staticmethod
    def _enforce_min_effect(
        baseline: np.ndarray, candidate: np.ndarray, target_effect: float
    ) -> np.ndarray:
        if target_effect <= 0.0 or candidate.size == 0:
            return candidate
        base = np.asarray(baseline, dtype=np.float64)
        result = np.asarray(candidate, dtype=np.float64).copy()
        delta = result - base
        peak = float(np.max(np.abs(delta)))
        if peak >= target_effect:
            return result
        if peak > 1e-12:
            scaled = base + delta * (target_effect / peak)
            if scaled.size > 0:
                scaled[0] = base[0]
                scaled[-1] = base[-1]
            return scaled

        n = result.size
        if n == 1:
            result[0] = base[0] + target_effect
            return result
        center = int(n // 2)
        impulse = np.zeros(n, dtype=np.float64)
        impulse[center] = target_effect
        if center > 0:
            impulse[center - 1] = -0.5 * target_effect
        if center + 1 < n:
            impulse[center + 1] = -0.5 * target_effect
        impulse[0] = 0.0
        impulse[-1] = 0.0
        return base + impulse

    @property
    def requires_period_start_position(self) -> bool:
        return True

    @staticmethod
    def get_parameter_class() -> Type[AnomalyPatternShiftParameters]:
        return AnomalyPatternShiftParameters
