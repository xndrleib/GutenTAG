from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
from scipy.signal import find_peaks

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
            if ecg.timeseries is not None:
                full_reference = np.asarray(ecg.timeseries, dtype=np.float64)
                variation = np.zeros_like(full_reference, dtype=np.float64)
                if getattr(ecg, "noise", None) is not None:
                    variation = variation + np.asarray(ecg.noise, dtype=np.float64)
                if getattr(ecg, "trend_series", None) is not None:
                    variation = variation + np.asarray(
                        ecg.trend_series, dtype=np.float64
                    )
                if getattr(ecg, "offset", None) is not None:
                    variation = variation + float(ecg.offset)
                visible_reference = full_reference + variation
                reference = visible_reference[
                    anomaly_protocol.start : anomaly_protocol.end
                ]
                subsequence = self._beat_aware_reference_window(
                    ecg=ecg,
                    full_reference=visible_reference,
                    start=anomaly_protocol.start,
                    end=anomaly_protocol.end,
                    frequency_factor=self.frequency_factor,
                )
                subsequence = self.blend_with_reference(
                    subsequence,
                    reference,
                    self._motif_transition_length(reference.shape[0]),
                )
                subsequence = (
                    subsequence
                    - variation[anomaly_protocol.start : anomaly_protocol.end]
                )
            else:
                subsequence = ecg.generate_only_base(
                    anomaly_protocol.ctx.to_bo(),
                    frequency=ecg.frequency * self.frequency_factor,
                )[anomaly_protocol.start : anomaly_protocol.end]
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
    def _motif_transition_length(length: int) -> int:
        if length <= 2:
            return 0
        return int(max(1, min(max(1, length // 12), round(0.04 * float(length)))))

    @staticmethod
    def _expected_period_samples(frequency: Optional[float]) -> Optional[float]:
        if frequency is None:
            return None
        freq = float(frequency)
        if not np.isfinite(freq) or freq <= 0:
            return None
        return 100.0 / freq

    @classmethod
    def _detect_ecg_peaks(
        cls, full_reference: np.ndarray, expected_period: Optional[float]
    ) -> np.ndarray:
        series = np.asarray(full_reference, dtype=np.float64)
        if series.size < 8:
            return np.zeros(0, dtype=int)
        centered = series - float(np.median(series))
        amplitude = float(np.max(centered) - np.min(centered))
        if not np.isfinite(amplitude) or amplitude <= 1e-8:
            return np.zeros(0, dtype=int)

        if expected_period is None:
            distance = max(2, int(round(series.shape[0] / 64.0)))
        else:
            distance = max(2, int(round(0.45 * float(expected_period))))
        prominence = max(1e-6, 0.12 * amplitude)

        positive_peaks, _ = find_peaks(
            centered, distance=distance, prominence=prominence
        )
        if positive_peaks.size >= 2:
            return positive_peaks.astype(int, copy=False)

        absolute_peaks, _ = find_peaks(
            np.abs(centered), distance=distance, prominence=prominence
        )
        return absolute_peaks.astype(int, copy=False)

    @staticmethod
    def _cap_anchor_scale(
        *,
        peaks: np.ndarray,
        midpoint: float,
        length: int,
        requested_scale: float,
    ) -> float:
        scale = float(max(1e-6, requested_scale))
        left_candidates = peaks[peaks < midpoint]
        if left_candidates.size > 0:
            left_extent = float(midpoint - float(np.min(left_candidates)))
            if left_extent > 1e-8:
                left_cap = (midpoint - 1.0) / left_extent
                scale = min(scale, left_cap)
        right_candidates = peaks[peaks > midpoint]
        if right_candidates.size > 0:
            right_extent = float(np.max(right_candidates) - midpoint)
            if right_extent > 1e-8:
                right_cap = (float(length - 2) - midpoint) / right_extent
                scale = min(scale, right_cap)
        return float(max(0.5, scale))

    @classmethod
    def _get_ecg_peaks(
        cls, ecg: ECG, full_reference: np.ndarray
    ) -> np.ndarray:
        source_id = id(full_reference)
        cached_source_id = getattr(ecg, "_frequency_peak_source_id", None)
        cached_peaks = getattr(ecg, "_frequency_peaks", None)
        if cached_source_id == source_id and isinstance(cached_peaks, np.ndarray):
            return cached_peaks

        peaks = cls._detect_ecg_peaks(
            full_reference,
            cls._expected_period_samples(getattr(ecg, "frequency", None)),
        )
        setattr(ecg, "_frequency_peak_source_id", source_id)
        setattr(ecg, "_frequency_peaks", peaks)
        return peaks

    @staticmethod
    def _smooth_baseline(
        window: np.ndarray, expected_period: Optional[float]
    ) -> np.ndarray:
        series = np.asarray(window, dtype=np.float64)
        if series.size <= 2:
            return series.astype(np.float64, copy=True)

        if expected_period is None:
            kernel_width = max(5, int(round(series.size / 6.0)))
        else:
            kernel_width = max(5, int(round(1.75 * float(expected_period))))
        kernel_width = min(kernel_width, series.size if series.size % 2 == 1 else series.size - 1)
        kernel_width = max(3, kernel_width)
        if kernel_width % 2 == 0:
            kernel_width = max(3, kernel_width - 1)
        if kernel_width >= series.size:
            kernel_width = series.size if series.size % 2 == 1 else series.size - 1
        if kernel_width <= 1:
            return np.full_like(series, np.mean(series))

        pad = kernel_width // 2
        padded = np.pad(series, pad_width=pad, mode="edge")
        kernel = np.ones(kernel_width, dtype=np.float64) / float(kernel_width)
        baseline = np.convolve(padded, kernel, mode="valid")
        return baseline.astype(np.float64, copy=False)

    @classmethod
    def _beat_aware_reference_window(
        cls,
        *,
        ecg: ECG,
        full_reference: np.ndarray,
        start: int,
        end: int,
        frequency_factor: float,
    ) -> np.ndarray:
        """Warp the local ECG window using beat anchors from the same event.

        A beat-aware frequency edit on ECG should change local pacing, not swap in a
        different waveform patch from elsewhere in the carrier. We therefore keep the
        source samples inside the current event window and move beat anchors by a
        monotone time warp around the window midpoint.
        """
        reference = np.asarray(full_reference, dtype=np.float64)
        left = int(max(0, start))
        right = int(min(reference.shape[0], end))
        length = int(max(0, right - left))
        if length <= 1:
            return np.array(reference[left:right], dtype=np.float64, copy=True)

        clean_window = np.asarray(reference[left:right], dtype=np.float64)
        expected_period = cls._expected_period_samples(getattr(ecg, "frequency", None))
        baseline = cls._smooth_baseline(clean_window, expected_period)
        residual = clean_window - baseline
        local_peaks = cls._detect_ecg_peaks(residual, expected_period).astype(np.float64)
        local_peaks = local_peaks[(local_peaks > 1.0) & (local_peaks < float(length - 2))]
        if local_peaks.size < 2:
            return cls._warp_reference_window(
                full_reference=clean_window,
                start=0,
                end=length,
                frequency_factor=frequency_factor,
                expected_period=expected_period,
            )

        midpoint = 0.5 * float(length - 1)
        requested_scale = 1.0 / float(np.clip(frequency_factor, 0.35, 3.5))
        scale = cls._cap_anchor_scale(
            peaks=local_peaks,
            midpoint=midpoint,
            length=length,
            requested_scale=requested_scale,
        )
        desired_peak_positions = midpoint + scale * (local_peaks - midpoint)
        desired_peak_positions = np.maximum.accumulate(desired_peak_positions)
        if np.any(np.diff(desired_peak_positions) < 1e-3):
            return cls._warp_reference_window(
                full_reference=clean_window,
                start=0,
                end=length,
                frequency_factor=frequency_factor,
                expected_period=expected_period,
            )

        source_anchors = np.concatenate(
            ([0.0], local_peaks.astype(np.float64), [float(length - 1)])
        )
        target_anchors = np.concatenate(
            ([0.0], desired_peak_positions.astype(np.float64), [float(length - 1)])
        )
        output_positions = np.arange(length, dtype=np.float64)
        source_positions = np.interp(output_positions, target_anchors, source_anchors)
        warped_residual = np.interp(
            source_positions,
            np.arange(length, dtype=np.float64),
            residual,
        ).astype(np.float64)
        residual_center = float(np.median(residual))
        warped_residual -= float(np.median(warped_residual))
        warped_residual += residual_center
        residual_scale = float(np.std(residual))
        warped_scale = float(np.std(warped_residual))
        if residual_scale > 1e-8 and warped_scale > 1e-8:
            warped_residual *= residual_scale / warped_scale
        warped = baseline + warped_residual
        warped += float(np.mean(clean_window) - np.mean(warped))
        return warped.astype(np.float64, copy=False)

    @staticmethod
    def _warp_reference_window(
        *,
        full_reference: np.ndarray,
        start: int,
        end: int,
        frequency_factor: float,
        expected_period: Optional[float] = None,
    ) -> np.ndarray:
        """Warp a clean motif-rich carrier window instead of resimulating it.

        For ECG-like carriers the morphology is more stable than the exact local
        beat timing. Reusing a wider or narrower clean context window and
        resampling it back to the target length keeps the motif family intact
        while still changing the local pacing inside the event support.
        """
        reference = np.asarray(full_reference, dtype=np.float64)
        left = int(max(0, start))
        right = int(min(reference.shape[0], end))
        length = int(max(0, right - left))
        if length <= 1:
            return np.array(reference[left:right], dtype=np.float64, copy=True)

        factor = float(np.clip(frequency_factor, 0.35, 3.5))
        source_length = int(round(float(length) * factor))
        source_length = int(max(2, min(reference.shape[0], source_length)))
        clean_window = np.asarray(reference[left:right], dtype=np.float64)
        if clean_window.shape[0] <= 1:
            return clean_window
        baseline = AnomalyFrequency._smooth_baseline(clean_window, expected_period)
        residual = clean_window - baseline
        max_source_start = max(0, reference.shape[0] - source_length)
        center = 0.5 * float(left + right)
        centered_start = int(round(center - 0.5 * float(source_length)))
        centered_start = int(np.clip(centered_start, 0, max_source_start))
        dst_x = np.linspace(0.0, 1.0, length, dtype=np.float64)

        def build_candidate(start_idx: int) -> np.ndarray:
            local_start = int(np.clip(start_idx, 0, max_source_start))
            source = np.asarray(reference[local_start : local_start + source_length], dtype=np.float64)
            if source.shape[0] <= 1:
                return clean_window
            source_baseline = AnomalyFrequency._smooth_baseline(source, expected_period)
            source_residual = source - source_baseline
            warped_residual = np.interp(
                dst_x,
                np.linspace(0.0, 1.0, source.shape[0], dtype=np.float64),
                source_residual,
            ).astype(np.float64)
            target_scale = float(np.std(residual))
            warped_scale = float(np.std(warped_residual))
            if target_scale > 1e-8 and warped_scale > 1e-8:
                warped_residual *= target_scale / warped_scale
            candidate = baseline + warped_residual
            candidate += float(np.mean(clean_window) - np.mean(candidate))
            return candidate

        candidates = [centered_start]
        max_shift = max(2, length // 3)
        step = max(1, length // 12)
        for offset in range(-max_shift, max_shift + 1, step):
            candidates.append(centered_start + offset)

        best = clean_window
        best_score = -np.inf
        for candidate_start in sorted(set(candidates)):
            warped = build_candidate(candidate_start)
            delta = warped - clean_window
            score = float(np.max(np.abs(delta))) + 0.25 * float(
                np.mean(np.abs(delta))
            )
            if score > best_score:
                best_score = score
                best = warped
        return best.astype(np.float64, copy=False)

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
