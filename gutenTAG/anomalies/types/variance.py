from dataclasses import dataclass
from typing import Type

import numpy as np

from . import BaseAnomaly
from .. import AnomalyProtocol
from ...base_oscillations import CylinderBellFunnel, RandomModeJump


@dataclass
class AnomalyVarianceParameters:
    variance: float = 0.0


class AnomalyVariance(BaseAnomaly):
    def __init__(self, parameters: AnomalyVarianceParameters):
        super().__init__()
        self.variance = parameters.variance

    def generate(self, anomaly_protocol: AnomalyProtocol) -> AnomalyProtocol:
        base = anomaly_protocol.base_oscillation
        if anomaly_protocol.base_oscillation_kind == RandomModeJump.KIND:
            self.logger.warn_false_combination(
                self.__class__.__name__, anomaly_protocol.base_oscillation_kind
            )

        elif anomaly_protocol.base_oscillation_kind == CylinderBellFunnel.KIND:
            subsequence = base.generate_only_base(
                anomaly_protocol.ctx.to_bo(), variance=self.variance
            )[anomaly_protocol.start : anomaly_protocol.end]
            anomaly_protocol.subsequences.append(subsequence)

        else:
            length = anomaly_protocol.end - anomaly_protocol.start
            if length <= 0:
                return anomaly_protocol

            clean_segment = base.timeseries[anomaly_protocol.start : anomaly_protocol.end]
            local_scale = float(np.std(clean_segment)) if clean_segment.size > 0 else 0.0
            amplitude = abs(float(getattr(base, "amplitude", 1.0)))
            # Keep anomaly strength comparable across base oscillation families.
            reference_scale = max(local_scale, 0.25 * amplitude, 1e-6)

            target_std = max(0.0, float(self.variance) * reference_scale)
            base_std = max(0.0, float(base.variance) * reference_scale)

            std_schedule = (
                self.generate_creeping(anomaly_protocol) * (target_std - base_std)
                + base_std
            )
            std_schedule = np.clip(std_schedule, 0.0, None)

            if target_std <= 0:
                base.noise[anomaly_protocol.start : anomaly_protocol.end] = 0.0
                return anomaly_protocol

            subsequence_noise = base.generate_noise(
                anomaly_protocol.ctx.to_bo(), target_std, length
            )
            base.noise[anomaly_protocol.start : anomaly_protocol.end] = (
                subsequence_noise * (std_schedule / target_std)
            )
        return anomaly_protocol

    @property
    def requires_period_start_position(self) -> bool:
        return False

    @staticmethod
    def get_parameter_class() -> Type[AnomalyVarianceParameters]:
        return AnomalyVarianceParameters
