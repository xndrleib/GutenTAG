from typing import Optional

import numpy as np

from . import BaseOscillation
from .sine import Sine, sine
from ..utils.global_variables import BASE_OSCILLATION_NAMES, PARAMETERS
from ..utils.types import BOGenerationContext


class SharedNoiseSine(Sine):
    """Sine carrier with lower deterministic amplitude and stronger noise by default.

    The shape remains periodic like a regular sine, but the default parameter
    regime is intended for structural/dependence anomalies where a shared
    stochastic layer should matter more than the clean deterministic waveform.
    """

    KIND = BASE_OSCILLATION_NAMES.SHARED_NOISE_SINE

    def __init__(self, *args, **kwargs) -> None:
        tuned = dict(kwargs)
        tuned.setdefault(PARAMETERS.FREQUENCY, 7.0)
        tuned.setdefault(PARAMETERS.AMPLITUDE, 0.55)
        tuned.setdefault(PARAMETERS.VARIANCE, 0.18)
        super().__init__(*args, **tuned)

    def get_base_oscillation_kind(self) -> str:
        return self.KIND

    def generate_only_base(
        self,
        ctx: BOGenerationContext,
        length: Optional[int] = None,
        frequency: Optional[float] = None,
        amplitude: Optional[float] = None,
        freq_mod: Optional[float] = None,
        phase: Optional[float] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        n: int = length or self.length
        f: float = frequency or self.frequency
        a: float = amplitude or self.amplitude
        v_freq_mod: float = freq_mod or self.freq_mod
        v_phase: float = phase or 0.0
        return shared_noise_sine(n, f, a, v_freq_mod, v_phase)


def shared_noise_sine(
    length: int,
    frequency: float = 7.0,
    amplitude: float = 0.55,
    freq_mod: float = 0.0,
    phase: float = 0.0,
) -> np.ndarray:
    return sine(length, frequency, amplitude, freq_mod, phase)


BaseOscillation.register(SharedNoiseSine.KIND, SharedNoiseSine)
