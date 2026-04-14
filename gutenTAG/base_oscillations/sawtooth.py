from functools import partial
from typing import Optional

import numpy as np
from scipy import signal

from . import BaseOscillation
from .interface import BaseOscillationInterface
from .utils.math_func_support import (
    prepare_base_signal,
    generate_periodic_signal,
    calc_n_periods,
)
from ..utils.default_values import default_values
from ..utils.global_variables import (
    BASE_OSCILLATION_NAMES,
    BASE_OSCILLATIONS,
    PARAMETERS,
)
from ..utils.types import BOGenerationContext


class Sawtooth(BaseOscillationInterface):
    KIND = BASE_OSCILLATION_NAMES.SAWTOOTH

    def get_base_oscillation_kind(self) -> str:
        return self.KIND

    def get_timeseries_periods(self) -> Optional[int]:
        return calc_n_periods(self.length, self.frequency)

    def generate_only_base(
        self,
        ctx: BOGenerationContext,
        length: Optional[int] = None,
        frequency: Optional[float] = None,
        amplitude: Optional[float] = None,
        freq_mod: Optional[float] = None,
        width: Optional[float] = None,
        phase: Optional[float] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        n: int = length if length is not None else self.length  # in points
        f: float = frequency if frequency is not None else self.frequency  # in Hz
        a: float = amplitude if amplitude is not None else self.amplitude
        v_freq_mod: float = freq_mod if freq_mod is not None else self.freq_mod
        v_width: float = width if width is not None else self.width
        v_phase: float = phase if phase is not None else self.phase

        return sawtooth(n, f, a, v_freq_mod, v_width, v_phase)


def sawtooth(
    length: int = default_values[BASE_OSCILLATIONS][PARAMETERS.LENGTH],
    frequency: float = default_values[BASE_OSCILLATIONS][PARAMETERS.FREQUENCY],
    amplitude: float = default_values[BASE_OSCILLATIONS][PARAMETERS.AMPLITUDE],
    freq_mod: float = default_values[BASE_OSCILLATIONS][PARAMETERS.FREQ_MOD],
    width: float = default_values[BASE_OSCILLATIONS][PARAMETERS.WIDTH],
    phase: float = 0.0,
) -> np.ndarray:
    base_ts = prepare_base_signal(length, frequency)
    func = partial(signal.sawtooth, width=width)
    return generate_periodic_signal(base_ts, func, amplitude, freq_mod, phase)


BaseOscillation.register(Sawtooth.KIND, Sawtooth)
