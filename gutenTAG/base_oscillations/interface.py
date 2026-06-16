from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..utils.default_values import default_values
from ..utils.global_variables import PARAMETERS, BASE_OSCILLATIONS
from ..utils.types import BOGenerationContext


class BaseOscillationInterface(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._init_core_parameters(kwargs)
        self._init_shape_parameters(kwargs)
        self._init_channel_parameters(kwargs)
        self._init_input_parameters(kwargs)
        self._init_components(kwargs)

    @staticmethod
    def _parameter_value(kwargs: dict, parameter):
        return kwargs.get(parameter, default_values[BASE_OSCILLATIONS][parameter])

    def _init_core_parameters(self, kwargs: dict) -> None:
        self.length = self._parameter_value(kwargs, PARAMETERS.LENGTH)
        self.frequency = self._parameter_value(kwargs, PARAMETERS.FREQUENCY)
        self.amplitude = self._parameter_value(kwargs, PARAMETERS.AMPLITUDE)
        self.variance = self._parameter_value(kwargs, PARAMETERS.VARIANCE)
        self.random_seed = self._parameter_value(kwargs, PARAMETERS.RANDOM_SEED)
        self.phase = self._parameter_value(kwargs, PARAMETERS.PHASE)
        self.offset = self._parameter_value(kwargs, PARAMETERS.OFFSET)

    def _init_shape_parameters(self, kwargs: dict) -> None:
        self.avg_pattern_length = self._parameter_value(
            kwargs,
            PARAMETERS.AVG_PATTERN_LENGTH,
        )
        self.variance_pattern_length = self._parameter_value(
            kwargs,
            PARAMETERS.VARIANCE_PATTERN_LENGTH,
        )
        self.variance_amplitude = self._parameter_value(
            kwargs,
            PARAMETERS.VARIANCE_AMPLITUDE,
        )
        self.freq_mod = self._parameter_value(kwargs, PARAMETERS.FREQ_MOD)
        self.polynomial = self._parameter_value(kwargs, PARAMETERS.POLYNOMIAL)
        self.trend: Optional[BaseOscillationInterface] = self._parameter_value(
            kwargs,
            PARAMETERS.TREND,
        )
        self.smoothing = self._parameter_value(kwargs, PARAMETERS.SMOOTHING)

    def _init_channel_parameters(self, kwargs: dict) -> None:
        self.channel_diff = self._parameter_value(kwargs, PARAMETERS.CHANNEL_DIFF)
        self.channel_offset = kwargs.get(PARAMETERS.CHANNEL_OFFSET, self.amplitude)

    def _init_input_parameters(self, kwargs: dict) -> None:
        self.formula = self._parameter_value(kwargs, PARAMETERS.FORMULA)
        self.ecg_sim_method = self._parameter_value(kwargs, PARAMETERS.ECG_SIM_METHOD)
        self.width = self._parameter_value(kwargs, PARAMETERS.WIDTH)
        self.duty = self._parameter_value(kwargs, PARAMETERS.DUTY)
        self.periodicity = self._parameter_value(kwargs, PARAMETERS.PERIODICITY)
        self.complexity = self._parameter_value(kwargs, PARAMETERS.COMPLEXITY)
        self.input_timeseries_path_train = self._parameter_value(
            kwargs,
            PARAMETERS.INPUT_TIMESERIES_PATH_TRAIN,
        )
        self.input_timeseries_path_test = self._parameter_value(
            kwargs,
            PARAMETERS.INPUT_TIMESERIES_PATH_TEST,
        )
        self.use_column_train = self._parameter_value(
            kwargs, PARAMETERS.USE_COLUMN_TRAIN
        )
        self.use_column_test = self._parameter_value(kwargs, PARAMETERS.USE_COLUMN_TEST)

    def _init_components(self, kwargs: dict) -> None:
        self.timeseries: Optional[np.ndarray] = None
        self.noise: Optional[np.ndarray] = None
        self.trend_series: Optional[np.ndarray] = None

    def generate_noise(
        self, ctx: BOGenerationContext, variance: float, length: int
    ) -> np.ndarray:
        return ctx.rng.normal(0, variance, length)

    def _generate_trend(self, ctx: BOGenerationContext) -> np.ndarray:
        trend_series = np.zeros(self.length)
        if self.trend:
            self.trend.length = self.length
            self.trend.generate_timeseries_and_variations(ctx)
            if self.trend.timeseries is not None:
                trend_series = self.trend.timeseries
            if self.trend.trend_series is not None:
                trend_series += self.trend.trend_series
        return trend_series

    def generate_timeseries_and_variations(
        self, ctx: BOGenerationContext, **kwargs
    ) -> BaseOscillationInterface:
        self.timeseries = self.generate_only_base(ctx, **kwargs)
        self.trend_series = self._generate_trend(ctx.to_trend())
        self.noise = self.generate_noise(
            ctx, self.variance * self.amplitude, self.length
        )
        return self

    def is_periodic(self) -> bool:
        periods = self.get_timeseries_periods()
        return periods is not None and periods > 1

    @abstractmethod
    def get_timeseries_periods(self) -> Optional[int]:
        """
        How many same-sized periods occur in the time series?

        If no periodicity is given, return None.
        :return: Optional[int]
        """
        raise NotImplementedError()

    def get_period_size(self) -> Optional[int]:
        """
        Return the number of points within one period.

        ``period_size * n_periods`` might not exactly equal the series length.
        If no periodicity is given, return None.
        """
        if self.is_periodic():
            return self.length // self.get_timeseries_periods()
        else:
            return None

    @abstractmethod
    def get_base_oscillation_kind(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def generate_only_base(
        self, ctx: BOGenerationContext, *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseOscillationInterface:
            if any("generate" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
