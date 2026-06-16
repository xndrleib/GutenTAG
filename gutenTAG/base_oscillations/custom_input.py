from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd
import warnings

from . import BaseOscillation
from .interface import BaseOscillationInterface
from ..utils.global_variables import BASE_OSCILLATION_NAMES
from ..utils.types import BOGenerationContext


@dataclass(frozen=True)
class _CustomInputOptions:
    """Resolved custom-input CSV read options."""

    path: Optional[str]
    use_column: Optional[Union[str, int]]
    length: int
    training: bool


class CustomInput(BaseOscillationInterface):
    KIND = BASE_OSCILLATION_NAMES.CUSTOM_INPUT

    def get_base_oscillation_kind(self) -> str:
        return self.KIND

    def get_timeseries_periods(self) -> Optional[int]:
        return None

    def generate_only_base(
        self,
        ctx: BOGenerationContext,
        input_timeseries_path_test: Optional[str] = None,
        use_column_test: Optional[Union[str, int]] = None,
        length: Optional[int] = None,
        input_timeseries_path_train: Optional[str] = None,
        use_column_train: Optional[Union[str, int]] = None,
        semi_supervised: Optional[bool] = None,
        supervised: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Generate a numpy array of timeseries data from a CSV file.

        The following requirements must be met by the input file:

        - CSV file with the first line as the header.
        - The file should not contain any anomalies. Otherwise, the label
          information provided by GutenTAG will be wrong.
        - If the extracted channel is specified using an integer, the first column is column 0.
        - `custom-input` can extract only a single column/channel/dimension from
          the input file at a time. If multiple channels are required, use the
          `custom-input` base oscillation multiple times.

        Arguments
        ---------
        ctx : BOGenerationContext
            An instance of the BOGenerationContext class.
        input_timeseries_path_test : str, optional
            The path to the test data CSV file. Defaults to None.
        use_column_test : Union[str, int], optional
            The name or index of the column containing the test data. Defaults to None.
        length : Optional[int], optional
            The desired length of the output time-series data. Defaults to None.
        input_timeseries_path_train : Optional[str], optional
            The path to the training data CSV file. Defaults to None.
        use_column_train : Optional[Union[str, int]], optional
            The name or index of the column containing the training data. Defaults to None.
        semi_supervised : Optional[bool], optional
            A flag to indicate if the model is trained in semi-supervised mode. Defaults to None.
        supervised : Optional[bool], optional
            A flag to indicate if the model is trained in supervised mode. Defaults to None.

        Returns
        -------
        np.ndarray
            A numpy array of the generated time-series data.

        Raises
        ------
        ValueError
            If the number of rows in the input timeseries file is less than the desired length.
        """
        options = self._resolve_input_options(
            length=length,
            input_timeseries_path_train=input_timeseries_path_train,
            input_timeseries_path_test=input_timeseries_path_test,
            use_column_train=use_column_train,
            use_column_test=use_column_test,
            semi_supervised=semi_supervised,
            supervised=supervised,
        )
        frame = self._read_input_frame(options)
        self._validate_frame_length(frame, options.length)
        return self._float_frame(frame).iloc[: options.length, 0]

    def _resolve_input_options(
        self,
        *,
        length: Optional[int],
        input_timeseries_path_train: Optional[str],
        input_timeseries_path_test: Optional[str],
        use_column_train: Optional[Union[str, int]],
        use_column_test: Optional[Union[str, int]],
        semi_supervised: Optional[bool],
        supervised: Optional[bool],
    ) -> _CustomInputOptions:
        resolved_train_path = (
            input_timeseries_path_train or self.input_timeseries_path_train
        )
        resolved_test_path = (
            input_timeseries_path_test or self.input_timeseries_path_test
        )
        training = bool(semi_supervised or supervised)
        return _CustomInputOptions(
            path=resolved_train_path if training else resolved_test_path,
            use_column=(
                (use_column_train or self.use_column_train)
                if training
                else (use_column_test or self.use_column_test)
            ),
            length=length or self.length,
            training=training,
        )

    @staticmethod
    def _read_input_frame(options: _CustomInputOptions) -> pd.DataFrame:
        if options.training:
            if options.path is None:
                raise ValueError(
                    "No path to an input timeseries file for the training timeseries specified!"
                )
        elif options.path is None:
            raise ValueError("No path to an input timeseries file specified!")
        if options.use_column is None:
            return pd.read_csv(options.path)
        return pd.read_csv(options.path, usecols=cast(Any, (options.use_column,)))

    @staticmethod
    def _validate_frame_length(frame: pd.DataFrame, length: int) -> None:
        if len(frame) < length:
            raise ValueError(
                "Number of rows in the input timeseries file is less than the desired length"
            )

    @staticmethod
    def _float_frame(frame: pd.DataFrame) -> pd.DataFrame:
        col_type = frame.dtypes.iloc[0]
        if col_type != np.float64:
            frame = frame.astype(np.float64)
            warnings.warn(
                f"Input data was of {col_type} type and has been automatically converted to float."
            )
        return frame


BaseOscillation.register(CustomInput.KIND, CustomInput)
