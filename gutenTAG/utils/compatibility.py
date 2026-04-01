import json
import warnings
from pathlib import Path

import pandas as pd

from ..utils.global_variables import BASE_OSCILLATION_NAMES, ANOMALY_TYPE_NAMES


class Compatibility:
    hard_combinations = pd.DataFrame(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],  # amplitude
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # correlation-flip
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # covariance-change
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # channel-rewiring
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # extremum
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  # frequency
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # lag-synchronization
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # mean
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],  # pattern
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  # pattern_shift
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # platform
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # shared-factor-break
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # trend
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # variance
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # mode_correlation
        ],
        columns=[
            BASE_OSCILLATION_NAMES.SINE,
            BASE_OSCILLATION_NAMES.COSINE,
            BASE_OSCILLATION_NAMES.SQUARE,
            BASE_OSCILLATION_NAMES.RANDOM_WALK,
            BASE_OSCILLATION_NAMES.CYLINDER_BELL_FUNNEL,
            BASE_OSCILLATION_NAMES.ECG,
            BASE_OSCILLATION_NAMES.POLYNOMIAL,
            BASE_OSCILLATION_NAMES.RANDOM_MODE_JUMP,
            BASE_OSCILLATION_NAMES.FORMULA,
            BASE_OSCILLATION_NAMES.SAWTOOTH,
            BASE_OSCILLATION_NAMES.DIRICHLET,
            BASE_OSCILLATION_NAMES.MLS,
            BASE_OSCILLATION_NAMES.CUSTOM_INPUT,
        ],
        index=[
            ANOMALY_TYPE_NAMES.AMPLITUDE,
            ANOMALY_TYPE_NAMES.CORRELATION_FLIP,
            ANOMALY_TYPE_NAMES.COVARIANCE_CHANGE,
            ANOMALY_TYPE_NAMES.CHANNEL_REWIRING,
            ANOMALY_TYPE_NAMES.EXTREMUM,
            ANOMALY_TYPE_NAMES.FREQUENCY,
            ANOMALY_TYPE_NAMES.LAG_SYNCHRONIZATION,
            ANOMALY_TYPE_NAMES.MEAN,
            ANOMALY_TYPE_NAMES.PATTERN,
            ANOMALY_TYPE_NAMES.PATTERN_SHIFT,
            ANOMALY_TYPE_NAMES.PLATFORM,
            ANOMALY_TYPE_NAMES.SHARED_FACTOR_BREAK,
            ANOMALY_TYPE_NAMES.TREND,
            ANOMALY_TYPE_NAMES.VARIANCE,
            ANOMALY_TYPE_NAMES.MODE_CORRELATION,
        ],
        dtype=bool,
    )
    recommended_combinations = hard_combinations.copy()
    _recommended_path = Path(__file__).with_name("recommended_compatibility.json")
    validated_combinations = hard_combinations.copy()
    _validated_path = Path(__file__).with_name("validated_compatibility.json")

    @staticmethod
    def get_matrix(mode: str = "hard") -> pd.DataFrame:
        """Return the compatibility matrix for the requested mode."""
        normalized = str(mode).strip().lower()
        if normalized == "hard":
            return Compatibility.hard_combinations
        if normalized == "recommended":
            return Compatibility.recommended_combinations
        if normalized == "validated":
            return Compatibility.validated_combinations
        raise ValueError(
            "Compatibility mode must be one of {'hard','recommended','validated'}"
        )

    @staticmethod
    def check(anomaly: str, base_oscillation: str, mode: str = "hard") -> bool:
        try:
            return bool(
                Compatibility.get_matrix(mode=mode).loc[anomaly, base_oscillation]
            )
        except KeyError:
            warnings.warn(
                message=(
                    f"No compatibility information for BO {base_oscillation} and "
                    f"anomaly {anomaly} found!"
                ),
                category=UserWarning,
            )
            return False


if Compatibility._recommended_path.exists():
    with Compatibility._recommended_path.open("r", encoding="utf-8") as handle:
        _recommended_payload = json.load(handle)
    if isinstance(_recommended_payload, dict):
        for _anomaly_name, _allowed_bases in _recommended_payload.items():
            if _anomaly_name not in Compatibility.recommended_combinations.index:
                continue
            Compatibility.recommended_combinations.loc[_anomaly_name, :] = False
            if isinstance(_allowed_bases, list):
                _valid_bases = [
                    _base_name
                    for _base_name in _allowed_bases
                    if _base_name in Compatibility.recommended_combinations.columns
                ]
                if len(_valid_bases) > 0:
                    Compatibility.recommended_combinations.loc[
                        _anomaly_name, _valid_bases
                    ] = True

Compatibility.validated_combinations.loc[:, :] = Compatibility.recommended_combinations

if Compatibility._validated_path.exists():
    with Compatibility._validated_path.open("r", encoding="utf-8") as handle:
        _validated_payload = json.load(handle)
    if isinstance(_validated_payload, dict):
        for _anomaly_name, _allowed_bases in _validated_payload.items():
            if _anomaly_name not in Compatibility.validated_combinations.index:
                continue
            Compatibility.validated_combinations.loc[_anomaly_name, :] = False
            if isinstance(_allowed_bases, list):
                _valid_bases = [
                    _base_name
                    for _base_name in _allowed_bases
                    if _base_name in Compatibility.validated_combinations.columns
                ]
                if len(_valid_bases) > 0:
                    Compatibility.validated_combinations.loc[
                        _anomaly_name, _valid_bases
                    ] = True
