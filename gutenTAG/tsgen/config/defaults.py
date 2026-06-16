"""Default runtime configuration values for TS dataset generation."""

from __future__ import annotations

from typing import Any

DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_DENSITY_RANGE: tuple[float, float] = (0.05, 0.10)
DEFAULT_SEGMENT_COUNT_RANGE: tuple[int, int] = (20, 50)
DEFAULT_PROFILES_PER_PAIR = 1

DEFAULT_BASE_OVERRIDES: dict[str, dict[str, Any]] = {
    "random-mode-jump": {"frequency": 250, "variance": 0.05, "random-seed": 7},
    "sine": {"frequency": 8.0, "variance": 0.03},
    "shared-noise-sine": {"frequency": 7.0, "amplitude": 0.55, "variance": 0.18},
    "cosine": {"frequency": 8.0, "variance": 0.03},
    "square": {"frequency": 8.0, "variance": 0.03},
    "sawtooth": {"frequency": 8.0, "variance": 0.03},
    "dirichlet": {"frequency": 10.0, "variance": 0.03},
    "ecg": {"frequency": 8.0, "variance": 0.03},
    "random-walk": {"smoothing": 0.01, "variance": 0.03},
    "polynomial": {"polynomial": [0.05, 0.4], "variance": 0.03},
    "cylinder-bell-funnel": {"variance": 0.03},
    "mls": {"complexity": 7, "variance": 0.03},
}

DEFAULT_ANOMALY_OVERRIDES: dict[str, dict[str, Any]] = {
    "amplitude": {"amplitude_factor": 2.0},
    "channel-rewiring": {"rotation_degrees": 25.0, "transition_length": 8},
    "correlation-flip": {"target_correlation": -0.85, "transition_length": 8},
    "covariance-change": {"coupling_strength": -0.9, "transition_length": 8},
    "frequency": {"frequency_factor": 2.0},
    "lag-synchronization": {"lag_steps": 6, "transition_length": 8},
    "mean": {"offset": 1.0},
    "pattern": {
        "sinusoid_k": 10.0,
        "cbf_pattern_factor": 2.0,
        "square_duty": 0.8,
        "sawtooth_width": 0.5,
    },
    "pattern-shift": {"shift_by": 4, "transition_window": 10},
    "platform": {"value": 3.0},
    "shared-factor-break": {"shared_factor_scale": 0.0, "transition_length": 8},
    "trend": {"oscillation": {"kind": "sine", "frequency": 2.0, "amplitude": 1.0}},
    "variance": {"variance": 1.0},
    "extremum": {"min": False, "local": False, "context_window": 10},
    "mode-correlation": {},
}

BASES_REQUIRING_EXTRA_CONFIG: tuple[str, ...] = ("custom-input", "formula")
ANOMALIES_INCOMPATIBLE_WITH_DENSITY_POLICY: tuple[str, ...] = ("extremum",)
