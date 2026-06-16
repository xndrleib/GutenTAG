"""Parameter template realization helpers."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np


def sample_between(lower: Any, upper: Any, rng: np.random.Generator) -> Any:
    """Sample a value between two scalar bounds.

    Parameters
    ----------
    lower : Any
        Lower bound, or first bound when values are unordered.
    upper : Any
        Upper bound, or second bound when values are unordered.
    rng : numpy.random.Generator
        Random number generator used for deterministic sampling.

    Returns
    -------
    Any
        Boolean, integer, or float value sampled with the historical generator
        semantics.
    """
    if isinstance(lower, bool) or isinstance(upper, bool):
        return bool(lower)
    if isinstance(lower, int) and isinstance(upper, int):
        if lower > upper:
            lower, upper = upper, lower
        return int(rng.integers(lower, upper + 1))
    lower_f = float(lower)
    upper_f = float(upper)
    if lower_f > upper_f:
        lower_f, upper_f = upper_f, lower_f
    return float(rng.uniform(lower_f, upper_f))


def realize_parameters(
    template: Mapping[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Realize a parameter template into concrete values."""

    return {
        str(key): _realize_parameter_value(value, rng)
        for key, value in dict(template).items()
    }


def _realize_parameter_value(value: Any, rng: np.random.Generator) -> Any:
    if isinstance(value, Mapping):
        mapping = dict(value)
        if "distribution" in mapping:
            return _sample_distribution(mapping, rng)
        if "min" in mapping and "max" in mapping and len(mapping) == 2:
            return sample_between(mapping["min"], mapping["max"], rng)
        return {
            str(key): _realize_parameter_value(nested, rng)
            for key, nested in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        return _realize_sequence_value(value, rng)
    return value


def _realize_sequence_value(
    value: list[Any] | tuple[Any, ...],
    rng: np.random.Generator,
) -> Any:
    if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        return sample_between(value[0], value[1], rng)
    return [_realize_parameter_value(item, rng) for item in value]


def _sample_distribution(
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> Any:
    distribution = str(spec.get("distribution", "")).lower()
    if distribution == "uniform":
        sampled = _sample_uniform_distribution(spec, rng)
    elif distribution == "int_uniform":
        sampled = _sample_int_uniform_distribution(spec, rng)
    elif distribution == "choice":
        sampled = _sample_choice_distribution(spec, rng)
    elif distribution == "bernoulli":
        sampled = _sample_bernoulli_distribution(spec, rng)
    elif distribution == "reject_if_abs_lt":
        sampled = _sample_reject_if_abs_lt_distribution(spec, rng)
    else:
        raise ValueError(f"Unsupported distribution primitive: '{distribution}'")

    if "reject_if_abs_lt" in spec and distribution != "reject_if_abs_lt":
        return _sample_with_abs_rejection(
            spec,
            threshold=float(spec["reject_if_abs_lt"]),
            rng=rng,
            nested=False,
        )
    return sampled


def _sample_uniform_distribution(
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    low = float(spec.get("low", spec.get("min")))
    high = float(spec.get("high", spec.get("max")))
    return float(rng.uniform(min(low, high), max(low, high)))


def _sample_int_uniform_distribution(
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> int:
    low = int(spec.get("low", spec.get("min")))
    high = int(spec.get("high", spec.get("max")))
    if low > high:
        low, high = high, low
    return int(rng.integers(low, high + 1))


def _sample_choice_distribution(
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> Any:
    values = spec.get("values", spec.get("choices"))
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("choice distribution requires non-empty 'values' list")
    idx = int(rng.integers(0, len(values)))
    return _realize_parameter_value(copy.deepcopy(values[idx]), rng)


def _sample_bernoulli_distribution(
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> bool:
    p = float(spec.get("p", 0.5))
    if p < 0.0 or p > 1.0:
        raise ValueError("bernoulli distribution requires p in [0, 1]")
    return bool(rng.random() < p)


def _sample_reject_if_abs_lt_distribution(
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> Any:
    threshold = float(spec.get("threshold", spec.get("reject_if_abs_lt", 0.0)))
    base_spec = spec.get("base")
    if base_spec is None:
        raise ValueError(
            "reject_if_abs_lt distribution requires nested 'base' specification"
        )
    return _sample_with_abs_rejection(base_spec, threshold=threshold, rng=rng)


def _sample_with_abs_rejection(
    base_spec: Any,
    threshold: float,
    rng: np.random.Generator,
    nested: bool = True,
    max_tries: int = 512,
) -> Any:
    threshold = max(0.0, float(threshold))
    for _ in range(max_tries):
        candidate = _sample_abs_rejection_candidate(base_spec, rng, nested)
        if _candidate_passes_abs_threshold(candidate, threshold):
            return candidate
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(sign * threshold)


def _sample_abs_rejection_candidate(
    base_spec: Any,
    rng: np.random.Generator,
    nested: bool,
) -> Any:
    if isinstance(base_spec, Mapping):
        spec_dict = dict(base_spec)
        if not nested:
            spec_dict.pop("reject_if_abs_lt", None)
        return _sample_distribution(spec_dict, rng)
    return _realize_parameter_value(base_spec, rng)


def _candidate_passes_abs_threshold(candidate: Any, threshold: float) -> bool:
    if isinstance(candidate, bool):
        return True
    if isinstance(candidate, (int, float)) and abs(float(candidate)) >= threshold:
        return True
    return False
