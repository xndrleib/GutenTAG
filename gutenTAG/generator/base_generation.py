"""Base-channel parameter and construction helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.random import SeedSequence

from ..base_oscillations import BaseOscillation
from ..utils.global_variables import PARAMETERS
from ..utils.types import GenerationContext
from .base_channels import apply_shared_noise_correlation, apply_variations

ParameterRealizer = Callable[[Mapping[str, Any], np.random.Generator], dict[str, Any]]
BaseFactory = Callable[..., Any]


@dataclass
class BaseInstanceSeries:
    """Generated base-channel arrays for one instance."""

    channel_bos: list[Any]
    base_values: np.ndarray
    observed_values: np.ndarray


@dataclass
class PreparedBaseInstance:
    """Resolved base parameters and generated clean base values for one instance."""

    base_parameters: dict[str, Any]
    base_channel_parameters: list[dict[str, Any]]
    base_parameters_per_channel: list[dict[str, Any]]
    split_phase_shift_info: dict[str, Any]
    series: BaseInstanceSeries


def realize_base_channel_parameters(
    *,
    template: Mapping[str, Any],
    channels: int,
    rng: np.random.Generator,
    realize_parameters: ParameterRealizer,
) -> list[dict[str, Any]]:
    """Realize base-channel parameter templates per channel.

    Parameters
    ----------
    template : Mapping[str, Any]
        Raw base-channel parameter template.
    channels : int
        Number of channels to realize.
    rng : numpy.random.Generator
        Random number generator for deterministic realization.
    realize_parameters : Callable[[Mapping[str, Any], numpy.random.Generator], dict]
        Parameter-template realization callback.

    Returns
    -------
    list[dict[str, Any]]
        Per-channel parameter dictionaries.
    """
    if len(template) == 0:
        return [{} for _ in range(int(channels))]
    shared_template: dict[str, Any] = {}
    per_channel_template: dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, Mapping) and bool(
            value.get("shared_across_channels", False)
        ):
            shared_spec = copy.deepcopy(dict(value))
            shared_spec.pop("shared_across_channels", None)
            shared_template[str(key)] = shared_spec
        else:
            per_channel_template[str(key)] = value

    shared_values = (
        realize_parameters(shared_template, rng) if len(shared_template) > 0 else {}
    )
    realized: list[dict[str, Any]] = []
    for _ in range(int(channels)):
        channel_params = (
            realize_parameters(per_channel_template, rng)
            if len(per_channel_template) > 0
            else {}
        )
        if len(shared_values) > 0:
            channel_params.update(copy.deepcopy(shared_values))
        realized.append(channel_params)
    return realized


def compose_base_parameters_per_channel(
    *,
    base_parameters: Mapping[str, Any],
    base_channel_parameters: Sequence[Mapping[str, Any]],
    channels: int,
    length: int,
) -> list[dict[str, Any]]:
    """Merge base and channel-specific parameters.

    Parameters
    ----------
    base_parameters : Mapping[str, Any]
        Shared base-oscillation parameters.
    base_channel_parameters : Sequence[Mapping[str, Any]]
        Optional per-channel overrides.
    channels : int
        Number of channels.
    length : int
        Time-series length.

    Returns
    -------
    list[dict[str, Any]]
        Per-channel base-oscillation parameters with ``length`` set.
    """
    channel_parameters: list[dict[str, Any]] = []
    for channel in range(int(channels)):
        params = copy.deepcopy(dict(base_parameters))
        if channel < len(base_channel_parameters):
            params.update(copy.deepcopy(dict(base_channel_parameters[channel])))
        params[PARAMETERS.LENGTH] = int(length)
        channel_parameters.append(params)
    return channel_parameters


def apply_split_phase_shift(
    *,
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    split: str,
    split_phase_shift: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply deterministic split-specific phase shifts.

    Parameters
    ----------
    base_parameters_per_channel : Sequence[Mapping[str, Any]]
        Per-channel base parameters.
    split : str
        Dataset split name.
    split_phase_shift : Mapping[str, Any]
        Split phase-shift configuration.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Shifted parameters and metadata describing the applied shift.
    """
    cfg = dict(split_phase_shift)
    enabled = bool(cfg.get("enabled", False))
    phase_modulo = float(cfg.get("phase_modulo", float(2.0 * np.pi)))
    offset = 0.0
    if enabled:
        offset = float(dict(cfg.get("values", {}))[split])

    realized: list[dict[str, Any]] = [
        copy.deepcopy(dict(params)) for params in base_parameters_per_channel
    ]
    channels_with_phase: list[int] = []
    for channel, params in enumerate(realized):
        if "phase" not in params:
            continue
        channels_with_phase.append(channel)
        if not enabled or abs(offset) <= 0.0:
            continue
        shifted_phase = float(params["phase"]) + offset
        params["phase"] = float(np.mod(shifted_phase, phase_modulo))

    info: dict[str, Any] = {
        "enabled": enabled,
        "split": split,
        "offset": float(offset),
        "phase_modulo": phase_modulo,
        "channels_with_phase": channels_with_phase,
        "phase_shift_applied": bool(enabled and abs(offset) > 0.0),
    }
    return realized, info


def generate_base_channels(
    *,
    base_kind: str,
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    seed: int,
    expected_length: int,
    base_factory: BaseFactory = BaseOscillation.from_key,
) -> list[Any]:
    """Generate base-oscillation objects for every channel.

    Parameters
    ----------
    base_kind : str
        Base-oscillation kind.
    base_parameters_per_channel : Sequence[Mapping[str, Any]]
        Per-channel constructor parameters.
    seed : int
        Seed used to initialize generation context.
    expected_length : int
        Required generated series length.
    base_factory : Callable[..., Any]
        Factory used to construct base-oscillation objects.

    Returns
    -------
    list[Any]
        Generated and sanitized base-oscillation objects.
    """
    ctx = GenerationContext(SeedSequence(seed))
    channels: list[Any] = []
    previous_channels: list[np.ndarray] = []
    for channel, channel_parameters in enumerate(base_parameters_per_channel):
        bo = base_factory(base_kind, **copy.deepcopy(dict(channel_parameters)))
        bo.generate_timeseries_and_variations(
            ctx.to_bo(channel=channel, previous_channels=previous_channels)
        )
        sanitize_generated_base_channel(
            base_kind=base_kind,
            bo=bo,
            expected_length=expected_length,
        )
        channels.append(bo)
        previous_channels.append(np.array(bo.timeseries, copy=True))
    return channels


def generate_base_instance_series(
    *,
    base_kind: str,
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    seed: int,
    shared_noise_seed: int,
    base_channel_correlation: Mapping[str, Any],
    expected_length: int,
    base_factory: BaseFactory = BaseOscillation.from_key,
) -> BaseInstanceSeries:
    """Generate base-channel objects, latent base values, and observed values.

    Parameters
    ----------
    base_kind : str
        Base-oscillation kind.
    base_parameters_per_channel : Sequence[Mapping[str, Any]]
        Per-channel constructor parameters.
    seed : int
        Seed used for base-channel generation.
    shared_noise_seed : int
        Seed used for shared-noise correlation.
    base_channel_correlation : Mapping[str, Any]
        Effective channel-correlation config.
    expected_length : int
        Required generated series length.
    base_factory : Callable[..., Any]
        Base-oscillation factory.

    Returns
    -------
    BaseInstanceSeries
        Generated channel objects, latent base values, and observed values after
        channel variations.
    """
    channel_bos = generate_base_channels(
        base_kind=base_kind,
        base_parameters_per_channel=base_parameters_per_channel,
        seed=seed,
        expected_length=expected_length,
        base_factory=base_factory,
    )
    apply_shared_noise_correlation(
        channel_bos=channel_bos,
        seed=shared_noise_seed,
        base_channel_correlation=base_channel_correlation,
    )
    base_values = stack_channel_timeseries(channel_bos)
    observed_values = apply_variations(base_values, channel_bos)
    return BaseInstanceSeries(
        channel_bos=channel_bos,
        base_values=base_values,
        observed_values=observed_values,
    )


def prepare_base_instance(
    *,
    base_kind: str,
    split: str,
    seeds: Mapping[str, int],
    base_parameter_template: Mapping[str, Any],
    base_channel_parameter_template: Mapping[str, Any],
    fixed_base_parameters: Mapping[str, Any] | None,
    fixed_base_channel_parameters: Sequence[Mapping[str, Any]] | None,
    base_parameter_policy: str,
    base_channel_parameter_policy: str,
    channels: int,
    length: int,
    split_phase_shift: Mapping[str, Any],
    base_channel_correlation: Mapping[str, Any],
    realize_parameters: ParameterRealizer,
    base_factory: BaseFactory = BaseOscillation.from_key,
) -> PreparedBaseInstance:
    """Resolve base parameters and generate one observed base instance."""

    base_parameter_rng = np.random.default_rng(int(seeds["base_params_seed"]))
    base_channel_parameter_rng = np.random.default_rng(
        int(seeds["base_channel_params_seed"])
    )
    base_parameters = _resolve_instance_base_parameters(
        policy=base_parameter_policy,
        template=base_parameter_template,
        fixed_parameters=fixed_base_parameters,
        rng=base_parameter_rng,
        realize_parameters=realize_parameters,
    )
    base_channel_parameters = _resolve_instance_base_channel_parameters(
        policy=base_channel_parameter_policy,
        template=base_channel_parameter_template,
        fixed_parameters=fixed_base_channel_parameters,
        channels=channels,
        rng=base_channel_parameter_rng,
        realize_parameters=realize_parameters,
    )
    base_parameters_per_channel, split_phase_shift_info = (
        _prepare_base_parameters_per_channel(
            base_parameters=base_parameters,
            base_channel_parameters=base_channel_parameters,
            channels=channels,
            length=length,
            split=split,
            split_phase_shift=split_phase_shift,
        )
    )
    series = _generate_prepared_base_series(
        base_kind=base_kind,
        base_parameters_per_channel=base_parameters_per_channel,
        seeds=seeds,
        base_channel_correlation=base_channel_correlation,
        length=length,
        base_factory=base_factory,
    )
    return PreparedBaseInstance(
        base_parameters=base_parameters,
        base_channel_parameters=base_channel_parameters,
        base_parameters_per_channel=base_parameters_per_channel,
        split_phase_shift_info=split_phase_shift_info,
        series=series,
    )


def _resolve_instance_base_parameters(
    *,
    policy: str,
    template: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any] | None,
    rng: np.random.Generator,
    realize_parameters: ParameterRealizer,
) -> dict[str, Any]:
    if policy == "fixed_per_variant" and fixed_parameters is not None:
        return copy.deepcopy(dict(fixed_parameters))
    return realize_parameters(template, rng)


def _resolve_instance_base_channel_parameters(
    *,
    policy: str,
    template: Mapping[str, Any],
    fixed_parameters: Sequence[Mapping[str, Any]] | None,
    channels: int,
    rng: np.random.Generator,
    realize_parameters: ParameterRealizer,
) -> list[dict[str, Any]]:
    if policy == "fixed_per_variant" and fixed_parameters is not None:
        return [copy.deepcopy(dict(params)) for params in fixed_parameters]
    return realize_base_channel_parameters(
        template=template,
        channels=channels,
        rng=rng,
        realize_parameters=realize_parameters,
    )


def _prepare_base_parameters_per_channel(
    *,
    base_parameters: Mapping[str, Any],
    base_channel_parameters: Sequence[Mapping[str, Any]],
    channels: int,
    length: int,
    split: str,
    split_phase_shift: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_parameters = compose_base_parameters_per_channel(
        base_parameters=base_parameters,
        base_channel_parameters=base_channel_parameters,
        channels=channels,
        length=length,
    )
    return apply_split_phase_shift(
        base_parameters_per_channel=raw_parameters,
        split=split,
        split_phase_shift=split_phase_shift,
    )


def _generate_prepared_base_series(
    *,
    base_kind: str,
    base_parameters_per_channel: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, int],
    base_channel_correlation: Mapping[str, Any],
    length: int,
    base_factory: BaseFactory,
) -> BaseInstanceSeries:
    return generate_base_instance_series(
        base_kind=base_kind,
        base_parameters_per_channel=base_parameters_per_channel,
        seed=int(seeds["base_seed"]),
        shared_noise_seed=int(seeds["base_shared_noise_seed"]),
        base_channel_correlation=base_channel_correlation,
        expected_length=length,
        base_factory=base_factory,
    )


def sanitize_generated_base_channel(
    *, base_kind: str, bo: Any, expected_length: int
) -> None:
    """Normalize generated base-channel arrays in-place.

    Parameters
    ----------
    base_kind : str
        Base-oscillation kind used in validation errors.
    bo : Any
        Generated base-oscillation object.
    expected_length : int
        Required generated series length.

    Raises
    ------
    ValueError
        If the base oscillation did not produce a valid time-series.
    """
    if bo.timeseries is None:
        raise ValueError(f"Base oscillation '{base_kind}' produced no timeseries.")
    if bo.timeseries.shape[0] != int(expected_length):
        raise ValueError(
            f"Base oscillation '{base_kind}' produced length {bo.timeseries.shape[0]}, "
            f"expected {int(expected_length)}."
        )
    if bo.noise is None:
        bo.noise = np.zeros(int(expected_length), dtype=np.float64)
    if bo.trend_series is None:
        bo.trend_series = np.zeros(int(expected_length), dtype=np.float64)
    bo.timeseries = bo.timeseries.astype(np.float64)
    bo.noise = bo.noise.astype(np.float64)
    bo.trend_series = bo.trend_series.astype(np.float64)


def stack_channel_timeseries(channel_bos: Sequence[Any]) -> np.ndarray:
    """Stack generated channel time-series into a multichannel array.

    Parameters
    ----------
    channel_bos : Sequence[Any]
        Base-oscillation objects with generated ``timeseries`` arrays.

    Returns
    -------
    numpy.ndarray
        ``float64`` array with one column per channel.
    """
    return np.column_stack([bo.timeseries for bo in channel_bos]).astype(np.float64)
