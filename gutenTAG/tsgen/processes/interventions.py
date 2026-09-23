"""Declared interventions on laws, signals, innovations, or observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .laws import Realization, ar_filter, require_spd
from ..sensor_artifacts import SENSOR_KINDS, apply_sensor_artifact

RELATION_ONLY = ("covariance-change", "precision-change")
SIGNAL_KINDS = (
    "mean",
    "amplitude",
    "noise-scale",
    "slope",
    "curvature",
    "bump",
    "impulse",
    "factor-loading",
)
MECHANISMS = ("null",) + RELATION_ONLY + SIGNAL_KINDS + SENSOR_KINDS


@dataclass(frozen=True)
class Intervention:
    """Half-open support fixed before drawing query observations.

    For mean/slope the signed strength is measured in marginal standard
    deviations; amplitude/noise-scale use a LOG standard-deviation ratio.
    Covariance strength is interpolation weight in [0,1]; zero is a true null.
    """

    kind: str
    start: int
    end: int
    strength: float
    channels: tuple[int, ...]
    transition: int = 0

    def validate(self, length: int, channels: int) -> None:
        if self.kind not in MECHANISMS or not 0 <= self.start < self.end <= length:
            raise ValueError("unknown intervention or invalid support")
        if not np.isfinite(self.strength) or not self.channels:
            raise ValueError("strength must be finite and channels nonempty")
        if len(set(self.channels)) != len(self.channels) or any(
            c < 0 or c >= channels for c in self.channels
        ):
            raise ValueError("invalid intervention channels")
        if self.transition < 0 or 2 * self.transition >= self.end - self.start:
            raise ValueError("transition must leave a nonempty event interior")
        if self.kind in RELATION_ONLY + SENSOR_KINDS and not 0 <= self.strength <= 1:
            raise ValueError("relation/sensor strength must be in [0,1]")
        if self.kind in ("amplitude", "noise-scale") and abs(self.strength) > 8:
            raise ValueError("log scale ratio exceeds the supported safe range")

    def envelope(self, length: int) -> np.ndarray:
        g = np.zeros(length)
        g[self.start : self.end] = 1
        if self.transition:
            ramp = np.linspace(0, 1, self.transition + 2)[1:-1]
            g[self.start : self.start + self.transition] = ramp
            g[self.end - self.transition : self.end] = ramp[::-1]
        return g


@dataclass
class InterventionResult:
    values: np.ndarray
    observed_mask: np.ndarray
    timestamps: np.ndarray
    intervention_mask: np.ndarray
    effect_mask: np.ndarray
    evaluation_mask: np.ndarray
    events: list[dict[str, Any]]


def apply_interventions(
    realization: Realization, events: list[Intervention], *, seed: int
) -> InterventionResult:
    """Apply a nonoverlapping plan without ever mutating the clean realization.

    AR covariance changes preserve diagonal innovations and hence complete
    one-channel trajectory laws, including boundaries. Recovery tails are not
    silently reset: the evaluation mask explicitly excludes post-event recovery.
    """
    law = realization.law
    n, c = realization.values.shape
    values = realization.values.copy()
    innovations = realization.noise.copy()
    mask = np.zeros((n, c), dtype=bool)
    evaluation = np.ones(n, dtype=bool)
    # Mixed sensor/process events require causal composition. Never silently
    # add an AR recovery tail after a sensor transformation of the same samples.
    if len(events) > 1 and any(
        e.kind in RELATION_ONLY + ("noise-scale",) and e.strength != 0 for e in events
    ):
        raise ValueError(
            "Multi-event innovation interventions require an explicit causal composition contract"
        )
    occupied = np.zeros(n, dtype=bool)
    times = np.broadcast_to(np.arange(n)[:, None], (n, c)).astype(float).copy()
    metadata = []
    rng = np.random.default_rng(seed)
    for index, event in enumerate(events):
        event.validate(n, c)
        if occupied[event.start : event.end].any():
            raise ValueError(
                "v13 plans must not overlap; composition requires an explicit contract"
            )
        occupied[event.start : event.end] = True
        g = event.envelope(n)
        active = event.kind != "null" and event.strength != 0
        selected = list(event.channels)
        pure = False
        extra: dict[str, Any] = {}
        if not active:
            pass
        elif event.kind in RELATION_ONLY:
            if law.family != "diagonal-ar" or c < 2:
                raise ValueError(
                    "exact relation-only requires diagonal-ar and >=2 channels"
                )
            sigma = law.covariance
            target = selected[0]
            if event.kind == "covariance-change":
                signs = np.ones(c)
                signs[target] = -1
                endpoint = sigma * np.outer(signs, signs)
            else:
                precision = np.linalg.inv(sigma)
                partner = (target + 1) % c
                v = np.zeros(c)
                v[target] = 1
                v[partner] = -1
                changed = np.linalg.inv(precision + 2 * np.outer(v, v))
                sd = np.sqrt(np.diag(changed) / np.diag(sigma))
                endpoint = changed / np.outer(sd, sd)
            require_spd(endpoint)
            for weight in np.unique(g[event.start : event.end]):
                where = np.flatnonzero((g == weight) & (g > 0))
                mixed = (
                    1 - weight * event.strength
                ) * sigma + weight * event.strength * endpoint
                innovations[where] = (
                    realization.standard_noise[where] @ np.linalg.cholesky(mixed).T
                )
            pure = True
            selected = list(range(c))
            extra = {
                "sigma_before": sigma.tolist(),
                "sigma_endpoint": endpoint.tolist(),
                "trajectory_marginals_preserved": True,
                "intervention_channel_semantics": "joint_reparameterization_not_causal_blame",
            }
            memory = float(np.max(np.abs(law.coefficients)))
            recovery = 0 if memory == 0 else int(np.ceil(np.log(1e-8) / np.log(memory)))
            evaluation[event.end : min(n, event.end + recovery)] = False
            extra["recovery_exclusion"] = [event.end, min(n, event.end + recovery)]
        elif event.kind == "factor-loading":
            if law.family != "gp-lmc":
                raise ValueError("factor-loading requires shared temporal GP factors")
            for ch in selected:
                values[:, ch] -= 2 * event.strength * g * realization.signal[:, ch]
            extra["trajectory_marginals_preserved"] = False
        elif event.kind in SENSOR_KINDS:
            for ch in selected:
                lo, hi = event.start, event.end
                sensor = apply_sensor_artifact(
                    values[lo:hi, ch],
                    kind=event.kind,
                    severity=event.strength,
                    rng=rng,
                    timestamps=times[lo:hi, ch],
                    previous_value=(
                        float(values[lo - 1, ch]) if lo else float(values[0, ch])
                    ),
                    reference_center=float(law.mean[ch]),
                    reference_scale=float(law.marginal_std[ch]),
                )
                values[lo:hi, ch], times[lo:hi, ch] = sensor.values, sensor.timestamps
            extra["sensor_calibration"] = "law_parameters"
        elif event.kind != "null":
            u = np.linspace(0, 1, event.end - event.start)
            shape = g.copy()
            if event.kind in ("slope", "curvature", "bump", "impulse"):
                shape[:] = 0
                local = {
                    "slope": u,
                    "curvature": u * u,
                    "bump": np.sin(np.pi * u) ** 2,
                    "impulse": (np.arange(u.size) == u.size // 2).astype(float),
                }[event.kind]
                shape[event.start : event.end] = local * g[event.start : event.end]
            for ch in selected:
                if event.kind == "amplitude":
                    values[:, ch] += (
                        np.expm1(event.strength * g) * realization.signal[:, ch]
                    )
                elif event.kind == "noise-scale":
                    if law.family == "diagonal-ar":
                        innovations[:, ch] *= np.exp(event.strength * g)
                    else:
                        values[:, ch] += (
                            np.expm1(event.strength * g) * realization.noise[:, ch]
                        )
                else:
                    values[:, ch] += event.strength * shape * law.marginal_std[ch]
        if active and event.kind == "noise-scale" and law.family == "diagonal-ar":
            memory = float(np.max(np.abs(law.coefficients[selected])))
            recovery = 0 if memory == 0 else int(np.ceil(np.log(1e-8) / np.log(memory)))
            evaluation[event.end : min(n, event.end + recovery)] = False
            extra["recovery_exclusion"] = [event.end, min(n, event.end + recovery)]
        if active:
            mask[event.start : event.end, selected] = True
        metadata.append(
            {
                **vars(event),
                **extra,
                "event_id": f"{realization.realization_id}:event:{index}",
                "law_id": law.law_id,
                "realization_id": realization.realization_id,
                "is_null": not active,
                "relation_only": pure,
                "requested_strength": event.strength,
                "effective_strength": event.strength,
            }
        )
    if law.family == "diagonal-ar":
        values += (
            ar_filter(innovations, law.coefficients, realization.initial)
            - realization.signal
        )
    effect = ~np.isclose(
        values, realization.values, rtol=0, atol=1e-12, equal_nan=True
    ) | (times != np.arange(n)[:, None])
    for record in metadata:
        lo, hi = record["start"], record["end"]
        delta = values[lo:hi] - realization.values[lo:hi]
        finite = delta[np.isfinite(delta)]
        record["realized_rms"] = (
            float(np.sqrt(np.mean(finite**2))) if finite.size else None
        )
    return InterventionResult(
        values, np.isfinite(values), times, mask, effect, evaluation, metadata
    )
