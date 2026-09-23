"""Executable split contracts; names alone never confer OOD or purity."""

from __future__ import annotations

import math
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..processes.interventions import MECHANISMS, RELATION_ONLY


class SplitSpec(BaseModel):
    """Parameter prior and identity constraints for one evaluation stratum."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal[
        "train",
        "validation",
        "calibration",
        "iid",
        "parameter_ood",
        "family_ood",
        "mechanism_ood",
        "relation_only",
    ]
    systems: int = Field(default=4, ge=1)
    families: tuple[str, ...] = ("diagonal-ar", "gp-lmc")
    mechanisms: tuple[str, ...] = ("mean", "noise-scale", "covariance-change")
    channels: tuple[int, ...] = (4, 8)
    length_scale: tuple[float, float] = (8.0, 32.0)
    memory: tuple[float, float] = (0.1, 0.7)
    noise_std: tuple[float, float] = (0.1, 0.4)
    ood_parameters: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_prior(self) -> SplitSpec:
        if not self.families or set(self.families) - {
            "diagonal-ar",
            "gp-lmc",
            "independent-gp",
        }:
            raise ValueError("families must be nonempty and registered")
        if not self.mechanisms or set(self.mechanisms) - set(MECHANISMS):
            raise ValueError("mechanisms must be nonempty and registered")
        if (
            not self.channels
            or min(self.channels) < 1
            or len(set(self.channels)) != len(self.channels)
        ):
            raise ValueError("channels must be distinct positive counts")
        for name in ("length_scale", "memory", "noise_std"):
            lo, hi = getattr(self, name)
            if not all(math.isfinite(x) for x in (lo, hi)) or not 0 <= lo <= hi:
                raise ValueError(f"{name} must be finite, nonnegative and ordered")
        if self.length_scale[0] <= 0 or self.memory[1] >= 1:
            raise ValueError("positive length scales and stable memory are required")
        if set(self.ood_parameters) - {
            "length_scale",
            "memory",
            "noise_std",
            "channels",
        }:
            raise ValueError("unknown OOD parameter")
        if self.role == "relation_only":
            if (
                set(self.families) != {"diagonal-ar"}
                or set(self.mechanisms) - set(RELATION_ONLY + ("null",))
                or min(self.channels) < 2
            ):
                raise ValueError(
                    "relation_only requires diagonal-ar covariance/precision/null and >=2 channels"
                )
        return self


class BenchmarkConfig(BaseModel):
    """Strict v13.1 schema with independent target references/calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["synthgen.v13.1"] = "synthgen.v13.1"
    master_seed: int = 20260924
    length: int = Field(default=512, ge=32)
    reference_length: int = Field(default=512, ge=32)
    calibration_realizations: int = Field(default=31, ge=1)
    replicas_per_genotype: int = Field(default=4, ge=1)
    event_fraction: tuple[float, float] = (0.15, 0.6)
    strength: tuple[float, float] = (0.25, 0.9)
    transition_fraction: tuple[float, float] = (0.0, 0.1)
    features: int = Field(default=48, ge=4)
    rank: int = Field(default=3, ge=1)
    window_lengths: tuple[int, ...] = (16, 32, 64)
    stride: int = Field(default=8, ge=1)
    alpha_grid: tuple[float, ...] = (0.1,)
    splits: dict[str, SplitSpec]

    @model_validator(mode="after")
    def validate_contracts(self) -> BenchmarkConfig:
        if not self.splits or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in self.splits
        ):
            raise ValueError("safe, nonempty split identifiers required")
        for name in ("event_fraction", "strength", "transition_fraction"):
            lo, hi = getattr(self, name)
            if not all(math.isfinite(x) for x in (lo, hi)) or not 0 <= lo <= hi <= 1:
                raise ValueError(f"invalid {name}")
        if (
            self.event_fraction[0] <= 0
            or self.event_fraction[1] > 0.8
            or self.transition_fraction[1] >= 0.5
        ):
            raise ValueError("events must leave normal context and nonempty interiors")
        if (
            not self.window_lengths
            or min(self.window_lengths) < 2
            or max(self.window_lengths) > min(self.length, self.reference_length)
        ):
            raise ValueError("window lengths must fit both reference and query")
        if not self.alpha_grid or any(
            not math.isfinite(a) or not 0 < a < 1 for a in self.alpha_grid
        ):
            raise ValueError("invalid alpha grid")
        roles = {s.role for s in self.splits.values()}
        if not {"train", "validation", "calibration"} <= roles:
            raise ValueError(
                "explicit train, validation and calibration roles required"
            )
        train = [s for s in self.splits.values() if s.role == "train"]
        train_families = set().union(*(set(s.families) for s in train))
        train_mechanisms = set().union(*(set(s.mechanisms) for s in train))
        for s in self.splits.values():
            if s.role == "iid":
                keys = (
                    "families",
                    "mechanisms",
                    "channels",
                    "length_scale",
                    "memory",
                    "noise_std",
                )
                if not any(
                    all(getattr(s, k) == getattr(t, k) for k in keys) for t in train
                ):
                    raise ValueError(
                        "IID split must use a training prior, not merely a new name"
                    )
            if s.role == "family_ood" and set(s.families) & train_families:
                raise ValueError("family-OOD leakage")
            if s.role == "mechanism_ood" and set(s.mechanisms) & train_mechanisms:
                raise ValueError("mechanism-OOD leakage")
            if s.role == "parameter_ood":
                if not s.ood_parameters:
                    raise ValueError(
                        "parameter_ood requires declared disjoint dimensions"
                    )
                for t in train:
                    for key in s.ood_parameters:
                        if key == "channels":
                            overlap = bool(set(s.channels) & set(t.channels))
                        else:
                            lo, hi = getattr(s, key)
                            a, b = getattr(t, key)
                            overlap = max(lo, a) <= min(hi, b)
                        if overlap:
                            raise ValueError(f"parameter-OOD overlap in {key}")
                if "noise_std" in s.ood_parameters and "diagonal-ar" in s.families:
                    raise ValueError(
                        "noise_std OOD is not a diagonal-ar prior parameter"
                    )
                if "memory" in s.ood_parameters and set(s.families) != {"diagonal-ar"}:
                    raise ValueError("memory OOD requires diagonal-ar")
                if "length_scale" in s.ood_parameters and "diagonal-ar" in s.families:
                    raise ValueError(
                        "length_scale OOD is not meaningful for diagonal-ar"
                    )
        return self
