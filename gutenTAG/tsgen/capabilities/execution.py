"""Execution context and profile selection for capability runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .array_store import ArrayStore
from .cache import CacheStore
from .dataset import DatasetIndex
from .hashes import code_version_hash, dataset_content_hash, protocol_hash
from .output import OutputWriter
from .protocol import CapabilityProtocol
from .windows import WindowLibrary


SUPPORTED_PROFILES: tuple[str, ...] = (
    "observability",
    "detectability",
    "corrected_detectability",
    "implementation_validity",
    "model_zoo",
    "law_observability",
    "identifiability",
    "describability",
    "annotation_alignment",
    "admission",
    "visual_audit",
)

PROFILE_PRESETS: dict[str, tuple[str, ...]] = {
    "smoke": ("observability",),
    "implementation_validity": (
        "observability",
        "detectability",
        "corrected_detectability",
        "implementation_validity",
    ),
    "calibration": ("detectability", "corrected_detectability"),
    "model_zoo": ("model_zoo",),
    "law_observability": ("law_observability",),
    "diagnosis": ("identifiability",),
    "repair": ("describability",),
    "maturity": ("identifiability", "describability"),
    "annotation": ("annotation_alignment",),
    "annotation_channels": ("annotation_alignment",),
    "admission": ("admission",),
    "visual_audit": ("visual_audit",),
    "full_certificate": SUPPORTED_PROFILES,
    "all": SUPPORTED_PROFILES,
}


@dataclass
class CapabilityRunContext:
    """Shared state used by capability profile computations."""

    dataset: DatasetIndex
    protocol: CapabilityProtocol
    arrays: ArrayStore
    cache: CacheStore
    windows: WindowLibrary
    output: OutputWriter
    rng: np.random.Generator
    n_jobs: int
    profile_names: tuple[str, ...]
    cache_dir: Path | None
    resume: bool
    run_manifest: dict[str, object]


def build_run_context(
    *,
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    output_dir: Path,
    profiles: Sequence[str] | None = None,
    profile_preset: str | None = None,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    materialize_arrays: bool = False,
    resume: bool = True,
    output_format: str = "csv",
    release_csv: bool = True,
    allow_parquet_fallback: bool = False,
    label_export: Literal["full", "diagnostics"] = "full",
    admission_policy_hash: str | None = None,
) -> CapabilityRunContext:
    """Create the shared execution context for a capability run."""

    resolved_profiles = resolve_profile_names(
        profiles=profiles,
        profile_preset=profile_preset,
    )
    active_cache_dir = cache_dir if cache_dir is not None else output_dir.parent / "cache"
    run_manifest = _build_run_manifest(
        dataset=dataset,
        protocol=protocol,
        profile_names=resolved_profiles,
        cache_dir=active_cache_dir,
        admission_policy_hash=admission_policy_hash,
        output_format=output_format,
        release_csv=release_csv,
        allow_parquet_fallback=allow_parquet_fallback,
        label_export=label_export,
    )
    cache = CacheStore(
        cache_dir=active_cache_dir,
        resume=resume,
        run_fingerprint=run_manifest["fingerprints"],
    )
    arrays = ArrayStore(cache_dir=active_cache_dir, mmap=True, lazy_materialize=True)
    if materialize_arrays:
        arrays.materialize(dataset.instances)
    return CapabilityRunContext(
        dataset=dataset,
        protocol=protocol,
        arrays=arrays,
        cache=cache,
        windows=WindowLibrary(),
        output=OutputWriter(
            output_dir=output_dir,
            output_format=output_format,
            release_csv=release_csv,
            allow_parquet_fallback=allow_parquet_fallback,
        ),
        rng=np.random.default_rng(protocol.random_seed),
        n_jobs=max(1, int(n_jobs)),
        profile_names=resolved_profiles,
        cache_dir=active_cache_dir,
        resume=bool(resume),
        run_manifest=run_manifest,
    )


def resolve_profile_names(
    *,
    profiles: Sequence[str] | None = None,
    profile_preset: str | None = None,
) -> tuple[str, ...]:
    """Normalize requested profile names and add required dependencies."""

    if profiles:
        requested = _expand_profiles(profiles)
    elif profile_preset:
        if profile_preset not in PROFILE_PRESETS:
            raise ValueError(f"Unknown profile preset: {profile_preset}")
        requested = list(PROFILE_PRESETS[profile_preset])
    else:
        requested = list(SUPPORTED_PROFILES)

    unknown = sorted(set(requested) - set(SUPPORTED_PROFILES))
    if unknown:
        raise ValueError(f"Unknown capability profiles: {unknown}")
    if "visual_audit" in requested and "admission" not in requested:
        requested.insert(0, "admission")
    if "admission" in requested:
        for dependency in (
            "annotation_alignment",
            "describability",
            "identifiability",
            "implementation_validity",
        ):
            if dependency not in requested:
                requested.insert(0, dependency)
    if "implementation_validity" in requested:
        requested.insert(0, "corrected_detectability")
        requested.insert(0, "detectability")
        requested.insert(0, "observability")
    if "identifiability" in requested or "describability" in requested:
        requested.insert(0, "observability")
    deduped = []
    for profile in requested:
        if profile not in deduped:
            deduped.append(profile)
    return tuple(deduped)


def _expand_profiles(profiles: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for profile in profiles:
        for part in str(profile).split(","):
            name = part.strip()
            if not name:
                continue
            if name == "all":
                expanded.extend(SUPPORTED_PROFILES)
            elif name in PROFILE_PRESETS:
                expanded.extend(PROFILE_PRESETS[name])
            else:
                expanded.append(name)
    return expanded


def _build_run_manifest(
    *,
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    profile_names: Sequence[str],
    cache_dir: Path,
    admission_policy_hash: str | None = None,
    output_format: str = "csv",
    release_csv: bool = True,
    allow_parquet_fallback: bool = False,
    label_export: Literal["full", "diagnostics"] = "full",
) -> dict[str, object]:
    code_root = Path(__file__).resolve().parent
    dataset_hash = dataset_content_hash(dataset.root)
    protocol_digest = protocol_hash(protocol)
    code_digest = code_version_hash((code_root,))
    fingerprints = {
        "dataset_hash": dataset_hash,
        "protocol_hash": protocol_digest,
        "code_hash": code_digest,
    }
    hash_inputs: dict[str, object] = {
        "dataset_hash_scope": "generated_dataset_artifacts_excluding_analysis",
        "code_hash_scope": str(code_root),
    }
    if admission_policy_hash is not None:
        fingerprints["admission_policy_hash"] = admission_policy_hash
        hash_inputs["admission_policy_hash_scope"] = "admission policy payload"
    return {
        "profile_run_manifest_version": "synthgen.capability.profile_run.v1",
        "dataset_root": str(dataset.root),
        "cache_dir": str(cache_dir),
        "profiles": list(profile_names),
        "output": {
            "output_format": output_format,
            "release_csv": bool(release_csv),
            "allow_parquet_fallback": bool(allow_parquet_fallback),
            "label_export": label_export,
        },
        "fingerprints": fingerprints,
        "hash_inputs": hash_inputs,
    }
