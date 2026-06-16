"""Deterministic seed derivation helpers for TS dataset generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class InstanceSpec:
    """Runtime identity for one split-local generated instance."""

    role: str
    index: int
    name: str
    seed_index: int


def derive_seed(seed: int, *parts: str) -> int:
    """Derive a stable 32-bit seed from a root seed and string parts.

    Parameters
    ----------
    seed : int
        Root seed.
    *parts : str
        Stable namespace parts.

    Returns
    -------
    int
        Derived seed in NumPy-compatible 32-bit range.
    """
    payload = f"{seed}|" + "|".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def split_seed_index(*, role: str, index: int, paired_count: int) -> int:
    """Return the stable seed index for paired or clean-only instances.

    Parameters
    ----------
    role : str
        Instance role, usually ``"paired"`` or ``"clean_only"``.
    index : int
        Role-local instance index.
    paired_count : int
        Number of paired instances preceding clean-only instances.

    Returns
    -------
    int
        Split-stable seed index.
    """
    if role == "clean_only":
        return int(paired_count) + int(index)
    return int(index)


def resolve_split_instance_specs(
    split_counts: Mapping[str, Any] | None,
    *,
    default_paired_instances: int,
) -> list[InstanceSpec]:
    """Resolve split-local instance identities and seed indices.

    Parameters
    ----------
    split_counts : Mapping[str, Any] or None
        Split generation counts. Missing paired counts fall back to
        ``default_paired_instances``; missing clean-only counts fall back to
        zero.
    default_paired_instances : int
        Legacy paired-instance count used when no explicit split count exists.

    Returns
    -------
    list[InstanceSpec]
        Ordered paired and clean-only instance specs. Clean-only seed indices
        are offset after paired instances to preserve split-stable seeding.
    """
    counts = dict(split_counts or {})
    paired_count = int(
        counts.get("paired_instances_per_variant", default_paired_instances)
    )
    clean_only_count = int(counts.get("clean_only_instances_per_variant", 0))
    paired = [
        InstanceSpec(
            role="paired",
            index=index,
            name=f"instance_{index:03d}",
            seed_index=split_seed_index(
                role="paired",
                index=index,
                paired_count=paired_count,
            ),
        )
        for index in range(paired_count)
    ]
    clean_only = [
        InstanceSpec(
            role="clean_only",
            index=index,
            name=f"clean_only_{index:03d}",
            seed_index=split_seed_index(
                role="clean_only",
                index=index,
                paired_count=paired_count,
            ),
        )
        for index in range(clean_only_count)
    ]
    return paired + clean_only


def derive_instance_seeds(
    *,
    master_seed: int,
    variant_id: str,
    split: str,
    instance_index: int,
) -> dict[str, int]:
    """Derive all runtime seeds for one generated instance.

    Parameters
    ----------
    master_seed : int
        Dataset-level seed.
    variant_id : str
        Stable variant identifier.
    split : str
        Dataset split name.
    instance_index : int
        Split-stable instance index.

    Returns
    -------
    dict[str, int]
        Named deterministic seeds used by generation substeps.
    """
    instance_label = f"instance_{instance_index:03d}"
    instance_seed = derive_seed(master_seed, variant_id, split, instance_label)
    split_stable_instance_seed = derive_seed(master_seed, variant_id, instance_label)
    return {
        "instance_seed": instance_seed,
        "split_stable_instance_seed": split_stable_instance_seed,
        "base_seed": derive_seed(instance_seed, "base"),
        "base_shared_noise_seed": derive_seed(instance_seed, "base-shared-noise"),
        "base_params_seed": derive_seed(
            split_stable_instance_seed, "base-parameter-sampling"
        ),
        "base_channel_params_seed": derive_seed(
            split_stable_instance_seed, "base-channel-parameter-sampling"
        ),
        "plan_seed": derive_seed(instance_seed, "segment-plan"),
        "anomaly_seed": derive_seed(instance_seed, "anomaly-transform"),
        "params_seed": derive_seed(instance_seed, "parameter-sampling"),
        "zoom_seed": derive_seed(instance_seed, "zoom-selection"),
    }
