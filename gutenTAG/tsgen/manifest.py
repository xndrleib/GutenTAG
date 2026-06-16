"""Manifest, provenance, and artifact hashing helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .io.strict_json import sanitize_json_value

DEFAULT_DEPENDENCIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
    "pydantic",
    "neurokit2",
)


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    """Return a path-independent SHA-256 hash for a JSON-compatible payload."""
    encoded = json.dumps(
        sanitize_json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def hash_generated_csv(root: Path) -> dict[str, Any]:
    """Hash generated CSV files under a dataset root.

    The returned digest is path-independent: relative paths and file bytes are
    both included in sorted order.
    """
    files = sorted(
        path
        for path in root.rglob("*.csv")
        if path.is_file() and _is_generated_csv(path)
    )
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        file_hash = _file_sha256(path)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        entries.append({"path": rel, "sha256": file_hash})
    return {
        "algorithm": "sha256",
        "scope": "generated_csv",
        "digest": "sha256:" + digest.hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def build_environment_metadata(
    dependencies: Iterable[str] = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    """Collect release-relevant Python environment metadata."""
    versions: dict[str, Optional[str]] = {}
    for package in dependencies:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": versions,
    }


def build_dataset_manifest(
    *,
    output_root: Path,
    repo_root: Path,
    dataset_version: str,
    library_version: str,
    config: Mapping[str, Any],
    generated_variant_entries: Iterable[Mapping[str, Any]],
    skipped_variants: Iterable[Mapping[str, Any]],
    disabled_anomaly_types: Iterable[str],
    aggregated_statistics: Mapping[str, Any],
    derived_seeds: Mapping[str, Any],
    metadata_registry: Mapping[str, Any],
    generator_sidecars: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the dataset-level manifest for generated TS artifacts."""

    config_dict = sanitize_json_value(config)
    variant_entries = [dict(entry) for entry in generated_variant_entries]
    sidecars = dict(generator_sidecars)
    manifest = {
        "dataset_schema_version": "synthgen.dataset.v1",
        "dataset_version": dataset_version,
        "generator": {
            "library_version": library_version,
            "git": resolve_git_metadata(repo_root),
        },
        "environment": build_environment_metadata(),
        "config": config_dict,
        "normalized_config_hash": canonical_json_hash(config_dict),
        "generated_variants": [entry["variant_id"] for entry in variant_entries],
        "variant_manifests": variant_entries,
        "skipped_variants": list(skipped_variants),
        "disabled_anomaly_types": list(disabled_anomaly_types),
        "aggregated_statistics": aggregated_statistics,
        "derived_seeds": derived_seeds,
        "metadata_registry": metadata_registry,
        "generator_sidecars": sidecars,
        "annotation_channels": sidecars["annotation_channels"],
        "law_level_replicates": sidecars["law_level_replicates"],
        "label_semantics": _label_semantics_manifest_section(),
        "onset_metadata": _onset_metadata_manifest_section(),
        "artifacts": _artifact_manifest_section(output_root),
    }
    return sanitize_json_value(manifest)


def _label_semantics_manifest_section() -> dict[str, Any]:
    return {
        "version": "label_semantics.v2",
        "labels_any": "event interval union across all events",
        "labels_intervention": (
            "operator target or perturbed channels; not necessarily the final "
            "benchmark target for relation-only anomalies"
        ),
        "labels_context": (
            "channels needed to interpret event-level or relation-level anomalies"
        ),
        "events": "event-level truth with group, operator-target, and context roles",
    }


def _onset_metadata_manifest_section() -> dict[str, Any]:
    return {
        "version": "onset_metadata.v1",
        "fields": {
            "requested_pre_context": (
                "Requested number of observations before the planned anomaly "
                "source starts; this is the experimental bucket value."
            ),
            "actual_source_start": (
                "Actual planned source start after any alignment strategy has "
                "been applied."
            ),
            "actual_support_start": (
                "First effective labeled support index after applying the "
                "anomaly operator and support-label policy."
            ),
            "onset_bucket": "Split bucket name that supplied requested_pre_context.",
            "alignment_strategy": (
                "Planner strategy used to place the first anomaly source, for "
                "example exact or mode_grid."
            ),
            "alignment_error": (
                "actual_source_start minus requested_pre_context; zero for exact "
                "placements."
            ),
        },
    }


def _artifact_manifest_section(output_root: Path) -> dict[str, Any]:
    return {
        "data_content_hash": hash_generated_csv(output_root),
    }


def resolve_git_metadata(repo_root: Path) -> dict[str, Any]:
    """Resolve git commit and dirty state for a repository."""
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo_root, "status", "--porcelain")),
        "branch": _git(repo_root, "branch", "--show-current") or None,
    }


def _is_generated_csv(path: Path) -> bool:
    return path.name in {
        "clean.csv",
        "anomalous.csv",
        "labels_pointwise.csv",
        "labels_any.csv",
        "labels_intervention.csv",
        "labels_context.csv",
        "labels_oracle_any.csv",
        "labels_oracle_intervention.csv",
        "labels_oracle_context.csv",
        "labels_event_only.csv",
        "labels_delayed.csv",
        "labels_weak_point.csv",
        "labels_visible_only.csv",
        "labels_noisy_boundary.csv",
        "labels_censored.csv",
        "law_level_replicates.csv",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
