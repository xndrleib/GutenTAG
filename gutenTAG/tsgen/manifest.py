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
