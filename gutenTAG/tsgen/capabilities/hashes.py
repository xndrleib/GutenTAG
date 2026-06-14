"""Hash helpers for capability cache invalidation and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..io import sanitize_json_value


HASHED_CODE_SUFFIXES: tuple[str, ...] = (".py", ".yaml", ".yml", ".json")
GENERATED_DATASET_SUFFIXES: tuple[str, ...] = (".csv", ".json", ".jsonl", ".yaml", ".yml")


def canonical_json_hash(obj: object) -> str:
    """Return a deterministic SHA-256 hash for a JSON-compatible object."""

    encoded = json.dumps(
        sanitize_json_value(obj),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def table_content_hash(
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
) -> str:
    """Return a deterministic logical-content hash for a pandas table."""

    active = frame.copy()
    if sort_by:
        missing = [column for column in sort_by if column not in active.columns]
        if missing:
            raise ValueError(f"Cannot sort table by missing columns: {missing}")
        active = active.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(",".join(map(str, active.columns)).encode("utf-8"))
    digest.update(b"\n")
    digest.update(active.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def protocol_hash(protocol: object) -> str:
    """Hash a capability protocol or protocol-like mapping."""

    if hasattr(protocol, "to_dict"):
        payload = protocol.to_dict()
    elif isinstance(protocol, Mapping):
        payload = dict(protocol)
    else:
        payload = protocol
    return canonical_json_hash(payload)


def code_version_hash(paths: Iterable[Path]) -> str:
    """Hash code/config files under the supplied paths."""

    files = _collect_files(paths, suffixes=HASHED_CODE_SUFFIXES)
    return _hash_file_set(files)


def dataset_content_hash(root: Path) -> str:
    """Hash generated dataset content relevant to capability profiles."""

    dataset_root = Path(root)
    files = _collect_files((dataset_root,), suffixes=GENERATED_DATASET_SUFFIXES)
    files = [
        path
        for path in files
        if _is_dataset_artifact(path)
    ]
    return _hash_file_set(files, root=dataset_root)


def cache_fingerprint_hash(
    *,
    dataset_hash: str,
    protocol_hash_value: str,
    code_hash: str,
    profile_name: str,
    partition_id: str = "all",
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Hash the standard cache invalidation tuple for one partition."""

    return canonical_json_hash(
        {
            "dataset_hash": dataset_hash,
            "protocol_hash": protocol_hash_value,
            "code_hash": code_hash,
            "profile_name": profile_name,
            "partition_id": partition_id,
            "extra": dict(extra or {}),
        }
    )


def _collect_files(
    paths: Iterable[Path],
    *,
    suffixes: Sequence[str],
) -> list[Path]:
    files: list[Path] = []
    suffix_set = set(suffixes)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix in suffix_set:
                files.append(path)
            continue
        files.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix in suffix_set
            and "__pycache__" not in candidate.parts
            and ".pytest_cache" not in candidate.parts
        )
    return sorted(files, key=lambda item: item.resolve().as_posix())


def _hash_file_set(files: Sequence[Path], root: Path | None = None) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = _relative_name(path, root)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _relative_name(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_posix()


def _is_dataset_artifact(path: Path) -> bool:
    if "analysis" in path.parts or ".cache" in path.parts:
        return False
    name = path.name
    if name in {
        "clean.csv",
        "anomalous.csv",
        "events.json",
        "instance_summary.json",
        "dataset_manifest.json",
    }:
        return True
    if name.startswith("labels_") and path.suffix == ".csv":
        return True
    if "metadata" in path.parts and path.suffix in {".json", ".jsonl"}:
        return True
    return False
