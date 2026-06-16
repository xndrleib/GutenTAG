"""Persistent cache helpers for capability profile partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ..io import write_json
from .hashes import canonical_json_hash, file_sha256, table_content_hash


@dataclass(frozen=True)
class CacheFingerprint:
    """Invalidation payload for one cached profile partition."""

    dataset_hash: str
    protocol_hash: str
    code_hash: str
    profile_name: str
    partition_id: str = "all"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible fingerprint payload."""

        return {
            "dataset_hash": self.dataset_hash,
            "protocol_hash": self.protocol_hash,
            "code_hash": self.code_hash,
            "profile_name": self.profile_name,
            "partition_id": self.partition_id,
            "extra": dict(self.extra),
        }

    @property
    def digest(self) -> str:
        """Return the canonical fingerprint hash."""

        return canonical_json_hash(self.to_dict())


class CacheStore:
    """Read and write resumable capability partition cache artifacts."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        resume: bool = True,
        run_fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.resume = bool(resume)
        self.run_fingerprint = dict(run_fingerprint or {})
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fingerprint(
        self,
        *,
        profile_name: str,
        partition_id: str = "all",
        extra: Mapping[str, Any] | None = None,
    ) -> CacheFingerprint:
        """Build a partition fingerprint from the run-level fingerprint."""

        return CacheFingerprint(
            dataset_hash=str(self.run_fingerprint.get("dataset_hash", "")),
            protocol_hash=str(self.run_fingerprint.get("protocol_hash", "")),
            code_hash=str(self.run_fingerprint.get("code_hash", "")),
            profile_name=profile_name,
            partition_id=partition_id,
            extra=dict(extra or {}),
        )

    def is_valid(
        self,
        *,
        profile_name: str,
        partition_id: str,
        fingerprint: CacheFingerprint | Mapping[str, Any],
    ) -> bool:
        """Return whether a cached partition exists and matches a fingerprint."""

        metadata = self._read_metadata(profile_name, partition_id)
        if not metadata:
            return False
        return metadata.get("status") == "complete" and metadata.get(
            "fingerprint_hash"
        ) == _fingerprint_hash(fingerprint)

    def get_or_compute_table(
        self,
        *,
        profile_name: str,
        partition_id: str,
        fingerprint: CacheFingerprint | Mapping[str, Any],
        compute: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Load a cached table partition or compute and store it."""

        path = self.table_path(profile_name, partition_id)
        if (
            self.resume
            and path.exists()
            and self.is_valid(
                profile_name=profile_name,
                partition_id=partition_id,
                fingerprint=fingerprint,
            )
        ):
            return pd.read_csv(path, compression="gzip")
        frame = compute()
        self.write_table_partition(
            profile_name=profile_name,
            partition_id=partition_id,
            fingerprint=fingerprint,
            frame=frame,
        )
        return frame

    def write_table_partition(
        self,
        *,
        profile_name: str,
        partition_id: str,
        fingerprint: CacheFingerprint | Mapping[str, Any],
        frame: pd.DataFrame,
    ) -> Path:
        """Write a compressed CSV table partition and metadata."""

        path = self.table_path(profile_name, partition_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, compression="gzip")
        self._write_metadata(
            profile_name=profile_name,
            partition_id=partition_id,
            fingerprint=fingerprint,
            artifacts={
                "table": {
                    "path": path.relative_to(self.cache_dir).as_posix(),
                    "format": "csv.gz",
                    "rows": int(len(frame)),
                    "sha256": file_sha256(path),
                    "content_hash": table_content_hash(frame),
                }
            },
        )
        return path

    def get_or_compute_json(
        self,
        *,
        profile_name: str,
        partition_id: str,
        fingerprint: CacheFingerprint | Mapping[str, Any],
        compute: Callable[[], Any],
    ) -> Any:
        """Load a cached JSON partition or compute and store it."""

        path = self.json_path(profile_name, partition_id)
        if (
            self.resume
            and path.exists()
            and self.is_valid(
                profile_name=profile_name,
                partition_id=partition_id,
                fingerprint=fingerprint,
            )
        ):
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        payload = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        self._write_metadata(
            profile_name=profile_name,
            partition_id=partition_id,
            fingerprint=fingerprint,
            artifacts={
                "json": {
                    "path": path.relative_to(self.cache_dir).as_posix(),
                    "sha256": file_sha256(path),
                }
            },
        )
        return payload

    def table_path(self, profile_name: str, partition_id: str) -> Path:
        """Return the cache path for a table partition."""

        return (
            self._partition_dir(profile_name)
            / f"{_safe_partition_id(partition_id)}.csv.gz"
        )

    def json_path(self, profile_name: str, partition_id: str) -> Path:
        """Return the cache path for a JSON partition."""

        return (
            self._partition_dir(profile_name)
            / f"{_safe_partition_id(partition_id)}.json"
        )

    def metadata_path(self, profile_name: str, partition_id: str) -> Path:
        """Return the metadata path for a cached partition."""

        return (
            self._partition_dir(profile_name)
            / f"{_safe_partition_id(partition_id)}.metadata.json"
        )

    def _partition_dir(self, profile_name: str) -> Path:
        return self.cache_dir / "profile_partitions" / _safe_partition_id(profile_name)

    def _read_metadata(
        self, profile_name: str, partition_id: str
    ) -> dict[str, Any] | None:
        path = self.metadata_path(profile_name, partition_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def _write_metadata(
        self,
        *,
        profile_name: str,
        partition_id: str,
        fingerprint: CacheFingerprint | Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> None:
        payload = _fingerprint_payload(fingerprint)
        write_json(
            self.metadata_path(profile_name, partition_id),
            {
                "cache_store_version": "synthgen.capability.cache.v1",
                "profile_name": profile_name,
                "partition_id": partition_id,
                "status": "complete",
                "fingerprint": payload,
                "fingerprint_hash": canonical_json_hash(payload),
                "artifacts": dict(artifacts),
            },
        )


def _fingerprint_payload(
    fingerprint: CacheFingerprint | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(fingerprint, CacheFingerprint):
        return fingerprint.to_dict()
    return dict(fingerprint)


def _fingerprint_hash(fingerprint: CacheFingerprint | Mapping[str, Any]) -> str:
    return canonical_json_hash(_fingerprint_payload(fingerprint))


def _safe_partition_id(value: str) -> str:
    cleaned = str(value).replace("\\", "/").strip("/")
    cleaned = cleaned.replace("/", "__").replace(" ", "_")
    return cleaned or "partition"
