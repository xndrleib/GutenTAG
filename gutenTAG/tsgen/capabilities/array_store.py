"""Cached array access for generated time-series instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from ..io import write_json
from .dataset import InstanceRecord, read_timeseries_csv
from .hashes import file_sha256

ArrayKind = Literal["clean", "anomalous"]


@dataclass(frozen=True)
class ArrayKey:
    """Stable key for one generated time-series array."""

    variant_id: str
    split: str
    instance_id: str
    kind: ArrayKind


class ArrayStore:
    """Read generated CSV arrays once and serve cached arrays afterwards."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        mmap: bool = True,
        lazy_materialize: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.mmap = bool(mmap)
        self.lazy_materialize = bool(lazy_materialize)
        self._memory_cache: dict[ArrayKey, np.ndarray] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def materialize(self, instances: list[InstanceRecord] | tuple[InstanceRecord, ...]) -> None:
        """Convert all instance CSV files into cached ``.npy`` arrays."""

        for instance in instances:
            self._materialize_one(instance, "clean")
            self._materialize_one(instance, "anomalous")

    def get(self, instance: InstanceRecord, kind: ArrayKind) -> np.ndarray:
        """Return an instance array from cache, memmap, or source CSV."""

        key = self._key(instance, kind)
        if self.cache_dir is None:
            if key not in self._memory_cache:
                self._memory_cache[key] = read_timeseries_csv(self._source_path(instance, kind))
            return self._memory_cache[key]

        array_path = self._array_path(key)
        if self._cache_is_valid(instance, kind):
            mmap_mode = "r" if self.mmap else None
            return np.load(array_path, mmap_mode=mmap_mode)

        if not self.lazy_materialize:
            raise FileNotFoundError(f"Cached array is missing or stale: {array_path}")
        return self._materialize_one(instance, kind)

    def _materialize_one(self, instance: InstanceRecord, kind: ArrayKind) -> np.ndarray:
        source_path = self._source_path(instance, kind)
        values = read_timeseries_csv(source_path)
        if self.cache_dir is None:
            self._memory_cache[self._key(instance, kind)] = values
            return values

        key = self._key(instance, kind)
        array_path = self._array_path(key)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_path, values)
        write_json(self._metadata_path(key), self._metadata_payload(source_path, values))
        mmap_mode = "r" if self.mmap else None
        return np.load(array_path, mmap_mode=mmap_mode)

    def _cache_is_valid(self, instance: InstanceRecord, kind: ArrayKind) -> bool:
        key = self._key(instance, kind)
        array_path = self._array_path(key)
        metadata_path = self._metadata_path(key)
        if not array_path.exists() or not metadata_path.exists():
            return False
        try:
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return metadata.get("source") == self._source_signature(self._source_path(instance, kind))

    def _array_path(self, key: ArrayKey) -> Path:
        if self.cache_dir is None:
            raise ValueError("ArrayStore has no cache directory")
        return self.cache_dir / "arrays" / _safe_component(key.variant_id) / key.split / key.instance_id / f"{key.kind}.npy"

    def _metadata_path(self, key: ArrayKey) -> Path:
        return self._array_path(key).with_suffix(".json")

    @staticmethod
    def _key(instance: InstanceRecord, kind: ArrayKind) -> ArrayKey:
        return ArrayKey(
            variant_id=instance.variant_id,
            split=instance.split,
            instance_id=instance.instance_id,
            kind=kind,
        )

    @staticmethod
    def _source_path(instance: InstanceRecord, kind: ArrayKind) -> Path:
        if kind == "clean":
            return instance.clean_path
        return instance.anomalous_path

    @staticmethod
    def _metadata_payload(source_path: Path, values: np.ndarray) -> dict[str, object]:
        return {
            "array_store_version": "synthgen.array_store.v2",
            "source": ArrayStore._source_signature(source_path),
            "dtype": str(values.dtype),
            "shape": list(values.shape),
        }

    @staticmethod
    def _source_signature(source_path: Path) -> Mapping[str, object]:
        stat = source_path.stat()
        return {
            "path": str(source_path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": file_sha256(source_path),
        }


def _safe_component(value: str) -> str:
    return str(value).replace("/", "__").replace("\\", "__")
