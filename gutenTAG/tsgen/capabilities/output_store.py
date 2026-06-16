"""Output storage for capability tables and release manifests."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..io import write_json

CsvCompression = Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"]


class ParquetOutputError(RuntimeError):
    """Raised when parquet output is requested but cannot be written."""


@dataclass(frozen=True)
class OutputTableSpec:
    """Describe a capability output table.

    Parameters
    ----------
    name:
        Stable logical table name.
    schema_version:
        Version of the table schema.
    primary_key:
        Columns that uniquely identify logical rows when known.
    partition_by:
        Columns used for future partitioned execution.
    release_csv:
        Whether this table should have a CSV release/export view.
    internal_format:
        Preferred canonical internal storage format.
    """

    name: str
    schema_version: str = "synthgen.capability.table.v1"
    primary_key: tuple[str, ...] = ()
    partition_by: tuple[str, ...] = ()
    release_csv: bool = True
    internal_format: Literal["parquet", "feather", "csv"] = "parquet"


@dataclass(frozen=True)
class OutputTableManifest:
    """Manifest entry for a finalized capability table."""

    name: str
    schema_version: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    partition_count: int
    content_hash: str
    canonical_format: str
    canonical_path: str
    file_hashes: dict[str, str]
    primary_key: tuple[str, ...] = ()
    partition_by: tuple[str, ...] = ()
    release_csv_path: str | None = None
    legacy_csv_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest record."""

        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "rows": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "primary_key": list(self.primary_key),
            "partition_by": list(self.partition_by),
            "partition_count": self.partition_count,
            "content_hash": self.content_hash,
            "canonical": {
                "format": self.canonical_format,
                "path": self.canonical_path,
                "sha256": self.file_hashes[self.canonical_path],
            },
            "release_csv": (
                {
                    "path": self.release_csv_path,
                    "sha256": self.file_hashes[self.release_csv_path],
                }
                if self.release_csv_path is not None
                else None
            ),
            "legacy_csv": (
                {
                    "path": self.legacy_csv_path,
                    "sha256": self.file_hashes[self.legacy_csv_path],
                }
                if self.legacy_csv_path is not None
                else None
            ),
            "file_hashes": dict(self.file_hashes),
            # Backward-compatible fields used by the first output manifest pass.
            "format": self.canonical_format,
            "path": self.canonical_path,
            "sha256": self.file_hashes[self.canonical_path],
        }


@dataclass
class OutputStore:
    """Write capability tables to canonical storage and release views."""

    output_dir: Path
    output_format: Literal["csv", "parquet", "both"] = "csv"
    release_csv: bool = True
    legacy_csv: bool = True
    allow_parquet_fallback: bool = False
    table_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    _tables_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    _partition_records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_format not in {"csv", "parquet", "both"}:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        if (
            self.output_format in {"parquet", "both"}
            and not self.allow_parquet_fallback
        ):
            _require_parquet_engine()

    def write_table(
        self,
        name: str,
        filename: str,
        frame: pd.DataFrame,
        spec: OutputTableSpec | None = None,
    ) -> OutputTableManifest:
        """Write a finalized table and return its manifest entry.

        Parameters
        ----------
        name:
            Stable logical table name.
        filename:
            Release CSV filename, relative to the analysis output directory.
        frame:
            Table rows.
        spec:
            Optional table specification. A default spec is created when absent.

        Returns
        -------
        OutputTableManifest
            Manifest entry for the table.
        """

        table_spec = spec or OutputTableSpec(name=name)
        canonical_path, canonical_format = self._write_canonical_table(
            filename, frame, table_spec
        )
        file_hashes = {
            self._relative_path(canonical_path): _file_sha256(canonical_path)
        }
        release_csv_path, legacy_csv_path = self._write_release_csv_views(
            filename,
            frame,
            table_spec,
            canonical_path,
            file_hashes,
        )
        manifest = self._build_table_manifest(
            table_spec=table_spec,
            frame=frame,
            canonical_format=canonical_format,
            canonical_path=canonical_path,
            file_hashes=file_hashes,
            release_csv_path=release_csv_path,
            legacy_csv_path=legacy_csv_path,
        )
        record = manifest.to_dict()
        self.table_records.append(record)
        self._tables_by_name[name] = record
        return manifest

    def _write_release_csv_views(
        self,
        filename: str,
        frame: pd.DataFrame,
        table_spec: OutputTableSpec,
        canonical_path: Path,
        file_hashes: dict[str, str],
    ) -> tuple[Path | None, Path | None]:
        release_csv_path: Path | None = None
        legacy_csv_path: Path | None = None
        if self.release_csv and table_spec.release_csv:
            release_csv_path = self.output_dir / "tables_csv" / filename
            if release_csv_path != canonical_path:
                _write_csv(release_csv_path, frame)
            file_hashes[self._relative_path(release_csv_path)] = _file_sha256(
                release_csv_path
            )
        if self.legacy_csv and self.release_csv and table_spec.release_csv:
            legacy_csv_path = self.output_dir / filename
            if (
                legacy_csv_path != release_csv_path
                and legacy_csv_path != canonical_path
            ):
                _write_csv(legacy_csv_path, frame)
            file_hashes[self._relative_path(legacy_csv_path)] = _file_sha256(
                legacy_csv_path
            )
        return release_csv_path, legacy_csv_path

    def _build_table_manifest(
        self,
        *,
        table_spec: OutputTableSpec,
        frame: pd.DataFrame,
        canonical_format: str,
        canonical_path: Path,
        file_hashes: dict[str, str],
        release_csv_path: Path | None,
        legacy_csv_path: Path | None,
    ) -> OutputTableManifest:
        return OutputTableManifest(
            name=table_spec.name,
            schema_version=table_spec.schema_version,
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            columns=tuple(map(str, frame.columns)),
            partition_count=1,
            content_hash=_frame_content_hash(frame),
            canonical_format=canonical_format,
            canonical_path=self._relative_path(canonical_path),
            file_hashes=file_hashes,
            primary_key=table_spec.primary_key,
            partition_by=table_spec.partition_by,
            release_csv_path=(
                self._relative_path(release_csv_path)
                if release_csv_path is not None
                else None
            ),
            legacy_csv_path=(
                self._relative_path(legacy_csv_path)
                if legacy_csv_path is not None
                else None
            ),
        )

    def write_partition(
        self,
        table: OutputTableSpec,
        partition_id: str,
        frame: pd.DataFrame,
    ) -> None:
        """Write an intermediate table partition for future parallel profiles."""

        partition_name = _safe_partition_id(partition_id)
        partition_dir = self.output_dir / "cache" / "profile_partitions" / table.name
        if self._prefers_parquet(table):
            path = partition_dir / f"{partition_name}.parquet"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                file_format = "parquet"
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                if not self.allow_parquet_fallback:
                    raise ParquetOutputError(
                        f"Failed to write parquet partition for table {table.name!r}: {exc}"
                    ) from exc
                self._record_warning(table.name, "parquet_fallback", str(exc))
                path = partition_dir / f"{partition_name}.csv.gz"
                _write_csv(path, frame, compression="gzip")
                file_format = "csv.gz"
        else:
            path = partition_dir / f"{partition_name}.csv"
            _write_csv(path, frame)
            file_format = "csv"
        self._partition_records.setdefault(table.name, []).append(
            {
                "partition_id": partition_name,
                "format": file_format,
                "path": self._relative_path(path),
                "rows": int(len(frame)),
                "sha256": _file_sha256(path),
            }
        )

    def finalize_table(
        self,
        table: OutputTableSpec,
        filename: str | None = None,
    ) -> OutputTableManifest:
        """Merge recorded partitions and write a finalized table."""

        records = self._partition_records.get(table.name, [])
        if not records:
            raise ValueError(
                f"No partitions have been written for table {table.name!r}"
            )
        frames = [
            self._read_path(self.output_dir / record["path"], record["format"])
            for record in records
        ]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.write_table(
            table.name, filename or f"{table.name}.csv", merged, table
        )

    def read_table(self, name: str) -> pd.DataFrame:
        """Read a table previously written by this store."""

        record = self._tables_by_name.get(name)
        if record is None:
            raise KeyError(f"Unknown output table: {name}")
        canonical = record["canonical"]
        return self._read_path(self.output_dir / canonical["path"], canonical["format"])

    def write_manifests(self) -> None:
        """Write output and hash manifests for all finalized tables."""

        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        file_hashes: dict[str, str] = {}
        table_hashes: dict[str, str] = {}
        for record in self.table_records:
            file_hashes.update(record["file_hashes"])
            table_hashes[record["name"]] = record["content_hash"]
        write_json(
            manifest_dir / "output_manifest.json",
            {
                "output_manifest_version": "synthgen.capability.output.v2",
                "layout_version": "synthgen.capability.layout.v12",
                "layout": {
                    "canonical_parquet_dir": "tables_parquet",
                    "release_csv_dir": "tables_csv",
                    "legacy_csv_dir": ".",
                    "partition_cache_dir": "cache/profile_partitions",
                },
                "requested_output_format": self.output_format,
                "release_csv": bool(self.release_csv),
                "legacy_csv": bool(self.legacy_csv),
                "allow_parquet_fallback": bool(self.allow_parquet_fallback),
                "warnings": list(self.warnings),
                "tables": self.table_records,
                "partitioned_intermediates": self._partition_records,
            },
        )
        write_json(
            manifest_dir / "table_hashes.json",
            {
                "table_hashes_version": "synthgen.capability.table_hashes.v2",
                "file_hashes": file_hashes,
                "table_content_hashes": table_hashes,
            },
        )

    def _write_canonical_table(
        self,
        filename: str,
        frame: pd.DataFrame,
        table: OutputTableSpec,
    ) -> tuple[Path, str]:
        if self._prefers_parquet(table):
            parquet_path = (
                self.output_dir
                / "tables_parquet"
                / filename.replace(".csv", ".parquet")
            )
            try:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(parquet_path, index=False)
                return parquet_path, "parquet"
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                if not self.allow_parquet_fallback:
                    raise ParquetOutputError(
                        f"Failed to write parquet table {table.name!r}: {exc}"
                    ) from exc
                self._record_warning(table.name, "parquet_fallback", str(exc))
                if parquet_path.exists():
                    parquet_path.unlink()
                fallback_path = self.output_dir / "tables_csv" / f"{filename}.gz"
                _write_csv(fallback_path, frame, compression="gzip")
                return fallback_path, "csv.gz"

        csv_path = self.output_dir / "tables_csv" / filename
        _write_csv(csv_path, frame)
        return csv_path, "csv"

    def _prefers_parquet(self, table: OutputTableSpec) -> bool:
        return (
            self.output_format in {"parquet", "both"}
            and table.internal_format == "parquet"
        )

    def _read_path(self, path: Path, file_format: str) -> pd.DataFrame:
        if file_format == "parquet":
            return pd.read_parquet(path)
        if file_format == "csv.gz":
            return pd.read_csv(path, compression="gzip")
        if file_format == "csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {file_format}")

    def _relative_path(self, path: Path | None) -> str:
        if path is None:
            raise ValueError("Cannot relativize an empty path")
        return path.relative_to(self.output_dir).as_posix()

    def _record_warning(self, table_name: str, code: str, message: str) -> None:
        self.warnings.append(
            {
                "table": table_name,
                "code": code,
                "message": message,
            }
        )


def _write_csv(
    path: Path, frame: pd.DataFrame, compression: CsvCompression | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression is None:
        frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False, compression=compression)


def _has_parquet_engine() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def _require_parquet_engine() -> None:
    if _has_parquet_engine():
        return
    raise ParquetOutputError(
        "Parquet output was requested, but neither pyarrow nor fastparquet is installed. "
        "Install one of them, use --output-format csv, or pass --allow-parquet-fallback "
        "for an explicit CSV/GZip fallback run."
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _frame_content_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(b"\n")
    digest.update(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _safe_partition_id(partition_id: str) -> str:
    cleaned = str(partition_id).replace("\\", "/").strip("/")
    cleaned = cleaned.replace("/", "__").replace(" ", "_")
    return cleaned or "partition"
