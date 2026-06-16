"""Mutable run state and telemetry for capability analysis."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

try:
    import resource
except ImportError:  # pragma: no cover - Unix-only optional telemetry.
    resource = None  # type: ignore[assignment]

import pandas as pd


def initial_runtime_record() -> dict[str, object]:
    """Return the initial runtime manifest payload."""

    return {
        "runtime_manifest_version": "synthgen.capability.runtime.v1",
        "started_at_unix": time.time(),
        "profiles": [],
        "current_rss_bytes": current_rss_bytes(),
        "peak_rss_bytes": peak_rss_bytes(),
    }


@dataclass
class CapabilityRunState:
    """Track tables, artifacts, and runtime telemetry during one analysis run."""

    output_dir: Path
    cache_dir: Path | None
    output: Any
    output_filenames: Mapping[str, str]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    extra_artifacts: dict[str, str] = field(default_factory=dict)
    written_tables: set[str] = field(default_factory=set)
    runtime: dict[str, object] = field(default_factory=initial_runtime_record)

    def attach_to_manifest(self, run_manifest: dict[str, Any]) -> None:
        """Attach this state's runtime record to the run manifest."""

        run_manifest["runtime"] = self.runtime

    def publish_tables(self, keys: Sequence[str]) -> None:
        """Write known, not-yet-written tables for the requested logical keys."""

        for key in keys:
            if key in self.written_tables or key not in self.tables:
                continue
            self.output.write_table(key, self.output_filenames[key], self.tables[key])
            self.written_tables.add(key)

    @contextmanager
    def profile_stage(self, profile_name: str, **details: object) -> Iterator[None]:
        """Record runtime telemetry around one capability profile stage."""

        before_tables = set(self.written_tables)
        before_perf = time.perf_counter()
        before_wall = time.time()
        status = "complete"
        error: dict[str, str] | None = None
        try:
            yield
        except Exception as exc:
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            elapsed = max(0.0, time.perf_counter() - before_perf)
            written_delta = sorted(self.written_tables - before_tables)
            record: dict[str, object] = {
                "profile": profile_name,
                "status": status,
                "started_at_unix": before_wall,
                "finished_at_unix": time.time(),
                "elapsed_seconds": elapsed,
                "written_tables": written_delta,
                "table_count_delta": len(written_delta),
                "table_rows": {
                    key: int(len(self.tables[key]))
                    for key in written_delta
                    if key in self.tables
                },
                "output_dir_size_bytes": directory_size_bytes(self.output_dir),
                "cache_dir_size_bytes": directory_size_bytes(self.cache_dir),
                "current_rss_bytes": current_rss_bytes(),
                "peak_rss_bytes": peak_rss_bytes(),
            }
            if details:
                record["details"] = dict(details)
            if error is not None:
                record["error"] = error
            profiles = self.runtime.get("profiles")
            if not isinstance(profiles, list):
                profiles = []
                self.runtime["profiles"] = profiles
            profiles.append(record)

    def finish_runtime(self) -> None:
        """Finalize aggregate runtime telemetry."""

        self.runtime["finished_at_unix"] = time.time()
        self.runtime["total_elapsed_seconds"] = max(
            0.0,
            float(cast(Any, self.runtime["finished_at_unix"]))
            - float(cast(Any, self.runtime["started_at_unix"])),
        )
        self.runtime["output_dir_size_bytes"] = directory_size_bytes(self.output_dir)
        self.runtime["cache_dir_size_bytes"] = directory_size_bytes(self.cache_dir)
        self.runtime["current_rss_bytes"] = current_rss_bytes()
        self.runtime["peak_rss_bytes"] = peak_rss_bytes()


def directory_size_bytes(path: Path | None) -> int | None:
    """Return the recursive size of an existing directory."""

    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(current_root) / filename).stat().st_size
            except OSError:
                continue
    return int(total)


def current_rss_bytes() -> int | None:
    """Return current resident-set size on Linux when available."""

    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            rss_pages = int(statm.read_text(encoding="utf-8").split()[1])
            return int(rss_pages * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, IndexError, ValueError):
            pass
    return None


def peak_rss_bytes() -> int | None:
    """Return peak resident-set size when provided by the platform."""

    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    if value <= 0:
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


__all__ = [
    "CapabilityRunState",
    "current_rss_bytes",
    "directory_size_bytes",
    "initial_runtime_record",
    "peak_rss_bytes",
]
