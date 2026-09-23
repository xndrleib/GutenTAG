#!/usr/bin/env python3
"""Check mypy base/head in one environment without hiding existing diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def diagnostics(root: Path, destination: Path) -> Counter:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "gutenTAG",
            "tests",
            "--no-pretty",
            "--no-incremental",
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    destination.write_text(result.stdout + result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"mypy did not execute normally: {destination}")
    lines = [
        re.sub(r":\d+(?::\d+)?: error:", ": error:", line)
        for line in result.stdout.splitlines()
        if ": error:" in line
    ]
    return Counter(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.base_sha):
        raise ValueError("base must be an exact commit SHA")
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="v13-type-base-") as temporary:
        base = Path(temporary) / "base"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(base), a.base_sha],
            cwd=root,
            check=True,
        )
        try:
            old = diagnostics(base, a.output / "base.txt")
            new = diagnostics(root, a.output / "head.txt")
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(base)],
                cwd=root,
                check=True,
            )
    added, resolved = list((new - old).elements()), list((old - new).elements())
    report = {
        "base_sha": a.base_sha,
        "head_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "base_errors": sum(old.values()),
        "head_errors": sum(new.values()),
        "new_errors": added,
        "resolved_errors": resolved,
        "full_mypy_clean": not new,
        "incremental_gate_passed": not added,
    }
    (a.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if added:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
