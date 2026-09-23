"""Fully blind reference-conditioned scans and independent-trajectory calibration.

Scoring accepts observations/reference/config only, never event metadata, true
channels, true duration, process parameters or clean query counterfactuals.
Each witness family's maximum over ALL configured windows/channels/pairs is
calibrated on whole independent normal trajectories from the same system.
Bonferroni corrects selection across witness families; raw units are not mixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any
import numpy as np

from ..benchmark.config import BenchmarkConfig
from ..benchmark.generation import file_hash, write_json


@dataclass(frozen=True)
class ScanResult:
    starts: np.ndarray
    ends: np.ndarray
    scores: np.ndarray
    components: np.ndarray

    @property
    def maximum(self) -> float:
        return float(self.scores.max())


def _window_mean(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    prefix = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    return (prefix[ends] - prefix[starts]) / (ends - starts)[:, None]


def blind_scan(
    reference: np.ndarray,
    values: np.ndarray,
    *,
    windows: tuple[int, ...],
    stride: int,
    timestamps: np.ndarray | None = None,
) -> ScanResult:
    """Scan a fixed grid; pair products are processed in bounded-memory chunks."""
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    if reference.ndim != 2 or values.ndim != 2 or reference.shape[1] != values.shape[1]:
        raise ValueError("reference/query channel dimensions must agree")
    if not np.isfinite(reference).all() or np.isinf(values).any():
        raise ValueError(
            "reference must be finite; query may contain NaNs, not infinities"
        )
    if stride < 1 or not windows or min(windows) < 2 or max(windows) > len(values):
        raise ValueError("invalid scan grid")
    starts: list[int] = []
    ends: list[int] = []
    for w in sorted(set(windows)):
        grid = sorted(set(range(0, len(values) - w + 1, stride)) | {len(values) - w})
        starts.extend(grid)
        ends.extend(s + w for s in grid)
    start, end = np.asarray(starts), np.asarray(ends)
    center = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    missing = ~np.isfinite(values)
    z = (np.where(missing, center, values) - center) / scale
    mean = _window_mean(z, start, end)
    var = np.maximum(_window_mean(z * z, start, end) - mean * mean, 1e-12)
    component = np.zeros((len(start), 5))
    component[:, 0] = np.max(np.abs(mean), axis=1)
    component[:, 1] = np.max(np.abs(np.log(var)), axis=1)
    component[:, 3] = _window_mean(missing.astype(float), start, end).max(axis=1)
    rz = np.sign((reference - center) / scale)
    qz = np.sign(z)
    pairs = list(combinations(range(values.shape[1]), 2))
    for i in range(0, len(pairs), 32):
        p = np.asarray(pairs[i : i + 32])
        rm = np.mean(rz[:, p[:, 0]] * rz[:, p[:, 1]], axis=0)
        qm = _window_mean(qz[:, p[:, 0]] * qz[:, p[:, 1]], start, end)
        component[:, 2] = np.maximum(component[:, 2], np.max(np.abs(qm - rm), axis=1))
    if timestamps is not None:
        t = np.asarray(timestamps, dtype=float)
        if (
            t.shape != values.shape
            or not np.isfinite(t).all()
            or np.any(np.diff(t, axis=0) <= 0)
        ):
            raise ValueError(
                "timestamps must be finite and strictly increasing per channel"
            )
        jitter = np.r_[np.zeros((1, values.shape[1])), np.abs(np.diff(t, axis=0) - 1)]
        component[:, 4] = _window_mean(jitter, start, end).max(axis=1)
    # Raw maxima are diagnostics ONLY. Decisions use calibrated families.
    return ScanResult(start, end, component.max(axis=1), component)


def empirical_p(null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Conservative finite Monte Carlo p-values including ties."""
    null = np.sort(np.asarray(null, dtype=float))
    if null.size == 0 or not np.isfinite(null).all():
        raise ValueError("independent calibration maxima must be nonempty and finite")
    return (1 + null.size - np.searchsorted(null, observed, side="left")) / (
        null.size + 1
    )


def evaluate_benchmark(
    root: Path, output: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    """Evaluate without query-label/clean leakage; then compute labelled metrics.

    The rank guarantee is conditional on the fixed reference and system law,
    for a query trajectory exchangeable with its normal calibration draws.
    It is not a guarantee under arbitrary drift or repeated future testing.
    """
    root, output = Path(root), Path(output)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    config = BenchmarkConfig.model_validate(manifest["config"])
    if verify_hashes:
        for relative, expected in manifest["sha256"].items():
            p = (root / relative).resolve()
            if not p.is_relative_to(root.resolve()) or file_hash(p) != expected:
                raise ValueError(f"artifact integrity failure: {relative}")
    rows = []
    systems = manifest["systems"]
    if len({s["law_id"] for s in systems}) != len(systems):
        raise ValueError("law reuse across split/system slots")
    for system in systems:
        folder = root / system["relative_path"]
        with np.load(folder / "reference.npz") as f:
            reference = f["values"]
        null = []
        for p in sorted((folder / "calibration").glob("*.npz")):
            with np.load(p) as f:
                null.append(
                    blind_scan(
                        reference,
                        f["values"],
                        windows=config.window_lengths,
                        stride=config.stride,
                    ).components.max(axis=0)
                )
        if len(null) != config.calibration_realizations:
            raise ValueError("missing calibration realizations; no test-data fallback")
        predictions = []
        for query in system["queries"]:
            with np.load(folder / "queries" / (query["id"] + ".npz")) as f:
                if not np.array_equal(f["observed_mask"], np.isfinite(f["values"])):
                    raise ValueError("invalid observed mask")
                scan = blind_scan(
                    reference,
                    f["values"],
                    windows=config.window_lengths,
                    stride=config.stride,
                    timestamps=f["timestamps"],
                )
            family_p = np.column_stack(
                [
                    empirical_p(np.asarray(null)[:, k], scan.components[:, k])
                    for k in range(scan.components.shape[1])
                ]
            )
            # Every family is scan-corrected; correct selection across families.
            pvalues = np.minimum(1.0, family_p.shape[1] * family_p.min(axis=1))
            predictions.append((query, scan, pvalues))
        for query, scan, pvalues in predictions:
            events = json.loads(
                (folder / "oracle" / (query["id"] + ".json")).read_text()
            )
            event = events[0]
            with np.load(folder / "oracle" / (query["id"] + ".npz")) as f:
                evaluable = f["evaluation_mask"]
            lo, hi = event["start"], event["end"]
            margin = event["transition"]
            if system["role"] == "relation_only" and not (
                event["relation_only"] or event["is_null"]
            ):
                raise ValueError(
                    "relation_only corpus contains a marginal/sensor event"
                )
            for alpha in config.alpha_grid:
                alerts = pvalues <= alpha
                endpoint_in_event = (scan.ends - 1 >= lo) & (scan.ends - 1 < hi)
                interior = (scan.starts >= lo + margin) & (scan.ends <= hi - margin)
                detected = alerts & endpoint_in_event & (not event["is_null"])
                outside = (
                    ((scan.ends <= lo) | (scan.starts >= hi))
                    if not event["is_null"]
                    else np.ones(len(alerts), bool)
                )
                # Entire windows, not just endpoints, must be normal and outside recovery.
                excluded = np.r_[0, np.cumsum(~evaluable)]
                false_alerts = (
                    alerts
                    & outside
                    & ((excluded[scan.ends] - excluded[scan.starts]) == 0)
                )
                rows.append(
                    {
                        "system_id": system["system_id"],
                        "law_id": system["law_id"],
                        "split": system["split"],
                        "role": system["role"],
                        "genotype_id": query["genotype_id"],
                        "query_id": query["id"],
                        "mechanism": query["mechanism"],
                        "alpha": alpha,
                        "strength": event["strength"],
                        "duration": hi - lo,
                        "source_length": hi - lo,
                        "transition_length": margin,
                        "is_null": event["is_null"],
                        "maximum": scan.maximum,
                        "scan_p": float(pvalues.min()),
                        "aggregation": "calibrated_family_scan_maxima_bonferroni",
                        "witness_family_count": scan.components.shape[1],
                        "event_detected": bool(detected.any()),
                        "interior_detected": bool(
                            (alerts & interior).any() and not event["is_null"]
                        ),
                        "interior_evaluable": bool(interior.any()),
                        "latency": (
                            int(scan.ends[detected].min() - lo)
                            if detected.any()
                            else None
                        ),
                        "false_alert_windows": int(false_alerts.sum()),
                        "calibration_count": len(null),
                        "calibration_resolution_ok": scan.components.shape[1]
                        / (len(null) + 1)
                        <= alpha,
                        "information_mode": "fully_blind_fixed_grid_reference_conditioned",
                    }
                )
    summary = {
        "schema_version": "synthgen.v13.evaluation.v1",
        "query_alpha_rows": len(rows),
        "independent_systems": len(systems),
        "calibration_resolution_ok": all(r["calibration_resolution_ok"] for r in rows),
        "scientific_admission": "not_automatic",
        "rows": rows,
    }
    write_json(output, summary)
    return summary
