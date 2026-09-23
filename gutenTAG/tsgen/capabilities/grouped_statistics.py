"""Group-held-out diagnostics and paired cluster uncertainty, without leakage."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def grouped_centroid_accuracy(
    x: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> float:
    """Balanced leave-one-GROUP-out accuracy with training-fold normalization.

    Both counterfactuals, all replicas and windows of a system must share a
    group. A class absent from training is counted as an error, not discarded.
    Constant or non-finite inputs do not silently become evidence of validity.
    """
    x = np.asarray(x, dtype=float)
    labels, groups = np.asarray(labels), np.asarray(groups)
    if (
        x.ndim != 2
        or len(labels) != len(x)
        or len(groups) != len(x)
        or not np.isfinite(x).all()
    ):
        raise ValueError("finite aligned features, labels and groups required")
    if len(np.unique(groups)) < 2:
        return float("nan")
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")
    correct = np.zeros(len(x), dtype=bool)
    total, squares = x.sum(axis=0), (x * x).sum(axis=0)
    sums = {k: x[labels == k].sum(axis=0) for k in classes}
    counts = {k: int((labels == k).sum()) for k in classes}
    for g in np.unique(groups):
        test = groups == g
        n = len(x) - int(test.sum())
        mean = (total - x[test].sum(axis=0)) / n
        var = (squares - (x[test] ** 2).sum(axis=0)) / n - mean * mean
        scale = np.sqrt(np.maximum(var, 0))
        scale[scale < 1e-12] = 1
        available, centers = [], []
        for k in classes:
            excluded = test & (labels == k)
            count = counts[k] - int(excluded.sum())
            if count:
                centers.append((sums[k] - x[excluded].sum(axis=0)) / count / scale)
                available.append(k)
        if available:
            dist = cdist(x[test] / scale, np.asarray(centers), metric="sqeuclidean")
            predicted = np.asarray(available)[np.argmin(dist, axis=1)]
            correct[test] = predicted == labels[test]
    return float(np.mean([correct[labels == k].mean() for k in classes]))


def paired_c2st(
    clean: np.ndarray, anomalous: np.ndarray, groups: np.ndarray | None = None
) -> float:
    """Classify paired laws with both counterparts held out together."""
    if clean.shape != anomalous.shape:
        raise ValueError("paired features must have identical shapes")
    n = len(clean)
    group = np.arange(n) if groups is None else np.asarray(groups)
    return grouped_centroid_accuracy(
        np.vstack([clean, anomalous]),
        np.r_[np.zeros(n), np.ones(n)],
        np.r_[group, group],
    )


def _distance_sum(a: np.ndarray, b: np.ndarray, block: int) -> float:
    total = 0.0
    for i in range(0, len(a), block):
        for j in range(0, len(b), block):
            total += float(cdist(a[i : i + block], b[j : j + block]).sum())
    return total


def cluster_energy_interval(
    clean: np.ndarray,
    anomalous: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    block: int = 256,
) -> tuple[float, float]:
    """Paired cluster bootstrap using cached group-level distance SUMS.

    Memory is O(G^2 + block^2), not O(N^2*features). Every bootstrap draw uses
    identical cluster multiplicities for clean and anomalous observations.
    The estimand weights observations; all events of a system are resampled
    together. No confidence interval is returned for fewer than two groups.
    """
    if clean.shape != anomalous.shape or len(groups) != len(clean):
        raise ValueError("paired features and group IDs must align")
    unique = np.unique(groups)
    if len(unique) < 2 or samples < 1:
        return float("nan"), float("nan")
    if not np.isfinite(clean).all() or not np.isfinite(anomalous).all():
        raise ValueError("energy features must be finite")
    parts = [np.flatnonzero(groups == g) for g in unique]
    counts = np.asarray([len(p) for p in parts])
    xx, yy, xy = (np.empty((len(parts), len(parts))) for _ in range(3))
    for i, pi in enumerate(parts):
        for j, pj in enumerate(parts):
            xx[i, j] = _distance_sum(clean[pi], clean[pj], block)
            yy[i, j] = _distance_sum(anomalous[pi], anomalous[pj], block)
            xy[i, j] = _distance_sum(clean[pi], anomalous[pj], block)
    draws = np.empty(samples)
    for k in range(samples):
        w = rng.multinomial(len(parts), np.full(len(parts), 1 / len(parts)))
        denom = float(w @ counts) ** 2
        draws[k] = max(0.0, float(w @ (2 * xy - xx - yy) @ w) / denom)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)
