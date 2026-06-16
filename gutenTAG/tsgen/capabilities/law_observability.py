"""Law-level observability profiles for generated benchmark variants."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import pandas as pd

from .array_store import ArrayStore
from .cache import CacheStore
from .dataset import DatasetIndex, EventGroup, InstanceRecord, read_timeseries_csv
from .numerics import finite_float
from .partitions import PartitionSpec, partition_sequence, run_partitions
from .protocol import CapabilityProtocol


@dataclass(frozen=True)
class LawObservabilityResult:
    """Distribution-level observability output tables."""

    profile: pd.DataFrame
    summary: pd.DataFrame


def compute_law_observability_profiles(
    dataset: DatasetIndex,
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None = None,
    cache: CacheStore | None = None,
    n_jobs: int = 1,
    partition_size: int = 4,
) -> LawObservabilityResult:
    """Compare clean and anomalous event-window laws by variant and split."""

    grouped = _group_instances(dataset.instances)
    keys = tuple(grouped)
    if cache is None and int(n_jobs) <= 1:
        profile = _law_observability_for_keys(
            keys=keys,
            grouped=grouped,
            protocol=protocol,
            arrays=arrays,
        )
        return LawObservabilityResult(profile=profile, summary=_summary(profile))
    partitions = partition_sequence(
        keys,
        partition_size=max(1, int(partition_size)),
        prefix="law_observability",
    )
    fingerprint_extra: dict[str, object] = {
        "bootstrap_samples": int(protocol.bootstrap_samples),
        "partition_size": int(partition_size),
    }

    def worker(
        partition: PartitionSpec[tuple[str, str, str, str, str]],
    ) -> pd.DataFrame:
        return _law_observability_for_keys(
            keys=partition.items,
            grouped=grouped,
            protocol=protocol,
            arrays=arrays,
        )

    results = run_partitions(
        partitions,
        worker,
        n_jobs=n_jobs,
        cache=cache,
        profile_name="law_observability_profile",
        fingerprint_extra=fingerprint_extra,
        table_worker=worker if cache is not None else None,
    )
    frames = [result.value for result in results if not result.value.empty]
    profile = _sort_profile(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    return LawObservabilityResult(profile=profile, summary=_summary(profile))


def _law_observability_for_keys(
    *,
    keys: Sequence[tuple[str, str, str, str, str]],
    grouped: dict[tuple[str, str, str, str, str], list[InstanceRecord]],
    protocol: CapabilityProtocol,
    arrays: ArrayStore | None,
) -> pd.DataFrame:
    """Compute law-observability profile rows for deterministic group keys."""

    rows: list[dict[str, object]] = []
    for key in keys:
        instances = grouped.get(key, [])
        variant_id, split, anomaly_type, constraint_tag, semantic_scope = key
        clean_features: list[np.ndarray] = []
        anomalous_features: list[np.ndarray] = []
        for instance in instances:
            clean = (
                arrays.get(instance, "clean")
                if arrays is not None
                else read_timeseries_csv(instance.clean_path)
            )
            anomalous = (
                arrays.get(instance, "anomalous")
                if arrays is not None
                else read_timeseries_csv(instance.anomalous_path)
            )
            for group in instance.event_groups:
                channels = _event_channels(group, instance.channels)
                clean_features.append(
                    _window_features(clean, group.start, group.end, channels)
                )
                anomalous_features.append(
                    _window_features(anomalous, group.start, group.end, channels)
                )
        clean_matrix, anomalous_matrix = _align_features(
            clean_features, anomalous_features
        )
        rows.append(
            _profile_row(
                variant_id=variant_id,
                split=split,
                anomaly_type=anomaly_type,
                constraint_tag=constraint_tag,
                semantic_scope=semantic_scope,
                clean=clean_matrix,
                anomalous=anomalous_matrix,
                bootstrap_samples=protocol.bootstrap_samples,
                rng=_rng_for_key(protocol.random_seed, key),
            )
        )
    return _sort_profile(pd.DataFrame(rows))


def _group_instances(
    instances: Sequence[InstanceRecord],
) -> dict[tuple[str, str, str, str, str], list[InstanceRecord]]:
    grouped: dict[tuple[str, str, str, str, str], list[InstanceRecord]] = {}
    for instance in instances:
        if not instance.event_groups:
            continue
        constraint_tag = (
            instance.event_groups[0].constraint_tag
            if instance.event_groups
            else "unknown"
        )
        semantic_scope = (
            instance.event_groups[0].semantic_scope
            if instance.event_groups
            else "unknown"
        )
        key = (
            instance.variant_id,
            instance.split,
            instance.anomaly_type,
            constraint_tag,
            semantic_scope,
        )
        grouped.setdefault(key, []).append(instance)
    return grouped


def _profile_row(
    *,
    variant_id: str,
    split: str,
    anomaly_type: str,
    constraint_tag: str,
    semantic_scope: str,
    clean: np.ndarray,
    anomalous: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    energy = _energy_distance(clean, anomalous)
    mmd = _mmd_rbf(clean, anomalous)
    c2st = _nearest_centroid_balanced_accuracy(clean, anomalous)
    gaussian = _gaussian_proxies(clean, anomalous)
    ci_low, ci_high = _bootstrap_ci(
        clean,
        anomalous,
        samples=int(bootstrap_samples),
        rng=rng,
    )
    status = _law_status(energy, c2st, clean.shape[0], anomalous.shape[0])
    return {
        "variant_id": variant_id,
        "split": split,
        "anomaly_type": anomaly_type,
        "constraint_tag": constraint_tag,
        "semantic_scope": semantic_scope,
        "feature_space": "event_window_summary",
        "clean_sample_count": int(clean.shape[0]),
        "anomalous_sample_count": int(anomalous.shape[0]),
        "feature_dim": int(clean.shape[1]) if clean.ndim == 2 else 0,
        "energy_distance": finite_float(energy, default=math.nan),
        "energy_distance_ci_low": finite_float(ci_low, default=math.nan),
        "energy_distance_ci_high": finite_float(ci_high, default=math.nan),
        "mmd_rbf": finite_float(mmd, default=math.nan),
        "c2st_balanced_accuracy": finite_float(c2st, default=math.nan),
        "gaussian_symmetric_kl": finite_float(
            gaussian["symmetric_kl"], default=math.nan
        ),
        "gaussian_hellinger_proxy": finite_float(
            gaussian["hellinger_proxy"], default=math.nan
        ),
        "law_observability_status": status,
    }


def _window_features(
    series: np.ndarray,
    start: int,
    end: int,
    channels: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    lo = max(0, min(int(start), values.shape[0]))
    hi = max(lo, min(int(end), values.shape[0]))
    segment = values[lo:hi, list(channels)] if channels else values[lo:hi]
    if segment.size == 0:
        return np.zeros(8, dtype=np.float64)
    matrix = np.atleast_2d(segment)
    flat = matrix.reshape(-1)
    diffs = np.diff(matrix, axis=0) if matrix.shape[0] >= 2 else np.zeros_like(matrix)
    corr_proxy = 0.0
    if matrix.shape[1] >= 2 and matrix.shape[0] >= 3:
        corr = np.asarray(np.corrcoef(matrix, rowvar=False), dtype=np.float64)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        corr_proxy = float(np.nanmean(np.abs(corr[mask])))
        if not math.isfinite(corr_proxy):
            corr_proxy = 0.0
    features = np.asarray(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.median(flat)),
            float(np.mean(np.abs(flat - np.median(flat)))),
            float(np.sqrt(np.mean(np.square(flat)))),
            float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
            corr_proxy,
            float(matrix.shape[0]),
        ],
        dtype=np.float64,
    )
    features[~np.isfinite(features)] = 0.0
    return features


def _align_features(
    clean_features: Sequence[np.ndarray],
    anomalous_features: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not clean_features or not anomalous_features:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    width = max(
        max(item.size for item in clean_features),
        max(item.size for item in anomalous_features),
    )
    clean = np.vstack([_pad(item, width) for item in clean_features])
    anomalous = np.vstack([_pad(item, width) for item in anomalous_features])
    return _standardize_pair(clean, anomalous)


def _standardize_pair(
    clean: np.ndarray, anomalous: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([clean, anomalous])
    center = np.mean(combined, axis=0)
    scale = np.std(combined, axis=0)
    scale[scale <= 1e-12] = 1.0
    return (clean - center) / scale, (anomalous - center) / scale


def _pad(values: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros(int(width), dtype=np.float64)
    result[: min(values.size, width)] = values[:width]
    return result


def _event_channels(group: EventGroup, channels: int) -> tuple[int, ...]:
    selected = sorted(
        set(group.group_channels)
        | set(group.context_channels)
        | set(group.intervention_channels)
    )
    return tuple(channel for channel in selected if 0 <= int(channel) < int(channels))


def _energy_distance(clean: np.ndarray, anomalous: np.ndarray) -> float:
    if clean.size == 0 or anomalous.size == 0:
        return math.nan
    cross = _mean_pairwise_distance(clean, anomalous)
    clean_self = _mean_pairwise_distance(clean, clean)
    anomalous_self = _mean_pairwise_distance(anomalous, anomalous)
    return float(max(0.0, 2.0 * cross - clean_self - anomalous_self))


def _mean_pairwise_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = left[:, None, :] - right[None, :, :]
    return float(np.mean(np.sqrt(np.sum(np.square(diff), axis=2))))


def _mmd_rbf(clean: np.ndarray, anomalous: np.ndarray) -> float:
    if clean.size == 0 or anomalous.size == 0:
        return math.nan
    gamma = _median_gamma(np.vstack([clean, anomalous]))
    k_xx = _rbf_kernel(clean, clean, gamma)
    k_yy = _rbf_kernel(anomalous, anomalous, gamma)
    k_xy = _rbf_kernel(clean, anomalous, gamma)
    return float(max(0.0, np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy)))


def _median_gamma(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 1.0
    distances = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            distances.append(float(np.sum(np.square(values[i] - values[j]))))
    median = float(np.median(distances)) if distances else 1.0
    return 1.0 / max(median, 1e-8)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    sqdist = np.sum(np.square(diff), axis=2)
    return np.exp(-float(gamma) * sqdist)


def _nearest_centroid_balanced_accuracy(
    clean: np.ndarray, anomalous: np.ndarray
) -> float:
    if clean.shape[0] < 2 or anomalous.shape[0] < 2:
        return math.nan
    clean_correct = 0
    for index in range(clean.shape[0]):
        clean_centroid = np.mean(np.delete(clean, index, axis=0), axis=0)
        anomalous_centroid = np.mean(anomalous, axis=0)
        clean_correct += int(
            _nearest_label(clean[index], clean_centroid, anomalous_centroid) == 0
        )
    anomalous_correct = 0
    for index in range(anomalous.shape[0]):
        clean_centroid = np.mean(clean, axis=0)
        anomalous_centroid = np.mean(np.delete(anomalous, index, axis=0), axis=0)
        anomalous_correct += int(
            _nearest_label(anomalous[index], clean_centroid, anomalous_centroid) == 1
        )
    sensitivity = anomalous_correct / max(float(anomalous.shape[0]), 1.0)
    specificity = clean_correct / max(float(clean.shape[0]), 1.0)
    return float(0.5 * (sensitivity + specificity))


def _nearest_label(
    value: np.ndarray, clean_centroid: np.ndarray, anomalous_centroid: np.ndarray
) -> int:
    clean_distance = float(np.linalg.norm(value - clean_centroid))
    anomalous_distance = float(np.linalg.norm(value - anomalous_centroid))
    return 0 if clean_distance <= anomalous_distance else 1


def _gaussian_proxies(clean: np.ndarray, anomalous: np.ndarray) -> dict[str, float]:
    if clean.shape[0] < 2 or anomalous.shape[0] < 2 or clean.shape[1] == 0:
        return {"symmetric_kl": math.nan, "hellinger_proxy": math.nan}
    mean_clean = np.mean(clean, axis=0)
    mean_anom = np.mean(anomalous, axis=0)
    cov_clean = _regularized_cov(clean)
    cov_anom = _regularized_cov(anomalous)
    kl_ca = _gaussian_kl(mean_clean, cov_clean, mean_anom, cov_anom)
    kl_ac = _gaussian_kl(mean_anom, cov_anom, mean_clean, cov_clean)
    symmetric = 0.5 * (kl_ca + kl_ac)
    hellinger = float(math.sqrt(max(0.0, 1.0 - math.exp(-min(symmetric, 700.0) / 8.0))))
    return {"symmetric_kl": symmetric, "hellinger_proxy": hellinger}


def _regularized_cov(values: np.ndarray) -> np.ndarray:
    cov = np.cov(values, rowvar=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov[~np.isfinite(cov)] = 0.0
    return cov + np.eye(cov.shape[0]) * 1e-6


def _gaussian_kl(
    mean_p: np.ndarray, cov_p: np.ndarray, mean_q: np.ndarray, cov_q: np.ndarray
) -> float:
    dim = int(mean_p.size)
    inv_q = np.linalg.pinv(cov_q)
    diff = mean_q - mean_p
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        return math.nan
    trace = float(np.trace(inv_q @ cov_p))
    mahal = float(diff.T @ inv_q @ diff)
    return float(0.5 * (trace + mahal - dim + logdet_q - logdet_p))


def _bootstrap_ci(
    clean: np.ndarray,
    anomalous: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples <= 0 or clean.shape[0] < 2 or anomalous.shape[0] < 2:
        return math.nan, math.nan
    draws = []
    count = min(int(samples), 200)
    for _ in range(count):
        clean_idx = rng.integers(0, clean.shape[0], size=clean.shape[0])
        anomalous_idx = rng.integers(0, anomalous.shape[0], size=anomalous.shape[0])
        draws.append(_energy_distance(clean[clean_idx], anomalous[anomalous_idx]))
    values = np.asarray(draws, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _law_status(
    energy: float, c2st: float, clean_count: int, anomalous_count: int
) -> str:
    if clean_count < 2 or anomalous_count < 2:
        return "insufficient_replicates"
    if math.isfinite(c2st) and c2st >= 0.80:
        return "law_observable"
    if math.isfinite(energy) and energy > 0.25:
        return "weakly_law_observable"
    return "law_not_observable_in_features"


def _summary(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = profile.groupby(
        ["variant_id", "anomaly_type", "constraint_tag", "semantic_scope"], dropna=False
    )
    for key, frame in grouped:
        variant_id, anomaly_type, constraint_tag, semantic_scope = cast(
            tuple[object, object, object, object], key
        )
        rows.append(
            {
                "variant_id": variant_id,
                "anomaly_type": anomaly_type,
                "constraint_tag": constraint_tag,
                "semantic_scope": semantic_scope,
                "split_count": int(frame["split"].nunique()),
                "median_energy_distance": float(frame["energy_distance"].median()),
                "median_mmd_rbf": float(frame["mmd_rbf"].median()),
                "median_c2st_balanced_accuracy": float(
                    frame["c2st_balanced_accuracy"].median()
                ),
                "median_gaussian_symmetric_kl": float(
                    frame["gaussian_symmetric_kl"].median()
                ),
                "observable_split_share": float(
                    frame["law_observability_status"]
                    .isin(["law_observable", "weakly_law_observable"])
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _rng_for_key(
    seed: int,
    key: tuple[str, str, str, str, str],
) -> np.random.Generator:
    payload = "|".join(map(str, (int(seed), *key))).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return np.random.default_rng(int(digest, 16) % (2**32))


def _sort_profile(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    columns = [
        column
        for column in (
            "variant_id",
            "split",
            "anomaly_type",
            "constraint_tag",
            "semantic_scope",
        )
        if column in frame.columns
    ]
    if not columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)
