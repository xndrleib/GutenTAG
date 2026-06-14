"""Correlation and covariance repair operators."""

from __future__ import annotations

import numpy as np

from .pattern import affine_channel_repair


def match_covariance(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Whiten anomalous channels and recolor them to the clean covariance."""

    if clean_segment.size == 0 or anomalous_segment.size == 0:
        return np.asarray(anomalous_segment, dtype=np.float64)
    if clean_segment.shape[1] < 2 or clean_segment.shape[0] < 3:
        return affine_channel_repair(clean_segment, anomalous_segment)
    candidates = [
        np.asarray(anomalous_segment, dtype=np.float64),
        affine_channel_repair(clean_segment, anomalous_segment),
    ]
    clean_mean = np.mean(clean_segment, axis=0)
    anomalous_mean = np.mean(anomalous_segment, axis=0)
    centered = np.asarray(anomalous_segment, dtype=np.float64) - anomalous_mean
    clean_cov = np.cov(clean_segment, rowvar=False) + np.eye(clean_segment.shape[1]) * 1e-6
    anomalous_cov = np.cov(anomalous_segment, rowvar=False) + np.eye(anomalous_segment.shape[1]) * 1e-6
    try:
        eval_anom, evec_anom = np.linalg.eigh(anomalous_cov)
        eval_clean, evec_clean = np.linalg.eigh(clean_cov)
        whitening = evec_anom @ np.diag(1.0 / np.sqrt(np.maximum(eval_anom, 1e-8))) @ evec_anom.T
        recolor = evec_clean @ np.diag(np.sqrt(np.maximum(eval_clean, 1e-8))) @ evec_clean.T
        candidates.append(centered @ whitening @ recolor + clean_mean)
    except np.linalg.LinAlgError:
        pass
    candidates.append(_multichannel_affine_repair(clean_segment, anomalous_segment))
    return min(candidates, key=lambda candidate: _rmse(candidate, clean_segment))


def match_correlation(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    """Restore the clean local correlation structure."""

    return match_covariance(clean_segment, anomalous_segment)


def _multichannel_affine_repair(clean_segment: np.ndarray, anomalous_segment: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean_segment, dtype=np.float64)
    anomalous = np.asarray(anomalous_segment, dtype=np.float64)
    if clean.shape != anomalous.shape or clean.size == 0:
        return np.asarray(anomalous_segment, dtype=np.float64)
    design = np.column_stack([np.ones(anomalous.shape[0]), anomalous])
    try:
        coefficients, *_ = np.linalg.lstsq(design, clean, rcond=None)
    except np.linalg.LinAlgError:
        return affine_channel_repair(clean, anomalous)
    return (design @ coefficients).astype(np.float64)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    residual = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))
