import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimate_ar1(context: np.ndarray) -> float:
    values = np.asarray(context, dtype=np.float64)
    if values.size < 4:
        return 0.0
    x_prev = values[:-1]
    x_next = values[1:]
    denom = float(np.dot(x_prev, x_prev))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_prev, x_next) / denom)


def _local_event_residuals(
    values: np.ndarray,
    source_start: int,
    source_end: int,
    context_size: int = 96,
) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    left_start = max(0, int(source_start) - int(context_size))
    context = series[left_start : int(source_start)]
    if context.size < 8:
        context = series[
            int(source_end) : min(series.shape[0], int(source_end) + int(context_size))
        ]
    phi = _estimate_ar1(context)
    if int(source_start) > 0:
        prev = series[int(source_start) - 1 : int(source_end) - 1]
        curr = series[int(source_start) : int(source_end)]
    else:
        prev = series[int(source_start) : int(source_end) - 1]
        curr = series[int(source_start) + 1 : int(source_end)]
    return curr - phi * prev


def _pair_residual_corr(
    values: np.ndarray,
    channels: list[int],
    source_start: int,
    source_end: int,
) -> float:
    first = _local_event_residuals(
        values[:, int(channels[0])], source_start, source_end
    )
    second = _local_event_residuals(
        values[:, int(channels[1])], source_start, source_end
    )
    length = min(first.shape[0], second.shape[0])
    if length < 3:
        return float("nan")
    return float(np.corrcoef(first[:length], second[:length])[0, 1])


def _realize_parameters(
    template: Mapping[str, Any],
    _rng: np.random.Generator,
) -> dict[str, Any]:
    return dict(template)


def _sanitize_parameters(
    _anomaly_type: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(params)


def base_config(output_root: Path) -> Dict:
    return {
        "generator": {
            "output_root": str(output_root),
            "master_seed": 1234,
            "overwrite_output": True,
            "log_level": "INFO",
            "on_variant_failure": "skip",
        },
        "dataset": {
            "length": 600,
            "channels": 3,
            "splits": ["train", "val"],
            "instances_per_split": 2,
        },
        "anomaly_policy": {
            "density_range": [0.05, 0.06],
            "density_tolerance": 0.005,
            "segment_count_range": [4, 6],
            "placement_policy": "uniform",
            "channel_policy": "single-random",
            "overlap_policy": "global",
            "length_normalization": "resample",
        },
        "variants": {
            "base_oscillations": ["sine"],
            "anomaly_types": ["mean"],
            "disabled_anomaly_types": [],
            "profiles_per_pair": 1,
            "pair_profiles": {},
            "base_parameter_policy": "fixed_per_variant",
            "anomaly_parameter_policy": "fixed_per_variant",
            "skip_base_oscillations": [],
            "skip_anomaly_types": [],
        },
        "plot": {
            "enabled": True,
            "zoom_count": 5,
            "zoom_fill_policy": "repeat",
            "zoom_margin": 32,
            "zoom_margin_min": 16,
            "zoom_margin_alpha": 0.5,
        },
    }


class TSDatasetGenerationConfigMixin:
    def _base_config(self, output_root: Path) -> Dict:
        return base_config(output_root)
