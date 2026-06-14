"""Descriptor identifiability and confusability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .ontology import repair_operator_for_anomaly
from .protocol import CapabilityProtocol


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Identifiability output tables."""

    event_embeddings: pd.DataFrame
    summary: pd.DataFrame
    pairwise: pd.DataFrame


EMBEDDING_WITNESSES: tuple[str, ...] = (
    "energy_delta",
    "mean_delta",
    "variance_delta",
    "shape_residual",
    "spectral_delta",
    "correlation_delta",
    "covariance_delta",
    "lag_correlation_delta",
)

DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "anomaly_type",
    "constraint_tag",
    "operator_family",
    "support_type",
    "semantic_scope",
    "repair_operator",
)


def compute_identifiability_profiles(
    observability: pd.DataFrame,
    event_summary: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> IdentifiabilityResult:
    """Compute observation-embedding based descriptor identifiability."""

    embeddings = _build_event_embeddings(observability, event_summary)
    summary = _leave_one_out_summary(embeddings)
    pairwise = _pairwise_class_distances(embeddings) if protocol.include_pairwise_identifiability else pd.DataFrame()
    return IdentifiabilityResult(event_embeddings=embeddings, summary=summary, pairwise=pairwise)


def _build_event_embeddings(observability: pd.DataFrame, event_summary: pd.DataFrame) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    grouped = observability.groupby("event_id") if not observability.empty else {}
    for _, row in event_summary.iterrows():
        event_id = str(row["event_id"])
        record: dict[str, object] = {
            "event_id": event_id,
            "variant_id": row.get("variant_id"),
            "split": row.get("split"),
            "instance_id": row.get("instance_id"),
            "base_oscillation": row.get("base_oscillation"),
            "anomaly_type": row.get("anomaly_type"),
            "constraint_tag": row.get("constraint_tag"),
            "operator_family": _operator_family(row.get("constraint_tag")),
            "support_type": _support_type(row),
            "semantic_scope": row.get("semantic_scope"),
            "repair_operator": repair_operator_for_anomaly(row.get("anomaly_type")),
            "length": row.get("length"),
            "D_s1": float(row.get("D_s1", 0.0)),
            "D_s2": float(row.get("D_s2", row.get("D_s1", 0.0))),
            "best_distance": float(row.get("best_distance", 0.0)),
            "witness_sufficiency_proxy": float(row.get("witness_sufficiency_proxy", 0.0)),
            "support_concentration_l2_max": float(row.get("support_concentration_l2_max", 0.0)),
        }
        if not observability.empty and event_id in grouped.groups:
            frame = grouped.get_group(event_id)
            for witness in EMBEDDING_WITNESSES:
                witness_frame = frame[frame["witness_family"] == witness]
                record[f"witness_{witness}"] = float(witness_frame["distance_value"].max()) if not witness_frame.empty else 0.0
        else:
            for witness in EMBEDDING_WITNESSES:
                record[f"witness_{witness}"] = 0.0
        records.append(record)
    return pd.DataFrame(records)


def _operator_family(constraint_tag: object) -> str:
    tag = str(constraint_tag or "unknown.generic")
    return tag.split(".", maxsplit=1)[0] if "." in tag else tag


def _support_type(row: pd.Series) -> str:
    semantic_scope = str(row.get("semantic_scope", "unknown"))
    if semantic_scope == "point":
        return "point"
    if semantic_scope in {"relation", "regime_relation"}:
        return "multi_channel_window"
    event_scope = str(row.get("event_scope", "unknown"))
    if event_scope not in {"", "unknown", "nan"}:
        return event_scope
    try:
        length = int(row.get("length", 0))
    except (TypeError, ValueError):
        length = 0
    if length <= 2:
        return "point"
    return "segment"


def _feature_matrix(embeddings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns = [
        column
        for column in embeddings.columns
        if column.startswith("witness_") or column in {"D_s1", "D_s2", "best_distance", "witness_sufficiency_proxy", "support_concentration_l2_max"}
    ]
    if not feature_columns:
        return np.zeros((len(embeddings), 0), dtype=np.float64), []
    matrix = embeddings[feature_columns].to_numpy(dtype=np.float64)
    matrix[~np.isfinite(matrix)] = 0.0
    scale = np.std(matrix, axis=0)
    scale[scale <= 1e-12] = 1.0
    center = np.mean(matrix, axis=0)
    return (matrix - center) / scale, feature_columns


def _leave_one_out_summary(embeddings: pd.DataFrame) -> pd.DataFrame:
    if embeddings.empty or len(embeddings) < 2:
        return pd.DataFrame()
    matrix, _ = _feature_matrix(embeddings)
    rows: list[dict[str, object]] = []
    for descriptor in DESCRIPTOR_COLUMNS:
        labels = embeddings[descriptor].astype(str).to_numpy()
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            rows.append(
                {
                    "descriptor": descriptor,
                    "class_count": len(unique_labels),
                    "event_count": int(len(labels)),
                    "loo_nearest_neighbor_accuracy": float("nan"),
                    "observation_quotient_impurity_proxy": 0.0,
                    "identifiability_status": "single_class",
                }
            )
            continue
        predictions = _leave_one_out_nearest_labels(matrix, labels)
        accuracy = float(np.mean(np.asarray(predictions, dtype=object) == labels))
        impurity = 1.0 - accuracy
        rows.append(
            {
                "descriptor": descriptor,
                "class_count": len(unique_labels),
                "event_count": int(len(labels)),
                "loo_nearest_neighbor_accuracy": accuracy,
                "observation_quotient_impurity_proxy": impurity,
                "identifiability_status": _status_from_accuracy(accuracy),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_out_nearest_labels(matrix: np.ndarray, labels: np.ndarray) -> list[str]:
    n_rows = int(len(labels))
    if n_rows < 2:
        return [str(label) for label in labels]
    if matrix.shape[1] == 0:
        return [str(labels[1 if idx == 0 else 0]) for idx in range(n_rows)]

    neighbors = NearestNeighbors(n_neighbors=min(2, n_rows), algorithm="auto", metric="euclidean")
    neighbors.fit(matrix)
    indices = neighbors.kneighbors(matrix, return_distance=False)
    predictions: list[str] = []
    for row_idx, neighbor_indices in enumerate(indices):
        nearest = next((int(candidate) for candidate in neighbor_indices if int(candidate) != row_idx), None)
        if nearest is None:
            nearest = 1 if row_idx == 0 else 0
        predictions.append(str(labels[nearest]))
    return predictions


def _pairwise_class_distances(embeddings: pd.DataFrame) -> pd.DataFrame:
    if embeddings.empty:
        return pd.DataFrame()
    matrix, _ = _feature_matrix(embeddings)
    rows: list[dict[str, object]] = []
    for descriptor in DESCRIPTOR_COLUMNS:
        labels = embeddings[descriptor].astype(str).to_numpy()
        classes = sorted(set(labels))
        centroids: dict[str, np.ndarray] = {
            label: np.mean(matrix[labels == label], axis=0) for label in classes if np.any(labels == label)
        }
        spreads: dict[str, float] = {
            label: _mean_radius(matrix[labels == label], centroids[label]) for label in centroids
        }
        for i, class_i in enumerate(classes):
            for class_j in classes[i + 1 :]:
                distance = float(np.linalg.norm(centroids[class_i] - centroids[class_j]))
                pooled_spread = float(spreads[class_i] + spreads[class_j] + 1e-12)
                rows.append(
                    {
                        "descriptor": descriptor,
                        "class_i": class_i,
                        "class_j": class_j,
                        "centroid_distance": distance,
                        "pooled_within_class_radius": pooled_spread,
                        "separation_ratio": distance / pooled_spread,
                        "confusability_proxy": 1.0 / (1.0 + distance / pooled_spread),
                        "count_i": int(np.sum(labels == class_i)),
                        "count_j": int(np.sum(labels == class_j)),
                    }
                )
    return pd.DataFrame(rows)


def _mean_radius(values: np.ndarray, centroid: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.sqrt(np.sum(np.square(values - centroid), axis=1))))


def _status_from_accuracy(accuracy: float) -> str:
    if not np.isfinite(accuracy):
        return "not_estimable"
    if accuracy >= 0.90:
        return "identifiable_in_embedding"
    if accuracy >= 0.70:
        return "partially_identifiable"
    return "confusable_in_embedding"
