"""Descriptor diagnosis and quotient identifiability profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .identifiability import DESCRIPTOR_COLUMNS
from .protocol import CapabilityProtocol


@dataclass(frozen=True)
class DiagnosisResult:
    """Diagnosis output tables."""

    confusion_matrix: pd.DataFrame
    quotient: pd.DataFrame
    summary: pd.DataFrame


def compute_diagnosis_profiles(
    event_embeddings: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> DiagnosisResult:
    """Compute descriptor-level diagnosis confusion and quotient impurity."""

    del protocol
    if event_embeddings.empty:
        return DiagnosisResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    embeddings = _ensure_descriptor_columns(event_embeddings)
    matrix, _ = _feature_matrix(embeddings)
    quotient = _epsilon_quotient(embeddings, matrix)
    confusion_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    quotient_by_descriptor = (
        quotient.groupby("descriptor", dropna=False)["mean_local_impurity"].mean().to_dict()
        if not quotient.empty
        else {}
    )
    for descriptor in DESCRIPTOR_COLUMNS:
        labels = embeddings[descriptor].astype(str).to_numpy()
        classes = sorted(set(labels))
        if len(classes) <= 1:
            label = classes[0] if classes else "unknown"
            confusion_rows.append(
                _confusion_row(
                    descriptor=descriptor,
                    true_label=label,
                    predicted_label=label,
                    count=len(labels),
                    event_count=len(labels),
                )
            )
            summary_rows.append(
                {
                    "descriptor": descriptor,
                    "class_count": len(classes),
                    "event_count": int(len(labels)),
                    "leave_one_variant_out_accuracy": float("nan"),
                    "held_out_base_family_accuracy": float("nan"),
                    "pairwise_c2st_balanced_accuracy": float("nan"),
                    "epsilon_quotient_impurity": 0.0,
                    "descriptor_diagnosis_loss": 0.0,
                    "diagnosis_status": "single_class",
                }
            )
            continue
        variant_predictions = _nearest_centroid_predictions(
            embeddings,
            matrix,
            descriptor=descriptor,
            holdout_column="variant_id",
        )
        base_predictions = _nearest_centroid_predictions(
            embeddings,
            matrix,
            descriptor=descriptor,
            holdout_column="base_oscillation",
            allow_event_fallback=False,
        )
        for row in _confusion_rows(descriptor, labels, variant_predictions):
            confusion_rows.append(row)
        variant_accuracy = _prediction_accuracy(labels, variant_predictions)
        base_accuracy = _prediction_accuracy(labels, base_predictions)
        pairwise_accuracy = _pairwise_c2st_accuracy(embeddings, matrix, descriptor)
        quotient_impurity = float(quotient_by_descriptor.get(descriptor, float("nan")))
        loss_terms = [
            1.0 - variant_accuracy if math.isfinite(variant_accuracy) else float("nan"),
            quotient_impurity,
        ]
        finite_loss_terms = [value for value in loss_terms if math.isfinite(value)]
        diagnosis_loss = float(np.mean(finite_loss_terms)) if finite_loss_terms else float("nan")
        summary_rows.append(
            {
                "descriptor": descriptor,
                "class_count": len(classes),
                "event_count": int(len(labels)),
                "leave_one_variant_out_accuracy": variant_accuracy,
                "held_out_base_family_accuracy": base_accuracy,
                "pairwise_c2st_balanced_accuracy": pairwise_accuracy,
                "epsilon_quotient_impurity": quotient_impurity,
                "descriptor_diagnosis_loss": diagnosis_loss,
                "diagnosis_status": _diagnosis_status(variant_accuracy, quotient_impurity),
            }
        )
    return DiagnosisResult(
        confusion_matrix=pd.DataFrame(confusion_rows),
        quotient=quotient,
        summary=pd.DataFrame(summary_rows),
    )


def merge_identifiability_summary(base: pd.DataFrame, diagnosis: pd.DataFrame) -> pd.DataFrame:
    """Merge legacy embedding summary with P4 diagnosis metrics."""

    if base.empty:
        return diagnosis.copy()
    if diagnosis.empty:
        return base.copy()
    merged = base.merge(diagnosis, on="descriptor", how="outer", suffixes=("", "_diagnosis"))
    for column in ("class_count", "event_count"):
        duplicate = f"{column}_diagnosis"
        if duplicate in merged.columns:
            merged[column] = merged[column].fillna(merged[duplicate])
            merged = merged.drop(columns=[duplicate])
    return merged


def _ensure_descriptor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "base_oscillation" not in result.columns:
        result["base_oscillation"] = "unknown"
    if "operator_family" not in result.columns:
        if "constraint_tag" in result.columns:
            result["operator_family"] = result["constraint_tag"].map(_operator_family)
        else:
            result["operator_family"] = "unknown"
    if "support_type" not in result.columns:
        result["support_type"] = "segment"
    if "repair_operator" not in result.columns:
        result["repair_operator"] = "unknown"
    for descriptor in DESCRIPTOR_COLUMNS:
        if descriptor not in result.columns:
            result[descriptor] = "unknown"
    return result


def _operator_family(value: object) -> str:
    tag = str(value or "unknown.generic")
    return tag.split(".", maxsplit=1)[0] if "." in tag else tag


def _feature_matrix(embeddings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns = [
        column
        for column in embeddings.columns
        if column.startswith("witness_")
        or column
        in {
            "D_s1",
            "D_s2",
            "best_distance",
            "witness_sufficiency_proxy",
            "support_concentration_l2_max",
        }
    ]
    if not feature_columns:
        return np.zeros((len(embeddings), 0), dtype=np.float64), []
    matrix = embeddings[feature_columns].to_numpy(dtype=np.float64)
    matrix[~np.isfinite(matrix)] = 0.0
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale <= 1e-12] = 1.0
    return (matrix - center) / scale, feature_columns


def _nearest_centroid_predictions(
    embeddings: pd.DataFrame,
    matrix: np.ndarray,
    *,
    descriptor: str,
    holdout_column: str,
    allow_event_fallback: bool = True,
) -> list[str | None]:
    labels = embeddings[descriptor].astype(str).to_numpy()
    groups = embeddings[holdout_column].astype(str).to_numpy() if holdout_column in embeddings else np.asarray([""] * len(labels))
    predictions: np.ndarray = np.full(len(labels), None, dtype=object)
    for group in sorted(set(groups)):
        target_indices = np.flatnonzero(groups == group)
        train_mask = groups != group
        if int(train_mask.sum()) == 0 and allow_event_fallback:
            for idx in target_indices:
                event_train_mask = np.ones(len(labels), dtype=bool)
                event_train_mask[idx] = False
                if int(event_train_mask.sum()) == 0:
                    continue
                predictions[idx] = _predict_from_centroids(matrix, labels, event_train_mask, idx)
            continue
        if int(train_mask.sum()) == 0:
            continue
        group_predictions = _predict_many_from_centroids(matrix, labels, train_mask, target_indices)
        predictions[target_indices] = group_predictions
    return [None if prediction is None else str(prediction) for prediction in predictions]


def _predict_from_centroids(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    target_index: int,
) -> str:
    return str(_predict_many_from_centroids(matrix, labels, train_mask, np.asarray([target_index], dtype=int))[0])


def _predict_many_from_centroids(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    train_labels = labels[train_mask]
    classes = sorted(set(map(str, train_labels)))
    predictions = np.empty(len(target_indices), dtype=object)
    if not classes:
        return np.asarray([str(labels[idx]) for idx in target_indices], dtype=object)
    if matrix.shape[1] == 0:
        counts = {label: int(np.sum(train_labels == label)) for label in classes}
        majority = sorted(classes, key=lambda label: (-counts[label], label))[0]
        predictions[:] = majority
        return predictions
    centroids: list[np.ndarray] = []
    active_classes: list[str] = []
    for label in classes:
        class_rows = matrix[train_mask & (labels == label)]
        if class_rows.size == 0:
            continue
        centroids.append(np.mean(class_rows, axis=0))
        active_classes.append(label)
    if not centroids:
        return np.asarray([str(labels[idx]) for idx in target_indices], dtype=object)
    centroid_matrix = np.vstack(centroids)
    target_matrix = matrix[target_indices]
    distances = np.sqrt(np.sum(np.square(target_matrix[:, None, :] - centroid_matrix[None, :, :]), axis=2))
    best_indices = np.argmin(distances, axis=1)
    for row_idx, best_idx in enumerate(best_indices):
        predictions[row_idx] = active_classes[int(best_idx)]
    return predictions


def _confusion_rows(
    descriptor: str,
    labels: np.ndarray,
    predictions: Sequence[str | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = [(str(label), str(prediction)) for label, prediction in zip(labels, predictions) if prediction is not None]
    event_count = len(valid)
    if event_count == 0:
        return rows
    counts: dict[tuple[str, str], int] = {}
    for label, prediction in valid:
        counts[(label, prediction)] = counts.get((label, prediction), 0) + 1
    for (true_label, predicted_label), count in sorted(counts.items()):
        rows.append(
            _confusion_row(
                descriptor=descriptor,
                true_label=true_label,
                predicted_label=predicted_label,
                count=count,
                event_count=event_count,
            )
        )
    return rows


def _confusion_row(
    *,
    descriptor: str,
    true_label: str,
    predicted_label: str,
    count: int,
    event_count: int,
) -> dict[str, object]:
    return {
        "descriptor": descriptor,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "count": int(count),
        "event_count": int(event_count),
        "share": float(count / max(event_count, 1)),
        "is_correct": str(true_label) == str(predicted_label),
    }


def _prediction_accuracy(labels: np.ndarray, predictions: Sequence[str | None]) -> float:
    valid = [(str(label), str(prediction)) for label, prediction in zip(labels, predictions) if prediction is not None]
    if not valid:
        return float("nan")
    return float(np.mean([label == prediction for label, prediction in valid]))


def _pairwise_c2st_accuracy(embeddings: pd.DataFrame, matrix: np.ndarray, descriptor: str) -> float:
    labels = embeddings[descriptor].astype(str).to_numpy()
    classes = sorted(set(labels))
    accuracies: list[float] = []
    for idx, class_i in enumerate(classes):
        for class_j in classes[idx + 1 :]:
            mask = (labels == class_i) | (labels == class_j)
            if int(mask.sum()) < 3:
                continue
            sub_matrix = matrix[mask]
            sub_labels = labels[mask]
            predictions = _binary_leave_one_out_centroid_predictions(sub_matrix, sub_labels)
            acc_i = _class_accuracy(sub_labels, predictions, class_i)
            acc_j = _class_accuracy(sub_labels, predictions, class_j)
            if math.isfinite(acc_i) and math.isfinite(acc_j):
                accuracies.append(float(0.5 * (acc_i + acc_j)))
    return float(np.median(accuracies)) if accuracies else float("nan")


def _binary_leave_one_out_centroid_predictions(matrix: np.ndarray, labels: np.ndarray) -> list[str]:
    classes = sorted(set(map(str, labels)))
    if not classes:
        return []
    if matrix.shape[1] == 0:
        predictions: list[str] = []
        for idx in range(len(labels)):
            train_labels = np.delete(labels, idx)
            if train_labels.size == 0:
                predictions.append(str(labels[idx]))
                continue
            counts = {label: int(np.sum(train_labels == label)) for label in classes}
            predictions.append(sorted(classes, key=lambda label: (-counts[label], label))[0])
        return predictions
    sums = {label: np.sum(matrix[labels == label], axis=0) for label in classes}
    counts = {label: int(np.sum(labels == label)) for label in classes}
    predictions = []
    for idx, label in enumerate(labels):
        target = matrix[idx]
        best_label = classes[0]
        best_distance = float("inf")
        for candidate in classes:
            count = counts[candidate]
            centroid_sum = sums[candidate]
            if str(label) == candidate:
                count -= 1
                centroid_sum = centroid_sum - target
            if count <= 0:
                continue
            centroid = centroid_sum / float(count)
            distance = float(np.linalg.norm(target - centroid))
            if distance < best_distance - 1e-12 or (abs(distance - best_distance) <= 1e-12 and candidate < best_label):
                best_label = candidate
                best_distance = distance
        predictions.append(best_label)
    return predictions


def _class_accuracy(labels: np.ndarray, predictions: Sequence[str], label: str) -> float:
    mask = labels == label
    if int(mask.sum()) == 0:
        return float("nan")
    pred = np.asarray(predictions, dtype=object)
    return float(np.mean(pred[mask] == label))


def _epsilon_quotient(embeddings: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    if embeddings.empty:
        return pd.DataFrame()
    epsilon = _epsilon_from_matrix(matrix)
    rows: list[dict[str, object]] = []
    zero_feature_neighbors = matrix.shape[1] == 0
    radius_neighbors: list[np.ndarray] | None = None
    if not zero_feature_neighbors and len(embeddings) > 0:
        neighbors = NearestNeighbors(radius=epsilon + 1e-12, algorithm="auto", metric="euclidean")
        neighbors.fit(matrix)
        radius_neighbors = list(neighbors.radius_neighbors(matrix, return_distance=False))
    for descriptor in DESCRIPTOR_COLUMNS:
        labels = embeddings[descriptor].astype(str).to_numpy()
        classes = sorted(set(labels))
        if zero_feature_neighbors:
            global_counts = {candidate: int(np.sum(labels == candidate)) for candidate in classes}
            global_majority = max(global_counts.values()) if global_counts else 0
            global_impurity = float(1.0 - global_majority / max(len(labels), 1))
            global_size = float(len(labels))
            for label in classes:
                rows.append(
                    {
                        "descriptor": descriptor,
                        "label": label,
                        "event_count": int(np.sum(labels == label)),
                        "epsilon": epsilon,
                        "mean_local_impurity": global_impurity,
                        "max_local_impurity": global_impurity,
                        "mean_quotient_cell_size": global_size,
                        "quotient_status": _quotient_status(global_impurity),
                    }
                )
            continue
        for label in classes:
            local_impurities: list[float] = []
            local_sizes: list[int] = []
            for idx, current_label in enumerate(labels):
                if current_label != label:
                    continue
                neighbor_indices = radius_neighbors[idx] if radius_neighbors is not None else np.asarray([idx], dtype=int)
                if neighbor_indices.size == 0:
                    neighbor_indices = np.asarray([idx], dtype=int)
                neighbor_labels = labels[neighbor_indices]
                majority = max(int(np.sum(neighbor_labels == candidate)) for candidate in set(neighbor_labels))
                local_impurities.append(float(1.0 - majority / max(len(neighbor_labels), 1)))
                local_sizes.append(int(len(neighbor_labels)))
            rows.append(
                {
                    "descriptor": descriptor,
                    "label": label,
                    "event_count": int(np.sum(labels == label)),
                    "epsilon": epsilon,
                    "mean_local_impurity": float(np.mean(local_impurities)) if local_impurities else float("nan"),
                    "max_local_impurity": float(np.max(local_impurities)) if local_impurities else float("nan"),
                    "mean_quotient_cell_size": float(np.mean(local_sizes)) if local_sizes else float("nan"),
                    "quotient_status": _quotient_status(float(np.mean(local_impurities)) if local_impurities else float("nan")),
                }
            )
    return pd.DataFrame(rows)


def _epsilon_from_matrix(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=min(2, matrix.shape[0]), algorithm="auto", metric="euclidean")
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix, return_distance=True)
    nearest: list[float] = []
    for row_idx, row_indices in enumerate(indices):
        for distance, candidate_idx in zip(distances[row_idx], row_indices):
            if int(candidate_idx) != row_idx:
                nearest.append(float(distance))
                break
    finite = np.asarray([value for value in nearest if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    return float(max(np.median(finite), 1e-12))


def _diagnosis_status(accuracy: float, impurity: float) -> str:
    if not math.isfinite(accuracy):
        return "not_estimable"
    if accuracy >= 0.90 and (not math.isfinite(impurity) or impurity <= 0.10):
        return "diagnosable"
    if accuracy >= 0.70 and (not math.isfinite(impurity) or impurity <= 0.30):
        return "partially_diagnosable"
    return "confusable"


def _quotient_status(impurity: float) -> str:
    if not math.isfinite(impurity):
        return "not_estimable"
    if impurity <= 0.10:
        return "low_impurity"
    if impurity <= 0.30:
        return "mixed_neighborhoods"
    return "high_impurity"
