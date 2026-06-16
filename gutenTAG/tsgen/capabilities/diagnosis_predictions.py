"""Centroid prediction helpers for descriptor diagnosis."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


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
    groups = (
        embeddings[holdout_column].astype(str).to_numpy()
        if holdout_column in embeddings
        else np.asarray([""] * len(labels))
    )
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
                predictions[idx] = _predict_from_centroids(
                    matrix,
                    labels,
                    event_train_mask,
                    int(idx),
                )
            continue
        if int(train_mask.sum()) == 0:
            continue
        group_predictions = _predict_many_from_centroids(
            matrix,
            labels,
            train_mask,
            target_indices,
        )
        predictions[target_indices] = group_predictions
    return [
        None if prediction is None else str(prediction) for prediction in predictions
    ]


def _predict_from_centroids(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    target_index: int,
) -> str:
    predictions = _predict_many_from_centroids(
        matrix,
        labels,
        train_mask,
        np.asarray([target_index], dtype=int),
    )
    return str(predictions[0])


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
    distances = np.sqrt(
        np.sum(
            np.square(target_matrix[:, None, :] - centroid_matrix[None, :, :]),
            axis=2,
        )
    )
    best_indices = np.argmin(distances, axis=1)
    for row_idx, best_idx in enumerate(best_indices):
        predictions[row_idx] = active_classes[int(best_idx)]
    return predictions


def _prediction_accuracy(
    labels: np.ndarray, predictions: Sequence[str | None]
) -> float:
    valid = [
        (str(label), str(prediction))
        for label, prediction in zip(labels, predictions)
        if prediction is not None
    ]
    if not valid:
        return float("nan")
    return float(np.mean([label == prediction for label, prediction in valid]))


def _pairwise_c2st_accuracy(
    embeddings: pd.DataFrame,
    matrix: np.ndarray,
    descriptor: str,
) -> float:
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
            predictions = _binary_leave_one_out_centroid_predictions(
                sub_matrix,
                sub_labels,
            )
            acc_i = _class_accuracy(sub_labels, predictions, class_i)
            acc_j = _class_accuracy(sub_labels, predictions, class_j)
            if math.isfinite(acc_i) and math.isfinite(acc_j):
                accuracies.append(float(0.5 * (acc_i + acc_j)))
    return float(np.median(accuracies)) if accuracies else float("nan")


def _binary_leave_one_out_centroid_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> list[str]:
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
            predictions.append(
                sorted(classes, key=lambda label: (-counts[label], label))[0]
            )
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
            tied_best = abs(distance - best_distance) <= 1e-12
            if distance < best_distance - 1e-12 or (
                tied_best and candidate < best_label
            ):
                best_label = candidate
                best_distance = distance
        predictions.append(best_label)
    return predictions


def _class_accuracy(
    labels: np.ndarray, predictions: Sequence[str], label: str
) -> float:
    mask = labels == label
    if int(mask.sum()) == 0:
        return float("nan")
    pred = np.asarray(predictions, dtype=object)
    return float(np.mean(pred[mask] == label))


__all__ = [
    "_feature_matrix",
    "_nearest_centroid_predictions",
    "_pairwise_c2st_accuracy",
    "_prediction_accuracy",
]
