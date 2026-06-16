"""Descriptor diagnosis and quotient identifiability profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .diagnosis_predictions import (
    _feature_matrix,
    _nearest_centroid_predictions,
    _pairwise_c2st_accuracy,
    _prediction_accuracy,
)
from .diagnosis_quotient import _epsilon_quotient
from .identifiability import DESCRIPTOR_COLUMNS
from .protocol import CapabilityProtocol


@dataclass(frozen=True)
class DiagnosisResult:
    """Diagnosis output tables."""

    confusion_matrix: pd.DataFrame
    quotient: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class _DiagnosisContext:
    """Prepared inputs shared by descriptor diagnosis calculations."""

    embeddings: pd.DataFrame
    matrix: np.ndarray
    quotient: pd.DataFrame
    quotient_by_descriptor: dict[str, float]


@dataclass(frozen=True)
class _DescriptorPredictions:
    """Holdout predictions for one descriptor."""

    variant: list[str | None]
    base_family: list[str | None]


def compute_diagnosis_profiles(
    event_embeddings: pd.DataFrame,
    protocol: CapabilityProtocol,
) -> DiagnosisResult:
    """Compute descriptor-level diagnosis confusion and quotient impurity."""

    del protocol
    if event_embeddings.empty:
        return DiagnosisResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    context = _diagnosis_context(event_embeddings)
    confusion_rows, summary_rows = _diagnosis_rows(context)
    return DiagnosisResult(
        confusion_matrix=pd.DataFrame(confusion_rows),
        quotient=context.quotient,
        summary=pd.DataFrame(summary_rows),
    )


def _diagnosis_context(event_embeddings: pd.DataFrame) -> _DiagnosisContext:
    embeddings = _ensure_descriptor_columns(event_embeddings)
    matrix, _ = _feature_matrix(embeddings)
    quotient = _epsilon_quotient(embeddings, matrix)
    return _DiagnosisContext(
        embeddings=embeddings,
        matrix=matrix,
        quotient=quotient,
        quotient_by_descriptor=_quotient_by_descriptor(quotient),
    )


def _quotient_by_descriptor(quotient: pd.DataFrame) -> dict[str, float]:
    if quotient.empty:
        return {}
    return (
        quotient.groupby("descriptor", dropna=False)["mean_local_impurity"]
        .mean()
        .to_dict()
    )


def _diagnosis_rows(
    context: _DiagnosisContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    confusion_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for descriptor in DESCRIPTOR_COLUMNS:
        descriptor_confusion, descriptor_summary = _descriptor_diagnosis_rows(
            context,
            descriptor,
        )
        confusion_rows.extend(descriptor_confusion)
        summary_rows.append(descriptor_summary)
    return confusion_rows, summary_rows


def _descriptor_diagnosis_rows(
    context: _DiagnosisContext,
    descriptor: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    labels = context.embeddings[descriptor].astype(str).to_numpy()
    classes = sorted(set(labels))
    if len(classes) <= 1:
        return _single_class_descriptor_rows(descriptor, labels, classes)
    predictions = _descriptor_predictions(context, descriptor)
    return (
        _confusion_rows(descriptor, labels, predictions.variant),
        _multiclass_descriptor_summary(
            context, descriptor, labels, classes, predictions
        ),
    )


def _single_class_descriptor_rows(
    descriptor: str,
    labels: np.ndarray,
    classes: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    label = classes[0] if classes else "unknown"
    confusion = [
        _confusion_row(
            descriptor=descriptor,
            true_label=label,
            predicted_label=label,
            count=len(labels),
            event_count=len(labels),
        )
    ]
    summary = {
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
    return confusion, summary


def _descriptor_predictions(
    context: _DiagnosisContext,
    descriptor: str,
) -> _DescriptorPredictions:
    return _DescriptorPredictions(
        variant=_nearest_centroid_predictions(
            context.embeddings,
            context.matrix,
            descriptor=descriptor,
            holdout_column="variant_id",
        ),
        base_family=_nearest_centroid_predictions(
            context.embeddings,
            context.matrix,
            descriptor=descriptor,
            holdout_column="base_oscillation",
            allow_event_fallback=False,
        ),
    )


def _multiclass_descriptor_summary(
    context: _DiagnosisContext,
    descriptor: str,
    labels: np.ndarray,
    classes: Sequence[str],
    predictions: _DescriptorPredictions,
) -> dict[str, object]:
    variant_accuracy = _prediction_accuracy(labels, predictions.variant)
    base_accuracy = _prediction_accuracy(labels, predictions.base_family)
    pairwise_accuracy = _pairwise_c2st_accuracy(
        context.embeddings,
        context.matrix,
        descriptor,
    )
    quotient_impurity = float(
        context.quotient_by_descriptor.get(descriptor, float("nan"))
    )
    diagnosis_loss = _descriptor_diagnosis_loss(variant_accuracy, quotient_impurity)
    return {
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


def _descriptor_diagnosis_loss(
    variant_accuracy: float, quotient_impurity: float
) -> float:
    loss_terms = [
        1.0 - variant_accuracy if math.isfinite(variant_accuracy) else float("nan"),
        quotient_impurity,
    ]
    finite_loss_terms = [value for value in loss_terms if math.isfinite(value)]
    return float(np.mean(finite_loss_terms)) if finite_loss_terms else float("nan")


def merge_identifiability_summary(
    base: pd.DataFrame, diagnosis: pd.DataFrame
) -> pd.DataFrame:
    """Merge legacy embedding summary with P4 diagnosis metrics."""

    if base.empty:
        return diagnosis.copy()
    if diagnosis.empty:
        return base.copy()
    merged = base.merge(
        diagnosis, on="descriptor", how="outer", suffixes=("", "_diagnosis")
    )
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


def _confusion_rows(
    descriptor: str,
    labels: np.ndarray,
    predictions: Sequence[str | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = [
        (str(label), str(prediction))
        for label, prediction in zip(labels, predictions)
        if prediction is not None
    ]
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


def _diagnosis_status(accuracy: float, impurity: float) -> str:
    if not math.isfinite(accuracy):
        return "not_estimable"
    if accuracy >= 0.90 and (not math.isfinite(impurity) or impurity <= 0.10):
        return "diagnosable"
    if accuracy >= 0.70 and (not math.isfinite(impurity) or impurity <= 0.30):
        return "partially_diagnosable"
    return "confusable"
