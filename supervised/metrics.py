from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def find_best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = -1.0
    for threshold in np.linspace(0.01, 0.99, 197):
        predictions = (probabilities >= threshold).astype(np.int64)
        precision, _, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=[1], average=None, zero_division=0
        )
        if f1[0] > best_f1 or (f1[0] == best_f1 and precision[0] > best_precision):
            best_threshold = float(threshold)
            best_f1 = float(f1[0])
            best_precision = float(precision[0])
    return best_threshold


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix": matrix.tolist(),
    }


def save_evaluation(
    output_dir: Path,
    split: str,
    paths: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    threshold = float(metrics["threshold"])
    predictions = (probabilities >= threshold).astype(np.int64)
    with (output_dir / f"{split}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "true_label", "predicted_class", "probability_class_1"])
        for path, label, prediction, probability in zip(
            paths, labels, predictions, probabilities
        ):
            writer.writerow([path, int(label), int(prediction), float(probability)])
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    with (output_dir / f"{split}_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", "0", "1"])
        writer.writerow(["0", *matrix[0].tolist()])
        writer.writerow(["1", *matrix[1].tolist()])
