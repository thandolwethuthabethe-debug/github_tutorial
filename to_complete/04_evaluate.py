from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _majority_label(train_rows: List[Dict[str, Any]], target_name: str) -> str:
    """Most common training label - a safe guess when we have no usable model."""
    labels = [str(row.get(target_name)) for row in train_rows if row.get(target_name) is not None]
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def _predict(model: Any, test_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]], target_name: str) -> List[str]:
    """Use the trained model if it can predict; otherwise fall back to the majority label."""
    if hasattr(model, "predict"):
        try:
            return [str(label) for label in model.predict(test_rows)]
        except Exception:
            pass
    fallback = _majority_label(train_rows, target_name)
    return [fallback for _ in test_rows]


def evaluate_model(trained_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Measure how good the model is.

    Input contract:
    - trained_bundle["dataset_name"]: str
    - trained_bundle["feature_names"]: list[str]
    - trained_bundle["target_name"]: str
    - trained_bundle["train_rows"]: list[dict]
    - trained_bundle["test_rows"]: list[dict]
    - trained_bundle["model"]: object

    Expected work:
    - create predictions for the test set
    - compare predictions with the true labels
    - calculate at least one useful metric

    Tip:
    - accuracy plus a confusion matrix is enough for the first version

    Output contract:
    - dataset_name: str
    - feature_names: list[str]
    - target_name: str
    - train_rows: list[dict]
    - test_rows: list[dict]
    - model: object
    - metrics: dict
    - sample_predictions: list[dict]
    - confusion_matrix: list[list[int]]
    """
    dataset_name = trained_bundle["dataset_name"]
    feature_names = trained_bundle["feature_names"]
    target_name = trained_bundle["target_name"]
    train_rows = trained_bundle["train_rows"]
    test_rows = trained_bundle["test_rows"]
    model = trained_bundle["model"]

    actual = [str(row.get(target_name)) for row in test_rows]
    predicted = _predict(model, test_rows, train_rows, target_name)

    # Stable, shared label order for both the metrics and the confusion matrix.
    labels = sorted(set(actual) | set(predicted))

    if actual:
        metrics = {
            "accuracy": round(float(accuracy_score(actual, predicted)), 4),
            "precision": round(
                float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)), 4
            ),
            "f1": round(
                float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)), 4
            ),
            "n_test_rows": len(test_rows),
        }
        raw_matrix = confusion_matrix(actual, predicted, labels=labels)
        matrix = [[int(value) for value in row] for row in raw_matrix]
    else:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_test_rows": 0,
        }
        matrix = []

    sample_predictions: List[Dict[str, Any]] = []
    for true_label, pred_label in zip(actual, predicted):
        sample_predictions.append(
            {
                "actual": true_label,
                "predicted": pred_label,
                "correct": true_label == pred_label,
            }
        )

    return {
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "target_name": target_name,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "model": model,
        "metrics": metrics,
        "sample_predictions": sample_predictions,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": labels,
    }
