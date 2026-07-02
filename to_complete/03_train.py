from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class WaterNeedModel:
    """
    Tiny wrapper around a fitted scikit-learn pipeline.

    It keeps the data contract simple: later steps can call
    ``model.predict(rows)`` with the same list-of-dict rows that flow
    through the whole pipeline, instead of having to rebuild a DataFrame.
    """

    def __init__(self, pipeline: Pipeline, feature_names: List[str], target_name: str, classes: List[str]) -> None:
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.target_name = target_name
        self.classes = list(classes)

    def predict(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        # Make sure every expected feature column exists, even if a row is missing it.
        for name in self.feature_names:
            if name not in frame.columns:
                frame[name] = None
        predictions = self.pipeline.predict(frame[self.feature_names])
        return [str(label) for label in predictions]


def _split_feature_types(rows: List[Dict[str, Any]], feature_names: List[str]) -> tuple[List[str], List[str]]:
    """Decide which features are numeric and which are categorical by looking at the values."""
    numeric: List[str] = []
    categorical: List[str] = []
    for name in feature_names:
        looks_numeric = True
        for row in rows:
            value = row.get(name)
            if value is None:
                continue
            # bool is a subclass of int, but here it is a category, not a number.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                looks_numeric = False
                break
        (numeric if looks_numeric else categorical).append(name)
    return numeric, categorical


def train_model(processed_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fit a model on the training rows.

    Input contract:
    - processed_bundle["dataset_name"]: str
    - processed_bundle["feature_names"]: list[str]
    - processed_bundle["target_name"]: str
    - processed_bundle["train_rows"]: list[dict]
    - processed_bundle["test_rows"]: list[dict]

    Expected work:
    - pick a simple model first
    - train it on the training rows
    - store anything needed for evaluation later

    Tip:
    - a small decision tree or logistic regression is enough for this workshop

    Output contract:
    - dataset_name: str
    - feature_names: list[str]
    - target_name: str
    - train_rows: list[dict]
    - test_rows: list[dict]
    - model: object
    - training_summary: dict
    """
    dataset_name = processed_bundle["dataset_name"]
    feature_names = list(processed_bundle["feature_names"])
    target_name = processed_bundle["target_name"]
    train_rows = processed_bundle["train_rows"]
    test_rows = processed_bundle["test_rows"]

    frame = pd.DataFrame(train_rows)
    features = frame[feature_names]
    target = frame[target_name].astype(str)

    numeric_cols, categorical_cols = _split_feature_types(train_rows, feature_names)

    # One-hot encode the categories, scale the numbers, then feed both into one model.
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    n_classes = int(target.nunique())
    if n_classes < 2:
        # Logistic regression needs at least two classes; fall back gracefully.
        estimator: Any = DummyClassifier(strategy="most_frequent")
        model_kind = "dummy_most_frequent"
    else:
        estimator = LogisticRegression(max_iter=1000)
        model_kind = "logistic_regression"

    pipeline = Pipeline([("prep", transformer), ("model", estimator)])
    pipeline.fit(features, target)

    train_predictions = pipeline.predict(features)
    train_accuracy = float((train_predictions == target.to_numpy()).mean())
    classes = sorted(target.unique().tolist())

    model = WaterNeedModel(pipeline, feature_names, target_name, classes)

    training_summary = {
        "model_kind": model_kind,
        "n_train_rows": int(len(train_rows)),
        "n_features": len(feature_names),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "classes": classes,
        "train_accuracy": round(train_accuracy, 4),
    }

    return {
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "target_name": target_name,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "model": model,
        "training_summary": training_summary,
    }
