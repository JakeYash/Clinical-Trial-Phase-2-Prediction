"""Compare Phase 2 success classifiers with and without PubMed sentiment.

Run ``python3 random_forest.py`` after feature_engineering.py. Inputs are
already preprocessed; this script preserves their row order and split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "processed" / "model_ready"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "random_forest_sentiment_comparison.json"
MODEL_PARAMS = {"n_estimators": 100, "random_state": 42}
SENTIMENT_COLUMNS = [
    f"pubmed_sentiment_{suffix}"
    for suffix in (
        "article_count", "mean_score", "mean_negative", "mean_neutral",
        "mean_positive", "negative_count", "neutral_count", "positive_count",
    )
]


def load_data(input_dir: Path) -> tuple:
    """Read and validate the existing model-ready split without preprocessing."""
    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")
    targets = []
    for split, X in (("train", X_train), ("test", X_test)):
        frame = pd.read_csv(input_dir / f"y_{split}.csv")
        if list(frame.columns) != ["label"]:
            raise ValueError(f"y_{split}.csv must contain exactly one 'label' column")
        y = frame["label"]
        if len(X) == 0 or len(X) != len(y):
            raise ValueError(f"{split}: features and targets must have matching nonzero rows")
        if not y.isin([0, 1]).all():
            raise ValueError(f"{split}: labels must be 0 (failure) or 1 (success)")
        if X.shape[1] == 0 or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise ValueError(f"{split}: features must be numeric and nonempty")
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(f"{split}: features must be finite with no missing values")
        targets.append(y.astype(int))
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns must match in name and order")
    return X_train, X_test, *targets


def evaluate(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """Use success as the positive class and a stable [failure, success] matrix."""
    predictions = model.predict(X)
    # A forest trained on one class returns only one probability column.
    if 1 in model.classes_:
        probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        probabilities = np.zeros(len(X))
    auc_defined = y.nunique() == 2
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if auc_defined else None,
        "roc_auc_note": None if auc_defined else "Undefined: test labels contain only one class.",
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions.tolist(),
        "success_probabilities": probabilities.tolist(),
    }


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the top_n most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(names, values)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved to: {output_path}")


def run(input_dir: Path = DEFAULT_INPUT_DIR, output_path: Path = DEFAULT_OUTPUT) -> dict:
    X_train, X_test, y_train, y_test = load_data(input_dir)
    sentiment = [c for c in X_train if c.startswith("pubmed_sentiment_")]
    missing = [c for c in SENTIMENT_COLUMNS if c not in X_train]
    notes = []
    if missing:
        notes.append(f"Missing sentiment columns (possibly removed as constant): {missing}")
    if not sentiment:
        notes.append("No sentiment features available; both models use identical features.")
    if y_train.nunique() == 1:
        notes.append("Training labels contain only one class; predictions are constant.")
    baseline = [c for c in X_train if c not in sentiment]
    if not baseline:
        raise ValueError("No features remain after removing PubMed sentiment")
    results = {}
    trained_models = {}
    for name, columns in (("with_sentiment", list(X_train.columns)),
                          ("without_sentiment", baseline)):
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train[columns], y_train)
        trained_models[name] = (model, columns)
        results[name] = {"feature_count": len(columns), "features": columns,
                         **evaluate(model, X_test[columns], y_test)}
    report = {
        "input_dir": str(input_dir.resolve()),
        "hyperparameters": MODEL_PARAMS.copy(),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "sentiment_columns_removed": sentiment,
        "missing_sentiment_columns": missing,
        "notes": notes,
        "metric_conventions": "Positive class: 1 (success); undefined precision/recall/F1: 0. Confusion matrix rows=true, columns=predicted, labels=[0, 1].",
        "test_labels": y_test.tolist(),
        "models": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(f"{name}: " + ", ".join(f"{key}={result[key]}" for key in
              ("accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix")))
    for note in notes:
        print(note)
    print(f"Comparison saved to: {output_path}")

    for name, (model, columns) in trained_models.items():
        plot_path = output_path.parent / f"random_forest_feature_importance_{name}.png"
        plot_feature_importance(model, columns, plot_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input_dir, args.output)
