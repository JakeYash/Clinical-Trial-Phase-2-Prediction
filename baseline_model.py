# Task 3: Baseline Logistic Regression Model
# Predict Phase 2 clinical trial success or failure

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "processed" / "model_ready"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "baseline_model" / "baseline_model_results.csv"
# Best C per variant, locked in after tuning on 100 trials with sentiment (ROC-AUC optimised).
BEST_C = {"without_sentiment": 1, "with_sentiment": 100}


def plot_coefficients(model: LogisticRegression, feature_names: list, output_path: Path, top_n: int = 20) -> None:
    """Save a horizontal bar chart of the top_n coefficients by absolute value."""
    coefs = model.coef_[0]
    indices = np.argsort(np.abs(coefs))[-top_n:]
    names = [feature_names[i] for i in indices]
    values = coefs[indices]
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (positive = favours success)")
    ax.set_title(f"Logistic Regression — Top {top_n} Coefficients")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Coefficient plot saved to: {output_path}")


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, c):
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    best_c = c
    cv_roc_auc = None

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = float("nan")

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)
    print("Number of features:", X_train.shape[1])
    print("Best C:", best_c, "  CV ROC-AUC:", cv_roc_auc)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    return model, {
        "Model": model_name,
        "Features": X_train.shape[1],
        "Best C": best_c,
        "CV ROC-AUC": cv_roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
    }


def run(input_dir: Path = DEFAULT_INPUT_DIR, output_path: Path = DEFAULT_OUTPUT):
    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv")["label"]
    y_test = pd.read_csv(input_dir / "y_test.csv")["label"]

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    sentiment_columns = [c for c in X_train.columns if c.startswith("pubmed_sentiment_")]
    print("\nSentiment features detected:", len(sentiment_columns))

    X_train_no_sent = X_train.drop(columns=sentiment_columns)
    X_test_no_sent = X_test.drop(columns=sentiment_columns)

    trained_models = {}
    results = []

    model, row = train_and_evaluate(
        X_train_no_sent, X_test_no_sent, y_train, y_test,
        "Logistic Regression Without Sentiment",
        c=BEST_C["without_sentiment"],
    )
    trained_models["without_sentiment"] = (model, list(X_train_no_sent.columns))
    results.append(row)

    if sentiment_columns:
        model, row = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            "Logistic Regression With Sentiment",
            c=BEST_C["with_sentiment"],
        )
        trained_models["with_sentiment"] = (model, list(X_train.columns))
        results.append(row)
    else:
        print("\nNo sentiment features available. Skipping model with sentiment.")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION MODEL COMPARISON")
    print("=" * 50)
    print(results_df.round(4).to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print("\nModel results saved successfully!")
    print("Results saved in:", output_path)

    for name, (model, features) in trained_models.items():
        pkl_path = output_path.parent / f"logistic_regression_{name}.pkl"
        joblib.dump({"model": model, "features": features}, pkl_path)
        print(f"Model saved to: {pkl_path}")
        plot_path = output_path.parent / f"logistic_regression_coefficients_{name}.png"
        plot_coefficients(model, features, plot_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline logistic regression model")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input_dir, args.output)