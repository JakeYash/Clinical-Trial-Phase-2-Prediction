# Task 3: Baseline Logistic Regression Model
# Predict Phase 2 clinical trial success or failure

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "processed" / "model_ready"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "baseline_model" / "baseline_model_results.csv"


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    base = LogisticRegression(max_iter=1000)

    if y_train.nunique() > 1:
        n_splits = min(5, int(y_train.value_counts().min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = GridSearchCV(base, {"C": [0.01, 0.1, 1, 10, 100]},
                              scoring="roc_auc", cv=cv, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_c = search.best_params_["C"]
        cv_roc_auc = round(search.best_score_, 4)
    else:
        base.fit(X_train, y_train)
        model = base
        best_c = None
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

    return {
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

    results = []
    results.append(train_and_evaluate(
        X_train_no_sent, X_test_no_sent, y_train, y_test,
        "Logistic Regression Without Sentiment",
    ))

    if sentiment_columns:
        results.append(train_and_evaluate(
            X_train, X_test, y_train, y_test,
            "Logistic Regression With Sentiment",
        ))
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

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline logistic regression model")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input_dir, args.output)