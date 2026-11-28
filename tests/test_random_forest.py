import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from random_forest import SENTIMENT_COLUMNS, MODEL_PARAMS, evaluate, load_data, run


class RandomForestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.X = pd.DataFrame({"enrollment_count": [10, 20, 30, 40, 50, 60]})
        for column in SENTIMENT_COLUMNS:
            self.X[column] = [0, 1, 0, 1, 0, 1]
        self.write_data(self.X)

    def write_data(self, X, train_labels=None, test_labels=None):
        X.to_csv(self.root / "X_train.csv", index=False)
        # Held-out values differ from every training row, including when the
        # caller retains only the baseline feature.
        X_test = X.iloc[:2].copy()
        if "enrollment_count" in X_test:
            X_test["enrollment_count"] = [15, 55]
        else:
            X_test.iloc[:, 0] = [15, 55]
        X_test.to_csv(self.root / "X_test.csv", index=False)
        pd.DataFrame({"label": train_labels or [0, 1, 0, 1, 0, 1]}).to_csv(
            self.root / "y_train.csv", index=False)
        pd.DataFrame({"label": test_labels or [0, 1]}).to_csv(
            self.root / "y_test.csv", index=False)

    def test_comparison_saved_and_reproducible(self):
        output = self.root / "out" / "comparison.json"
        fast_params = {k: {**v, "n_estimators": 10} for k, v in MODEL_PARAMS.items()}
        with patch("random_forest.MODEL_PARAMS", fast_params):
            report = run(self.root, output)
        self.assertEqual(report, json.loads(output.read_text()))
        with patch("random_forest.MODEL_PARAMS", fast_params):
            self.assertEqual(report, run(self.root, output))
        self.assertEqual(report["sentiment_columns_removed"], SENTIMENT_COLUMNS)
        self.assertEqual(report["train_rows"], 6)
        self.assertEqual(report["test_rows"], 2)
        self.assertEqual(report["test_labels"], [0, 1])
        for name, result in report["models"].items():
            self.assertEqual(len(result["predictions"]), 2)
            self.assertIn("model_params", result)
        self.assertEqual(report["models"]["with_sentiment"]["feature_count"], len(self.X.columns))
        self.assertEqual(report["models"]["without_sentiment"]["feature_count"], 1)

    def test_evaluation_with_asymmetric_errors_and_probability_auc(self):
        X = pd.DataFrame({"enrollment_count": [11, 21, 31, 41, 51, 61]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.6, 0.9, 0.4, 0.7])
        model = Mock()
        model.classes_ = np.array([0, 1])
        model.predict.return_value = np.array([0, 1, 1, 1, 0, 1])
        model.predict_proba.return_value = np.column_stack(
            (1 - probabilities, probabilities)
        )

        result = evaluate(model, X, y)

        # TN=1, FP=2, FN=1, TP=2; six of nine positive/negative
        # probability pairs are ranked correctly. Hard-label AUC would be 1/2.
        self.assertEqual(result["confusion_matrix"], [[1, 2], [1, 2]])
        self.assertAlmostEqual(result["accuracy"], 1 / 2)
        self.assertAlmostEqual(result["precision"], 1 / 2)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1_score"], 4 / 7)
        self.assertAlmostEqual(result["roc_auc"], 2 / 3)
        self.assertIsNone(result["roc_auc_note"])
        self.assertEqual(result["success_probabilities"], probabilities.tolist())
        model.predict.assert_called_once_with(X)
        model.predict_proba.assert_called_once_with(X)

    def test_single_class_test_with_two_class_training(self):
        for label in (0, 1):
            with self.subTest(label=label):
                self.write_data(self.X, test_labels=[label, label])
                report = run(self.root, self.root / "result.json")
                self.assertEqual(report["train_class_counts"], {"0": 3, "1": 3})
                for result in report["models"].values():
                    self.assertIsNone(result["roc_auc"])
                    self.assertEqual(
                        result["roc_auc_note"],
                        "Undefined: test labels contain only one class.",
                    )
                    self.assertEqual(np.shape(result["confusion_matrix"]), (2, 2))
                    self.assertEqual(sum(map(sum, result["confusion_matrix"])), 2)
                self.assertEqual(
                    json.loads((self.root / "result.json").read_text()), report
                )

    def test_partial_or_absent_sentiment(self):
        for columns in (["enrollment_count"], ["enrollment_count", SENTIMENT_COLUMNS[0]]):
            with self.subTest(columns=columns):
                self.write_data(self.X[columns])
                report = run(self.root, self.root / "result.json")
                self.assertEqual(report["models"]["without_sentiment"]["feature_count"], 1)
                self.assertTrue(report["notes"])
                self.assertEqual(len(report["missing_sentiment_columns"]), 9 - len(columns))
                if len(columns) == 1:
                    self.assertEqual(report["models"]["with_sentiment"]["features"],
                                     report["models"]["without_sentiment"]["features"])

    def test_single_class_train_and_test(self):
        for label in (0, 1):
            with self.subTest(label=label):
                self.write_data(self.X, [label] * 6, [label] * 2)
                report = run(self.root, self.root / "result.json")
                for result in report["models"].values():
                    self.assertIsNone(result["roc_auc"])
                    self.assertTrue(result["roc_auc_note"])
                    self.assertEqual(result["success_probabilities"], [float(label)] * 2)
                    self.assertEqual(result["confusion_matrix"],
                                     [[2, 0], [0, 0]] if label == 0 else [[0, 0], [0, 2]])
                    self.assertEqual(result["precision"], float(label))
                # A one-class training set still supports AUC on two-class test labels.
                model = RandomForestClassifier(random_state=42, n_estimators=10).fit(self.X, [label] * 6)
                result = evaluate(model, pd.read_csv(self.root / "X_test.csv"), pd.Series([0, 1]))
                self.assertEqual(result["roc_auc"], 0.5)

    def test_rejects_mismatched_features(self):
        self.X.iloc[:2, ::-1].to_csv(self.root / "X_test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "columns must match"):
            load_data(self.root)

    def test_rejects_invalid_labels_and_row_counts(self):
        for labels in ([0, 2], [0], [0, None]):
            with self.subTest(labels=labels):
                self.write_data(self.X, test_labels=labels)
                with self.assertRaises(ValueError):
                    load_data(self.root)

    def test_rejects_nonfinite_features(self):
        X = self.X.astype(float)
        X.loc[0, "enrollment_count"] = float("inf")
        self.write_data(X)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_data(self.root)

    def test_rejects_empty_baseline(self):
        self.write_data(self.X[SENTIMENT_COLUMNS])
        with self.assertRaisesRegex(ValueError, "No features remain"):
            run(self.root, self.root / "result.json")


if __name__ == "__main__":
    unittest.main()
