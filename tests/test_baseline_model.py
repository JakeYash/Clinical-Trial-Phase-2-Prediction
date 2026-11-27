import unittest
import tempfile
from pathlib import Path

import pandas as pd

from baseline_model import run


class TestBaselineModel(unittest.TestCase):

    def _write_data(self, root, include_sentiment):
        data_folder = root / "data/processed/model_ready"
        data_folder.mkdir(parents=True)

        X_train = pd.DataFrame({"trial_feature": [-3, -2, -1, 1, 2, 3]})
        X_test = pd.DataFrame({"trial_feature": [-2, 2]})

        if include_sentiment:
            X_train["pubmed_sentiment_mean_score"] = [-0.8, -0.6, -0.4, 0.4, 0.6, 0.8]
            X_test["pubmed_sentiment_mean_score"] = [-0.5, 0.5]

        X_train.to_csv(data_folder / "X_train.csv", index=False)
        X_test.to_csv(data_folder / "X_test.csv", index=False)
        pd.DataFrame({"label": [0, 0, 0, 1, 1, 1]}).to_csv(data_folder / "y_train.csv", index=False)
        pd.DataFrame({"label": [0, 1]}).to_csv(data_folder / "y_test.csv", index=False)
        return data_folder

    def run_baseline(self, include_sentiment):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_folder = self._write_data(root, include_sentiment)
            output = root / "outputs/baseline_model/baseline_model_results.csv"
            run(data_folder, output)
            return pd.read_csv(output)

    def test_without_sentiment(self):
        results = self.run_baseline(False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["Features"], 1)
        self.assertEqual(results.iloc[0]["Model"], "Logistic Regression Without Sentiment")
        self.assertIn("Best C", results.columns)
        self.assertIn("CV ROC-AUC", results.columns)
        self.assertIsNotNone(results.iloc[0]["Best C"])

    def test_with_sentiment(self):
        results = self.run_baseline(True)
        self.assertEqual(len(results), 2)
        self.assertEqual(results.iloc[0]["Features"], 1)
        self.assertEqual(results.iloc[1]["Features"], 2)
        self.assertEqual(results.iloc[1]["Model"], "Logistic Regression With Sentiment")

    def test_evaluation_metrics_present(self):
        results = self.run_baseline(True)
        for col in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Best C", "CV ROC-AUC"):
            self.assertIn(col, results.columns)

    def test_cv_roc_auc_is_none(self):
        results = self.run_baseline(False)
        import math
        self.assertTrue(math.isnan(results.iloc[0]["CV ROC-AUC"]))

    def test_roc_auc_in_valid_range(self):
        results = self.run_baseline(True)
        for _, row in results.iterrows():
            self.assertGreaterEqual(row["ROC-AUC"], 0.0)
            self.assertLessEqual(row["ROC-AUC"], 1.0)


if __name__ == "__main__":
    unittest.main()
