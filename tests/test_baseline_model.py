import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baseline_model import run


class TestBaselineModel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "results.csv"

    def write_data(self, include_sentiment):
        X_train = pd.DataFrame({"trial_feature": [-3, -2, -1, 1, 2, 3]})
        X_test = pd.DataFrame({"trial_feature": [-2, 2]})
        if include_sentiment:
            X_train["pubmed_sentiment_mean_score"] = [-0.8, -0.6, -0.4, 0.4, 0.6, 0.8]
            X_test["pubmed_sentiment_mean_score"] = [-0.5, 0.5]
        X_train.to_csv(self.root / "X_train.csv", index=False)
        X_test.to_csv(self.root / "X_test.csv", index=False)
        pd.DataFrame({"label": [0, 0, 0, 1, 1, 1]}).to_csv(self.root / "y_train.csv", index=False)
        pd.DataFrame({"label": [0, 1]}).to_csv(self.root / "y_test.csv", index=False)

    def test_without_sentiment(self):
        self.write_data(False)
        results = run(self.root, self.output)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["Features"], 1)
        self.assertEqual(results.iloc[0]["Model"], "Logistic Regression Without Sentiment")

    def test_with_sentiment(self):
        self.write_data(True)
        results = run(self.root, self.output)
        self.assertEqual(len(results), 2)
        self.assertEqual(results.iloc[0]["Features"], 1)
        self.assertEqual(results.iloc[1]["Features"], 2)
        self.assertEqual(results.iloc[1]["Model"], "Logistic Regression With Sentiment")

    def test_evaluation_metrics(self):
        self.write_data(True)
        results = run(self.root, self.output)
        for metric in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"):
            self.assertIn(metric, results.columns)
            for value in results[metric]:
                self.assertAlmostEqual(value, 1.0)

    def test_results_saved_to_file(self):
        self.write_data(False)
        run(self.root, self.output)
        self.assertTrue(self.output.exists())
        saved = pd.read_csv(self.output)
        self.assertEqual(len(saved), 1)


if __name__ == "__main__":
    unittest.main()
    