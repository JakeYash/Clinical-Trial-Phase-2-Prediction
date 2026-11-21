import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


class TestBaselineModel(unittest.TestCase):

    # Create temporary feature matrices for testing
    def run_baseline(self, include_sentiment):

        with tempfile.TemporaryDirectory() as temp_dir:

            root = Path(temp_dir)
            data_folder = root / "data/processed/model_ready"
            data_folder.mkdir(parents=True)

            # Create sample training and testing features
            X_train = pd.DataFrame({
                "trial_feature": [-3, -2, -1, 1, 2, 3]
            })

            X_test = pd.DataFrame({
                "trial_feature": [-2, 2]
            })

            # Add sentiment features when required
            if include_sentiment:
                X_train["pubmed_sentiment_mean_score"] = [
                    -0.8, -0.6, -0.4, 0.4, 0.6, 0.8
                ]

                X_test["pubmed_sentiment_mean_score"] = [
                    -0.5, 0.5
                ]

            # Create success/failure labels
            y_train = pd.DataFrame({
                "label": [0, 0, 0, 1, 1, 1]
            })

            y_test = pd.DataFrame({
                "label": [0, 1]
            })

            # Save temporary feature matrices
            X_train.to_csv(data_folder / "X_train.csv", index=False)
            X_test.to_csv(data_folder / "X_test.csv", index=False)
            y_train.to_csv(data_folder / "y_train.csv", index=False)
            y_test.to_csv(data_folder / "y_test.csv", index=False)

            # Locate and run the baseline model script
            script = Path(__file__).resolve().parents[1] / "baseline_model.py"

            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                capture_output=True,
                text=True
            )

            # Check that the script ran successfully
            self.assertEqual(
                completed.returncode, 0, completed.stderr
            )

            # Check that the results file was generated
            results_file = (
                root / "outputs/baseline_model/baseline_model_results.csv"
            )

            self.assertTrue(results_file.exists())

            return pd.read_csv(results_file)


    # Test the model when sentiment is unavailable
    def test_without_sentiment(self):

        results = self.run_baseline(False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["Features"], 1)

        self.assertEqual(
            results.iloc[0]["Model"],
            "Logistic Regression Without Sentiment"
        )


    # Test both models when sentiment is available
    def test_with_sentiment(self):

        results = self.run_baseline(True)

        self.assertEqual(len(results), 2)
        self.assertEqual(results.iloc[0]["Features"], 1)
        self.assertEqual(results.iloc[1]["Features"], 2)

        self.assertEqual(
            results.iloc[1]["Model"],
            "Logistic Regression With Sentiment"
        )


    # Test evaluation metrics
    def test_evaluation_metrics(self):

        results = self.run_baseline(True)

        metrics = [
            "Accuracy", "Precision", "Recall",
            "F1-score", "ROC-AUC"
        ]

        for metric in metrics:

            self.assertTrue(metric in results.columns)

            for value in results[metric]:
                self.assertAlmostEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
    