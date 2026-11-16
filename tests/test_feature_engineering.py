import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    create_feature_matrices,
    merge_sentiment_features,
    run,
)


class FeatureEngineeringTests(unittest.TestCase):
    def test_creates_clean_matrices_without_leakage(self):
        df = pd.DataFrame(
            {
                "nct_id": [f"NCT{i:08d}" for i in range(12)],
                "label": [0, 1] * 6,
                "top_split": ["train"] * 10 + ["test"] * 2,
                "overall_status": ["COMPLETED", "TERMINATED"] * 6,
                "why_stopped": [None, "failed"] * 6,
                "has_results": [True, False] * 6,
                "primary_completion_date": ["2021-01-01"] * 12,
                "completion_date": ["2022-01-01"] * 12,
                "start_date": ["2010-01-01", "2011-06-01"] * 6,
                "enrollment_count": [10, 20, np.nan, 40, 50, 60] * 2,
                "number_of_arms": [1, 2] * 6,
                "min_age": ["18 Years", "6 Months"] * 6,
                "max_age": ["65 Years", "70 Years", None] * 4,
                "allocation": ["RANDOMIZED", "NON_RANDOMIZED"] * 6,
                "lead_sponsor_class": ["INDUSTRY", "OTHER"] * 6,
                "average_sentiment": [0.1, -0.2, np.nan, 0.4, 0.2, -0.1] * 2,
                "constant": [1] * 12,
            }
        )

        X_train, X_test, y_train, y_test, summary, _ = (
            create_feature_matrices(df)
        )

        self.assertEqual(len(X_train), 10)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 10)
        self.assertEqual(len(y_test), 2)
        self.assertEqual(list(X_train.columns), list(X_test.columns))
        self.assertFalse(X_train.isna().any().any())
        self.assertFalse(X_test.isna().any().any())
        self.assertFalse(any("overall_status" in c for c in X_train.columns))
        self.assertFalse(any("why_stopped" in c for c in X_train.columns))
        self.assertFalse(any("completion_date" in c for c in X_train.columns))
        self.assertIn("start_days_since_2000", X_train.columns)
        self.assertIn("min_age_years", X_train.columns)
        self.assertIn("max_age_years", X_train.columns)
        self.assertIn("average_sentiment", summary["sentiment_columns_detected"])


SENTIMENT_COLUMNS = [
    "nct_id",
    "pubmed_sentiment_article_count",
    "pubmed_sentiment_mean_score",
    "pubmed_sentiment_mean_negative",
    "pubmed_sentiment_mean_neutral",
    "pubmed_sentiment_mean_positive",
    "pubmed_sentiment_negative_count",
    "pubmed_sentiment_neutral_count",
    "pubmed_sentiment_positive_count",
]


def _trials(n=12):
    return pd.DataFrame(
        {
            "nct_id": [f"NCT{i:08d}" for i in range(n)],
            "label": [0, 1] * (n // 2),
            "top_split": ["train"] * (n - 2) + ["test"] * 2,
            "enrollment_count": np.arange(n, dtype=float),
            "allocation": ["RANDOMIZED", "NON_RANDOMIZED"] * (n // 2),
        }
    )


def _sentiment(ids):
    rows = []
    for i, nct_id in enumerate(ids):
        if i % 3 == 0:  # no scorable articles: zero counts, blank means
            rows.append([nct_id, 0, None, None, None, None, 0, 0, 0])
        else:
            rows.append([nct_id, 2, 0.1 * i, 0.2 + 0.01 * i, 0.3 - 0.01 * i, 0.5, i % 2, 1, 1 + i % 3])
    return pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)


class SentimentMergeTests(unittest.TestCase):
    def test_merge_keeps_all_trials_and_adds_columns(self):
        trials = _trials()
        # Drop one trial from the sentiment file: it must stay, with NaNs.
        sentiment = _sentiment(trials["nct_id"]).iloc[1:]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            sentiment.to_csv(path, index=False)
            merged = merge_sentiment_features(trials, path)
        self.assertEqual(len(merged), len(trials))
        self.assertEqual(list(merged["nct_id"]), list(trials["nct_id"]))
        self.assertTrue(
            set(SENTIMENT_COLUMNS[1:]).issubset(merged.columns)
        )
        self.assertTrue(merged.loc[0, SENTIMENT_COLUMNS[1:]].isna().all())

    def test_duplicate_ids_rejected(self):
        trials = _trials()
        sentiment = pd.concat([_sentiment(trials["nct_id"])] * 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            sentiment.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                merge_sentiment_features(trials, path)

    def test_run_end_to_end_includes_sentiment(self):
        trials = _trials()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trials.to_csv(root / "trials.csv", index=False)
            _sentiment(trials["nct_id"]).to_csv(root / "s.csv", index=False)
            out = root / "out"
            run(root / "trials.csv", out, root / "s.csv")
            X_train = pd.read_csv(out / "X_train.csv")
            X_test = pd.read_csv(out / "X_test.csv")
            summary = json.loads(
                (out / "feature_engineering_summary.json").read_text()
            )
        self.assertIn("pubmed_sentiment_mean_score", X_train.columns)
        self.assertEqual(len(summary["sentiment_columns_detected"]), 7)
        self.assertEqual(list(X_train.columns), list(X_test.columns))
        self.assertFalse(X_train.isna().any().any())
        self.assertFalse(X_test.isna().any().any())

    def test_run_without_sentiment_file_still_works(self):
        trials = _trials()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trials.to_csv(root / "trials.csv", index=False)
            run(root / "trials.csv", root / "out", root / "missing.csv")
            X_train = pd.read_csv(root / "out" / "X_train.csv")
        self.assertFalse(
            any("sentiment" in c for c in X_train.columns)
        )


if __name__ == "__main__":
    unittest.main()
