import unittest

import numpy as np
import pandas as pd

from feature_engineering import create_feature_matrices


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


if __name__ == "__main__":
    unittest.main()
