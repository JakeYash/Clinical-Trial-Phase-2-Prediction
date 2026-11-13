import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from score_sentiment import score_pubmed


class TestSentimentOutput(unittest.TestCase):
    def test_scores_cached_across_trials_and_missing_text_stays_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials.csv"
            trials.write_text("nct_id,label\nA,0\nB,1\nC,1\n", encoding="utf-8")
            articles = root / "articles.jsonl"
            shared = {"pmid": "1", "title": "Study title", "abstract": "Study results"}
            articles.write_text("\n".join(map(json.dumps, [
                {"nct_id": "A", **shared},
                {"nct_id": "A", **shared},
                {"nct_id": "B", **shared},
                {"nct_id": "B", "pmid": "2", "abstract": "  "},
                {"nct_id": "B", "pmid": "3", "title": "Title only"},
            ])), encoding="utf-8")
            model = Mock()
            model.score.return_value = [(0.1, 0.2, 0.7), (0.9, 0.05, 0.05)]

            score_pubmed(articles, trials, root, lambda: model)

            model.score.assert_called_once_with(["Study title\n\nStudy results", "Title only"])
            records = [
                json.loads(line)
                for line in (root / "pubmed_article_sentiment.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["sentiment_label"] for row in records],
                             ["positive", "positive", "negative"])
            with (root / "pubmed_sentiment_features.csv").open(newline="") as result:
                first, second, empty = csv.DictReader(result)
            self.assertAlmostEqual(float(first["pubmed_sentiment_mean_score"]), 0.6)
            self.assertAlmostEqual(float(second["pubmed_sentiment_mean_score"]), -0.125)
            self.assertEqual(second["pubmed_sentiment_article_count"], "2")
            self.assertEqual(empty["pubmed_sentiment_article_count"], "0")
            self.assertEqual(empty["pubmed_sentiment_mean_score"], "")

    def test_empty_article_file_does_not_load_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials.csv"
            trials.write_text("nct_id\nA\n", encoding="utf-8")
            articles = root / "articles.jsonl"
            articles.touch()
            load_model = Mock()

            score_pubmed(articles, trials, root, load_model)

            load_model.assert_not_called()
            self.assertEqual((root / "pubmed_article_sentiment.jsonl").read_bytes(), b"")
