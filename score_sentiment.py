"""Generate article-level PubMed sentiment and features keyed by trial ID."""

import argparse
import csv
import hashlib
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

import config
from src.sentiment import LABELS, PubMedSentimentModel

FEATURE_COLUMNS = (
    "nct_id",
    "pubmed_sentiment_article_count",
    "pubmed_sentiment_mean_score",
    *(f"pubmed_sentiment_mean_{label}" for label in LABELS),
    *(f"pubmed_sentiment_{label}_count" for label in LABELS),
)


def iter_articles(path, trial_ids):
    """Yield a single row per trial and PMID from the retained Entrez records."""
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            nct_id = record["nct_id"]
            if nct_id not in trial_ids:
                raise ValueError(f"{path}:{line_number}: {nct_id} is absent from the trial CSV")
            fields = [(record.get(field) or "").strip() for field in ("title", "abstract")]
            article = "\n\n".join(filter(None, fields))
            if not article:
                continue
            pmid = record.get("pmid")
            text_key = hashlib.sha256(article.encode("utf-8")).digest()
            article_key = (nct_id, pmid or text_key)
            if article_key not in seen:
                seen.add(article_key)
                yield nct_id, pmid, text_key, article


def score_pubmed(articles_path, trials_path, output_dir, load_model):
    with trials_path.open(encoding="utf-8", newline="") as source:
        trials = csv.DictReader(source)
        if "nct_id" not in (trials.fieldnames or []):
            raise ValueError(f"{trials_path}: missing nct_id column")
        trial_ids = dict.fromkeys(trial["nct_id"] for trial in trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    articles = iter_articles(articles_path, trial_ids)
    model = None
    cached_scores = {}
    trial_totals = {}
    with TemporaryDirectory(dir=output_dir) as staging, tqdm(unit="article") as progress:
        article_output = Path(staging) / "pubmed_article_sentiment.jsonl"
        feature_output = Path(staging) / "pubmed_sentiment_features.csv"
        with article_output.open("w", encoding="utf-8") as sink:
            while batch := list(islice(articles, 32)):
                pending = {key: text for _, _, key, text in batch if key not in cached_scores}
                if pending:
                    if model is None:
                        model = load_model()
                    cached_scores.update(zip(pending, model.score(list(pending.values()))))

                for nct_id, pmid, key, _ in batch:
                    scores = cached_scores[key]
                    label_index = max(range(len(LABELS)), key=scores.__getitem__)
                    polarity = scores[2] - scores[0]
                    sink.write(json.dumps({
                        "nct_id": nct_id,
                        "pmid": pmid,
                        "sentiment_label": LABELS[label_index],
                        "sentiment_score": polarity,
                        "sentiment_scores": dict(zip(LABELS, scores)),
                    }) + "\n")
                    sums, counts = trial_totals.setdefault(nct_id, ([0.0] * 3, [0] * 3))
                    for index, score in enumerate(scores):
                        sums[index] += score
                    counts[label_index] += 1
                progress.update(len(batch))

        with feature_output.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(FEATURE_COLUMNS)
            for nct_id in trial_ids:
                sums, counts = trial_totals.get(nct_id, ([0.0] * 3, [0] * 3))
                total = sum(counts)
                means = [""] * (1 + len(LABELS))
                if total:
                    means = [(sums[2] - sums[0]) / total, *(value / total for value in sums)]
                writer.writerow([nct_id, total, *means, *counts])

        for path in (article_output, feature_output):
            target = path.replace(output_dir / path.name)
            print(f"Wrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=config.RAW_PUBMED_PATH)
    parser.add_argument("--trials", type=Path, default=config.FINAL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.DATA_PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, help="Local copy of the pinned model")
    args = parser.parse_args()
    score_pubmed(args.articles, args.trials, args.output_dir,
                 lambda: PubMedSentimentModel(args.model_path))


if __name__ == "__main__":
    main()
