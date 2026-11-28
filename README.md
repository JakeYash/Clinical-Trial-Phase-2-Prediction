# Phase 2 Trial Success — Data Collection Pipeline

Collects a public, pre-labeled phase 2 clinical trial dataset (12,494 trials,
from TrialBench) and enriches it with pre-trial PubMed discussion of each
drug. Run sentiment scoring after collecting the data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your email (and optional NCBI API key)
```

## Run

```bash
python3 pipeline.py --max-trials 50             # quick test run
python3 pipeline.py --condition "breast cancer"  # scope by disease text
python3 pipeline.py                               # full ~12,500-trial run
```

A full unscoped run does ~12,500 ClinicalTrials.gov lookups + ~12,500 PubMed
round trips — roughly two hours. Use `--max-trials` for a first test, or
`--condition` to scope to a therapeutic area.

## Pipeline steps

1. **Load TrialBench's phase 2 dataset** (`src/top_dataset.py`) — downloads
   `{train,test}_{x,y}.csv`, joins features to labels. 58 columns, including
   `label` (0=failure, 1=success), `diseases`, `drugs`, `icdcodes`,
   `smiless`, `criteria`, `brief_summary/textblock`.
2. **Enrich with ClinicalTrials.gov** (`src/ctgov_client.py`) — one lookup
   per `nct_id` to get `start_date` (needed to window the PubMed search) and
   a few extra design fields. A handful of old NCT IDs no longer resolve —
   dropped and logged.
3. **Filter/clean** (`src/filters.py`) — dedupe + drop rows missing an
   essential field.
4. **Collect PubMed discussion** (`src/pubmed_client.py`) — for each trial's
   primary drug (skipping placebo/comparator arms), pulls titles/abstracts
   dated before that trial's `start_date`. Drug names are resolved to PubChem's
   preferred synonym (catching brand names and alternate spellings); disease names
   are resolved to MeSH terms (falling back to text search if no entry is found).
   Uses a two-stage search: disease-specific articles (drug + indication) are
   prioritized, then backfilled with broader drug-only results. Articles with
   disclosed ties to the trial's industry sponsor or known manufacturers are
   filtered out and written to `data/raw/pubmed_excluded.jsonl` for transparency.
5. **Merge** (`src/merge.py`) — joins PubMed summary stats onto the trial
   table, writes `data/processed/trials_dataset.csv`.

## Output: `data/processed/trials_dataset.csv`

One row per phase 2 trial. Key columns:

| Column | Description |
|---|---|
| `nct_id` | Trial identifier |
| `label` | **Target.** 0 = failure, 1 = success |
| `top_split` | Original TrialBench train/test assignment |
| `overall_status`, `why_stopped`, `has_results` | Fresh CT.gov status fields |
| `start_date`, `primary_completion_date`, `completion_date` | Trial timeline |
| `diseases`, `icdcodes`, `condition` | Disease/indication |
| `drugs`, `smiless`, `intervention_name` | Drug name(s), SMILES |
| `criteria`, `brief_summary/textblock`, `brief_title` | Free text |
| `*Arm Number`, `*intervention Number` columns | Per-arm-type counts |
| `phase`, `study_type`, `study_design_info/allocation`, `MaskingType-*` | Design fields |
| `enrollment`, `enrollment_count`, `number_of_arms` | Size |
| `sponsors/lead_sponsor/agency_class`, `lead_sponsor_name` | Sponsor |
| `eligibility/*`, `min_age`, `max_age`, `sex` | Eligibility |
| `n_countries`, `countries` | Geographic spread |
| `oversight_info/is_fda_regulated_drug` | Regulatory |
| `pubmed_mention_count`, `pubmed_with_abstract_count`, `pubmed_pmids` | PubMed discussion volume |
| `pubmed_collection_status` | `collected`, `missing_or_invalid_start_date`, or `not_collected` |

Raw abstract text for sentiment scoring lives in
`data/raw/pubmed_abstracts.jsonl` (`{nct_id, pmid, title, abstract, journal,
pub_year}`), keyed back to `trials_dataset.csv` by `nct_id`.

## Feature engineering

After generating `trials_dataset.csv`, create model-ready matrices with:

```bash
python3 feature_engineering.py
```

The script preserves TrialBench's original `top_split`, fits all preprocessing
on the training rows only, and then applies the same preprocessing to the test
rows. It:

- removes outcome-leaking fields such as `overall_status`, `why_stopped`,
  `has_results`, and completion dates;
- converts `start_date` into time relative to January 1, 2000;
- converts minimum and maximum ages to numeric years;
- fills missing numerical values with the training median and standardizes
  numerical columns;
- drops columns that are constant (or entirely blank) in the training rows,
  since they cannot help a model; this uses training rows only, and the dropped
  columns are listed under `constant_or_empty_columns_removed` in the summary
  JSON; this can remove some sentiment columns on very small samples;
- fills missing categories and one-hot encodes categorical columns; and
- joins `data/processed/pubmed_sentiment_features.csv` (from
  `score_sentiment.py`) onto the trials by `nct_id` when that file exists, so the
  `pubmed_sentiment_*` columns are included. Use `--sentiment PATH` to point at
  a different file.

The scorer produces 8 sentiment feature columns: article count, mean polarity
(`mean_score`), mean negative/neutral/positive scores, and counts by top label.
They overlap (the means sum to 1, and the counts sum to the article count), so
`mean_score` and `article_count` are the most compact summary.

Output is written to `data/processed/model_ready/`:

- `X_train.csv` and `X_test.csv`: clean feature matrices;
- `y_train.csv` and `y_test.csv`: success/failure targets;
- `preprocessor.pkl`: fitted transformations for later models (see the note
  below); and
- `feature_engineering_summary.json`: columns and dimensions used.

> **Note:** `preprocessor.pkl` is a pickled scikit-learn object, so it can only be
> loaded reliably with the same scikit-learn (and pandas/numpy) versions that
> created it. Load it in the same environment, or regenerate it by re-running
> `python3 feature_engineering.py` after upgrading. Only load pickle files you
> created yourself.


## Score PubMed sentiment

Run the scorer against the matching outputs from a collection run:

```bash
pip install -r requirements-sentiment.txt
python3 score_sentiment.py
```

The scorer uses the [DeBERTa V3 zero-shot classifier](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33)
with a pinned checkpoint. The first run downloads the model (roughly 370 MB),
then inference runs locally on CPU. Use `--articles` and `--trials` to select a
different pair of saved inputs, and `--output-dir` to choose where results go.

Each retained PubMed article is scored using its title and abstract together,
or whichever field is available. For long articles, the scorer evaluates every
token chunk and averages the chunk scores by token count. This English model
provides zero-shot estimates of scientific tone. Its accuracy on this
project's PubMed corpus has yet to be measured against human labels.

The scorer writes two files to `data/processed/`:

- `pubmed_article_sentiment.jsonl` contains the top sentiment label, all three
  class scores, and a polarity score (positive minus negative, from −1 to 1)
  for each scored trial–PMID pair.
- `pubmed_sentiment_features.csv` contains one row per `nct_id`. Join it to
  `trials_dataset.csv` for the next feature engineering step. The
  `pubmed_sentiment_*` columns contain the scored article count, mean polarity,
  mean class scores, and counts by top label. Trials without scorable text
  have zero counts and blank means. Use `pubmed_collection_status` from the
  trial table to distinguish uncollected trials from collected trials.

## Baseline Logistic Regression Model

The baseline model predicts Phase 2 clinical trial success or failure
using logistic regression. Two versions of the model are evaluated:

- Logistic regression without PubMed sentiment features.
- Logistic regression with PubMed sentiment features.

### Run

After generating the model-ready feature matrices, run:

```bash
python3 baseline_model.py
```

The script reads the following files from
`data/processed/model_ready/`:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

Both models preserve TrialBench's original train/test split and use
the same logistic regression settings and clinical trial observations.
The model without sentiment excludes columns beginning with
`pubmed_sentiment_`.

If sentiment features are unavailable, the script evaluates only the
model without sentiment and displays a message explaining that the
second model was skipped.

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix

Precision, recall, and F1-score in the comparison table use clinical
trial success (label 1) as the positive class.

### Output

Model performance results are saved to:

`outputs/baseline_model/baseline_model_results.csv`

This file can be used for model comparisons and performance
visualizations.

Results generated from small test samples should be treated as
preliminary. A larger dataset is needed to reliably evaluate
predictive performance and the contribution of sentiment features.


## Random Forest comparison

After feature engineering, run:

```bash
python3 random_forest.py
```

This trains two Random Forest classifiers using the existing model-ready
train/test split: all available features, and the same features with
`pubmed_sentiment_*` columns removed. No preprocessing is refitted.
Other PubMed features, such as mention counts, remain in both models.
Hyperparameters were locked in after a tuning run (ROC-AUC optimised):
100 trees / max depth 10 without sentiment; 200 trees / max depth 10 with
sentiment. Use `--input-dir PATH` or `--output PATH` to override the defaults.

The report at `outputs/random_forest_sentiment_comparison.json` contains
accuracy, precision, recall, F1, ROC-AUC, confusion matrices, predictions,
success probabilities, feature lists, and hyperparameters. Confusion matrices
use rows for actual labels and columns for predictions, ordered `[0, 1]`.
Success (1) is the positive class; undefined precision/recall/F1 are set to 0.
ROC-AUC is JSON `null` with an explanation when the test set has one class.
Missing sentiment columns are reported and available columns are removed; if
none are present, both models use identical inputs. The generated report is
ignored by Git under the existing output rule.

Small sample results are only a smoke test and cannot establish whether
sentiment improves generalization. The comparison measures the incremental
contribution of sentiment alongside the other retained features.

## Tests

```bash
python3 -m unittest discover -s tests
```

## Possible Issues / Future Additions

1. **Entity resolution has coverage gaps** — Drug names are resolved via
   PubChem (catches brand names and alternate spellings) and disease names via
   MeSH, but both fall back to text search when no entry is found. Very new
   drugs may not yet be in PubChem; rare diseases may have no MeSH term.
   PubChem's preferred synonym is the first entry in its synonym list, which
   may not be the most common name in the literature.
2. **Manufacturer filtering strength depends on `DRUG_MANUFACTURERS` config** —
   By default only CT.gov's registered industry sponsors/collaborators are used to
   screen articles. Subsidiaries and historical company names won't be caught unless
   manually added to `DRUG_MANUFACTURERS` in `config.py`.
3. **Maybe add other forms of public sentiment**
   We could try looking at more informal sources like reddit maybe?

## Next Steps
1. **Start thinking about how we are going to analyze this data for the impact of the outside source articles**
