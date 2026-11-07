# Phase 2 Trial Success — Data Collection Pipeline

Collects a public, pre-labeled phase 2 clinical trial dataset (12,494 trials,
from TrialBench) and enriches it with pre-trial PubMed discussion of each
drug. **This pipeline only collects and cleans data — no sentiment scoring
or modeling is performed here.**

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
   dated before that trial's `start_date`. Uses a two-stage search: disease-specific
   articles (drug + indication) are prioritized, then backfilled with broader drug-only results.
   Articles with disclosed ties to the trial's industry sponsor or known manufacturers
   are filtered out and written to `data/raw/pubmed_excluded.jsonl` for transparency.
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


## Tests

```bash
python3 -m unittest discover -s tests
```

## Possible Issues / Future Additions

1. **PubMed drug-name matching is free-text search**, not entity-linked —
   The search uses disease context to prioritize relevant articles, but matching
   is still text-based rather than using a controlled vocabulary (e.g. MeSH terms).
2. **Manufacturer filtering strength depends on `DRUG_MANUFACTURERS` config** —
   By default only CT.gov's registered industry sponsors/collaborators are used to
   screen articles. Subsidiaries and historical company names won't be caught unless
   manually added to `DRUG_MANUFACTURERS` in `config.py`.
3. **Maybe add other forms of public sentiment**
   We could try looking at more informal sources like reddit maybe?

## Next Steps
1. **Visualize the data**
   Just a little bit to actually parse those csvs and get an idea of the shape of our data. I would like if this was done in a separate py file. Also, getting the overall stats on the dataset is a must (number of points for example). 
3. **Start thinking about how we are going to analyze this data for the impact of the outside source articles**
