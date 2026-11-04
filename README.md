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
   dated strictly before that trial's `start_date`, excluding disclosed ties
   to industry sponsors/collaborators and configured manufacturers.
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

Raw abstract text for sentiment scoring lives in
`data/raw/pubmed_abstracts.jsonl` (`{nct_id, pmid, title, abstract, journal,
pub_year}`), keyed back to `trials_dataset.csv` by `nct_id`.


## Possible Issues / Future Additions

1. **PubMed drug-name matching is free-text search**, not entity-linked —
   This just means the data we pull is based on text in the article title instead of a field for the article.
2. **Maybe add other forms of public sentiment**
   We could try looking at more informal sources like reddit maybe?

## Next Steps
1. **Heavier filtering implemented** — see the screening and missing-date policy below.
2. **Visualize the data**
   Just a little bit to actually parse those csvs and get an idea of the shape of our data. I would like if this was done in a separate py file. Also, getting the overall stats on the dataset is a must (number of points for example). 
3. **Start thinking about how we are going to analyze this data for the impact of the outside source articles**



## Article screening and missing dates

The pipeline extracts author/investigator affiliations, grant agencies, and
conflict-of-interest statements from PubMed XML. Articles matching an industry
lead sponsor, industry collaborator, or a configured drug manufacturer in those
fields are excluded. A company appearing only in a title or abstract is not
sufficient to exclude an article. Matching ignores capitalization, punctuation,
and common corporate suffixes, and uses token boundaries.

Add reviewed manufacturer names and aliases in `config.py`, for example:

```python
DRUG_MANUFACTURERS = {
    "example drug": ["Example Pharma", "Example Pharma's former name"],
}
```

Use the primary drug name, lower-cased, as the key. Industry sponsors are only
proxies for manufacturers; this configuration is needed for known makers that
are not listed as sponsors/collaborators, including academic-led trials. No
manufacturer database or ownership/alias inference is built in.

Retained articles in `data/raw/pubmed_abstracts.jsonl` include affiliations,
funding agencies, conflict disclosures, and `industry_screening`:
`no_disclosed_match` or `no_known_organizations`. Neither proves independence:
PubMed disclosures may be missing/incomplete. Conservative disclosure matching
can also exclude statements that negate a relationship; review exclusions as
needed. Excluded articles and matching organization/field pairs are saved to
`data/raw/pubmed_excluded.jsonl`, linked by `nct_id` and `pmid`.

Trials missing a start date remain in the dataset. Their PubMed search is
skipped, `pubmed_collection_status` is `missing_or_invalid_start_date`, and
article counts are blank (unknown), **not zero**. Successful collections have
status `collected`; zero then means no retained articles from the capped search.
Downstream analysis should explicitly handle missing counts or restrict to
`collected` trials. No dates are inferred from completion dates or today's date.

Month-only and year-only start dates use the first day of that month/year as
the earliest possible start. The inclusive Entrez upper bound is one day before
that cutoff, excluding same-day publications. The existing three-year lookback
and 200-result cap remain. This is a publication-date window, not proof of when
an article first became publicly accessible.

Run the offline regression tests:

```bash
python -m unittest discover -s tests -v
```
