"""
Loads the public, pre-labeled phase 2 trial dataset from **TrialBench's own
"trial approval forecasting" task** (Chen, Hu, Fu et al., Nature Scientific
Data 2025), published alongside the paper and redistributed as plain CSVs:

    Repo: https://github.com/ML2Health/ML2ClinicalTrials
    Path: Trialbench/data/trial-approval-forecasting/Phase2/

This supersedes the first version of this module, which used the smaller
TOP/HINT benchmark (~6,105 phase 2 trials). TrialBench's phase 2 approval
set has **12,494 trials** (9,980 train / 2,514 test) and far richer
pre-curated features (58 columns: MeSH terms, intervention-type counts,
masking broken out by role, brief summary text, SMILES, ICD-10 codes, etc.)
-- roughly double the data, for the same "just download some CSVs" effort.

WHY THIS INSTEAD OF A HAND-ROLLED PROXY LABEL
------------------------------------------------
IQVIA's own label data is commercial and not reachable here. TrialBench
redistributes a *processed, labeled* result in the open -- free, no login,
no git even required, just plain HTTPS downloads of four CSV files per
phase. We fetch those directly instead of rebuilding labels ourselves.

IMPORTANT PROVENANCE CAVEAT (please read before using this in a write-up)
----------------------------------------------------------------------------
Per the TrialBench paper's own Methods section, the "trial approval outcome"
label comes from two sources: (1) the HINT paper's benchmark, whose approval
labels were themselves sourced from TrialTrove (a commercial, IQVIA-adjacent
database), redistributed here for research use; and (2) ClinicalTrials.gov's
own "why stopped" field for trials terminated for business/safety/efficacy
reasons, which TrialBench added as negative samples. So part of this label
is public-data-derived and part is commercially-sourced-then-redistributed.
There is no LICENSE file in the repo stating redistribution terms, though it
is a newly published (2025), peer-reviewed (Nature Scientific Data) academic
benchmark intended for exactly this kind of reuse. Cite the TrialBench paper
if you use this, and check with your instructor on whether that provenance
is acceptable for your project.
"""
import logging
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger(__name__)

TRIALBENCH_BASE_URL = (
    "https://raw.githubusercontent.com/ML2Health/ML2ClinicalTrials/main/"
    "Trialbench/data/trial-approval-forecasting/Phase2"
)
TRIALBENCH_FILES = {
    "train": (f"{TRIALBENCH_BASE_URL}/train_x.csv", f"{TRIALBENCH_BASE_URL}/train_y.csv"),
    "test": (f"{TRIALBENCH_BASE_URL}/test_x.csv", f"{TRIALBENCH_BASE_URL}/test_y.csv"),
}


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True,
)
def _fetch_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    # index_col=0: the NCT ID is an unnamed index column in TrialBench's files
    return pd.read_csv(StringIO(resp.text), index_col=0)


def load_phase2_labeled_trials(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Downloads and combines TrialBench's phase 2 train/test x+y CSVs.
    Adds a `top_split` column (kept under this name for backward
    compatibility with the rest of the pipeline) so the original split is
    preserved for reproducible comparison against published numbers.

    label: 0 = failure, 1 = success (see provenance caveat above).
    """
    frames = []
    for split, (x_url, y_url) in TRIALBENCH_FILES.items():
        logger.info("Downloading TrialBench phase 2 %s features from %s", split, x_url)
        x_df = _fetch_csv(x_url)
        logger.info("Downloading TrialBench phase 2 %s labels from %s", split, y_url)
        y_df = _fetch_csv(y_url)

        df = x_df.join(y_df, how="inner")
        df = df.rename(columns={"outcome": "label"})
        df["top_split"] = split
        df = df.reset_index().rename(columns={"index": "nct_id"})
        frames.append(df)

        if cache_dir:
            import os

            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(os.path.join(cache_dir, f"trialbench_phase2_{split}.csv"), index=False)

    combined = pd.concat(frames, ignore_index=True)
    # TrialBench's own naming -> keep a couple of columns under the names
    # the rest of this pipeline (filters.py etc.) already expects.
    combined = combined.rename(
        columns={
            "condition": "diseases",
            "intervention/intervention_name": "drugs",
            "eligibility/criteria/textblock": "criteria",
            "icdcode": "icdcodes",
        }
    )

    before = len(combined)
    combined = combined.drop_duplicates(subset=["nct_id"])
    if len(combined) != before:
        logger.info("Dropped %d duplicate nct_id rows across splits", before - len(combined))

    logger.info(
        "Loaded %d labeled phase 2 trials (label=1 success: %d, label=0 failure: %d)",
        len(combined),
        int((combined["label"] == 1).sum()),
        int((combined["label"] == 0).sum()),
    )
    return combined


# Comparator/control arms that show up in the `drugs` list but aren't the
# investigational drug -- searching PubMed for "placebo" would be meaningless.
_NON_DRUG_TERMS = {
    "placebo", "standard of care", "best supportive care", "saline",
    "normal saline", "sham", "no intervention", "observation",
}


def primary_drug_name(drugs_field) -> Optional[str]:
    """
    TrialBench's `drugs` (intervention/intervention_name) column is a
    stringified Python list, e.g. "['placebo', 'drugA']", and sometimes
    repeats the same name per arm, e.g. "['Topotecan', 'Topotecan']".
    Returns the first *actual* drug name (skipping placebo/comparator arms)
    as plain text for use in a PubMed search.
    """
    if not drugs_field or not isinstance(drugs_field, str):
        return None
    import ast

    candidates = []
    try:
        parsed = ast.literal_eval(drugs_field)
        if isinstance(parsed, list):
            candidates = [str(x).strip() for x in parsed if x]
    except (ValueError, SyntaxError):
        cleaned = drugs_field.strip("[]").split(",")
        candidates = [c.strip(" '\"") for c in cleaned if c.strip(" '\"")]

    for name in candidates:
        if name.lower() not in _NON_DRUG_TERMS:
            return name
    return candidates[0] if candidates else None
