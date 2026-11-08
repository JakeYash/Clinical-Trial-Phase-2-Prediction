"""
End-to-end data COLLECTION pipeline for the MLDS phase-2-trial-success project.

This script only gathers and cleans data. It does not score sentiment, train
any model, or compute anything beyond the label already published by the
TOP benchmark -- those are left for later, per spec.

Pipeline:
    1. Download the TOP benchmark's labeled phase 2 trial CSVs (public,
       keyless, plain HTTPS -- see src/top_dataset.py for the label's
       provenance caveat).
    2. Enrich each trial with a fresh CT.gov API v2 lookup (mainly for
       start_date, which the TOP CSVs don't include and which we need to
       keep the PubMed search from leaking post-trial discussion).
    3. Filter/clean.
    4. Collect PubMed discussion of each trial's primary drug, dated before
       that trial's start date -- the novel "twist" feature for this project.
    5. Merge everything into one final CSV.

Usage:
    python pipeline.py --max-trials 300            # quick test run
    python pipeline.py --condition "breast cancer"  # scope by disease text
    python pipeline.py                               # full ~6,100-trial run

Outputs:
    data/raw/top_benchmark/top_phase_II_{split}.csv  - TOP CSVs as downloaded
    data/raw/phase2_trials_enriched.jsonl            - + CT.gov enrichment
    data/raw/pubmed_abstracts.jsonl                  - one record per (nct_id, pmid)
    data/processed/phase2_trials_filtered.csv        - after cleaning
    data/processed/trials_dataset.csv                - final model-ready table
"""
import argparse
import json
import logging


import pandas as pd
from tqdm import tqdm

import config
from src.top_dataset import load_phase2_labeled_trials, primary_drug_name
from src.ctgov_client import get_study_by_nctid, flatten_study
from src.pubmed_client import collect_drug_discussion, trial_organizations, _date_window, get_entity_resolution_stats
from src.filters import run_all_filters
from src.merge import build_pubmed_summary_features, merge_trials_with_pubmed

_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=_log_fmt)  # terminal: warnings+ only
_file_handler = logging.FileHandler(config.LOG_PATH, mode="w", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(_log_fmt))
logging.getLogger().addHandler(_file_handler)
logger = logging.getLogger("pipeline")


def step1_load_top_dataset(condition: str, max_trials: int) -> pd.DataFrame:
    logger.info("STEP 1: Downloading TOP benchmark's labeled phase 2 trials")
    df = load_phase2_labeled_trials(cache_dir=str(config.TOP_CACHE_DIR))

    if condition:
        mask = df["diseases"].astype(str).str.contains(condition, case=False, na=False)
        before = len(df)
        df = df[mask].reset_index(drop=True)
        logger.info("Scoped to condition '%s': %d / %d trials kept", condition, len(df), before)

    if max_trials and len(df) > max_trials:
        df = df.sample(n=max_trials, random_state=42).reset_index(drop=True)
        logger.info("Sampled down to --max-trials cap of %d", max_trials)

    return df


def step2_enrich_with_ctgov(df: pd.DataFrame) -> pd.DataFrame:
    """
    The TOP CSVs don't carry start_date (needed to window the PubMed search)
    or several design fields. We look each nct_id up individually via CT.gov's
    single-study endpoint -- cheap, one call per trial, no broad search.
    """
    logger.info("STEP 2: Enriching each trial with a fresh CT.gov API v2 lookup")
    enriched_rows = []
    n_missing = 0
    with open(config.RAW_ENRICHED_TRIALS_PATH, "w") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CT.gov enrichment"):
            raw = get_study_by_nctid(row["nct_id"])
            if raw is None:
                n_missing += 1
                continue
            flat = flatten_study(raw)
            f.write(json.dumps(flat) + "\n")
            enriched_rows.append(flat)

    if n_missing:
        logger.warning(
            "%d / %d NCT IDs from the TOP benchmark no longer resolve on CT.gov "
            "(older registry entries withdrawn/merged/retracted since the benchmark's "
            "Feb-2021 snapshot) -- dropped.",
            n_missing, len(df),
        )

    enriched_df = pd.DataFrame(enriched_rows)
    # Carry the TOP label, split, disease/drug/SMILES/criteria text, status
    # fields forward, preferring the fresh CT.gov values only where TOP had none.
    merged = df.merge(
        enriched_df, on="nct_id", how="inner", suffixes=("_top", "")
    )
    logger.info("Enriched %d / %d trials -> %s", len(merged), len(df), config.RAW_ENRICHED_TRIALS_PATH)
    return merged


def step3_filter(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("STEP 3: Filtering / cleaning trial table")
    filtered = run_all_filters(df)
    filtered.to_csv(config.FILTERED_TRIALS_PATH, index=False)
    logger.info("Filtered trial table -> %s (%d rows)", config.FILTERED_TRIALS_PATH, len(filtered))
    return filtered


def step4_collect_pubmed(df: pd.DataFrame) -> dict:
    logger.info("STEP 4: Collecting PubMed discussion prior to each trial's start date")
    if not config.NCBI_EMAIL:
        logger.warning(
            "NCBI_EMAIL is not set (see .env.example). Proceeding without it is against "
            "NCBI's usage policy and you may get throttled or blocked."
        )

    pubmed_by_trial = {}
    with open(config.RAW_PUBMED_PATH, "w") as f, open(config.RAW_PUBMED_EXCLUDED_PATH, "w") as rejected:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="pubmed collection"):
            nct_id = row["nct_id"]
            drug_name = primary_drug_name(row.get("drugs")) or (row.get("intervention_name") or "").split("|")[0].strip()
            if _date_window(row.get("start_date")) is None:
                pubmed_by_trial[nct_id] = None
                continue
            excluded = []
            records = collect_drug_discussion(
                drug_name,
                row.get("start_date"),
                row.get("diseases"),
                trial_organizations(row, drug_name),
                excluded,
            )
            pubmed_by_trial[nct_id] = records
            for rec in records:
                f.write(json.dumps({"nct_id": nct_id, **rec}) + "\n")
            for rec in excluded:
                rejected.write(json.dumps({"nct_id": nct_id, **rec}) + "\n")

    logger.info("PubMed abstracts -> %s", config.RAW_PUBMED_PATH)

    stats = get_entity_resolution_stats()
    drug_total = stats["drug_resolved"] + stats["drug_text_fallback"]
    disease_total = stats["disease_resolved"] + stats["disease_text_fallback"]
    print(
        f"\nEntity resolution summary (unique lookups, cached across trials):\n"
        f"  Drugs   : {stats['drug_resolved']}/{drug_total} resolved via PubChem"
        f", {stats['drug_text_fallback']}/{drug_total} fell back to text search\n"
        f"  Diseases: {stats['disease_resolved']}/{disease_total} resolved via MeSH"
        f", {stats['disease_text_fallback']}/{disease_total} fell back to text search\n"
        f"  Full details -> {config.LOG_PATH}"
    )

    return pubmed_by_trial


def step5_merge_and_save(trials_df: pd.DataFrame, pubmed_by_trial: dict) -> pd.DataFrame:
    logger.info("STEP 5: Merging trial table with PubMed summary features")
    pubmed_summary = build_pubmed_summary_features(pubmed_by_trial)
    final_df = merge_trials_with_pubmed(trials_df, pubmed_summary)
    final_df.to_csv(config.FINAL_DATASET_PATH, index=False)
    logger.info("Final dataset -> %s (%d rows, %d columns)",
                config.FINAL_DATASET_PATH, len(final_df), len(final_df.columns))
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Phase 2 clinical trial data collection pipeline")
    parser.add_argument(
        "--condition", default=None,
        help="Substring to filter the TOP dataset's 'diseases' text field (e.g. 'breast cancer')",
    )
    parser.add_argument("--max-trials", type=int, default=None, help="Cap on number of trials processed")
    args = parser.parse_args()

    top_df = step1_load_top_dataset(args.condition, args.max_trials)
    if top_df.empty:
        logger.error("No trials matched -- check --condition scope.")
        return

    enriched_df = step2_enrich_with_ctgov(top_df)
    if enriched_df.empty:
        logger.error("No trials survived CT.gov enrichment.")
        return

    filtered_df = step3_filter(enriched_df)
    if filtered_df.empty:
        logger.error("All trials were filtered out.")
        return

    pubmed_by_trial = step4_collect_pubmed(filtered_df)
    step5_merge_and_save(filtered_df, pubmed_by_trial)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
