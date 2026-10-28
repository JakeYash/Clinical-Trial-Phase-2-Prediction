"""
Central configuration for the clinical trial data collection pipeline.
Adjust the constants below to change scope without touching pipeline logic.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TOP_CACHE_DIR = DATA_RAW_DIR / "top_benchmark"
RAW_ENRICHED_TRIALS_PATH = DATA_RAW_DIR / "phase2_trials_enriched.jsonl"
RAW_PUBMED_PATH = DATA_RAW_DIR / "pubmed_abstracts.jsonl"
FILTERED_TRIALS_PATH = DATA_PROCESSED_DIR / "phase2_trials_filtered.csv"
FINAL_DATASET_PATH = DATA_PROCESSED_DIR / "trials_dataset.csv"

# --- ClinicalTrials.gov API v2 ------------------------------------------
# Public, keyless REST API: https://clinicaltrials.gov/api/v2/studies
CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_REQUEST_DELAY_SEC = 0.2  # be polite; no published hard rate limit but avoid hammering

# --- Filtering: minimum completeness after CT.gov enrichment -------------
REQUIRED_NON_NULL_FIELDS = [
    "nct_id",
    "label",
    "start_date",
    "condition",
    "intervention_name",
]

# --- PubMed / NCBI Entrez -----------------------------------------------
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENTREZ_REQUEST_DELAY_SEC = 0.11 if NCBI_API_KEY else 0.35  # 10/sec with key, 3/sec without
PUBMED_RETMAX_PER_DRUG = 200          # cap per intervention to keep volume sane
PUBMED_LOOKBACK_DAYS = 3 * 365        # only discussion in the N days before phase 2 start
