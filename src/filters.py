"""
Filtering and cleaning rules for the enriched phase 2 trial table.
Most heavy filtering (phase, resolved-status, dedup against IQVIA labels) is
already baked into the TOP benchmark's CSVs -- what's left here is just
making sure CT.gov enrichment actually succeeded for a row before it goes
into the final dataset, since a handful of older NCT IDs in a benchmark
built from a Feb-2021 ClinicalTrials.gov snapshot no longer resolve via the
current API (withdrawn/merged/retracted registry entries).
"""
import logging
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _log_drop(df_before: pd.DataFrame, df_after: pd.DataFrame, reason: str):
    dropped = len(df_before) - len(df_after)
    logger.info("Filter [%s]: dropped %d / %d rows", reason, dropped, len(df_before))


def filter_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any field considered essential for modeling."""
    mask = pd.Series(True, index=df.index)
    for field in config.REQUIRED_NON_NULL_FIELDS:
        if field not in df.columns:
            logger.warning("Required field '%s' not present in dataframe", field)
            continue
        col = df[field]
        mask &= col.notna() & (col.astype(str).str.strip() != "")
    out = df[mask]
    _log_drop(df, out, "required fields present")
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset=["nct_id"])
    _log_drop(df, out, "duplicate nct_id")
    return out


def run_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the full filter chain in order and returns the cleaned dataframe."""
    logger.info("Starting filter chain with %d rows", len(df))
    df = deduplicate(df)
    df = filter_required_fields(df)
    logger.info("Filter chain complete: %d rows remain", len(df))
    return df.reset_index(drop=True)
