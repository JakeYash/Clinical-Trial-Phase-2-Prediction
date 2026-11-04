"""
Merges the filtered CT.gov trial table with PubMed discussion summary
features. Raw abstract text is kept in a separate JSONL file (one record per
PMID, linked by nct_id) rather than crammed into the CSV, since sentiment
scoring will need the full text later and abstracts can be long.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def build_pubmed_summary_features(pubmed_records_by_trial: dict) -> pd.DataFrame:
    """
    pubmed_records_by_trial: {nct_id: [ {pmid, title, abstract, journal, pub_year}, ... ]}
    Returns one summary row per nct_id. No sentiment is computed here -- only
    counts/metadata that are themselves useful control features (e.g. prior
    discussion volume correlates with how established a drug already is).
    """
    rows = []
    for nct_id, records in pubmed_records_by_trial.items():
        if records is None:
            rows.append({"nct_id": nct_id, "pubmed_mention_count": pd.NA,
                         "pubmed_with_abstract_count": pd.NA, "pubmed_pmids": "",
                         "pubmed_collection_status": "missing_or_invalid_start_date"})
            continue
        n_with_abstract = sum(1 for r in records if r.get("abstract"))
        rows.append(
            {
                "nct_id": nct_id,
                "pubmed_collection_status": "collected",
                "pubmed_mention_count": len(records),
                "pubmed_with_abstract_count": n_with_abstract,
                "pubmed_pmids": "|".join(r["pmid"] for r in records if r.get("pmid")),
            }
        )
    return pd.DataFrame(rows, columns=["nct_id", "pubmed_collection_status",
                                      "pubmed_mention_count", "pubmed_with_abstract_count", "pubmed_pmids"])


def merge_trials_with_pubmed(
    trials_df: pd.DataFrame, pubmed_summary_df: pd.DataFrame
) -> pd.DataFrame:
    merged = trials_df.merge(pubmed_summary_df, on="nct_id", how="left")
    merged["pubmed_mention_count"] = merged["pubmed_mention_count"].astype("Int64")
    merged["pubmed_with_abstract_count"] = (
        merged["pubmed_with_abstract_count"].astype("Int64")
    )
    merged["pubmed_collection_status"] = merged["pubmed_collection_status"].fillna("not_collected")
    merged["pubmed_pmids"] = merged["pubmed_pmids"].fillna("")
    logger.info("Merged trial table with PubMed features: %d rows", len(merged))
    return merged

