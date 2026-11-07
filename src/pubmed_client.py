"""
NCBI Entrez client for pulling non-official PubMed discussion about a drug
prior to its phase 2 trial start date. This only COLLECTS titles/abstracts;
no sentiment scoring happens here (that's left for a later step, per spec).

Usage requires an identifying email (NCBI policy); an API key is optional
but raises the rate limit from 3 req/sec to 10 req/sec. Both are read from
environment variables via config.py / .env -- see .env.example.
"""
import logging
import re
import time
import ast
import unicodedata
from datetime import datetime, timedelta
from typing import List, Optional
from xml.etree import ElementTree as ET

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger(__name__)

if not config.NCBI_EMAIL:
    logger.warning(
        "NCBI_EMAIL is not set. NCBI requests you identify yourself for Entrez API "
        "use. Copy .env.example to .env and fill it in before running at scale."
    )


def _common_params() -> dict:
    params = {"email": config.NCBI_EMAIL, "tool": "mlds_phase2_pipeline"}
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    return params


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True,
)
def _get(endpoint: str, params: dict) -> requests.Response:
    url = f"{config.ENTREZ_BASE_URL}/{endpoint}"
    resp = requests.get(url, params={**_common_params(), **params}, timeout=30)
    resp.raise_for_status()
    return resp


def _date_window(before_date: Optional[str]) -> Optional[tuple]:
    """
    Returns (mindate, maxdate) in YYYY/MM/DD, ending strictly before `before_date`
    (phase 2 start, using the first day for month/year-only dates). Returns None
    when `before_date` is missing/unparseable -- NEVER falls back to today's date,
    since that would silently pull in discussion published during or after the trial
    (e.g. phase 4 articles) instead of skipping the trial.
    """
    if not isinstance(before_date, str):
        return None
    value = before_date.strip()
    # Use the earliest possible start for partial registry dates.
    formats = {4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d"}
    if not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        return None
    try:
        cutoff = datetime.strptime(value, formats[len(value)])
        end = cutoff - timedelta(days=1)  # Entrez maxdate is inclusive.
        start = cutoff - timedelta(days=config.PUBMED_LOOKBACK_DAYS)
    except (ValueError, OverflowError):
        return None
    return start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")


def parse_diseases(value):
    """Convert a disease name or list-like string into a list."""
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(d).strip() for d in value if str(d).strip()]

    if not isinstance(value, str) or not value.strip():
        return []

    value = value.strip()

    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(d).strip() for d in parsed if str(d).strip()]
        except (ValueError, SyntaxError):
            logger.warning("Could not parse diseases value: %r -- falling back to drug-only search", value)
        return []

    return [value]


def search_pmids(
    drug_name: str,
    before_date: Optional[str],
    disease: Optional[str] = None,
) -> List[str]:
    """Prioritize disease-specific articles, then add broader drug articles."""
    if not drug_name:
        return []

    window = _date_window(before_date)
    if window is None:
        logger.warning(
            "No valid start_date for drug=%s -- skipping PubMed search to avoid leakage",
            drug_name,
        )
        return []

    mindate, maxdate = window
    max_articles = config.PUBMED_RETMAX_PER_DRUG

    def run_search(query):
        params = {
            "db": "pubmed",
            "term": f"({query})",
            "retmax": max_articles,
            "retmode": "json",
            "datetype": "pdat",
            "mindate": mindate,
            "maxdate": maxdate,
        }
        resp = _get("esearch.fcgi", params)
        time.sleep(config.ENTREZ_REQUEST_DELAY_SEC)

        try:
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except ValueError:
            logger.warning("Non-JSON esearch response for query=%s", query)
            return []

    # Search for articles mentioning the drug
    drug_query = f'"{drug_name}"[Title/Abstract]'

    # Convert disease information into a list
    diseases = parse_diseases(disease)

    # If no disease is provided, use drug-only search
    if not diseases:
        return run_search(drug_query)

    # Stage 1: Drug + disease(s)
    disease_terms = [
        f'"{d}"[Title/Abstract]' for d in diseases
    ]

    disease_query = (
        f'{drug_query} AND ({" OR ".join(disease_terms)})'
    )

    # Do more general search only if necessary
    specific_pmids = run_search(disease_query)
    if len(specific_pmids) >= max_articles:
        return specific_pmids[:max_articles]
    general_pmids = run_search(drug_query)

    # Prioritize specific articles and remove duplicates
    combined_pmids = list(dict.fromkeys(specific_pmids + general_pmids))

    return combined_pmids[:max_articles]


def fetch_abstracts(pmids: List[str]) -> List[dict]:
    """efetch: returns [{pmid, title, abstract, journal, pub_year, affiliations,
    conflict_of_interest, funding_agencies}, ...] for given PMIDs."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = _get("efetch.fcgi", params)
    time.sleep(config.ENTREZ_REQUEST_DELAY_SEC)

    records = []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        logger.warning("Failed to parse efetch XML for %d pmids", len(pmids))
        return records

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//Abstract/AbstractText")
        journal_el = article.find(".//Journal/Title")
        pub_year_el = article.find(".//JournalIssue/PubDate/Year")

        abstract_text = " ".join(
            "".join(el.itertext()) for el in abstract_parts if el is not None
        ).strip()

        records.append(
            {
                "pmid": pmid_el.text if pmid_el is not None else None,
                "title": "".join(title_el.itertext()).strip() if title_el is not None else None,
                "abstract": abstract_text or None,
                "journal": journal_el.text if journal_el is not None else None,
                "pub_year": pub_year_el.text if pub_year_el is not None else None,
                "affiliations": ["".join(el.itertext()).strip()
                                 for el in article.findall(".//Affiliation")],
                "conflict_of_interest": " ".join(
                    "".join(el.itertext()).strip()
                    for el in article.findall(".//CoiStatement")),
                "funding_agencies": ["".join(el.itertext()).strip()
                                     for el in article.findall(".//Grant/Agency")],
            }
        )
    return records


def collect_drug_discussion(
    drug_name: str,
    before_date: Optional[str],
    disease: Optional[str] = None,
    organizations=(),
    excluded_records=None,
) -> List[dict]:
    """Full esearch -> efetch round trip for one drug.

    Uses disease context for two-stage search when provided, then filters out
    articles with disclosed ties to any of the given organizations.
    Excluded articles are appended to `excluded_records` if supplied.
    """
    pmids = search_pmids(drug_name, before_date, disease)

    all_records = []
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        all_records.extend(fetch_abstracts(batch))

    if not organizations:
        for record in all_records:
            record["industry_screening"] = "no_known_organizations"
        return all_records

    kept = []
    for record in all_records:
        matches = organization_matches(record, organizations)
        if matches:
            if excluded_records is not None:
                excluded_records.append({**record, "exclusion_matches": matches})
        else:
            record["industry_screening"] = "no_disclosed_match"
            kept.append(record)
    return kept


def _normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def organization_matches(record, organizations):
    """Match disclosed ties, never a company's mention in title/abstract alone.

    Whole-token matching avoids e.g. matching Merck to Merckx. Strip corporate
    suffixes so registry 'Acme, Inc.' also matches affiliation 'Acme'. Aliases
    and ownership relationships require a reviewed configuration entry.
    """
    fields = {"affiliation": record.get("affiliations", []),
              "funding": record.get("funding_agencies", []),
              "conflict_of_interest": [record.get("conflict_of_interest", "")]}
    matches = []
    for organization in sorted(set(organizations)):
        name = _normalize(organization)
        tokens = name.split()
        while tokens and tokens[-1] in {"inc", "incorporated", "ltd", "limited",
                                        "llc", "corp", "corporation", "plc"}:
            tokens.pop()
        name = " ".join(tokens)
        if not name:
            continue
        for field, values in fields.items():
            if any(" " + name + " " in " " + _normalize(v) + " " for v in values):
                matches.append({"organization": organization, "field": field})
    return matches


def trial_organizations(row, drug_name):
    """Industry sponsors/collaborators are proxies, supplemented by known makers."""
    names = list(config.DRUG_MANUFACTURERS.get(drug_name.lower(), []))
    if row.get("lead_sponsor_class") == "INDUSTRY":
        sponsor = row.get("lead_sponsor_name")
        if isinstance(sponsor, str) and sponsor.strip():
            names.append(sponsor.strip())
    collaborators = row.get("industry_collaborators")
    if isinstance(collaborators, str):
        names.extend(n.strip() for n in collaborators.split("|") if n.strip())
    return sorted(set(names))
