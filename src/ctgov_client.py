"""
Thin client around the ClinicalTrials.gov API v2 (https://clinicaltrials.gov/api/v2/studies).
Public and keyless -- no wget / manual download needed, everything here is programmatic.
"""
import time
import logging
from typing import Iterator, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger(__name__)


def _dig(d: dict, *path, default=None):
    """Safely walk a chain of nested dict keys; returns `default` on any miss."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True,
)
def _get(params: dict) -> dict:
    resp = requests.get(config.CTGOV_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def iter_raw_studies(
    query_cond: Optional[str] = None,
    query_intr: Optional[str] = None,
    phase_filter: Optional[set] = None,
    page_size: int = None,
) -> Iterator[dict]:
    """
    Yields raw (unflattened) study JSON objects from the CT.gov API v2,
    paging through results via the nextPageToken cursor.
    """
    page_size = page_size or config.CTGOV_PAGE_SIZE
    params = {"pageSize": page_size, "countTotal": "true"}
    if query_cond:
        params["query.cond"] = query_cond
    if query_intr:
        params["query.intr"] = query_intr
    if phase_filter:
        # Essie expression: AREA[Phase](PHASE1 OR PHASE2 ...)
        phase_expr = " OR ".join(sorted(phase_filter))
        params["filter.advanced"] = f"AREA[Phase]({phase_expr})"

    page_token = None
    total_seen = 0
    while True:
        if page_token:
            params["pageToken"] = page_token
        data = _get(params)

        studies = data.get("studies", [])
        for study in studies:
            yield study
        total_seen += len(studies)

        total_count = data.get("totalCount")
        if total_count is not None:
            logger.info("Fetched %d / %s studies", total_seen, total_count)

        page_token = data.get("nextPageToken") or data.get("pageToken")
        if not page_token or not studies:
            break
        time.sleep(config.CTGOV_REQUEST_DELAY_SEC)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True,
)
def get_study_by_nctid(nct_id: str) -> Optional[dict]:
    """
    Single-study lookup: GET https://clinicaltrials.gov/api/v2/studies/{NCT_ID}.
    Used to enrich an already-known trial (e.g. from the TOP benchmark) with
    start_date and design fields, without doing a broad search.
    Returns None (not raises) on a 404 -- some NCT IDs in older benchmark
    exports no longer resolve (withdrawn/merged/retracted records).
    """
    url = f"{config.CTGOV_BASE_URL}/{nct_id}"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def flatten_study(study: dict) -> dict:
    """Pull the fields we care about out of a raw nested CT.gov study record."""
    protocol = study.get("protocolSection", {})

    interventions = _dig(protocol, "armsInterventionsModule", "interventions", default=[]) or []
    drug_interventions = [
        iv.get("name") for iv in interventions if iv.get("type") in ("DRUG", "BIOLOGICAL")
    ]

    conditions = _dig(protocol, "conditionsModule", "conditions", default=[]) or []
    collaborators = _dig(protocol, "sponsorCollaboratorsModule", "collaborators", default=[]) or []
    locations = _dig(protocol, "contactsLocationsModule", "locations", default=[]) or []
    countries = sorted({loc.get("country") for loc in locations if loc.get("country")})
    primary_outcomes = _dig(protocol, "outcomesModule", "primaryOutcomes", default=[]) or []
    secondary_outcomes = _dig(protocol, "outcomesModule", "secondaryOutcomes", default=[]) or []

    return {
        "nct_id": _dig(protocol, "identificationModule", "nctId"),
        "brief_title": _dig(protocol, "identificationModule", "briefTitle"),
        "overall_status": _dig(protocol, "statusModule", "overallStatus"),
        "why_stopped": _dig(protocol, "statusModule", "whyStopped"),
        "start_date": _dig(protocol, "statusModule", "startDateStruct", "date"),
        "primary_completion_date": _dig(
            protocol, "statusModule", "primaryCompletionDateStruct", "date"
        ),
        "completion_date": _dig(protocol, "statusModule", "completionDateStruct", "date"),
        "phase": "|".join(_dig(protocol, "designModule", "phases", default=[]) or []),
        "study_type": _dig(protocol, "designModule", "studyType"),
        "allocation": _dig(protocol, "designModule", "designInfo", "allocation"),
        "intervention_model": _dig(protocol, "designModule", "designInfo", "interventionModel"),
        "masking": _dig(protocol, "designModule", "designInfo", "maskingInfo", "masking"),
        "primary_purpose": _dig(protocol, "designModule", "designInfo", "primaryPurpose"),
        "enrollment_count": _dig(protocol, "designModule", "enrollmentInfo", "count"),
        "enrollment_type": _dig(protocol, "designModule", "enrollmentInfo", "type"),
        "condition": "|".join(conditions),
        "intervention_name": "|".join([n for n in drug_interventions if n]),
        "n_arms": len(_dig(protocol, "armsInterventionsModule", "armGroups", default=[]) or []),
        "lead_sponsor_name": _dig(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "lead_sponsor_class": _dig(protocol, "sponsorCollaboratorsModule", "leadSponsor", "class"),
        "industry_collaborators": "|".join(
            c["name"] for c in collaborators
            if c.get("class") == "INDUSTRY" and c.get("name")
        ),
        "n_collaborators": len(collaborators),
        "n_primary_outcomes": len(primary_outcomes),
        "n_secondary_outcomes": len(secondary_outcomes),
        "min_age": _dig(protocol, "eligibilityModule", "minimumAge"),
        "max_age": _dig(protocol, "eligibilityModule", "maximumAge"),
        "sex": _dig(protocol, "eligibilityModule", "sex"),
        "healthy_volunteers": _dig(protocol, "eligibilityModule", "healthyVolunteers"),
        "eligibility_criteria_len": len(
            _dig(protocol, "eligibilityModule", "eligibilityCriteria", default="") or ""
        ),
        "n_countries": len(countries),
        "countries": "|".join(countries),
        "is_fda_regulated_drug": _dig(protocol, "oversightModule", "isFdaRegulatedDrug"),
        "fdaaa801_violation": _dig(protocol, "oversightModule", "fdaaa801Violation"),
        "has_results": study.get("hasResults", False),
    }

