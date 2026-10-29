#!/usr/bin/env python3
"""
AtlasNLP content-country re-extractor v2.

Purpose
-------
Rebuild represented-country metadata from paper evidence while preventing the
main failure modes found in the Stage-1 random audit.

Key safeguards
--------------
1. NEVER sends old AtlasNLP content-country or producer-country fields to the model.
2. Removes the author/affiliation header from page 1 when an Abstract marker exists.
3. Classifies target-dataset relationship before deciding eligibility.
4. Requires per-country evidence.
5. Uses a second, independent verification call only for candidate countries.
6. Final country sets are constructed deterministically in Python.
7. Reuse-only / ambiguous / target-not-found records are excluded from primary analysis.
8. Tracks token usage and estimated API cost per paper.

Default model is pinned for reproducibility:
    gpt-5.4-mini-2026-03-17

The script can reuse <paper_id>.pages.json caches created by the original AtlasNLP
metadata extractor. If no cache is available, use --allow_download to fetch the PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from pypdf import PdfReader


MODEL_DEFAULT = "gpt-5.4-mini-2026-03-17"
PROMPT_VERSION = "content-country-v2.2"

# Current GPT-5.4-mini standard API pricing as of 2026-08-26.
# Update these constants if pricing changes.
PRICE_INPUT_PER_M = 0.75
PRICE_CACHED_INPUT_PER_M = 0.075
PRICE_OUTPUT_PER_M = 4.50

REQUEST_TIMEOUT = 60
USER_AGENT = "atlasnlp-content-country-v2/2.0"

ELIGIBLE_RELATIONSHIPS = {
    "INTRODUCED_HERE",
    "EXTENDED_HERE",
    "NEW_COMPILATION_HERE",
}

EXCLUDED_RELATIONSHIPS = {
    "REUSED_EXISTING",
    "MULTIPLE_DATASETS_AMBIGUOUS",
    "TARGET_DATASET_NOT_FOUND",
}

VALID_EVIDENCE_TYPES = [
    # Explicit evidence
    "EXPLICIT_DATASET_PROVENANCE",
    "PARTICIPANT_LOCATION",
    "COUNTRY_SPECIFIC_SOURCE",
    "COUNTRY_SPLIT_TABLE",
    "REPRESENTED_POPULATION",
    "GEOGRAPHIC_COLLECTION_FILTER",
    "EXPLICIT_TARGET_POPULATION",
    # Indirect evidence
    "LANGUAGE_COMMUNITY",
    "DIALECT_GEOGRAPHY",
    "REGIONAL_SOURCE",
    "SPEECH_COMMUNITY",
    "OTHER_INDIRECT",
    # Invalid / prohibited evidence
    "AUTHOR_AFFILIATION",
    "FUNDING_LOCATION",
    "CONFERENCE_LOCATION",
    "CITATION_LOCATION",
    "INCIDENTAL_MENTION",
    "EXAMPLE_OR_QUESTION_CONTENT",
    "LANGUAGE_GEOGRAPHY_ONLY",
    "MODEL_PROVIDER_LOCATION",
    "ANNOTATOR_LOCATION_NOT_REPRESENTED_POPULATION",
    "OTHER_INVALID",
]

EXPLICIT_TYPES = {
    "EXPLICIT_DATASET_PROVENANCE",
    "PARTICIPANT_LOCATION",
    "COUNTRY_SPECIFIC_SOURCE",
    "COUNTRY_SPLIT_TABLE",
    "REPRESENTED_POPULATION",
    "GEOGRAPHIC_COLLECTION_FILTER",
    "EXPLICIT_TARGET_POPULATION",
}

INFERRED_TYPES = {
    "LANGUAGE_COMMUNITY",
    "DIALECT_GEOGRAPHY",
    "REGIONAL_SOURCE",
    "SPEECH_COMMUNITY",
    "OTHER_INDIRECT",
}

INVALID_TYPES = set(VALID_EVIDENCE_TYPES) - EXPLICIT_TYPES - INFERRED_TYPES

COUNTRY_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Russian Federation": "Russia",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "South Korea": "South Korea",
    "Korea": "South Korea",
    "Iran, Islamic Republic of": "Iran",
    "Viet Nam": "Vietnam",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d’Ivoire": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "State of Palestine": "Palestine",
    "Türkiye": "Turkey",
    "UAE": "United Arab Emirates",
}


EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dataset_match_status": {
            "type": "string",
            "enum": ["MATCHED", "POSSIBLE_MATCH", "NOT_FOUND", "AMBIGUOUS"],
        },
        "dataset_relationship": {
            "type": "string",
            "enum": [
                "INTRODUCED_HERE",
                "EXTENDED_HERE",
                "NEW_COMPILATION_HERE",
                "REUSED_EXISTING",
                "MULTIPLE_DATASETS_AMBIGUOUS",
                "TARGET_DATASET_NOT_FOUND",
            ],
        },
        "relationship_evidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pdf_page": {"type": ["integer", "null"]},
                "quote": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["pdf_page", "quote", "explanation"],
        },
        "candidate_country_assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "country": {"type": "string"},
                    "proposed_support_level": {
                        "type": "string",
                        "enum": ["EXPLICIT", "INFERRED", "INVALID"],
                    },
                    "evidence_type": {
                        "type": "string",
                        "enum": VALID_EVIDENCE_TYPES,
                    },
                    "pdf_page": {"type": "integer"},
                    "quote": {"type": "string"},
                    "context": {"type": "string"},
                    "dataset_tie": {"type": "string"},
                },
                "required": [
                    "country",
                    "proposed_support_level",
                    "evidence_type",
                    "pdf_page",
                    "quote",
                    "context",
                    "dataset_tie",
                ],
            },
        },
        "needs_human_review": {"type": "boolean"},
        "review_reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
        "notes": {"type": "string"},
    },
    "required": [
        "dataset_match_status",
        "dataset_relationship",
        "relationship_evidence",
        "candidate_country_assertions",
        "needs_human_review",
        "review_reasons",
        "notes",
    ],
}


VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_key": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["ACCEPT_EXPLICIT", "ACCEPT_INFERRED", "REJECT"],
                    },
                    "reason_code": {
                        "type": "string",
                        "enum": [
                            "DIRECT_PROVENANCE",
                            "DIRECT_PARTICIPANT_OR_POPULATION",
                            "DIRECT_COUNTRY_SOURCE",
                            "DIRECT_COUNTRY_TABLE_OR_SCOPE",
                            "INDIRECT_LANGUAGE_OR_COMMUNITY",
                            "INDIRECT_REGIONAL_SIGNAL",
                            "AUTHOR_AFFILIATION",
                            "FUNDING_OR_INSTITUTION",
                            "INCIDENTAL_MENTION",
                            "EXAMPLE_OR_QUESTION_CONTENT",
                            "LANGUAGE_GEOGRAPHY_ONLY",
                            "ANNOTATOR_OR_WORKER_LOCATION",
                            "REUSED_DATASET_PROVENANCE_UNRESOLVED",
                            "NOT_TIED_TO_TARGET_DATASET",
                            "INSUFFICIENT_EVIDENCE",
                            "OTHER",
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["candidate_key", "decision", "reason_code", "reason"],
            },
        },
        "needs_human_review": {"type": "boolean"},
        "review_reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["decisions", "needs_human_review", "review_reasons"],
}


EXTRACTION_INSTRUCTIONS = """\
You are the evidence-extraction stage of a scientific reliability correction pipeline.

Your task is narrow: for ONE TARGET DATASET named by the user, determine the
dataset's relationship to the current paper and find evidence about the geographic
population/content represented by that target dataset.

CRITICAL RULES
1. Do not guess. Abstention is preferred to unsupported attribution.
2. Ignore any previous knowledge about AtlasNLP. The user will not provide old
   AtlasNLP country labels or producer-country labels.
3. A country is represented only when evidence is tied to the TARGET DATASET's
   data provenance, collection population, represented population, source corpus,
   geographic scope, or explicit target population.
4. AUTHOR AFFILIATIONS ARE NEVER CONTENT-COUNTRY EVIDENCE. Neither are author
   institutions, email domains, funders, conference locations, model-provider
   locations, or the country of a cited paper.
5. A country appearing in an example, prompt, question, generated sentence,
   topic, biography, or other item CONTENT is not provenance by itself.
6. Language alone is never EXPLICIT country evidence. A specific language,
   dialect, or speech community may be recorded as INFERRED only when the
   geographic association is plausible and relevant to the dataset.
   LANGUAGE VS. REPRESENTED-POPULATION IDENTITY

    Distinguish language evidence from explicit population identity.

    A language name or native-speaker description alone does NOT establish a
    represented country.

    Examples:
    - "German native speakers" -> Germany is normally INFERRED, not EXPLICIT.
    - "French-speaking participants" -> France is not established.
    - "Arabic-language texts" -> no specific country is established.

    However, an explicit description of the PARTICIPANTS OR DATA POPULATION by
    nationality, national origin, or country-linked population identity can establish
    represented-country evidence.

    Examples:
    - "Vietnamese students" -> Vietnam may be EXPLICIT as REPRESENTED_POPULATION.
    - "Japanese participants" -> Japan may be EXPLICIT as REPRESENTED_POPULATION.
    - "participants from Kenya" -> Kenya may be EXPLICIT.
    - "Brazilian respondents" -> Brazil may be EXPLICIT.

    Do not confuse this with language:
    - "Vietnamese speakers" is ambiguous and should normally be INFERRED.
    - "Vietnamese students" identifies the population itself and may be EXPLICIT.

    When wording is ambiguous between language community and nationality/population,
    prefer INFERRED.
7. COUNTRY-SPECIFIC SOURCE EVIDENCE
    A source may support EXPLICIT represented-country attribution when BOTH conditions hold:

    a. The source itself is explicitly tied to a particular country rather than merely
    to a language; AND
    b. The source directly contributes the data contained in the TARGET DATASET.

    Examples that may qualify as EXPLICIT:
    - questions drawn from a named national examination system
    - records from an explicitly identified country's government or national archive
    - data collected from an explicitly country-specific institutional or public source
    - text gathered from media outlets that the paper explicitly identifies as
    country-specific sources

    Examples that do NOT qualify:
    - "Russian-language text" -> Russia is not established
    - "Spanish news" -> Spain is not established
    - an author's university is in Turkey -> Turkey is not established
    - a cited dataset was developed in Japan -> Japan is not established
    - a website happens to use a country-code domain unless the paper establishes
    that the source represents that country's data

    When country specificity is uncertain, use INFERRED or return no assertion.
8. Annotator/crowdworker location is not represented-population evidence unless
   those people themselves are the population whose responses/language/behavior
   constitute the dataset.
   COLLECTION LOCATION IS NOT AUTOMATICALLY REPRESENTED COUNTRY

    The physical place where participants were recruited, interviewed, tested, or
    where annotation/data collection occurred is distinct from the population the
    dataset represents.

    Do NOT assign a content country solely because data collection occurred there.

    Example:
    - "Vietnamese students studying in Australia were recruited..."
    -> Vietnam may be represented-country evidence from the explicitly Vietnamese
        population.
    -> Australia is collection location and must NOT be assigned as represented
        country unless the dataset is also explicitly framed as representing an
        Australian population.

    Participant location can support EXPLICIT represented-country attribution only
    when the location is constitutive of the population or geographic data source,
    for example:
    - participants explicitly recruited as residents of Kenya for a Kenya-focused study;
    - survey respondents sampled from the population of Mexico;
    - geographically filtered social-media posts originating in Nigeria.

    If location merely describes where an otherwise distinct population happened to
    be recruited or studied, do not treat it as content-country evidence.
9. Multi-country attribution is PER COUNTRY. Every country must have its own
   evidence item. Never expand a language or region into all countries where it
   might occur.
10. For synthetic data, assign geographic content only if the dataset explicitly
   targets a country/population/community. The generator/provider location is irrelevant.

11. TARGET DATASET IDENTITY CHECK: Before extracting any geographic evidence, verify that the TARGET DATASET NAME
    actually refers to a dataset, corpus, benchmark, data collection, annotation
    resource, or newly released dataset artifact in this paper.

    - A model, method, algorithm, system, framework, toolkit, metric, task,
    shared-task name, or experimental setup is NOT a dataset merely because the
    paper also uses or creates data.
    - If the supplied target name refers to a non-dataset artifact while the paper
    discusses a different dataset or an unnamed dataset, do NOT transfer that
    dataset's geography to the supplied target.
    - In that case return:
        dataset_match_status = "NOT_FOUND"
        dataset_relationship = "TARGET_DATASET_NOT_FOUND"
    - If you cannot determine whether the supplied target is the dataset itself,
    prefer AMBIGUOUS / MULTIPLE_DATASETS_AMBIGUOUS rather than guessing.

    Example:
    If TARGET DATASET NAME is the name of a language model, but the paper separately
    constructs an unnamed corpus, the model name must NOT inherit the corpus's
    represented country.
12. Distinguish reuse from construction:
    - REUSED_EXISTING: paper only uses/evaluates/analyzes a previously existing
      dataset and creates no new dataset artifact for this target.
    - EXTENDED_HERE: paper materially adds/releases new examples, annotations,
      languages, populations, splits, or a new version of the target dataset.
    - NEW_COMPILATION_HERE: paper constructs/releases a new dataset artifact by
      combining or transforming existing sources.
    - INTRODUCED_HERE: target dataset is newly introduced/collected/released here.
    - MULTIPLE_DATASETS_AMBIGUOUS: target cannot be unambiguously isolated.
    - TARGET_DATASET_NOT_FOUND: target dataset cannot be located in the paper.
13. Quotes must come from the provided paper text. Keep quote <= 25 words and
    context concise.
14. INVALID evidence may be logged, but it must be labeled INVALID and must not
    be treated as support.

SUPPORT LEVEL
- EXPLICIT: paper directly establishes represented geography for the target dataset.
- INFERRED: plausible indirect geographic association, but paper does not directly
  establish represented geography.
- INVALID: prohibited or irrelevant evidence.

Be conservative. Returning zero country assertions is a valid and often desirable result.
"""


VERIFIER_INSTRUCTIONS = """\
You are the independent evidence verifier in a scientific metadata correction pipeline.

You are given candidate geographic assertions for ONE TARGET DATASET. You did not
generate these candidates. Evaluate each assertion independently and conservatively.

Your decision for each candidate must be:
- ACCEPT_EXPLICIT: the supplied evidence directly establishes that the target
  dataset represents data/populations/sources/geographic scope from that country.
- ACCEPT_INFERRED: the evidence supports only a plausible indirect geographic
  association, such as a specific dialect/speech community, without direct country
  provenance.
- REJECT: the evidence is insufficient, prohibited, incidental, or not tied to the
  target dataset.

HARD REJECTION RULES
- Author affiliation, author institution, email domain, funder, conference, or
  model-provider location cannot establish represented country.
- Example/question/prompt/topic content cannot establish provenance by itself.
- Language identity alone cannot be ACCEPT_EXPLICIT.
- Distinguish "Vietnamese speakers" (linguistic evidence, normally INFERRED)
  from "Vietnamese students/participants/respondents" when the latter explicitly
  identifies the represented population (may be EXPLICIT).
- Annotator/crowdworker location is not content provenance unless the workers are
  themselves the represented population whose responses/language constitute the data.
- A reused dataset with unresolved upstream provenance must not inherit the current
  paper authors' geography.
- Every country must be justified separately. Do not infer a multi-country set from
  a region or language.
- Phrases such as "German native speakers", "French speakers", "Arabic speakers",
  or similar language/native-speaker descriptions do not establish residence,
  nationality, or collection country. Without separate geographic provenance,
  they can be at most ACCEPT_INFERRED.
- COUNTRY_SPECIFIC_SOURCE may be ACCEPT_EXPLICIT only when the source is explicitly
  country-bound and directly supplies data to the target dataset. A source described
  only by language is not country-specific provenance.
  REPRESENTED POPULATION VS. COLLECTION LOCATION

Evaluate represented country, not merely physical collection location.

- Explicit nationality, national-origin, or country-linked population identity
  may support ACCEPT_EXPLICIT when it describes the target dataset's represented
  participants/population.
- A recruitment, interview, testing, or study location alone does NOT establish
  represented country.
- If participants belong to one explicitly identified national population but
  are temporarily located in another country, the temporary/study location
  should not replace their represented-population country.

Example:
"Vietnamese students studying in Australia"
-> Vietnam can be ACCEPT_EXPLICIT as REPRESENTED_POPULATION.
-> Australia should be REJECTED as represented-country evidence unless the paper
   independently frames the dataset as representing an Australian population.

If evidence is borderline, downgrade EXPLICIT to INFERRED or REJECT.
Do not preserve the extractor's proposed support level out of deference.
Your job is to challenge the proposal independently, especially when the evidence
could reflect language geography, author geography, or incidental country mentions.
"""


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def canonical_paper_id(url: str) -> str:
    url = str(url).strip()
    url = re.sub(r"\.pdf$", "", url, flags=re.I)
    url = url.rstrip("/")
    return url.split("/")[-1]


def resolve_pdf_url(url: str) -> str:
    url = str(url).strip()
    if "aclanthology.org" in url and not url.lower().endswith(".pdf"):
        return url.rstrip("/") + ".pdf"
    return url


def fetch_pdf(url: str) -> bytes:
    r = requests.get(
        resolve_pdf_url(url),
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.content


def extract_text_pages_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return pages


def load_pages_json(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pages = obj.get("pages", obj if isinstance(obj, list) else [])
    if not isinstance(pages, list):
        raise ValueError(f"Invalid pages cache: {path}")
    return [str(x) for x in pages]


def find_cached_pages(cache_items_dir: Optional[Path], paper_id: str) -> Optional[Path]:
    if cache_items_dir is None:
        return None

    candidates = [
        cache_items_dir / f"{paper_id}.pages.json",
        cache_items_dir / paper_id / f"{paper_id}.pages.json",
        cache_items_dir / "items" / f"{paper_id}.pages.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def sanitize_first_page(page: str) -> Tuple[str, bool]:
    """
    Remove page-1 author/affiliation header when a clear Abstract marker exists.
    We intentionally do NOT use a risky fallback deletion when Abstract cannot be found.
    """
    m = re.search(r"(?im)^\s*abstract\s*$", page)
    if not m:
        m = re.search(r"(?i)\babstract\b", page[:12000])

    if m:
        return (
            "[PAGE-1 AUTHOR/AFFILIATION HEADER REMOVED]\n" + page[m.start():],
            True,
        )
    return page, False


def join_pages_for_model(pages: List[str], max_chars: int = 1_200_000) -> Tuple[str, bool, bool]:
    prepared: List[str] = []
    affiliation_sanitized = False

    for i, page in enumerate(pages, start=1):
        if i == 1:
            page, affiliation_sanitized = sanitize_first_page(page)
        prepared.append(f"=== PDF PAGE {i} ===\n{page}")

    full = "\n\n".join(prepared)
    truncated = False

    # Very large papers are rare. Do not silently exceed context.
    if len(full) > max_chars:
        truncated = True
        # Keep first 60% and last 40% so appendices are not automatically discarded.
        first_n = int(max_chars * 0.60)
        last_n = max_chars - first_n
        full = (
            full[:first_n]
            + "\n\n=== MIDDLE TEXT OMITTED DUE TO EXTREME LENGTH ===\n\n"
            + full[-last_n:]
        )

    return full, affiliation_sanitized or False, truncated


def normalize_country(country: str) -> str:
    c = re.sub(r"\s+", " ", str(country).strip())
    return COUNTRY_ALIASES.get(c, c)


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in values:
        x = normalize_country(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_usage_fields(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    inp_details = getattr(usage, "input_tokens_details", None)
    cached = int(getattr(inp_details, "cached_tokens", 0) or 0) if inp_details else 0

    out_details = getattr(usage, "output_tokens_details", None)
    reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0) if out_details else 0

    uncached = max(0, input_tokens - cached)
    cost = (
        uncached * PRICE_INPUT_PER_M
        + cached * PRICE_CACHED_INPUT_PER_M
        + output_tokens * PRICE_OUTPUT_PER_M
    ) / 1_000_000

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "estimated_cost_usd": cost,
    }


def call_structured(
    client: Any,
    *,
    model: str,
    instructions: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    reasoning_effort: str,
    max_output_tokens: int,
    prompt_cache_key: str,
    retries: int = 6,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_text,
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
                max_output_tokens=max_output_tokens,
                store=False,
                prompt_cache_key=prompt_cache_key,
            )

            if getattr(response, "status", None) not in (None, "completed"):
                raise RuntimeError(
                    f"Response status={getattr(response, 'status', None)} "
                    f"incomplete={getattr(response, 'incomplete_details', None)}"
                )

            raw = response.output_text
            obj = json.loads(raw)
            usage = get_usage_fields(response)
            response_id = str(getattr(response, "id", ""))
            return obj, usage, response_id

        except Exception as e:
            last_error = e
            if attempt == retries - 1:
                break
            backoff = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(backoff)

    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")


def build_extraction_user(
    record_id: str,
    paper_id: str,
    paper_url: str,
    dataset_name: str,
    dataset_link: str,
    paper_text: str,
) -> str:
    return f"""\
RECORD ID: {record_id}
PAPER ID: {paper_id}
PAPER URL: {paper_url}
TARGET DATASET NAME: {dataset_name}
TARGET DATASET LINK: {dataset_link or "Not provided"}

The target dataset name/link above are identifiers only. They are NOT geographic evidence.

PAPER TEXT WITH PDF PAGE MARKERS:
{paper_text}
"""


def make_candidate_key(i: int, candidate: Dict[str, Any]) -> str:
    # Opaque identifier only.
    # Country/page information is already included elsewhere in the payload.
    return f"C{i:03d}"


def build_verifier_user(
    record_id: str,
    dataset_name: str,
    relationship: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    keyed: List[Dict[str, Any]] = []

    for i, c in enumerate(candidates):
        item = dict(c)
        item["country"] = normalize_country(item.get("country", ""))
        item["candidate_key"] = make_candidate_key(i, item)
        keyed.append(item)

    payload = {
        "record_id": record_id,
        "target_dataset_name": dataset_name,
        "dataset_relationship": relationship,
        "candidates": keyed,
    }

    return (
        "Evaluate every candidate in the exact payload below. "
        "Return one decision for every candidate_key and no others.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2),
        keyed,
    )


def validate_verifier_keys(
    keyed_candidates: List[Dict[str, Any]],
    verification: Dict[str, Any],
) -> None:
    expected = {c["candidate_key"] for c in keyed_candidates}
    got = {d.get("candidate_key") for d in verification.get("decisions", [])}

    if expected != got:
        missing = expected - got
        extra = got - expected
        raise ValueError(f"Verifier candidate-key mismatch; missing={missing}, extra={extra}")


def deterministic_finalize(
    extraction: Dict[str, Any],
    verification: Optional[Dict[str, Any]],
    keyed_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    relationship = extraction["dataset_relationship"]
    eligible = (
        extraction["dataset_match_status"] == "MATCHED"
        and relationship in ELIGIBLE_RELATIONSHIPS
    )

    if not eligible:
        if relationship == "REUSED_EXISTING":
            status = "EXCLUDED_REUSE"
        elif relationship == "TARGET_DATASET_NOT_FOUND":
            status = "EXCLUDED_TARGET_NOT_FOUND"
        else:
            status = "EXCLUDED_AMBIGUOUS"
        return {
            "primary_eligible": False,
            "content_country_status": status,
            "explicit_content_countries": [],
            "inferred_content_countries": [],
            "accepted_assertions": [],
            "rejected_assertions": [],
        }

    if verification is None:
        return {
            "primary_eligible": True,
            "content_country_status": "UNATTRIBUTED",
            "explicit_content_countries": [],
            "inferred_content_countries": [],
            "accepted_assertions": [],
            "rejected_assertions": [],
        }

    by_key = {c["candidate_key"]: c for c in keyed_candidates}
    explicit: List[str] = []
    inferred: List[str] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for d in verification["decisions"]:
        key = d["candidate_key"]
        cand = by_key[key]
        merged = {**cand, **d}

        decision = d["decision"]
        country = normalize_country(cand["country"])

        # Defense in depth: invalid evidence types can never become explicit.
        evidence_type = cand.get("evidence_type")
        if evidence_type in INVALID_TYPES:
            decision = "REJECT"
            merged["decision"] = "REJECT"
            merged["reason_code"] = "INSUFFICIENT_EVIDENCE"
            merged["reason"] = (
                "Deterministic guard: extraction evidence type is prohibited/invalid."
            )

        # Defense in depth: language/community-only evidence can never become explicit.
        if evidence_type in INFERRED_TYPES and decision == "ACCEPT_EXPLICIT":
            decision = "ACCEPT_INFERRED"
            merged["decision"] = "ACCEPT_INFERRED"
            merged["reason"] = (
                "Deterministic downgrade: indirect language/community evidence "
                "cannot be explicit."
            )

        if decision == "ACCEPT_EXPLICIT":
            explicit.append(country)
            accepted.append(merged)
        elif decision == "ACCEPT_INFERRED":
            inferred.append(country)
            accepted.append(merged)
        else:
            rejected.append(merged)

    explicit = unique_preserve_order(explicit)
    inferred = [c for c in unique_preserve_order(inferred) if c not in set(explicit)]

    if explicit:
        status = "EXPLICIT"
    elif inferred:
        status = "INFERRED_ONLY"
    else:
        status = "UNATTRIBUTED"

    return {
        "primary_eligible": True,
        "content_country_status": status,
        "explicit_content_countries": explicit,
        "inferred_content_countries": inferred,
        "accepted_assertions": accepted,
        "rejected_assertions": rejected,
    }


def get_input_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    def pick(*names: str, required: bool = False) -> Optional[str]:
        for name in names:
            if name in df.columns:
                return name
        if required:
            raise ValueError(f"Missing required column; expected one of {names}")
        return None

    return {
        "record_id": pick("audit_record_id", "record_id", "source_row_index"),
        "paper_url": pick("paper_url", "Paper link", "paper_link", required=True),
        "dataset_name": pick("Dataset name", "dataset_name", required=True),
        "dataset_link": pick("Dataset Link (URL)", "Dataset link", "dataset_link"),
    }


def summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    meta = result["meta"]
    extraction = result["extraction"]
    final = result["final"]
    usage = result["usage"]

    return {
        "record_id": meta["record_id"],
        "paper_id": meta["paper_id"],
        "paper_url": meta["paper_url"],
        "dataset_name": meta["dataset_name"],
        "dataset_link": meta["dataset_link"],
        "dataset_match_status": extraction["dataset_match_status"],
        "dataset_relationship": extraction["dataset_relationship"],
        "primary_eligible": final["primary_eligible"],
        "content_country_status": final["content_country_status"],
        "explicit_content_countries": json.dumps(
            final["explicit_content_countries"], ensure_ascii=False
        ),
        "inferred_content_countries": json.dumps(
            final["inferred_content_countries"], ensure_ascii=False
        ),
        "n_explicit": len(final["explicit_content_countries"]),
        "n_inferred": len(final["inferred_content_countries"]),
        "extractor_needs_human_review": extraction["needs_human_review"],
        "verifier_needs_human_review": (
            result.get("verification", {}).get("needs_human_review", False)
            if result.get("verification")
            else False
        ),
        "affiliation_header_sanitized": meta["affiliation_header_sanitized"],
        "paper_text_truncated": meta["paper_text_truncated"],
        "elapsed_seconds": meta.get("elapsed_seconds", None),
        "extractor_input_tokens": usage["extractor"]["input_tokens"],
        "extractor_output_tokens": usage["extractor"]["output_tokens"],
        "extractor_reasoning_tokens": usage["extractor"]["reasoning_tokens"],
        "verifier_input_tokens": usage["verifier"]["input_tokens"],
        "verifier_output_tokens": usage["verifier"]["output_tokens"],
        "verifier_reasoning_tokens": usage["verifier"]["reasoning_tokens"],
        "estimated_cost_usd": usage["total_estimated_cost_usd"],
    }


def write_run_summary(out_dir: Path, item_results: List[Dict[str, Any]]) -> None:
    if not item_results:
        return

    rows = [summary_row(x) for x in item_results]
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    total_cost = sum(x["usage"]["total_estimated_cost_usd"] for x in item_results)
    total_input = sum(
        x["usage"]["extractor"]["input_tokens"]
        + x["usage"]["verifier"]["input_tokens"]
        for x in item_results
    )
    total_output = sum(
        x["usage"]["extractor"]["output_tokens"]
        + x["usage"]["verifier"]["output_tokens"]
        for x in item_results
    )
    verifier_calls = sum(1 for x in item_results if x.get("verification") is not None)
    n = len(item_results)
    elapsed_values = [
        float(x["meta"]["elapsed_seconds"])
        for x in item_results
        if x.get("meta", {}).get("elapsed_seconds") is not None
    ]

    run = {
        "prompt_version": PROMPT_VERSION,
        "model": item_results[0]["meta"]["model"],
        "records_completed": n,
        "verifier_calls": verifier_calls,
        "verifier_call_rate": verifier_calls / n,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_total_cost_usd": total_cost,
        "estimated_cost_per_completed_record_usd": total_cost / n,
        "naive_full_18035_extrapolation_usd": total_cost / n * 18035,
        "mean_sequential_seconds_per_record": (
            sum(elapsed_values) / len(elapsed_values) if elapsed_values else None
        ),
        "note": (
            "The full-corpus extrapolation is a planning estimate. Actual cost depends "
            "on paper length, cache hits, exclusion rate, and verifier-call rate."
        ),
    }

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV of paper-level records")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--cache_items_dir",
        default=None,
        help="Directory containing <paper_id>.pages.json from old extractor",
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Download/parse PDF when pages cache is missing",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--extractor_reasoning",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="low",
    )
    parser.add_argument(
        "--verifier_reasoning",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= index < num_shards")

    from openai import OpenAI

    client = OpenAI()
    input_df = pd.read_csv(args.input)
    cols = get_input_columns(input_df)

    # Deterministic sharding by input row position.
    input_df = input_df.iloc[
        [i for i in range(len(input_df)) if i % args.num_shards == args.shard_index]
    ].copy()

    if args.max_items is not None:
        input_df = input_df.head(args.max_items)

    out_dir = Path(args.out)
    item_dir = out_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_items_dir) if args.cache_items_dir else None

    item_results: List[Dict[str, Any]] = []

    for local_i, (_, row) in enumerate(input_df.iterrows(), start=1):
        paper_url = str(row[cols["paper_url"]]).strip()
        dataset_name = str(row[cols["dataset_name"]]).strip()
        dataset_link = (
            ""
            if cols["dataset_link"] is None or pd.isna(row[cols["dataset_link"]])
            else str(row[cols["dataset_link"]]).strip()
        )
        record_id = (
            str(row[cols["record_id"]]).strip()
            if cols["record_id"] is not None and not pd.isna(row[cols["record_id"]])
            else f"row_{row.name}"
        )
        paper_id = canonical_paper_id(paper_url)

        safe_record = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)
        item_path = item_dir / f"{safe_record}.v2.json"

        if item_path.exists():
            with open(item_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            item_results.append(result)
            print(f"[{local_i}/{len(input_df)}] {record_id}: cached", flush=True)
            continue

        t0 = time.time()

        try:
            cached_path = find_cached_pages(cache_dir, paper_id)

            if cached_path is not None:
                pages = load_pages_json(cached_path)
                paper_source = str(cached_path)
            elif args.allow_download:
                pdf_bytes = fetch_pdf(paper_url)
                pages = extract_text_pages_from_pdf(pdf_bytes)
                local_cache = out_dir / "downloaded_pages"
                local_cache.mkdir(exist_ok=True)
                local_pages_path = local_cache / f"{paper_id}.pages.json"
                with open(local_pages_path, "w", encoding="utf-8") as f:
                    json.dump({"pages": pages}, f, ensure_ascii=False)
                paper_source = str(local_pages_path)
            else:
                raise FileNotFoundError(
                    f"No pages cache for {paper_id}. Supply --cache_items_dir "
                    f"or use --allow_download."
                )

            paper_text, affiliation_sanitized, truncated = join_pages_for_model(pages)
            text_hash = stable_hash(paper_text)

            extraction_user = build_extraction_user(
                record_id=record_id,
                paper_id=paper_id,
                paper_url=paper_url,
                dataset_name=dataset_name,
                dataset_link=dataset_link,
                paper_text=paper_text,
            )

            extraction, usage1, extraction_response_id = call_structured(
                client,
                model=args.model,
                instructions=EXTRACTION_INSTRUCTIONS,
                user_text=extraction_user,
                schema_name="atlasnlp_country_evidence_v2",
                schema=EXTRACTION_SCHEMA,
                reasoning_effort=args.extractor_reasoning,
                max_output_tokens=5000,
                prompt_cache_key="atlasnlp-country-v2-extractor",
                retries=args.retries,
            )

            # Normalize candidate country strings immediately.
            for c in extraction["candidate_country_assertions"]:
                c["country"] = normalize_country(c["country"])

            relationship = extraction["dataset_relationship"]
            eligible = (
                extraction["dataset_match_status"] == "MATCHED"
                and relationship in ELIGIBLE_RELATIONSHIPS
            )

            # Do not spend a verifier call on prohibited/invalid candidate evidence.
            verifier_candidates = [
                c
                for c in extraction["candidate_country_assertions"]
                if c.get("proposed_support_level") != "INVALID"
                and c.get("evidence_type") not in INVALID_TYPES
            ]

            verification = None
            keyed_candidates: List[Dict[str, Any]] = []
            usage2 = {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
            verifier_response_id = ""

            if eligible and verifier_candidates:
                verifier_user, keyed_candidates = build_verifier_user(
                    record_id=record_id,
                    dataset_name=dataset_name,
                    relationship=relationship,
                    candidates=verifier_candidates,
                )

                verification, usage2, verifier_response_id = call_structured(
                    client,
                    model=args.model,
                    instructions=VERIFIER_INSTRUCTIONS,
                    user_text=verifier_user,
                    schema_name="atlasnlp_country_verification_v2",
                    schema=VERIFIER_SCHEMA,
                    reasoning_effort=args.verifier_reasoning,
                    max_output_tokens=3500,
                    prompt_cache_key="atlasnlp-country-v2-verifier",
                    retries=args.retries,
                )

                validate_verifier_keys(keyed_candidates, verification)

            final = deterministic_finalize(
                extraction=extraction,
                verification=verification,
                keyed_candidates=keyed_candidates,
            )

            result = {
                "meta": {
                    "record_id": record_id,
                    "paper_id": paper_id,
                    "paper_url": paper_url,
                    "dataset_name": dataset_name,
                    "dataset_link": dataset_link,
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "extractor_reasoning": args.extractor_reasoning,
                    "verifier_reasoning": args.verifier_reasoning,
                    "paper_source": paper_source,
                    "paper_text_hash": text_hash,
                    "affiliation_header_sanitized": affiliation_sanitized,
                    "paper_text_truncated": truncated,
                    "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                    "extractor_response_id": extraction_response_id,
                    "verifier_response_id": verifier_response_id,
                    "elapsed_seconds": time.time() - t0,
                },
                "extraction": extraction,
                "verification": verification,
                "final": final,
                "usage": {
                    "extractor": usage1,
                    "verifier": usage2,
                    "total_estimated_cost_usd": (
                        usage1["estimated_cost_usd"] + usage2["estimated_cost_usd"]
                    ),
                },
            }

            with open(item_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            item_results.append(result)

            print(
                f"[{local_i}/{len(input_df)}] {record_id}: "
                f"{relationship} -> {final['content_country_status']} "
                f"| explicit={len(final['explicit_content_countries'])} "
                f"inferred={len(final['inferred_content_countries'])} "
                f"| ${result['usage']['total_estimated_cost_usd']:.4f} "
                f"| {time.time()-t0:.1f}s",
                flush=True,
            )

            if args.sleep_s:
                time.sleep(args.sleep_s)

        except Exception as e:
            failure = {
                "record_id": record_id,
                "paper_id": paper_id,
                "paper_url": paper_url,
                "dataset_name": dataset_name,
                "error": repr(e),
            }
            failures = out_dir / "failures.jsonl"
            with open(failures, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(
                f"[{local_i}/{len(input_df)}] {record_id}: FAILED {e}",
                flush=True,
            )

        # Rebuild summary after each item so an interrupted pilot still has usable totals.
        if item_results:
            write_run_summary(out_dir, item_results)

    if item_results:
        write_run_summary(out_dir, item_results)
        print(f"\nWrote {out_dir/'summary.csv'}")
        print(f"Wrote {out_dir/'run_summary.json'}")


if __name__ == "__main__":
    main()
