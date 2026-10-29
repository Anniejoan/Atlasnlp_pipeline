#!/usr/bin/env python3
"""
author_affiliation_extract.py — Extract author institution affiliations from PDFs

Paper section: §3.3 "Automated Dataset Expansion" (producer-country attribution)
               §5.2 "Dataset Production and Representation are Asymmetric"
Role in pipeline: Step 9a — downloads each ACL paper PDF and regex-extracts
  author institution names. This is the first pass (regex-based); entries that
  fail are passed to the LLM fallback (run_author_affiliation_llm_fallback.py).
  Combined output feeds compute_worldbench_country_coverage.py which maps
  institution names to countries (producer geography used in Figure 3 and Table 3).

Pipeline within author_affiliation/:
  1. make_author_affiliation_shards.py  — split master into shards
  2. run_author_affiliation_shards.sh    — run shards in parallel (SLURM or local)
  3. THIS SCRIPT                         — runs on each shard
  4. combine_author_affiliation_shards.py— merge shard outputs
  5. extract_author_institutions_regex.py— normalize/clean institutions
  6. clean_author_institutions_regex.py  — further cleaning
  7. prepare_author_affiliation_llm_fallback.py — flag failures for LLM
  8. run_author_affiliation_llm_fallback.py — LLM fallback for hard cases
  9. merge_author_affiliation_regex_llm.py — combine regex + LLM results
  10. aggregate_author_affiliation_diagnostics.py — run diagnostics
  11. compute_worldbench_country_coverage.py — final country coverage stats

Features:
  - resumable (skips already-processed rows)
  - shardable by row range (--start-row, --end-row)
  - checkpoint saving every N rows
  - diagnostics report

Example:
    python author_affiliation_extract.py \\
        --input master.csv \\
        --output outputs/author_affiliation_shard_001.csv \\
        --pdf-dir pdfs \\
        --url-column "Paper Link (URL)" \\
        --start-row 0 \\
        --end-row 999 \\
        --save-every 100 \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd
import requests

# ----------------------------
# Optional imports
# ----------------------------
HAVE_FITZ = False
HAVE_PDFMINER = False
HAVE_PYCOUNTRY = False

try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    pass

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAVE_PDFMINER = True
except Exception:
    pass

try:
    import pycountry
    HAVE_PYCOUNTRY = True
except Exception:
    pass


# ----------------------------
# Regex / constants
# ----------------------------
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
GROUPED_EMAIL_PATTERN = re.compile(r"\{([^{}]+)\}@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
WHITESPACE_RE = re.compile(r"\s+")
DIGIT_PREFIX_RE = re.compile(r"^\s*[\d,*†‡§¶]+\s*")
TRAILING_FOOTNOTE_RE = re.compile(r"[\d*†‡§¶]+$")

SECTION_BOUNDARY_RE = re.compile(
    r"^\s*(abstract|introduction|keywords?|1\s+[A-Z]|1\.\s+[A-Z]|\d+\.\d+\s+[A-Z])\b",
    re.IGNORECASE,
)

HARD_STOP_LINE_RE = re.compile(
    r"(research interests|biography|acknowledg|references|related work|background)",
    re.IGNORECASE,
)

WORKSHOP_TITLE_RE = re.compile(
    r"(proceedings of|association for computational linguistics|pages \d+[-–]\d+|workshop on|joint workshop)",
    re.IGNORECASE,
)

CORRESPONDENCE_RE = re.compile(r"^\s*correspondence\s*:", re.IGNORECASE)

ACADEMIC_KEYWORDS = [
    "university", "université", "universitat", "universidad",
    "college", "school", "faculty", "department", "dept",
    "institute", "institut", "academy", "polytechnic",
    "laboratory", "lab", "centre", "center", "hospital",
    "research center", "research centre", "graduate school",
    "applied sciences"
]

INDUSTRY_KEYWORDS = [
    "google", "microsoft", "meta", "openai", "amazon", "aws", "ibm",
    "nvidia", "intel", "samsung", "huawei", "deepmind", "apple",
    "salesforce", "adobe", "oracle", "baidu", "alibaba", "tencent",
    "bytedance", "uber", "airbnb", "bosch", "siemens",
    "corp", "corporation", "inc", "ltd", "llc", "gmbh",
    "technologies", "technology", "systems", "solutions",
    "modulabs", "tcs research", "kt", "cairo"
]

AFFILIATION_HINTS = [
    "university", "université", "universitat", "universidad",
    "institute", "institut", "college", "school", "faculty",
    "department", "laboratory", "lab", "centre", "center",
    "academy", "polytechnic", "hospital",
    "corporation", "corp", "inc", "ltd", "llc", "gmbh",
    "technologies", "technology", "systems", "solutions",
    "applied sciences"
]

INSTITUTION_COUNTRY_HINTS = {
    "university of cambridge": "United Kingdom",
    "cardiff university": "United Kingdom",
    "cardiff nlp": "United Kingdom",
    "university of leeds": "United Kingdom",
    "university of bath": "United Kingdom",
    "university of mannheim": "Germany",
    "university of stuttgart": "Germany",
    "ghent university": "Belgium",
    "university of groningen": "Netherlands",
    "university of turin": "Italy",
    "bocconi university": "Italy",
    "johns hopkins university": "United States",
    "university of rochester": "United States",
    "texas state university": "United States",
    "university of arizona": "United States",
    "fudan university": "China",
    "harbin institute of technology": "China",
    "hong kong polytechnic university": "Hong Kong",
    "the hong kong polytechnic university": "Hong Kong",
    "national institute of informatics": "Japan",
    "university of electro-communications": "Japan",
    "ewha womans university": "South Korea",
    "sookmyung women's university": "South Korea",
    "sookmyung women’s university": "South Korea",
    "constructor university": "Germany",
    "international institute of information technology bhubaneswar": "India",
    "iiit bhubaneswar": "India",
    "iiit hyderabad": "India",
    "nitc": "India",
    "national institute of technology calicut": "India",
    "iiser bhopal": "India",
    "iiserb": "India",
    "universidad nacional de la plata": "Argentina",
    "conicet": "Argentina",
    "nantes université": "France",
    "nantes university": "France",
    "ecole centrale nantes": "France",
    "école centrale nantes": "France",
    "université côte d’azur": "France",
    "université côte d'azur": "France",
    "hochschule mittweida": "Germany",
    "hochschule darmstadt": "Germany",
    "hochschule fresenius": "Germany",
    "tcs research": "India",
    "tu dresden": "Germany",
    "tu-dresden": "Germany",
    "scads.ai": "Germany",
    "modulabs": "South Korea",
    "kt": "South Korea",
    "postech": "South Korea",
}

EMAIL_DOMAIN_COUNTRY_HINTS = {
    ".edu": "United States",
    ".edu.ng": "Nigeria",
    ".ac.uk": "United Kingdom",
    ".ac.jp": "Japan",
    ".edu.cn": "China",
    ".ac.in": "India",
    ".edu.au": "Australia",
    ".ac.kr": "South Korea",
    ".edu.sg": "Singapore",
    ".ac.za": "South Africa",
    ".edu.br": "Brazil",
    ".edu.mx": "Mexico",
    ".edu.tr": "Turkey",
    ".edu.pk": "Pakistan",
}

COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "republic of korea": "South Korea",
    "uae": "United Arab Emirates",
    "prc": "China",
    "p.r. china": "China",
}

FALLBACK_COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Nigeria", "India", "China",
    "Japan", "Germany", "France", "Italy", "Spain", "Brazil", "Mexico",
    "South Korea", "Australia", "Netherlands", "Sweden", "Switzerland",
    "Austria", "Belgium", "Denmark", "Finland", "Norway", "Poland",
    "Portugal", "Ireland", "Turkey", "Russia", "South Africa", "Kenya",
    "Ghana", "Uganda", "Tanzania", "Ethiopia", "Egypt", "Morocco",
    "Algeria", "Tunisia", "Argentina", "Chile", "Colombia", "Peru",
    "Pakistan", "Bangladesh", "Indonesia", "Thailand", "Vietnam",
    "Philippines", "Malaysia", "Singapore", "New Zealand", "Israel",
    "Saudi Arabia", "United Arab Emirates", "Qatar", "Hong Kong"
]

COMMON_TITLE_WORDS = {
    "dataset", "datasets", "model", "models", "language", "languages",
    "retrieval", "generation", "translation", "classification", "analysis",
    "learning", "sign", "speech", "hate", "content", "document", "event",
    "summarization", "resource", "evaluation", "augmented", "synthetic"
}


# ----------------------------
# Helpers
# ----------------------------
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def get_country_list() -> List[str]:
    countries = set(FALLBACK_COUNTRIES)
    if HAVE_PYCOUNTRY:
        for c in pycountry.countries:
            countries.add(c.name)
            if hasattr(c, "official_name"):
                countries.add(c.official_name)
            if hasattr(c, "common_name"):
                countries.add(c.common_name)
    countries.update(["United States of America", "USA", "UK", "Hong Kong"])
    return sorted(countries, key=lambda x: len(x), reverse=True)


COUNTRY_LIST = get_country_list()


def build_pdf_url(acl_url: str) -> str:
    acl_url = safe_str(acl_url).strip()
    if not acl_url:
        return ""
    return acl_url.rstrip("/") + ".pdf"


def make_safe_filename_from_url(url: str) -> str:
    url = safe_str(url).rstrip("/")
    name = url.split("/")[-1]
    if not name.endswith(".pdf"):
        name += ".pdf"
    return name


def download_pdf(pdf_url: str, out_path: Path, timeout: int = 30, sleep_sec: float = 0.2) -> Tuple[bool, str]:
    if out_path.exists():
        return True, "already_exists"
    try:
        response = requests.get(pdf_url, timeout=timeout)
        if response.status_code == 200 and response.content:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(response.content)
            time.sleep(sleep_sec)
            return True, "downloaded"
        return False, f"http_{response.status_code}"
    except Exception as e:
        return False, f"download_error:{type(e).__name__}"


def extract_page_text_pymupdf(pdf_path: Path, page_num: int = 0) -> str:
    if not HAVE_FITZ:
        return ""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) <= page_num:
            return ""
        return doc[page_num].get_text() or ""
    except Exception:
        return ""


def extract_text_pdfminer(pdf_path: Path) -> str:
    if not HAVE_PDFMINER:
        return ""
    try:
        return pdfminer_extract_text(str(pdf_path)) or ""
    except Exception:
        return ""


def extract_page1_text(pdf_path: Path) -> Tuple[str, str]:
    text = extract_page_text_pymupdf(pdf_path, page_num=0)
    if text.strip():
        return text, "pymupdf_page1"

    text = extract_text_pdfminer(pdf_path)
    if text.strip():
        return text, "pdfminer_full"
    return "", "none"


def split_lines(text: str) -> List[str]:
    return [normalize_text(line) for line in str(text).splitlines() if normalize_text(line)]


def clean_line(line: str) -> str:
    line = normalize_text(line)
    line = DIGIT_PREFIX_RE.sub("", line)
    line = TRAILING_FOOTNOTE_RE.sub("", line).strip()
    return normalize_text(line)


def get_header_lines(page1_text: str, max_lines: int = 45) -> List[str]:
    lines = split_lines(page1_text)
    header = []

    for line in lines[:max_lines]:
        raw = clean_line(line)
        if not raw:
            continue

        if SECTION_BOUNDARY_RE.search(raw):
            break

        if HARD_STOP_LINE_RE.search(raw):
            break

        header.append(raw)

    return header


def expand_grouped_emails(text: str) -> List[str]:
    emails = []
    emails.extend(EMAIL_PATTERN.findall(text or ""))

    for local_block, domain in GROUPED_EMAIL_PATTERN.findall(text or ""):
        locals_ = [normalize_text(x).strip(",;") for x in local_block.split(",")]
        for local in locals_:
            local = local.strip()
            if local:
                emails.append(f"{local}@{domain}")

    cleaned = []
    for e in emails:
        e = e.strip(" ;,()[]{}")
        e = e.replace(" ", "")
        if "@" in e:
            cleaned.append(e)

    return sorted(set(cleaned))


def line_looks_like_body_text(line: str) -> bool:
    low = line.lower()

    body_phrases = [
        "in this paper", "we propose", "we present", "our results", "this work",
        "we introduce", "we analyze", "dataset is available", "code is available",
        "research interests", "shared task", "for retrieval-augmented generation",
        "through step-wise reasoning", "with revision instruction", "a dataset and a study",
    ]
    if any(p in low for p in body_phrases):
        return True

    if len(line) > 140 and "@" not in line and not any(h in low for h in AFFILIATION_HINTS):
        return True

    return False


def line_looks_like_title_or_venue(line: str) -> bool:
    low = line.lower()

    if WORKSHOP_TITLE_RE.search(line):
        return True

    if "@xllm" in low or "@wslp" in low or "@woah" in low or "@acl" in low:
        return True

    words = re.findall(r"[A-Za-z]+", line)
    if len(words) >= 6:
        overlap = sum(1 for w in words if w.lower() in COMMON_TITLE_WORDS)
        if overlap >= 2 and "@" not in line and not any(h in low for h in AFFILIATION_HINTS):
            return True

    return False


def line_looks_like_author_list(line: str) -> bool:
    low = line.lower()

    if "@" in low:
        return False

    if any(h in low for h in AFFILIATION_HINTS):
        return False

    if "correspondence:" in low:
        return False

    if line.count(",") >= 2 and len(line) < 140:
        return True

    tokens = re.findall(r"\b[A-Z][a-zA-Z’'`-]+\b", line)
    if len(tokens) >= 4 and len(line) < 120:
        return True

    return False


def normalize_affiliation_line(line: str) -> str:
    line = clean_line(line)
    line = line.replace("Correspondence:", "").replace("correspondence:", "")
    return normalize_text(line)


def looks_like_affiliation_line(line: str) -> bool:
    low = line.lower()

    if line_looks_like_body_text(line):
        return False

    if line_looks_like_title_or_venue(line):
        return False

    if line_looks_like_author_list(line):
        return False

    if CORRESPONDENCE_RE.search(line):
        return True

    if "@" in line:
        return True

    if any(h in low for h in AFFILIATION_HINTS):
        return True

    if any(k in low for k in INSTITUTION_COUNTRY_HINTS):
        return True

    if len(line) < 110 and line.count(",") >= 1:
        if any(token in low for token in [
            "university", "institute", "department", "school", "lab",
            "centre", "center", "france", "germany", "japan", "uk", "usa", "india"
        ]):
            return True

    return False


def extract_affiliation_lines_from_header(header_lines: List[str], max_lines: int = 10) -> List[str]:
    candidates = []
    for line in header_lines:
        if looks_like_affiliation_line(line):
            normalized = normalize_affiliation_line(line)
            if normalized:
                candidates.append(normalized)

    seen = set()
    out = []
    for line in candidates:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)

    return out[:max_lines]


# Country names that appear in institution names located elsewhere.
# Key: lowercase country name. Value: list of lowercase institution-name fragments
# that indicate the institution is NOT in that country.
INSTITUTION_EXCLUSION_PATTERNS: Dict[str, List[str]] = {
    # Georgia (Caucasus) vs. US institutions whose names start with "Georgia"
    "georgia": [
        "georgia institute of technology", "georgia tech", "gatech",
        "georgia state university", "university of georgia",
        "georgia southern", "georgia college", "georgia gwinnett",
        "kennesaw state",  # part of the University System of Georgia
    ],
    # Mexico vs. "New Mexico" US institutions
    "mexico": [
        "new mexico state", "new mexico institute", "university of new mexico",
        "new mexico tech", "new mexico highlands",
    ],
}

# Pre-compile word-boundary patterns for all countries/aliases once at import time.
# Negative lookbehind/lookahead on [a-z] ensures we don't match partial words
# (e.g. "niger" inside "nigeria", "dominica" inside "dominican").
_COUNTRY_PATTERNS: List[tuple] = []
_ALIAS_PATTERNS: List[tuple] = []


def _build_country_patterns() -> None:
    for alias, canon in COUNTRY_ALIASES.items():
        pat = re.compile(r"(?<![a-z])" + re.escape(alias.lower()) + r"(?![a-z])")
        _ALIAS_PATTERNS.append((pat, canon))
    for country in COUNTRY_LIST:
        pat = re.compile(r"(?<![a-z])" + re.escape(country.lower()) + r"(?![a-z])")
        _COUNTRY_PATTERNS.append((pat, country))


_build_country_patterns()


def extract_countries_from_text(text: str) -> List[str]:
    low = text.lower()
    found = []

    for pat, canon in _ALIAS_PATTERNS:
        if pat.search(low):
            found.append(canon)

    for pat, country in _COUNTRY_PATTERNS:
        if pat.search(low):
            # Check whether this match comes only from a known non-country institution name.
            excl_frags = INSTITUTION_EXCLUSION_PATTERNS.get(country.lower(), [])
            if excl_frags:
                # Remove all exclusion fragments from the text, then re-test.
                scrubbed = low
                for frag in excl_frags:
                    scrubbed = scrubbed.replace(frag, " ")
                if not pat.search(scrubbed):
                    continue  # match was entirely inside excluded institution names
            found.append(country)

    normalized = []
    for c in found:
        if c == "United States of America":
            c = "United States"
        elif c == "UK":
            c = "United Kingdom"
        normalized.append(c)

    return sorted(set(normalized))


def infer_countries_from_emails(emails: List[str]) -> List[str]:
    found = []
    for email in emails:
        low = email.lower()
        domain = low.split("@")[-1] if "@" in low else low

        for suffix, country in EMAIL_DOMAIN_COUNTRY_HINTS.items():
            if domain.endswith(suffix.lstrip(".")) or domain.endswith(suffix):
                found.append(country)

        # Conservative ccTLD fallback: only domain ending, not substring
        cc_map = {
            ".ng": "Nigeria",
            ".jp": "Japan",
            ".uk": "United Kingdom",
            ".cn": "China",
            ".in": "India",
            ".au": "Australia",
            ".kr": "South Korea",
            ".za": "South Africa",
            ".br": "Brazil",
            ".mx": "Mexico",
            ".tr": "Turkey",
            ".pk": "Pakistan",
            ".fr": "France",
            ".de": "Germany",
            ".ar": "Argentina",
            ".be": "Belgium",
            ".nl": "Netherlands",
            ".it": "Italy",
        }
        for suffix, country in cc_map.items():
            if domain.endswith(suffix):
                found.append(country)

    return sorted(set(found))


def infer_countries_from_institutions(affiliation_lines: List[str]) -> List[str]:
    text = " | ".join(affiliation_lines).lower()
    found = []
    for inst, country in INSTITUTION_COUNTRY_HINTS.items():
        if inst in text:
            found.append(country)
    return sorted(set(found))


def classify_affiliation_type(affiliation_lines: List[str]) -> str:
    typing_lines = []
    for line in affiliation_lines:
        low = line.lower()
        if "@" in low and not any(h in low for h in AFFILIATION_HINTS) and not any(k in low for k in INSTITUTION_COUNTRY_HINTS):
            continue
        if "correspondence" in low and not any(h in low for h in AFFILIATION_HINTS):
            continue
        typing_lines.append(line)

    text = " ".join(typing_lines).lower()

    has_academic = any(k in text for k in ACADEMIC_KEYWORDS)
    has_industry = any(k in text for k in INDUSTRY_KEYWORDS)

    if has_academic and not any(k in text for k in [
        "google", "microsoft", "meta", "openai", "amazon", "aws", "ibm",
        "nvidia", "intel", "samsung", "huawei", "deepmind", "apple",
        "tcs research", "modulabs", "kt", "cairo",
        "corp", "corporation", "inc", "ltd", "llc", "gmbh"
    ]):
        has_industry = False

    if has_academic and has_industry:
        return "mixed"
    if has_academic:
        return "academia"
    if has_industry:
        return "industry"
    return "unknown"


def process_row(
    row: pd.Series,
    url_column: str,
    pdf_dir: Path,
    download_timeout: int,
    force_redownload: bool = False,
) -> Dict[str, str]:
    paper_url = safe_str(row.get(url_column, "")).strip()
    pdf_url = build_pdf_url(paper_url)
    pdf_filename = make_safe_filename_from_url(pdf_url)
    pdf_path = pdf_dir / pdf_filename

    result = {
        "paper_url": paper_url,
        "pdf_url": pdf_url,
        "pdf_path": str(pdf_path),
        "pdf_downloaded": False,
        "pdf_download_status": "",
        "text_extraction_method": "",
        "header_text": "",
        "emails": "",
        "author_affiliations_raw": "",
        "author_affiliation_countries": "",
        "author_institution_countries": "",
        "author_email_countries": "",
        "author_all_inferred_countries": "",
        "author_country_inference_method": "",
        "author_affiliation_type": "",
        "affiliation_extracted": False,
    }

    if not paper_url:
        result["pdf_download_status"] = "missing_paper_url"
        return result

    if force_redownload and pdf_path.exists():
        try:
            pdf_path.unlink()
        except Exception:
            pass

    ok, status = download_pdf(pdf_url, pdf_path, timeout=download_timeout)
    result["pdf_downloaded"] = ok
    result["pdf_download_status"] = status

    if not ok or not pdf_path.exists():
        return result

    page1_text, extraction_method = extract_page1_text(pdf_path)
    result["text_extraction_method"] = extraction_method

    if not page1_text.strip():
        result["pdf_download_status"] = f"{status}|no_text"
        return result

    header_lines = get_header_lines(page1_text)
    header_text = " | ".join(header_lines)
    result["header_text"] = header_text

    emails = expand_grouped_emails(header_text)
    aff_lines = extract_affiliation_lines_from_header(header_lines)

    aff_text = " | ".join(aff_lines)
    aff_countries = extract_countries_from_text(aff_text)
    inst_countries = infer_countries_from_institutions(aff_lines)
    email_countries = infer_countries_from_emails(emails)

    all_countries = sorted(set(aff_countries + inst_countries + email_countries))

    if aff_countries:
        method = "inferred_from_author_affiliation"
    elif inst_countries:
        method = "inferred_from_author_institution_hint"
    elif email_countries:
        method = "inferred_from_author_email_domain"
    else:
        method = ""

    result.update({
        "emails": "; ".join(emails),
        "author_affiliations_raw": aff_text,
        "author_affiliation_countries": "; ".join(aff_countries),
        "author_institution_countries": "; ".join(inst_countries),
        "author_email_countries": "; ".join(email_countries),
        "author_all_inferred_countries": "; ".join(all_countries),
        "author_country_inference_method": method,
        "author_affiliation_type": classify_affiliation_type(aff_lines),
        "affiliation_extracted": bool(aff_lines or emails),
    })

    return result


# ----------------------------
# Resume / diagnostics helpers
# ----------------------------
def load_processed_urls(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    try:
        prev = pd.read_csv(output_path, usecols=["paper_url"])
        return set(prev["paper_url"].dropna().astype(str).tolist())
    except Exception:
        return set()


def summarize_series(series: pd.Series) -> Dict[str, int]:
    vc = series.value_counts(dropna=False)
    out = {}
    for k, v in vc.items():
        key = "NaN" if pd.isna(k) else str(k)
        out[key] = int(v)
    return out


def compute_diagnostics(df: pd.DataFrame) -> Dict:
    total = len(df)
    if total == 0:
        return {"total_rows": 0}

    covered_mask = df["author_all_inferred_countries"].fillna("").str.strip() != ""
    missing_mask = ~covered_mask

    diagnostics = {
        "total_rows": int(total),
        "rows_with_any_inferred_author_country": int(covered_mask.sum()),
        "coverage_rate": float(round(covered_mask.mean(), 4)),
        "pdf_download_status_counts": summarize_series(df["pdf_download_status"]),
        "country_inference_method_counts": summarize_series(
            df["author_country_inference_method"].replace("", pd.NA)
        ),
        "affiliation_type_counts": summarize_series(df["author_affiliation_type"]),
        "affiliation_extracted_true": int(df["affiliation_extracted"].fillna(False).sum()),
        "rows_missing_any_author_country": int(missing_mask.sum()),
    }

    # simple missing-pattern diagnostics
    missing_df = df.loc[missing_mask].copy()

    if len(missing_df) > 0:
        diagnostics["missing_text_extraction_method_counts"] = summarize_series(
            missing_df["text_extraction_method"]
        )
        diagnostics["missing_affiliation_type_counts"] = summarize_series(
            missing_df["author_affiliation_type"]
        )

        missing_email_domains = []
        for val in missing_df["emails"].fillna(""):
            for email in str(val).split(";"):
                email = email.strip()
                if "@" in email:
                    missing_email_domains.append(email.split("@")[-1].lower())

        if missing_email_domains:
            ser = pd.Series(missing_email_domains)
            diagnostics["top_missing_email_domains"] = {
            str(k): int(v) for k, v in ser.value_counts().head(20).items()
                }
        else:
            diagnostics["top_missing_email_domains"] = {}

    return diagnostics

def make_json_safe(obj):
    """
    Recursively convert pandas/numpy objects to plain Python types
    so they can be serialized by json.dump.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif pd.isna(obj):
        return None

    # numpy / pandas scalars
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass

    # fallback for normal Python scalars/strings
    return obj


def write_diagnostics(output_csv: Path, df: pd.DataFrame) -> None:
    diagnostics = compute_diagnostics(df)

    json_path = output_csv.with_suffix(".diagnostics.json")
    txt_path = output_csv.with_suffix(".summary.txt")

    safe_diagnostics = make_json_safe(diagnostics)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_diagnostics, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"Output file: {output_csv}")
    lines.append(f"Total rows: {diagnostics.get('total_rows', 0)}")
    lines.append(
        f"Rows with any inferred author country: "
        f"{diagnostics.get('rows_with_any_inferred_author_country', 0)} / {diagnostics.get('total_rows', 0)}"
    )
    lines.append(f"Coverage rate: {diagnostics.get('coverage_rate', 0.0):.4f}")
    lines.append("")
    lines.append("PDF download status counts:")
    for k, v in diagnostics.get("pdf_download_status_counts", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Country inference method counts:")
    for k, v in diagnostics.get("country_inference_method_counts", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Affiliation type counts:")
    for k, v in diagnostics.get("affiliation_type_counts", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Missing affiliation type counts:")
    for k, v in diagnostics.get("missing_affiliation_type_counts", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Top missing email domains:")
    for k, v in diagnostics.get("top_missing_email_domains", {}).items():
        lines.append(f"  {k}: {v}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input master CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--pdf-dir", default="pdfs", help="Directory to store PDFs")
    parser.add_argument("--url-column", default="Paper Link (URL)", help="Paper URL column name")

    parser.add_argument("--start-row", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--end-row", type=int, default=None, help="End row index (inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after slicing")
    parser.add_argument("--download-timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N rows")
    parser.add_argument("--force-redownload", action="store_true", help="Redownload PDFs even if present")
    parser.add_argument("--resume", action="store_true", help="Skip rows whose paper_url already exists in output")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    pdf_dir = Path(args.pdf_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)

    if args.url_column not in df.columns:
        print(
            f"ERROR: URL column '{args.url_column}' not found.\n"
            f"Available columns:\n{list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    start = max(0, args.start_row)
    end = len(df) - 1 if args.end_row is None else min(args.end_row, len(df) - 1)

    if start > end:
        print(f"ERROR: start-row {start} is greater than end-row {end}", file=sys.stderr)
        sys.exit(1)

    df = df.iloc[start:end + 1].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    processed_urls = set()
    existing_rows = pd.DataFrame()

    if args.resume and output_path.exists():
        try:
            existing_rows = pd.read_csv(output_path)
            if "paper_url" in existing_rows.columns:
                processed_urls = set(existing_rows["paper_url"].dropna().astype(str))
        except Exception:
            processed_urls = set()
            existing_rows = pd.DataFrame()

    if processed_urls:
        before = len(df)
        df = df[~df[args.url_column].astype(str).isin(processed_urls)].copy()
        skipped = before - len(df)
    else:
        skipped = 0

    results = existing_rows.to_dict(orient="records") if len(existing_rows) > 0 else []
    total_to_process = len(df)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"PDF directory: {pdf_dir}")
    print(f"Row slice: {start} to {end}")
    print(f"Rows to process in this run: {total_to_process}")
    print(f"Rows skipped due to resume: {skipped}")
    print(f"PyMuPDF available: {HAVE_FITZ}")
    print(f"pdfminer available: {HAVE_PDFMINER}")
    print(f"pycountry available: {HAVE_PYCOUNTRY}")

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            result = process_row(
                row=row,
                url_column=args.url_column,
                pdf_dir=pdf_dir,
                download_timeout=args.download_timeout,
                force_redownload=args.force_redownload,
            )
            results.append(result)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
            break
        except Exception as e:
            paper_url = safe_str(row.get(args.url_column, ""))
            results.append({
                "paper_url": paper_url,
                "pdf_url": build_pdf_url(paper_url),
                "pdf_path": "",
                "pdf_downloaded": False,
                "pdf_download_status": f"row_error:{type(e).__name__}",
                "text_extraction_method": "",
                "header_text": "",
                "emails": "",
                "author_affiliations_raw": "",
                "author_affiliation_countries": "",
                "author_institution_countries": "",
                "author_email_countries": "",
                "author_all_inferred_countries": "",
                "author_country_inference_method": "",
                "author_affiliation_type": "",
                "affiliation_extracted": False,
            })

        if idx % args.save_every == 0 or idx == total_to_process:
            out_df = pd.DataFrame(results)
            out_df.to_csv(output_path, index=False)
            write_diagnostics(output_path, out_df)
            print(f"[{idx}/{total_to_process}] checkpoint saved -> {output_path}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    write_diagnostics(output_path, out_df)

    diagnostics = compute_diagnostics(out_df)

    print("\nDone.")
    print(f"Output written to: {output_path}")
    print(f"Diagnostics JSON: {output_path.with_suffix('.diagnostics.json')}")
    print(f"Summary TXT: {output_path.with_suffix('.summary.txt')}")

    print("\nQuick summary:")
    for k, v in diagnostics.get("pdf_download_status_counts", {}).items():
        print(f"  {k}: {v}")

    print("\nCountry inference method counts:")
    for k, v in diagnostics.get("country_inference_method_counts", {}).items():
        print(f"  {k}: {v}")

    print("\nAffiliation type counts:")
    for k, v in diagnostics.get("affiliation_type_counts", {}).items():
        print(f"  {k}: {v}")

    print(
        f"\nRows with any inferred author country: "
        f"{diagnostics.get('rows_with_any_inferred_author_country', 0)} / {diagnostics.get('total_rows', 0)}"
    )


if __name__ == "__main__":
    main()