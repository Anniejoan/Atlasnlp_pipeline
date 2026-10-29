#!/usr/bin/env python3
"""
author_affiliation_extract_v3.py

Improved author affiliation extraction pipeline for ACL Anthology papers.

What this version improves over v2:
- tighter header parsing
- stronger stop boundaries
- better filtering of title lines and author-name lines
- cleaner handling of correspondence/email lines
- stricter affiliation-line selection

Usage:
    python author_affiliation_extract_v3.py \
        --input master.csv \
        --output author_affiliation_test_output_v3.csv \
        --pdf-dir pdfs \
        --url-column "Paper Link (URL)" \
        --limit 200
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    """
    Keep the top portion of page 1 until a strong section/bio boundary.
    """
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

    # many commas / separators and no affiliation keywords often means author line
    if line.count(",") >= 2 and len(line) < 140:
        return True

    # Many short capitalized tokens can indicate names
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

    # remove plain correspondence-only lines from affiliation typing later,
    # but keep them available for email extraction
    seen = set()
    out = []
    for line in candidates:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)

    return out[:max_lines]


def extract_countries_from_text(text: str) -> List[str]:
    low = f" {text.lower()} "
    found = []

    for alias, canon in COUNTRY_ALIASES.items():
        if f" {alias.lower()} " in low:
            found.append(canon)

    for country in COUNTRY_LIST:
        c_low = country.lower()
        if f" {c_low} " in low or f" {c_low}," in low or f", {c_low}" in low:
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

        for suffix, country in EMAIL_DOMAIN_COUNTRY_HINTS.items():
            if low.endswith(suffix):
                found.append(country)

        domain = low.split("@")[-1] if "@" in low else low
        if domain.endswith(".ng"):
            found.append("Nigeria")
        elif domain.endswith(".jp"):
            found.append("Japan")
        elif domain.endswith(".uk"):
            found.append("United Kingdom")
        elif domain.endswith(".cn"):
            found.append("China")
        elif domain.endswith(".in"):
            found.append("India")
        elif domain.endswith(".au"):
            found.append("Australia")
        elif domain.endswith(".kr"):
            found.append("South Korea")
        elif domain.endswith(".za"):
            found.append("South Africa")
        elif domain.endswith(".br"):
            found.append("Brazil")
        elif domain.endswith(".mx"):
            found.append("Mexico")
        elif domain.endswith(".tr"):
            found.append("Turkey")
        elif domain.endswith(".pk"):
            found.append("Pakistan")
        elif domain.endswith(".fr"):
            found.append("France")
        elif domain.endswith(".de"):
            found.append("Germany")
        elif domain.endswith(".ar"):
            found.append("Argentina")
        elif domain.endswith(".be"):
            found.append("Belgium")
        elif domain.endswith(".nl"):
            found.append("Netherlands")
        elif domain.endswith(".it"):
            found.append("Italy")

    return sorted(set(found))


def infer_countries_from_institutions(affiliation_lines: List[str]) -> List[str]:
    text = " | ".join(affiliation_lines).lower()
    found = []
    for inst, country in INSTITUTION_COUNTRY_HINTS.items():
        if inst in text:
            found.append(country)
    return sorted(set(found))


def classify_affiliation_type(affiliation_lines: List[str]) -> str:
    """
    Conservative classification using only cleaned lines,
    excluding correspondence-only lines.
    """
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

    # suppress overly generic hits
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input master CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--pdf-dir", default="pdfs", help="Directory to store PDFs")
    parser.add_argument("--url-column", default="Paper Link (URL)", help="Paper URL column name")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process")
    parser.add_argument("--download-timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N rows")
    parser.add_argument("--force-redownload", action="store_true", help="Redownload PDFs even if present")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    pdf_dir = Path(args.pdf_dir)

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

    if args.limit is not None:
        df = df.head(args.limit).copy()

    results = []
    total = len(df)

    print(f"Rows to process: {total}")
    print(f"Using URL column: {args.url_column}")
    print(f"PDF directory: {pdf_dir}")
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

        if idx % args.save_every == 0 or idx == total:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"[{idx}/{total}] saved checkpoint -> {output_path}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    print("\nDone.")
    print(f"Output written to: {output_path}")

    if len(out_df) > 0:
        print("\nQuick summary:")
        print(out_df["pdf_download_status"].value_counts(dropna=False).head(20).to_string())

        print("\nCountry inference method counts:")
        print(out_df["author_country_inference_method"].replace("", pd.NA).value_counts(dropna=False).to_string())

        print("\nAffiliation type counts:")
        print(out_df["author_affiliation_type"].value_counts(dropna=False).to_string())

        covered = (out_df["author_all_inferred_countries"].fillna("").str.strip() != "").sum()
        print(f"\nRows with any inferred author country: {covered} / {len(out_df)}")


if __name__ == "__main__":
    main()