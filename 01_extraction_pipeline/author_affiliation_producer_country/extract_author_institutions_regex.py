#!/usr/bin/env python3
"""
extract_author_institutions_regex.py

Post-process existing author affiliation regex output to extract
institution/company/university names for all rows.

Input:
- combined regex output CSV

Output:
- same rows plus institution extraction columns

New columns:
- author_institutions_regex
- author_institutions_regex_source
- author_is_individual_regex

Usage:
    python extract_author_institutions_regex.py --input author_affiliation_all_regex.csv --output author_affiliation_all_regex_with_institutions.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


ORG_HINTS = [
    "university", "université", "universitat", "universidad",
    "college", "school", "faculty", "department", "dept",
    "institute", "institut", "academy", "polytechnic",
    "laboratory", "lab", "centre", "center", "hospital",
    "research center", "research centre", "applied sciences",
    "corporation", "corp", "inc", "ltd", "llc", "gmbh",
    "technologies", "technology", "systems", "solutions",
    "research", "ai", "foundation", "group"
]

# Strong known short-form orgs or branded labs that may not contain "university"
KNOWN_ORGS = [
    "openai", "google", "google research", "deepmind", "meta", "amazon",
    "aws", "microsoft", "microsoft research", "ibm", "apple", "nvidia",
    "oracle", "salesforce", "allenai", "bytedance", "tencent", "alibaba",
    "huawei", "samsung", "bloomberg", "bosch", "siemens", "dataminr",
    "duolingo", "elsevier", "airi", "mbzuai", "allen institute for ai",
    "tcs research", "modulabs", "postech", "cairo", "scads.ai", "conicet",
    "ufal", "epfl", "fbk", "athenarc", "adapt centre", "racai", "vicomtech"
]

INDIVIDUAL_MARKERS = [
    "independent researcher",
    "independent research",
    "independent scholar",
    "freelance researcher",
]

NOISY_PATTERNS = [
    r"^proceedings of\b",
    r"association for computational linguistics",
    r"^pages?\b",
    r"^august\b",
    r"^december\b",
    r"^january\b",
    r"^february\b",
    r"^march\b",
    r"^april\b",
    r"^may\b",
    r"^june\b",
    r"^july\b",
    r"^september\b",
    r"^october\b",
    r"^november\b",
    r"^correspondence\b",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
BRACE_EMAIL_RE = re.compile(r"\{([^{}]+)\}@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_pipe_text(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [normalize_text(x) for x in text.split("|") if normalize_text(x)]


def remove_email_text(line: str) -> str:
    line = EMAIL_RE.sub("", line)
    line = BRACE_EMAIL_RE.sub("", line)
    line = re.sub(r"\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", line)
    line = re.sub(r"\s+", " ", line).strip(" ,;|")
    return line.strip()


def is_noisy_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    for pat in NOISY_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            return True
    return False


def looks_like_person_line(line: str) -> bool:
    """
    Heuristic to avoid returning pure author-name lines as institutions.
    """
    line = normalize_text(line)
    low = line.lower()

    if "@" in line:
        return False

    if any(h in low for h in ORG_HINTS):
        return False
    if any(k in low for k in KNOWN_ORGS):
        return False

    # Many capitalized words and commas can indicate author lists
    tokens = re.findall(r"\b[A-Z][a-zA-Z’'`\-]+\b", line)
    if len(tokens) >= 4 and len(line) < 140:
        return True

    return False


def clean_inst_line(line: str) -> str:
    line = normalize_text(line)
    line = remove_email_text(line)
    line = re.sub(r"^\d+\s*", "", line)
    line = re.sub(r"[\*\u2020\u2021]+", "", line)
    line = re.sub(r"\s+", " ", line).strip(" ,;:-")
    return line


def extract_org_candidates_from_lines(lines: List[str]) -> List[str]:
    candidates = []

    for raw in lines:
        line = clean_inst_line(raw)
        if not line:
            continue
        if is_noisy_line(line):
            continue
        if looks_like_person_line(line):
            continue

        low = line.lower()

        if any(marker in low for marker in INDIVIDUAL_MARKERS):
            candidates.append("individual")
            continue

        if any(h in low for h in ORG_HINTS) or any(k in low for k in KNOWN_ORGS):
            candidates.append(line)
            continue

        # Short branded org-like lines
        if len(line) <= 80 and re.match(r"^[A-Za-z0-9&.,'’\-\(\) ]+$", line):
            if line.isupper() or any(k in low for k in KNOWN_ORGS):
                candidates.append(line)

    return dedupe_institutions(candidates)


def dedupe_institutions(items: List[str]) -> List[str]:
    out = []
    seen = set()

    for item in items:
        item = normalize_text(item).strip(" ,;:-")
        if not item:
            continue

        low = item.lower()

        # normalize individual
        if low in INDIVIDUAL_MARKERS or low == "individual":
            item = "individual"
            low = "individual"

        # skip generic weak fragments
        weak = {
            "department", "school", "faculty", "research", "lab",
            "center", "centre", "hospital", "ai"
        }
        if low in weak:
            continue

        if low not in seen:
            seen.add(low)
            out.append(item)

    # If we found real orgs, drop "individual"
    if len(out) > 1 and "individual" in [x.lower() for x in out]:
        out = [x for x in out if x.lower() != "individual"]

    return out


def extract_institutions_for_row(row: pd.Series) -> tuple[str, str, int]:
    aff_lines = split_pipe_text(row.get("author_affiliations_raw", ""))
    header_lines = split_pipe_text(row.get("header_text", ""))

    aff_orgs = extract_org_candidates_from_lines(aff_lines)
    if aff_orgs:
        is_individual = int(aff_orgs == ["individual"])
        return "; ".join(aff_orgs), "author_affiliations_raw", is_individual

    header_orgs = extract_org_candidates_from_lines(header_lines)
    if header_orgs:
        is_individual = int(header_orgs == ["individual"])
        return "; ".join(header_orgs), "header_text", is_individual

    # If nothing explicit found, do not force "individual"
    return "", "", 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Combined regex output CSV")
    parser.add_argument("--output", required=True, help="Output CSV with institution columns")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    results = df.apply(extract_institutions_for_row, axis=1, result_type="expand")
    results.columns = [
        "author_institutions_regex",
        "author_institutions_regex_source",
        "author_is_individual_regex",
    ]

    out_df = pd.concat([df, results], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    covered = (out_df["author_institutions_regex"].fillna("").astype(str).str.strip() != "").sum()
    individuals = out_df["author_is_individual_regex"].fillna(0).astype(int).sum()

    print(f"Input rows: {len(out_df)}")
    print(f"Rows with regex institutions: {covered}")
    print(f"Rows marked individual: {individuals}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()