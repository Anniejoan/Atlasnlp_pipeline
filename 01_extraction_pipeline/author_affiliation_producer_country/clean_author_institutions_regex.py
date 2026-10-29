#!/usr/bin/env python3
"""
clean_author_institutions_regex.py

Clean and normalize author_institutions_regex produced by the regex pass.

Input:
- CSV with author_institutions_regex column

Output:
- same CSV plus cleaned institution columns

New columns:
- author_institutions_regex_clean
- author_institutions_regex_clean_source
- author_institutions_regex_clean_count
- author_institutions_regex_was_modified

Usage:
    python clean_author_institutions_regex.py --input author_affiliation_all_regex_with_institutions.csv --output author_affiliation_all_regex_with_institutions_clean.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


NOISE_PATTERNS = [
    r"^proceedings of\b",
    r"association for computational linguistics",
    r"^pages?\b",
    r"^august\b",
    r"^september\b",
    r"^october\b",
    r"^november\b",
    r"^december\b",
    r"^january\b",
    r"^february\b",
    r"^march\b",
    r"^april\b",
    r"^may\b",
    r"^june\b",
    r"^july\b",
    r"^correspondence\b",
    r"^reading between the lines\b",
    r"^a dataset\b",
    r"^attacks on\b",
    r"^identifying and analyzing\b",
    r"^translation estimation systems\b",
    r"^towards\b",
    r"^beyond the binary\b",
    r"^low-resource\b",
    r"^cross-document\b",
    r"^typed-rag\b",
    r"^docie@",
    r"^llmsr@",
    r"^narra",
    r"^finding common patterns\b",
]

LOCATION_ONLY_PATTERNS = [
    r"^[A-Z][a-z]+,\s*[A-Z][a-z]+$",
    r"^[A-Z][a-z]+,\s*[A-Z][a-z]+\s*,\s*[A-Z][a-z]+$",
]

COUNTRY_WORDS = {
    "united states", "united kingdom", "uk", "usa", "germany", "france",
    "japan", "china", "india", "italy", "spain", "canada", "argentina",
    "mexico", "belgium", "netherlands", "switzerland", "austria",
    "lithuania", "south korea", "korea", "hong kong", "ireland",
    "denmark", "sweden", "norway", "finland", "australia", "brazil",
    "pakistan", "bangladesh", "singapore", "turkey"
}

GENERIC_SUBUNITS = {
    "department", "department of computer science", "department of linguistics",
    "school", "school of computer science", "school of engineering and sciences",
    "faculty", "lab", "laboratory", "research", "research lab", "research center",
    "research centre", "centre", "center", "institute for", "division"
}

ORG_CUES = [
    "university", "université", "universidad", "universitat",
    "institute", "institut", "college", "school", "faculty",
    "academy", "polytechnic", "hospital", "lab", "laboratory",
    "centre", "center", "research", "corporation", "corp", "inc",
    "ltd", "llc", "gmbh", "technologies", "technology", "systems",
    "solutions", "foundation", "group", "institute of", "department of"
]

KNOWN_ORGS = [
    "openai", "google", "google research", "deepmind", "meta", "amazon",
    "aws", "microsoft", "microsoft research", "ibm", "apple", "nvidia",
    "oracle", "salesforce", "allenai", "bytedance", "tencent", "alibaba",
    "huawei", "samsung", "bloomberg", "bosch", "siemens", "dataminr",
    "duolingo", "elsevier", "airi", "mbzuai", "tcs research", "modulabs",
    "postech", "cairo", "scads.ai", "conicet", "ufal", "epfl", "fbk",
    "athenarc", "adapt centre", "racai", "vicomtech", "centai", "cemas",
    "cardiff nlp", "public health wales"
]


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_semicolon(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [normalize_text(x) for x in text.split(";") if normalize_text(x)]


def clean_basic(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[\*\u2020\u2021]+", "", text)
    text = re.sub(r"^\d+\s*", "", text)
    text = re.sub(r"\b\d+([A-Z])", r"\1", text)  # 2University -> University
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    return text


def split_numbered_institutions(text: str) -> List[str]:
    """
    Split strings like:
    'Fudan University 2University of Arizona'
    'Université Côte d’Azur, 2Ghent University, 3University of Turin'
    """
    text = clean_basic(text)
    if not text:
        return []

    # insert separator before numbered org starts
    text = re.sub(r"\s+\d+\s*(?=[A-Z])", " ||| ", text)
    parts = [clean_basic(x) for x in text.split("|||") if clean_basic(x)]
    return parts if parts else [text]


def is_noise(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            return True
    return False


def looks_like_person_list(text: str) -> bool:
    low = text.lower()

    if any(cue in low for cue in ORG_CUES):
        return False
    if any(org in low for org in KNOWN_ORGS):
        return False
    if low == "individual":
        return False

    # many person-name-like tokens
    tokens = re.findall(r"\b[A-Z][a-zA-Z’'`\-]+\b", text)
    if len(tokens) >= 4 and len(text) < 160:
        return True

    return False


def is_location_only(text: str) -> bool:
    low = text.lower().strip()

    # direct location-only pattern
    for pat in LOCATION_ONLY_PATTERNS:
        if re.match(pat, text):
            return True

    # plain country only
    if low in COUNTRY_WORDS:
        return True

    return False


def strip_trailing_location(text: str) -> str:
    """
    Remove trailing ', Country' or ', City, Country' where possible,
    but keep the institution core.
    """
    text = clean_basic(text)

    # Remove terminal country
    for country in sorted(COUNTRY_WORDS, key=len, reverse=True):
        pattern = rf",\s*{re.escape(country)}\.?$"
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" ,;:-")
            break

    # Remove common city-country endings after institution
    text = re.sub(r",\s*[A-Z][a-zA-Z .'-]+,\s*[A-Z][a-zA-Z .'-]+$", "", text).strip(" ,;:-")
    return text


def prefer_parent_orgs(items: List[str]) -> List[str]:
    """
    If both a department/school and its parent university are present,
    prefer the parent university and drop the generic subunit.
    """
    cleaned = [clean_basic(x) for x in items if clean_basic(x)]
    lowers = [x.lower() for x in cleaned]

    out = []
    for i, item in enumerate(cleaned):
        low = lowers[i]

        # Keep individual only if it is alone later
        if low == "individual":
            out.append(item)
            continue

        # generic subunit without clear org identity: drop if a stronger org exists
        if any(gs == low for gs in GENERIC_SUBUNITS) or low.startswith("department of ") or low.startswith("school of "):
            stronger_exists = any(
                (j != i) and (
                    "university" in lowers[j]
                    or "institute" in lowers[j]
                    or "college" in lowers[j]
                    or "corporation" in lowers[j]
                    or "inc" in lowers[j]
                    or any(k in lowers[j] for k in KNOWN_ORGS)
                )
                for j in range(len(cleaned))
            )
            if stronger_exists:
                continue

        out.append(item)

    # dedupe preserve order
    deduped = []
    seen = set()
    for item in out:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            deduped.append(item)

    # if real orgs exist, drop individual
    if len(deduped) > 1:
        deduped = [x for x in deduped if x.lower() != "individual"]

    return deduped


def clean_institution_list(raw_text: str) -> List[str]:
    items = split_semicolon(raw_text)
    expanded = []

    for item in items:
        expanded.extend(split_numbered_institutions(item))

    cleaned = []
    for item in expanded:
        item = clean_basic(item)
        if not item:
            continue
        if is_noise(item):
            continue
        if looks_like_person_list(item):
            continue
        if is_location_only(item):
            continue

        # normalize explicit independent
        if "independent researcher" in item.lower() or "independent scholar" in item.lower():
            cleaned.append("individual")
            continue

        item = strip_trailing_location(item)
        item = clean_basic(item)

        if not item:
            continue
        if is_noise(item):
            continue
        if is_location_only(item):
            continue

        cleaned.append(item)

    cleaned = prefer_parent_orgs(cleaned)

    # final dedupe
    out = []
    seen = set()
    for item in cleaned:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            out.append(item)

    return out


def process_row(row: pd.Series) -> tuple[str, str, int, int]:
    raw = normalize_text(row.get("author_institutions_regex", ""))
    if not raw:
        return "", "", 0, 0

    cleaned = clean_institution_list(raw)
    cleaned_text = "; ".join(cleaned)

    modified = int(cleaned_text != raw)
    count = len(cleaned)

    return cleaned_text, "cleaned_from_author_institutions_regex", count, modified


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    results = df.apply(process_row, axis=1, result_type="expand")
    results.columns = [
        "author_institutions_regex_clean",
        "author_institutions_regex_clean_source",
        "author_institutions_regex_clean_count",
        "author_institutions_regex_was_modified",
    ]

    out_df = pd.concat([df, results], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    covered = (out_df["author_institutions_regex_clean"].fillna("").astype(str).str.strip() != "").sum()
    modified = out_df["author_institutions_regex_was_modified"].fillna(0).astype(int).sum()

    print(f"Input rows: {len(out_df)}")
    print(f"Rows with cleaned regex institutions: {covered}")
    print(f"Rows modified during cleaning: {modified}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()