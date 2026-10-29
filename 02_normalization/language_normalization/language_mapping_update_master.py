#!/usr/bin/env python3
"""
language_mapping_update_master.py — Apply language→country mapping to the master dataset

Paper section: §3.3 "Automated Dataset Expansion" (language-informed attribution)
Role in pipeline: Step 7d — joins the language→country mapping table produced by
  language_country_mapping.py back onto every row in the master dataset.
  Adds inferred content-country assignments for datasets missing explicit attribution.
  Produces updated_master_with_language_mapping_v2.csv which is the master file
  used for all downstream analysis (link inference, task normalization, figures).

Run from compiled_ACL/:
  python language_mapping_update_master.py

Apply token-level language -> country mapping back to the full WorldBench master dataset.

Inputs:
    1. updated_master.csv
    2. language_country_mapping_table.csv

Outputs:
    1. updated_master_with_language_mapping.csv
    2. language_mapping_rowlevel_diagnostics.csv

Behavior:
    - Preserves all existing country information from multiple country columns
    - Adds strict auto-expand language-based country assignments
    - Stores low-spread and candidate-only language signals in Candidate Country From Language
    - Updates country_attribution_method with any newly applied inference methods
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

## run: python language_mapping_update_master.py

# =========================
# CONFIG
# =========================
MASTER_PATH = "country_second_pass/country_second_pass_merged/updated_master.csv"
MAPPING_PATH = "language_country_mapping_outputs/language_country_mapping_table.csv"   # use the full mapping table output
OUTPUT_PATH = "updated_master_with_language_mapping_v2.csv"
DIAGNOSTICS_PATH = "language_mapping_rowlevel_diagnostics.csv"

COUNTRY_COLS = [
    "all_countries_covered",
    "All Countries Covered (you can select multiple countries)",
    "Normalized Countries",
    "Assigned country",
]

NORMALIZED_LANG_COL = "Normalized Languages"
RAW_LANG_COL = "Languages covered (separate using semicolon)"

MISSING_STRINGS = {
    "", "not stated", "nan", "none", "null", "unknown", "n/a"
}

# Alias map aligned with your deterministic cleanup
ALIAS_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "el": "Greek",
    "tr": "Turkish",
    "bg": "Bulgarian",
    "da": "Danish",
    "lv": "Latvian",
    "sk": "Slovak",
    "is": "Icelandic",
    "fi": "Finnish",
    "hu": "Hungarian",
    "ve": "Venda",
    "tn": "Tswana",
    "ee": "Ewe",
    "ak": "Akan",
    "ff": "Fulah",
    "lg": "Ganda",
    "ln": "Lingala",
    "bm": "Bambara",
    "zul": "Zulu",
    "pes": "Persian",
    "ind": "Indonesian",
    "vie": "Vietnamese",
    "tha": "Thai",
    "fil": "Filipino",
    "zsm": "Malay",
    "bangla": "Bengali",
    "farsi": "Persian",
    "mandarin": "Chinese",
    "mandarin chinese": "Chinese",
    "chinese mandarin": "Chinese",
    "indonesian bahasa": "Indonesian",
    "slovene": "Slovenian",
    "british english": "English (UK)",
    "argentine spanish": "Spanish (Argentina)",
    "argentinian spanish": "Spanish (Argentina)",
    "spanish (castilian)": "Spanish",
    "swiss-german": "German (Switzerland)",
    "egyptian dialectal arabic": "Arabic (Egyptian)",
    "kurdish (kurmanji)": "Kurmanji",
    "akan (twi)": "Akan; Twi",
    "punjabi (indian)": "Punjabi",
    "eastern punjabi": "Punjabi",
    "welsh (north wales)": "Welsh",
    "welsh (south wales)": "Welsh",
    "nigerian-pidgin": "Nigerian Pidgin",
    "serbo-croatian (cyrillic)": "Serbo-Croatian",
    "serbo-croatian (latin)": "Serbo-Croatian",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "dgs": "German Sign Language",
    "vgt": "Flemish Sign Language",
    "mas": "Masai",
    "sun": "Sundanese",
}


# =========================
# HELPERS
# =========================

def clean_token(x) -> str:
    """Basic string cleanup with quote stripping."""
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r'^[\'"]+', '', x)
    x = re.sub(r'[\'"]+$', '', x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def is_missing(val) -> bool:
    if pd.isna(val):
        return True
    val = clean_token(val).lower()
    return val in MISSING_STRINGS


def split_multi(val) -> list[str]:
    """
    Split semicolon-delimited values and clean.
    We keep this strict because your country columns are semicolon-based.
    """
    if is_missing(val):
        return []
    out = []
    for p in str(val).split(";"):
        p = clean_token(p)
        if p and p.lower() not in MISSING_STRINGS:
            out.append(p)
    return out


def join_multi(values) -> str:
    """Deduplicate while preserving order."""
    seen = []
    for v in values:
        v = clean_token(v)
        if v and v.lower() not in MISSING_STRINGS and v not in seen:
            seen.append(v)
    return "; ".join(seen)


def normalize_locale(token: str) -> str:
    """
    Normalize locale-like forms:
      pt-br -> pt-BR
      zh-hant-hk -> zh-Hant-HK
    """
    parts = token.split("-")
    norm = []
    for i, p in enumerate(parts):
        if i == 0:
            norm.append(p.lower())
        elif len(p) == 4 and p.isalpha():
            norm.append(p.title())
        elif (len(p) == 2 and p.isalpha()) or (len(p) == 3 and p.isdigit()):
            norm.append(p.upper())
        else:
            norm.append(p.title())
    return "-".join(norm)


def normalize_lookup_token(x: str) -> str:
    """
    Normalize a language token from the master file into the same format
    used by the token-level mapping table.
    """
    x = clean_token(x)
    if not x:
        return ""

    # locale-like
    if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){1,3}", x):
        return normalize_locale(x)

    # uppercase short code -> lowercase for alias lookup
    if re.fullmatch(r"[A-Z]{2,4}", x):
        x = x.lower()

    # title-preserving-ish normalization
    parts = []
    for part in x.split():
        low = part.lower()
        if re.fullmatch(r"[a-z]{2,4}", low):
            parts.append(low)
        else:
            parts.append(part.title())
    x = " ".join(parts)

    low = x.lower()
    if low in ALIAS_MAP:
        return ALIAS_MAP[low]

    return x


def get_existing_countries(row) -> list[str]:
    """
    Pull best-available countries from all existing country columns.
    """
    countries = []
    for col in COUNTRY_COLS:
        if col in row.index:
            countries.extend(split_multi(row[col]))
    return list(dict.fromkeys(countries))


def get_languages(row) -> tuple[list[str], str]:
    """
    Return normalized language tokens plus which source column was used.
    Fallback to raw language column if normalized language is missing.
    """
    source_val = row.get(NORMALIZED_LANG_COL, None)
    source_name = NORMALIZED_LANG_COL

    if is_missing(source_val):
        source_val = row.get(RAW_LANG_COL, None)
        source_name = RAW_LANG_COL

    if is_missing(source_val):
        return [], "none"

    raw_tokens = split_multi(source_val)
    normed = []
    for t in raw_tokens:
        nt = normalize_lookup_token(t)
        if nt and nt.lower() not in MISSING_STRINGS:
            normed.append(nt)

    # dedupe preserve order
    normed = list(dict.fromkeys(normed))
    return normed, source_name


def has_country(val) -> bool:
    return len(split_multi(val)) > 0


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(MASTER_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)

# normalize column names minimally
df.columns = [c.strip() for c in df.columns]
mapping_df.columns = [c.strip() for c in mapping_df.columns]

# =========================
# BUILD MAPPING DICTIONARY
# =========================

mapping_dict = {}
for _, row in mapping_df.iterrows():
    token = normalize_lookup_token(row["normalized_token"])
    mapping_dict[token] = {
        "candidate_country": clean_token(row.get("candidate_country", "")),
        "language_signal_countries": split_multi(row.get("language_signal_countries", "")),
        "method": clean_token(row.get("country_attribution_method", "")),
        "should_expand": bool(row.get("should_auto_expand", False)),
        "bucket": clean_token(row.get("mapping_bucket", "")),
    }

# diagnostics on mapping table
mapped_expand_tokens = {
    k for k, v in mapping_dict.items()
    if v["should_expand"] and v["candidate_country"]
}

print("Unique mapping tokens:", len(mapping_dict))
print("Auto-expand mapping tokens:", len(mapped_expand_tokens))

# =========================
# APPLY MAPPING
# =========================

final_all_countries = []
final_candidate_signal = []
final_method = []

# diagnostics
diag_rows = []
matched = 0
expand_matched = 0
auto_expand_hits = Counter()
low_spread_hits = Counter()
candidate_only_hits = Counter()

for idx, row in df.iterrows():
    existing_countries = get_existing_countries(row)
    langs, lang_source = get_languages(row)

    inferred_countries = []
    signal_countries = []
    methods_used = []

    existing_method = row.get("country_attribution_method", "")
    if not is_missing(existing_method):
        methods_used.extend(split_multi(existing_method))

    matched_tokens_this_row = []
    auto_expand_tokens_this_row = []
    low_spread_tokens_this_row = []
    candidate_only_tokens_this_row = []

    for lang in langs:
        if lang not in mapping_dict:
            continue

        matched += 1
        matched_tokens_this_row.append(lang)

        entry = mapping_dict[lang]

        if entry["should_expand"] and entry["candidate_country"]:
            inferred_countries.append(entry["candidate_country"])
            auto_expand_tokens_this_row.append(lang)
            auto_expand_hits[lang] += 1
            expand_matched += 1
            if entry["method"]:
                methods_used.append(entry["method"])

        elif entry["bucket"] == "low_spread_signal_only":
            signal_countries.extend(entry["language_signal_countries"])
            low_spread_tokens_this_row.append(lang)
            low_spread_hits[lang] += 1
            if entry["method"]:
                methods_used.append(entry["method"])

        elif entry["bucket"] == "candidate_only_signal" and entry["candidate_country"]:
            signal_countries.append(entry["candidate_country"])
            candidate_only_tokens_this_row.append(lang)
            candidate_only_hits[lang] += 1
            if entry["method"]:
                methods_used.append(entry["method"])

    merged_countries = existing_countries + inferred_countries
    merged_signal = signal_countries

    final_all_countries.append(join_multi(merged_countries))
    final_candidate_signal.append(join_multi(merged_signal))
    final_method.append(join_multi(methods_used))

    diag_rows.append({
        "row_index": idx,
        "paper_url": row.get("paper_url", row.get("Paper Link (URL)", "")),
        "language_source_used": lang_source,
        "normalized_languages_used_for_lookup": "; ".join(langs),
        "matched_mapping_tokens": "; ".join(matched_tokens_this_row),
        "auto_expand_tokens_matched": "; ".join(auto_expand_tokens_this_row),
        "low_spread_tokens_matched": "; ".join(low_spread_tokens_this_row),
        "candidate_only_tokens_matched": "; ".join(candidate_only_tokens_this_row),
        "existing_countries_before": join_multi(existing_countries),
        "inferred_countries_added": join_multi(inferred_countries),
        "language_signal_countries_added": join_multi(merged_signal),
        "final_all_countries_covered": join_multi(merged_countries),
    })

# =========================
# WRITE BACK
# =========================

df["all_countries_covered"] = final_all_countries
df["Candidate Country From Language"] = final_candidate_signal
df["country_attribution_method"] = final_method

df.to_csv(OUTPUT_PATH, index=False)
pd.DataFrame(diag_rows).to_csv(DIAGNOSTICS_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Saved diagnostics to: {DIAGNOSTICS_PATH}")

# =========================
# METRICS
# =========================

rows_with_existing_before = sum(len(get_existing_countries(row)) > 0 for _, row in df.iterrows())
rows_with_country_after = df["all_countries_covered"].apply(has_country).sum()

print("\nRows with existing countries before mapping:", rows_with_existing_before)
print("\nMatched language tokens in master:", matched)
print("Matched auto-expand tokens in master:", expand_matched)

print("\nTop matched auto-expand languages in master:")
for lang, cnt in auto_expand_hits.most_common(25):
    print(f"  {lang}: {cnt}")

print("\nTop matched low-spread languages in master:")
for lang, cnt in low_spread_hits.most_common(25):
    print(f"  {lang}: {cnt}")

print("\nTop matched candidate-only languages in master:")
for lang, cnt in candidate_only_hits.most_common(25):
    print(f"  {lang}: {cnt}")

print("\nCoverage after mapping:")
print(f"{rows_with_country_after} / {len(df)} rows have country ({rows_with_country_after/len(df):.2%})")