#!/usr/bin/env python3
"""
merge_normalized_language.py — Write normalized language tokens back to the master dataset

Paper section: §3.1 "Annotation Framework and Schema Design" (language normalization)
               pipeline_tracker.md: Stage 3.5
Role in pipeline: Step 7e — adds the 'Normalized Languages' column to the master CSV,
  converting raw language strings (from LLM extraction) into cleaned, canonical tokens.
  This is the output used by language_country_mapping.py and all downstream steps.
  Run after deterministic_language_cleanup.py and before language_country_mapping.py.

Run from compiled_ACL/:
  python merge_normalized_language.py

Input:
    updated_master.csv  (from country_second_pass/country_second_pass_merged/)

Output:
    updated_master_with_normalized_languages.csv  — primary master file for all downstream analysis
    normalized_languages_rowlevel_diagnostics.csv — per-row diagnostics on normalization applied
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# CONFIG
# =========================

MASTER_PATH = "/home/jnwatu/Worldbench/compiled_ACL/updated_master_with_language_mapping_v2.csv"
OUTPUT_PATH = "updated_master_with_normalized_languages.csv"
DIAGNOSTICS_PATH = "normalized_languages_rowlevel_diagnostics.csv"

RAW_LANG_COL = "Languages covered (separate using semicolon)"
NORM_LANG_COL = "Normalized Languages"

MISSING_VALUES = {
    "",
    "not stated",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "not available",
    "unspecified",
}

EXACT_INVALID_VALUES = {
    "no",
    "not english",
    "synthetic languages",
}

INVALID_SUBSTRINGS = [
    " languages",
    "different low-resource languages",
    "others",
]

INVALID_PATTERNS = [
    r"^\d+\s+languages$",
    r"^\d+\s+different.*languages$",
    r"^and\s+\d+\s+others$",
]

NON_NATURAL_LANGUAGES = {
    "python",
    "php",
    "html",
    "css",
    "javascript",
    "sql",
    "xml",
    "json",
    "yaml",
    "markdown",
    "csv",
    "cuda",
    "abap",
    "cpp",
    "rexx",
    "sas",
    "vb",
    "vhdl",
    "xslt",
    "jcl",
    "tptp",
    "acsl",
}

SCRIPT_KEYWORDS = {
    "simplified",
    "traditional",
    "katakana",
    "latin",
    "cyrillic",
    "nkoo",
    "deva",
    "latn",
    "script",
}

VARIANT_HINTS = {
    "argentina",
    "argentine",
    "argentinian",
    "swiss",
    "egyptian",
    "north",
    "south",
    "belgian",
    "gulf",
    "levantine",
    "north-african",
    "historical",
    "old",
    "middle",
    "neo",
    "standard",
    "contemporary",
    "eastern",
    "southern",
    "western",
    "peripheral",
}

MULTI_SPLIT_PATTERN = re.compile(r"\s*(?:[;,/|]|\band\b|\+)\s*", flags=re.IGNORECASE)
LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){1,3}$")
UPPER_CODE_PATTERN = re.compile(r"^[A-Z]{2,4}$")

ALIAS_MAP: Dict[str, str] = {
    # common codes
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
    "ota": "Ottoman Turkish",
    "orv": "Old East Slavic",
    "lfn": "Lingua Franca Nova",
    "doi": "Dogri",
    "ccp": "Chakma",
    "xnr": "Kangri",
    "mjl": "Mandeali",
    "lif-limb": "Limbu",
    "bgq": "Bagri",
    "kru": "Kurukh",
    "hoc-wara": "Ho; Warang Citi",
    "bra": "Braj",
    "bns": "Bundeli",
    "mag": "Magahi",
    "wbr": "Wagdi",
    "bfy": "Bagheli",
    "bho": "Bhojpuri",
    "mtr": "Mewari",
    "tcy": "Tulu",
    "ahr": "Ahirani",
    "dhd": "Dhundhari",
    "bfq": "Badaga",
    "kfy": "Kumaoni",
    "trp": "Kok Borok",
    "cgg": "Chiga",
    "din": "Dinka",
    "tiv": "Tiv",
    "kr": "Kanuri",
    "alz": "Alur",
    "bem": "Bemba",
    "tum": "Tumbuka",
    "dyu": "Dyula",
    "ach": "Acholi",
    "dje": "Zarma",
    "aa": "Afar",
    "bci": "Baoulé",
    "sus": "Susu",
    "gaa": "Ga",
    "lu": "Luba-Katanga",
    "noe": "Nimadi",
    "scl": "Shina",
    "lep": "Lepcha",
    "sjp": "Surjapuri",
    "apd": "Sudanese Arabic",
    "ayl": "Libyan Arabic",
    "kl": "Kalaallisut",
    "nbl": "Southern Ndebele",
    "ssw": "Swati",
    "ind": "Indonesian",
    "vie": "Vietnamese",
    "tha": "Thai",
    "fil": "Filipino",
    "zsm": "Malay",
    "sv": "Swedish",
    "et": "Estonian",
    "sl": "Slovenian",
    "sr": "Serbian",
    "th": "Thai",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "ca": "Catalan",
    "lt": "Lithuanian",
    "ml": "Malayalam",
    "qu": "Quechua",
    "sw": "Swahili",
    "ta": "Tamil",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "eng": "English",
    "fin": "Finnish",
    "gre": "Greek",
    "heb": "Hebrew",
    "hun": "Hungarian",
    "ita": "Italian",
    "jpn": "Japanese",
    "pol": "Polish",
    "rus": "Russian",
    "spa": "Spanish",
    "swe": "Swedish",
    "tur": "Turkish",
    "dan": "Danish",
    "nor": "Norwegian",
    "khm": "Khmer",
    "mya": "Burmese",
    "jav": "Javanese",
    "zlm": "Malay",
    "eo": "Esperanto",
    "fa": "Persian",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "dgs": "German Sign Language",
    "vgt": "Flemish Sign Language",
    "mas": "Masai",
    "sun": "Sundanese",

    # synonyms and variants
    "farsi": "Persian",
    "bangla": "Bengali",
    "mandarin": "Chinese",
    "mandarin chinese": "Chinese",
    "chinese mandarin": "Chinese",
    "indonesian bahasa": "Indonesian",
    "slovene": "Slovenian",
    "serbo-crotian": "Serbo-Croatian",
    "serbo-croation": "Serbo-Croatian",
    "simpliﬁed chinese": "Simplified Chinese",
    "simplified chinese": "Simplified Chinese",
    "chinese-simplified": "Simplified Chinese",
    "chinese-traditional": "Traditional Chinese",
    "british english": "English (UK)",
    "argentine spanish": "Spanish (Argentina)",
    "argentinian spanish": "Spanish (Argentina)",
    "spanish (castilian)": "Spanish",
    # add to ALIAS_MAP
    "java": "Javanese",
    "odia": "Odia",
    "indian sign language": "Indian Sign Language",
     "swiss-german": "German (Switzerland)",
    "egyptian dialectal arabic": "Arabic (Egyptian)",
    "kurdish (kurmanji)": "Kurmanji",
    "akan (twi)": "Akan; Twi",
    "punjabi (indian)": "Punjabi",
    "eastern punjabi": "Punjabi",
    "welsh (north wales)": "Welsh",
    "welsh (south wales)": "Welsh",
    "french belgian sign language": "Belgian French Sign Language",
    "russia buriat": "Buriat",
    "old bulgarian": "Old Church Slavonic",
    "old slavic": "Old Church Slavonic",
    "historical slovene": "Slovenian (Historical)",
    "middle english": "Middle English",
    "old french": "Old French",
    "old spanish": "Old Spanish",
    "middle russian": "Middle Russian",
    "contemporary standard russian": "Russian",
    "old east slavonic": "Old East Slavic",
    "proto indo-european": "Proto-Indo-European",
    "kanien'kéha": "Kanien’kéha",
    "mbyá guaraní": "Mbyá Guaraní",
    "nigerian-pidgin": "Nigerian Pidgin",
    "serbo-croatian (cyrillic)": "Serbo-Croatian",
    "serbo-croatian (latin)": "Serbo-Croatian",
}


# =========================
# HELPERS
# =========================

def clean_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "")
    replacements = {
        "ï¬": "fi",
        "Ã©": "é",
        "Ã¡": "á",
        "Ã­": "í",
        "Ã¼": "ü",
        "Ã±": "ñ",
        "Ã³": "ó",
        "Ã ": "à",
        "Ã¨": "è",
        "Ã¢": "â",
        "Ãª": "ê",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def basic_clean(text: str) -> str:
    text = clean_unicode(text)
    text = str(text).strip()
    text = re.sub(r'^[\'"]+', '', text)
    text = re.sub(r'[\'"]+$', '', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(";,/|")
    return text


def normalize_locale(text: str) -> str:
    pieces = text.split("-")
    norm = []
    for i, p in enumerate(pieces):
        if i == 0:
            norm.append(p.lower())
        elif len(p) == 4 and p.isalpha():
            norm.append(p.title())
        elif (len(p) == 2 and p.isalpha()) or (len(p) == 3 and p.isdigit()):
            norm.append(p.upper())
        else:
            norm.append(p.title())
    return "-".join(norm)


def title_preserving_special_forms(text: str) -> str:
    if LOCALE_PATTERN.match(text):
        return normalize_locale(text)

    parts = []
    for part in text.split():
        low = part.lower()
        if re.fullmatch(r"[a-z]{2,4}", low):
            parts.append(low)
        else:
            parts.append(part.title())
    return " ".join(parts)


def looks_invalid_metadata(text: str) -> bool:
    low = text.lower()
    if low in EXACT_INVALID_VALUES:
        return True
    if low in MISSING_VALUES:
        return True
    if low in NON_NATURAL_LANGUAGES:
        return True
    for pat in INVALID_PATTERNS:
        if re.fullmatch(pat, low):
            return True
    for s in INVALID_SUBSTRINGS:
        if s in low:
            return True
    return False


def split_raw_language_cell(text: str) -> List[str]:
    """
    Split the raw language cell using the same pattern used earlier.
    This supports semicolons, commas, slashes, pipes, 'and', plus.
    """
    text = basic_clean(text)
    if not text or text.lower() in MISSING_VALUES:
        return []

    parts = [p.strip() for p in MULTI_SPLIT_PATTERN.split(text) if p.strip()]
    return parts if parts else [text]


def classify_single_token(token: str) -> Tuple[Optional[str], str, str]:
    token = basic_clean(token)

    if not token:
        return None, "missing", "empty after cleaning"

    low = token.lower()

    if low in MISSING_VALUES:
        return None, "missing", "placeholder missing value"

    if low in NON_NATURAL_LANGUAGES:
        return None, "non_natural_language", "programming or non-natural language"

    if looks_invalid_metadata(token):
        return None, "invalid_metadata", "metadata/count/invalid entry"

    if LOCALE_PATTERN.match(token):
        return normalize_locale(token), "locale_specific", "locale/script tag preserved"

    if UPPER_CODE_PATTERN.match(token):
        low_code = token.lower()
        if low_code in ALIAS_MAP:
            return ALIAS_MAP[low_code], "iso_mapped", "uppercase code mapped via alias table"
        return token, "needs_review", "uppercase code-like token unresolved"

    token_norm = title_preserving_special_forms(token)
    low_norm = token_norm.lower()

    if low_norm in ALIAS_MAP:
        mapped = ALIAS_MAP[low_norm]
        if LOCALE_PATTERN.match(mapped):
            return normalize_locale(mapped), "locale_specific", "alias mapped to locale"
        if "sign language" in mapped.lower():
            return mapped, "canonical_language", "sign language alias mapped"
        if any(k in low_norm for k in SCRIPT_KEYWORDS):
            return mapped, "script_variant", "alias mapped from script/orthography variant"
        if any(k in low_norm for k in VARIANT_HINTS) or "pidgin" in low_norm or "creole" in low_norm:
            return mapped, "dialect_or_variant", "alias mapped from dialect/regional/historical/pidgin variant"
        return mapped, "alias_mapped", "alias or synonym mapped"

    if "sign language" in low_norm:
        return token_norm, "canonical_language", "sign language"

    if any(k in low_norm for k in SCRIPT_KEYWORDS):
        return token_norm, "script_variant", "script/orthography-related entry"

    if "pidgin" in low_norm or "creole" in low_norm:
        return token_norm, "dialect_or_variant", "pidgin or creole variant"

    if any(k in low_norm for k in VARIANT_HINTS):
        return token_norm, "dialect_or_variant", "regional/dialect/historical variant"

    if "(" in token_norm or ")" in token_norm:
        return token_norm, "dialect_or_variant", "parenthetical qualifier preserved"

    if "-" in token_norm and not LOCALE_PATTERN.match(token_norm):
        return token_norm, "canonical_language", "hyphenated language name preserved"

    if re.search(r"\d", token_norm):
        return None, "invalid_metadata", "contains digits; likely metadata"

    return token_norm, "canonical_language", "kept as cleaned canonical token"


def normalize_language_cell(raw_value) -> Tuple[str, List[dict]]:
    """
    Normalize one row's raw language cell.
    Returns:
      normalized_languages_string
      diagnostics list of token-level decisions
    """
    if pd.isna(raw_value):
        return "Not stated", []

    raw_text = basic_clean(raw_value)
    if not raw_text or raw_text.lower() in MISSING_VALUES:
        return "Not stated", []

    raw_tokens = split_raw_language_cell(raw_text)

    normalized_tokens = []
    diagnostics = []

    for tok in raw_tokens:
        norm, entry_type, note = classify_single_token(tok)
        diagnostics.append({
            "raw_token": tok,
            "normalized_token": norm,
            "entry_type": entry_type,
            "note": note,
        })

        # Keep all valid normalized language labels, not just canonical
        if norm is not None and entry_type in {
            "canonical_language",
            "alias_mapped",
            "iso_mapped",
            "locale_specific",
            "dialect_or_variant",
            "script_variant",
        }:
            if norm not in normalized_tokens:
                normalized_tokens.append(norm)

    if not normalized_tokens:
        return "Not stated", diagnostics

    return "; ".join(normalized_tokens), diagnostics


# =========================
# MAIN
# =========================

df = pd.read_csv(MASTER_PATH)
df.columns = [c.strip() for c in df.columns]

if RAW_LANG_COL not in df.columns:
    raise ValueError(f"Missing required column: {RAW_LANG_COL}")

normalized_values = []
diag_rows = []

for idx, row in df.iterrows():
    raw_val = row.get(RAW_LANG_COL, None)
    normalized_str, token_diags = normalize_language_cell(raw_val)
    normalized_values.append(normalized_str)

    diag_rows.append({
        "row_index": idx,
        "paper_url": row.get("paper_url", row.get("Paper Link (URL)", "")),
        "raw_language_cell": raw_val,
        "normalized_languages": normalized_str,
        "token_count": len(token_diags),
        "token_details": " || ".join(
            f"{d['raw_token']} => {d['normalized_token']} [{d['entry_type']}]"
            for d in token_diags
        ),
    })

df[NORM_LANG_COL] = normalized_values

df.to_csv(OUTPUT_PATH, index=False)
pd.DataFrame(diag_rows).to_csv(DIAGNOSTICS_PATH, index=False)

# Summary
total_rows = len(df)
not_stated_count = (df[NORM_LANG_COL].fillna("Not stated").astype(str).str.strip() == "Not stated").sum()
with_norm = total_rows - not_stated_count

print(f"Saved to: {OUTPUT_PATH}")
print(f"Saved diagnostics to: {DIAGNOSTICS_PATH}")
print("\nNormalized Languages summary:")
print(f"Rows with normalized languages: {with_norm} / {total_rows} ({with_norm/total_rows:.2%})")
print(f"Rows still 'Not stated': {not_stated_count} / {total_rows} ({not_stated_count/total_rows:.2%})")