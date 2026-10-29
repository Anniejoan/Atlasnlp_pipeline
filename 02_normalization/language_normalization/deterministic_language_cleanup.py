#!/usr/bin/env python3
"""
deterministic_language_cleanup.py — Rule-based normalization of raw language tokens

Paper section: §3.1 "Annotation Framework and Schema Design" (language metadata)
Role in pipeline: Step 7b — takes the raw language frequency table (from
  unique_languages.py) and deterministically classifies each token as:
  canonical, locale-specific, dialect/variant, invalid, or needing review.
  Output feeds into language_country_mapping.py.

Notes:
  - Conservative: preserves locale-specific forms like pt-BR, es-MX, zh-Hant-HK
  - Does NOT infer countries from language tokens (that is language_country_mapping.py)
  - Does NOT aggressively collapse dialects

Usage:
    python deterministic_language_cleanup.py \\
        --input raw_language_frequency.csv \\
        --output_dir language_cleanup_outputs

Input:
    raw_language_frequency.csv — token frequency table (from unique_languages.py)

Output (all in language_cleanup_outputs/):
    1. raw_entry_deterministic_classification.csv — per-token classification
    2. normalized_token_frequency.csv — canonical tokens with frequencies (→ language_country_mapping.py)
    3. unresolved_or_review_tokens.csv — tokens needing manual/LLM review
    4. invalid_or_metadata_tokens.csv — garbage / missing / non-language tokens
    5. locale_specific_tokens.csv — locale forms (e.g., pt-BR) for country mapping
    6. dialect_or_variant_tokens.csv — dialect/script/historical variants
    7. canonical_language_tokens.csv — clean canonical language names
    8. cleanup_summary.csv — summary counts
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

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

ALIAS_MAP: Dict[str, str] = {
    # common 2-letter / short codes
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
    # 3-letter ISO
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

NON_NATURAL_LANGUAGES = {
    "python",
    "php",
    "html",
    "css",
    "javascript",
    "java",
    "sql",
    "xml",
    "json",
    "yaml",
    "markdown",
    "abap", 
    "cpp", 
    "cuda", 
    "csv", 
    "vhdl",
    "xslt", 
    "sas", 
    "rexx", 
    "vb", 
    "jcl",
    "acsl",
    "tptp"
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

COUNTRY_CODES = {"cn", "us", "uk", "in", "au", "ca"}

MULTI_SPLIT_PATTERN = re.compile(r"\s*(?:[;,/|]|\band\b|\+)\s*", flags=re.IGNORECASE)

LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){1,3}$")
UPPER_CODE_PATTERN = re.compile(r"^[A-Z]{2,4}$")


# -----------------------------
# Helpers
# -----------------------------

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
    text = text.strip()
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
        if re.fullmatch(r"[a-z]{2,4}", part):
            parts.append(part.lower())
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


def looks_multilingual(text: str) -> bool:
    parts = [p.strip() for p in MULTI_SPLIT_PATTERN.split(text) if p.strip()]
    return len(parts) > 1


def split_multilingual(text: str) -> List[str]:
    parts = [p.strip() for p in MULTI_SPLIT_PATTERN.split(text) if p.strip()]
    return parts if parts else [text]


def classify_single_token(token: str) -> Tuple[Optional[str], str, str]:
    """
    Returns:
        normalized_value, entry_type, note

    entry_type in:
        missing
        invalid_metadata
        non_natural_language
        locale_specific
        iso_mapped
        alias_mapped
        script_variant
        dialect_or_variant
        canonical_language
        needs_review
    """
    token = basic_clean(token)

    if not token:
        return None, "missing", "empty after cleaning"

    low = token.lower()

    if low in MISSING_VALUES:
        return None, "missing", "placeholder missing value"
    
    if low in COUNTRY_CODES:
        return None, "invalid_metadata", "country code, not language"

    if low in NON_NATURAL_LANGUAGES:
        return None, "non_natural_language", "programming or non-natural language"

    if looks_invalid_metadata(token):
        return None, "invalid_metadata", "metadata/count/invalid entry"

    # Locale-specific forms like pt-BR, zh-Hant-HK
    if LOCALE_PATTERN.match(token):
        return normalize_locale(token), "locale_specific", "locale/script tag preserved"

    # Uppercase code-like tokens: EN, FR, DE, IND, VIE, THA, FIL
    if UPPER_CODE_PATTERN.match(token):
        low_code = token.lower()
        if low_code in ALIAS_MAP:
            return ALIAS_MAP[low_code], "iso_mapped", "uppercase code mapped via alias table"
        return token, "needs_review", "uppercase code-like token; unresolved deterministically"

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

    # Sign languages are valid languages
    if "sign language" in low_norm:
        return token_norm, "canonical_language", "sign language"

    # Script / orthography variants
    if any(k in low_norm for k in SCRIPT_KEYWORDS):
        return token_norm, "script_variant", "script/orthography-related entry"

    # Pidgin / creole / dialect / regional / historical
    if "pidgin" in low_norm or "creole" in low_norm:
        return token_norm, "dialect_or_variant", "pidgin or creole variant"

    if any(k in low_norm for k in VARIANT_HINTS):
        return token_norm, "dialect_or_variant", "regional/dialect/historical variant"

    # Parenthetical qualifiers are often variants, not invalid
    if "(" in token_norm or ")" in token_norm:
        return token_norm, "dialect_or_variant", "parenthetical qualifier preserved"

    # Hyphenated language names that are not locale tags are often still valid
    if "-" in token_norm and not LOCALE_PATTERN.match(token_norm):
        return token_norm, "canonical_language", "hyphenated language name preserved"

    if re.search(r"\d", token_norm):
        return None, "invalid_metadata", "contains digits; likely metadata"

    return token_norm, "canonical_language", "kept as cleaned canonical token"


def classify_raw_entry(raw_value: str, frequency: int) -> List[Dict]:
    cleaned = basic_clean(str(raw_value))

    if not cleaned:
        return [{
            "raw_value": raw_value,
            "frequency": frequency,
            "cleaned_raw_value": cleaned,
            "is_multilingual_entry": False,
            "raw_token": "",
            "normalized_token": None,
            "entry_type": "missing",
            "note": "empty after cleaning",
        }]

    if looks_invalid_metadata(cleaned):
        return [{
            "raw_value": raw_value,
            "frequency": frequency,
            "cleaned_raw_value": cleaned,
            "is_multilingual_entry": False,
            "raw_token": cleaned,
            "normalized_token": None,
            "entry_type": "invalid_metadata",
            "note": "metadata/count/invalid full-cell entry",
        }]

    is_multi = looks_multilingual(cleaned)
    tokens = split_multilingual(cleaned) if is_multi else [cleaned]

    rows = []
    for tok in tokens:
        norm, entry_type, note = classify_single_token(tok)
        rows.append({
            "raw_value": raw_value,
            "frequency": frequency,
            "cleaned_raw_value": cleaned,
            "is_multilingual_entry": is_multi,
            "raw_token": tok,
            "normalized_token": norm,
            "entry_type": entry_type,
            "note": note,
        })
    return rows


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw language frequency CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--language_col", default="Language", help="Language column name")
    parser.add_argument("--frequency_col", default="Frequency", help="Frequency column name")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    if args.language_col not in df.columns or args.frequency_col not in df.columns:
        raise ValueError(
            f"Input CSV must contain columns '{args.language_col}' and '{args.frequency_col}'. "
            f"Found: {list(df.columns)}"
        )

    df = df[[args.language_col, args.frequency_col]].copy()
    df[args.language_col] = df[args.language_col].astype(str)
    df[args.frequency_col] = pd.to_numeric(df[args.frequency_col], errors="coerce").fillna(0).astype(int)

    classified_rows = []
    for _, row in df.iterrows():
        classified_rows.extend(
            classify_raw_entry(
                raw_value=row[args.language_col],
                frequency=row[args.frequency_col],
            )
        )

    classified = pd.DataFrame(classified_rows)

    valid_inventory_types = {
        "canonical_language",
        "alias_mapped",
        "iso_mapped",
        "locale_specific",
        "script_variant",
        "dialect_or_variant",
        "needs_review",
    }

    valid_for_inventory = classified[
        classified["normalized_token"].notna() &
        classified["entry_type"].isin(valid_inventory_types)
    ].copy()

    token_freq = (
        valid_for_inventory
        .groupby(["normalized_token", "entry_type"], dropna=False)["frequency"]
        .sum()
        .reset_index()
        .sort_values(["frequency", "normalized_token"], ascending=[False, True])
    )

    unresolved = classified[
        classified["entry_type"] == "needs_review"
    ].copy().sort_values(["frequency", "raw_token"], ascending=[False, True])

    invalid = classified[
        classified["entry_type"].isin({"missing", "invalid_metadata", "non_natural_language"})
    ].copy().sort_values(["frequency", "raw_token"], ascending=[False, True])

    locales = classified[
        classified["entry_type"] == "locale_specific"
    ].copy().sort_values(["frequency", "normalized_token"], ascending=[False, True])

    variants = classified[
        classified["entry_type"].isin({"dialect_or_variant", "script_variant"})
    ].copy().sort_values(["frequency", "normalized_token"], ascending=[False, True])

    canonicals = classified[
        classified["entry_type"].isin({"canonical_language", "alias_mapped", "iso_mapped"})
    ].copy().sort_values(["frequency", "normalized_token"], ascending=[False, True])

    classified.sort_values(["frequency", "raw_value"], ascending=[False, True]).to_csv(
        outdir / "raw_entry_deterministic_classification.csv",
        index=False
    )

    token_freq.to_csv(
        outdir / "normalized_token_frequency.csv",
        index=False
    )

    unresolved.to_csv(
        outdir / "unresolved_or_review_tokens.csv",
        index=False
    )

    invalid.to_csv(
        outdir / "invalid_or_metadata_tokens.csv",
        index=False
    )

    locales.to_csv(
        outdir / "locale_specific_tokens.csv",
        index=False
    )

    variants.to_csv(
        outdir / "dialect_or_variant_tokens.csv",
        index=False
    )

    canonicals.to_csv(
        outdir / "canonical_language_tokens.csv",
        index=False
    )

    summary = {
        "raw_unique_full_cell_entries": int(df.shape[0]),
        "raw_total_weighted_frequency": int(df[args.frequency_col].sum()),
        "classified_rows_after_tokenization": int(classified.shape[0]),
        "unique_normalized_tokens": int(token_freq["normalized_token"].nunique()),
        "weighted_frequency_valid_inventory": int(valid_for_inventory["frequency"].sum()),
        "weighted_frequency_multilingual_entries": int(
            classified.loc[classified["is_multilingual_entry"], "frequency"].sum()
        ),
        "weighted_frequency_missing": int(
            classified.loc[classified["entry_type"] == "missing", "frequency"].sum()
        ),
        "weighted_frequency_invalid_metadata": int(
            classified.loc[classified["entry_type"].isin({"invalid_metadata", "non_natural_language"}), "frequency"].sum()
        ),
        "weighted_frequency_locale_specific": int(
            classified.loc[classified["entry_type"] == "locale_specific", "frequency"].sum()
        ),
        "weighted_frequency_needs_review": int(
            classified.loc[classified["entry_type"] == "needs_review", "frequency"].sum()
        ),
        "weighted_frequency_canonical_language": int(
            classified.loc[classified["entry_type"] == "canonical_language", "frequency"].sum()
        ),
        "weighted_frequency_alias_mapped": int(
            classified.loc[classified["entry_type"] == "alias_mapped", "frequency"].sum()
        ),
        "weighted_frequency_iso_mapped": int(
            classified.loc[classified["entry_type"] == "iso_mapped", "frequency"].sum()
        ),
        "weighted_frequency_dialect_or_variant": int(
            classified.loc[classified["entry_type"] == "dialect_or_variant", "frequency"].sum()
        ),
        "weighted_frequency_script_variant": int(
            classified.loc[classified["entry_type"] == "script_variant", "frequency"].sum()
        ),
    }

    summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    summary_df.to_csv(outdir / "cleanup_summary.csv", index=False)

    print("\nDeterministic cleanup complete.")
    print(f"Outputs written to: {outdir.resolve()}")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()