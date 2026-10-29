#!/usr/bin/env python3
"""
language_country_mapping.py — Map normalized language tokens to countries

Paper section: §3.3 "Automated Dataset Expansion" (language-informed attribution)
               §5.5 "Language Metadata Overestimates Geographic Representation"
Role in pipeline: Step 7c — implements the conservative language→country inference
  policy described in the paper. Widely-distributed languages (English, Spanish,
  French, Arabic, Portuguese) are NOT assigned to specific countries without
  supporting context. Only high-concentration or locale-specific languages
  receive auto-expanded country assignments.
  Output feeds into language_mapping_update_master.py.

Run: python language_country_mapping.py --input language_cleanup_outputs/normalized_token_frequency.csv --output_dir language_country_mapping_outputs

Input:
    normalized_token_frequency.csv
    Expected columns:
        - normalized_token
        - entry_type
        - frequency

Output:
    1. language_country_mapping_table.csv
       Main mapping table with one row per normalized token

    2. auto_expand_language_tokens.csv
       Tokens that should automatically expand into primary country assignments

    3. candidate_only_language_tokens.csv
       Tokens with a single candidate_country but no auto-expansion

    4. ambiguous_language_tokens.csv
       Tokens that should not be mapped to country

    5. low_spread_language_signal_tokens.csv
       Tokens with language-based regional signal across 2-3 countries,
       not used for strict country assignment but preserved for analysis

    6. mapping_summary.csv
       Summary counts

Policy:
    STRICT PRIMARY COUNTRY ASSIGNMENT:
        - locale_specific -> auto-expand if region maps to one allowed country
        - high-concentration canonical languages -> assign ONE dominant country, auto-expand
        - explicit regional/dialect variants -> assign one country, auto-expand if explicit enough

    SOFT LANGUAGE SIGNAL LAYER:
        - low-spread languages -> preserve language_signal_countries, do not auto-expand
        - candidate-only signals -> preserve a single candidate_country, do not auto-expand

    NO COUNTRY MAPPING:
        - script variants
        - globally distributed canonical languages
        - unresolved / ambiguous cases
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# --------------------------------------------------
# Allowed countries
# --------------------------------------------------

ALLOWED_COUNTRIES = {
    "Nigeria", "South Africa", "Angola", "Burkina Faso", "Cabo Verde", "Morocco",
    "Egypt", "Tunisia", "Benin", "Malawi", "South Sudan",
    "Democratic Republic of the Congo", "Kenya", "Algeria", "Mozambique", "Zambia",
    "Guinea", "Senegal", "Madagascar", "Cameroon", "Cote d'Ivoire", "Guinea-Bissau",
    "São Tomé and Príncipe", "Libya", "Ethiopia", "Uganda", "Rwanda", "Tanzania",
    "Togo", "Burundi", "Ghana", "Eritrea", "Niger", "Somalia", "Zimbabwe",
    "Djibouti", "Mali", "Central African Republic", "Namibia", "Sierra Leone",
    "Comoros", "Mauritius", "Seychelles", "Gambia", "Liberia", "Equatorial Guinea",
    "Sudan", "Gabon", "Botswana", "Chad", "Congo (Congo-Brazzaville)", "Eswatini",
    "Lesotho", "Mauritania", "United States", "Canada", "Mexico", "Brazil",
    "Paraguay", "Peru", "Bolivia", "Colombia", "Chile", "Guatemala", "Ecuador",
    "Argentina", "Costa Rica", "Cuba", "Dominican Republic", "El Salvador",
    "Honduras", "Nicaragua", "Panama", "Uruguay", "Venezuela", "Haiti", "Jamaica",
    "Suriname", "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Dominica",
    "Grenada", "Guyana", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Trinidad and Tobago", "India",
    "Philippines", "Singapore", "China", "Japan", "South Korea", "Myanmar",
    "Indonesia", "Vietnam", "Turkey", "Iraq", "Saudi Arabia", "Malaysia", "Iran",
    "Bangladesh", "Pakistan", "Taiwan", "Thailand", "Kazakhstan", "Kyrgyzstan",
    "Timor-Leste", "Lebanon", "Syria", "Yemen", "Bahrain", "Jordan", "Kuwait",
    "Oman", "Qatar", "United Arab Emirates", "Sri Lanka", "Tajikistan",
    "Afghanistan", "Israel", "Azerbaijan", "Uzbekistan", "Nepal", "Brunei",
    "Georgia", "Cambodia", "Armenia", "Mongolia", "Laos", "Maldives",
    "Turkmenistan", "Bhutan", "Cyprus", "North Korea", "Palestine",
    "United Kingdom", "Ireland", "Spain", "Italy", "Germany", "Russia", "France",
    "Switzerland", "Poland", "Netherlands", "Latvia", "Portugal", "Romania",
    "Greece", "Slovenia", "Serbia", "Slovakia", "Luxembourg", "Belgium", "Ukraine",
    "Belarus", "Austria", "Czechia", "Denmark", "Sweden", "Norway", "Finland",
    "Hungary", "Bulgaria", "Estonia", "Iceland", "Albania", "Croatia",
    "Lithuania", "Malta", "North Macedonia", "Vatican City",
    "Bosnia and Herzegovina", "Moldova", "Montenegro", "Andorra", "Kosovo",
    "Liechtenstein", "Monaco", "San Marino", "Australia", "New Zealand",
    "Papua New Guinea", "Samoa", "Fiji", "Tonga", "Vanuatu", "Marshall Islands",
    "Micronesia", "Nauru", "Solomon Islands", "Palau", "Kiribati", "Tuvalu",
}


# --------------------------------------------------
# Policy lookup tables
# --------------------------------------------------

REGION_TO_COUNTRY = {
    "BR": "Brazil", "PT": "Portugal", "MX": "Mexico", "HK": "China", "TW": "Taiwan",
    "CN": "China", "JP": "Japan", "KR": "South Korea", "VN": "Vietnam",
    "TH": "Thailand", "IN": "India", "PK": "Pakistan", "BD": "Bangladesh",
    "ID": "Indonesia", "MY": "Malaysia", "SG": "Singapore", "PH": "Philippines",
    "NG": "Nigeria", "ZA": "South Africa", "GB": "United Kingdom", "UK": "United Kingdom",
    "US": "United States", "CA": "Canada", "IE": "Ireland", "DE": "Germany",
    "FR": "France", "ES": "Spain", "IT": "Italy", "NL": "Netherlands",
    "BE": "Belgium", "CH": "Switzerland", "AT": "Austria", "PL": "Poland",
    "CZ": "Czechia", "SK": "Slovakia", "HU": "Hungary", "RO": "Romania",
    "BG": "Bulgaria", "GR": "Greece", "SE": "Sweden", "NO": "Norway",
    "DK": "Denmark", "FI": "Finland", "EE": "Estonia", "LV": "Latvia",
    "LT": "Lithuania", "UA": "Ukraine", "RU": "Russia", "RS": "Serbia",
    "SI": "Slovenia", "HR": "Croatia", "MK": "North Macedonia", "AL": "Albania",
    "TR": "Turkey", "IR": "Iran", "IQ": "Iraq", "SA": "Saudi Arabia",
    "AE": "United Arab Emirates", "IL": "Israel", "NP": "Nepal", "LK": "Sri Lanka",
    "MM": "Myanmar", "KH": "Cambodia", "LA": "Laos", "MN": "Mongolia",
    "KZ": "Kazakhstan", "UZ": "Uzbekistan", "KG": "Kyrgyzstan", "TJ": "Tajikistan",
    "AZ": "Azerbaijan", "GE": "Georgia", "AM": "Armenia", "ET": "Ethiopia",
    "KE": "Kenya", "TZ": "Tanzania", "UG": "Uganda", "RW": "Rwanda", "GH": "Ghana",
    "SN": "Senegal", "ML": "Mali", "CM": "Cameroon", "MA": "Morocco",
    "DZ": "Algeria", "TN": "Tunisia", "EG": "Egypt", "AU": "Australia",
    "NZ": "New Zealand", "FJ": "Fiji", "PG": "Papua New Guinea",
}

LOCALE_NOTES = {
    "HK": "Locale indicates Hong Kong; mapped to China because Hong Kong is not a separate country in the allowed country list.",
}

# Strict one-country auto-expansion
HIGH_CONCENTRATION_LANGUAGE_TO_COUNTRY = {
    "Albanian": "Albania",
    "Amharic": "Ethiopia",
    "Armenian": "Armenia",
    "Azerbaijani": "Azerbaijan",
    "Basque": "Spain",
    "Bulgarian": "Bulgaria",
    "Burmese": "Myanmar",
    "Croatian": "Croatia",
    "Czech": "Czechia",
    "Danish": "Denmark",
    "Estonian": "Estonia",
    "Finnish": "Finland",
    "Georgian": "Georgia",
    "Greek": "Greece",
    "Gujarati": "India",
    "Hebrew": "Israel",
    "Hungarian": "Hungary",
    "Icelandic": "Iceland",
    "Japanese": "Japan",
    "Kannada": "India",
    "Kazakh": "Kazakhstan",
    "Khmer": "Cambodia",
    "Korean": "South Korea",
    "Kyrgyz": "Kyrgyzstan",
    "Latvian": "Latvia",
    "Lithuanian": "Lithuania",
    "Macedonian": "North Macedonia",
    "Malayalam": "India",
    "Maltese": "Malta",
    "Marathi": "India",
    "Mongolian": "Mongolia",
    "Nepali": "Nepal",
    "Norwegian": "Norway",
    "Pashto": "Afghanistan",
    "Persian": "Iran",
    "Polish": "Poland",
    "Romanian": "Romania",
    "Russian": "Russia",
    "Serbian": "Serbia",
    "Sinhala": "Sri Lanka",
    "Slovak": "Slovakia",
    "Slovenian": "Slovenia",
    "Swedish": "Sweden",
    "Telugu": "India",
    "Thai": "Thailand",
    "Turkish": "Turkey",
    "Ukrainian": "Ukraine",
    "Urdu": "Pakistan",
    "Uzbek": "Uzbekistan",
    "Vietnamese": "Vietnam",
    "Welsh": "United Kingdom",
    "Zulu": "South Africa",
}

# Strict explicit variants
VARIANT_TO_COUNTRY = {
    "Arabic (Egyptian)": "Egypt",
    "English (UK)": "United Kingdom",
    "German (Switzerland)": "Switzerland",
    "Nigerian Pidgin": "Nigeria",
    "Spanish (Argentina)": "Argentina",
}

# Low-spread soft regional signal only, not strict assignment
LOW_SPREAD_LANGUAGE_TO_COUNTRIES = {
    "Hausa": ["Nigeria", "Niger"],
    "Yoruba": ["Nigeria", "Benin"],
    "Bengali": ["Bangladesh", "India"],
    "Tamil": ["India", "Sri Lanka"],
    "Dutch": ["Netherlands", "Belgium"],
}

# Optional single-country soft signal only
CANDIDATE_ONLY_LANGUAGE_TO_COUNTRY = {
    "Catalan": "Spain",
   # "Portuguese": "Brazil", 
}

GLOBAL_OR_BROAD_LANGUAGES = {
    "Arabic", "Chinese", "English", "French", "German", "Hindi",
    "Indonesian", "Italian", "Malay", "Punjabi", "Spanish", "Swahili", "Portuguese", # Portuguese added
}

SCRIPT_VARIANTS_NO_COUNTRY = {
    "Simplified Chinese",
    "Traditional Chinese",
    "Katakana",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def safe_country(country: Optional[str]) -> Optional[str]:
    if country is None:
        return None
    return country if country in ALLOWED_COUNTRIES else None


def safe_countries(countries: List[str]) -> List[str]:
    return [c for c in countries if c in ALLOWED_COUNTRIES]


def parse_locale_country(token: str) -> Tuple[Optional[str], str]:
    parts = token.split("-")
    if len(parts) < 2:
        return None, "No region subtag found."

    region = None
    for p in reversed(parts[1:]):
        if re.fullmatch(r"[A-Z]{2}", p):
            region = p
            break

    if region is None:
        return None, "No mappable region subtag found."

    country = safe_country(REGION_TO_COUNTRY.get(region))
    if country is None:
        return None, f"Region subtag {region} not mapped to an allowed country."

    note = LOCALE_NOTES.get(region, f"Mapped from locale region subtag {region}.")
    return country, note


def map_token_to_country(token: str, entry_type: str) -> Dict[str, object]:
    result = {
        "candidate_country": None,
        "language_signal_countries": [],
        "country_attribution_method": None,
        "should_auto_expand": False,
        "confidence": None,
        "note": None,
        "mapping_bucket": None,
    }

    # 1. Locale-specific: strongest strict signal
    if entry_type == "locale_specific":
        country, note = parse_locale_country(token)
        if country:
            result.update({
                "candidate_country": country,
                "country_attribution_method": "inferred_from_language_locale",
                "should_auto_expand": True,
                "confidence": "high",
                "note": note,
                "mapping_bucket": "auto_locale",
            })
        else:
            result.update({
                "country_attribution_method": "locale_unresolved",
                "should_auto_expand": False,
                "confidence": "low",
                "note": note,
                "mapping_bucket": "unresolved_locale",
            })
        return result

    # 2. Script variants: no country inference
    if entry_type == "script_variant" or token in SCRIPT_VARIANTS_NO_COUNTRY:
        result.update({
            "country_attribution_method": "script_variant_no_country_inference",
            "should_auto_expand": False,
            "confidence": "none",
            "note": "Script or orthography label is not treated as country evidence.",
            "mapping_bucket": "script_no_map",
        })
        return result

    # 3. Explicit variant: strict one-country assignment
    if token in VARIANT_TO_COUNTRY:
        country = safe_country(VARIANT_TO_COUNTRY[token])
        if country:
            result.update({
                "candidate_country": country,
                "country_attribution_method": "inferred_from_language_variant",
                "should_auto_expand": True,
                "confidence": "medium",
                "note": "Variant or regional form strongly indicates one country.",
                "mapping_bucket": "auto_variant",
            })
            return result

    # 4. High-concentration canonical language: strict one-country assignment
    if token in HIGH_CONCENTRATION_LANGUAGE_TO_COUNTRY:
        country = safe_country(HIGH_CONCENTRATION_LANGUAGE_TO_COUNTRY[token])
        if country:
            result.update({
                "candidate_country": country,
                "country_attribution_method": "inferred_from_language_high_concentration",
                "should_auto_expand": True,
                "confidence": "medium",
                "note": "Mapped to one dominant country to preserve signal while avoiding multi-country over-attribution.",
                "mapping_bucket": "auto_high_concentration",
            })
            return result

    # 5. Low-spread language signal only: preserve, do not auto-expand
    if token in LOW_SPREAD_LANGUAGE_TO_COUNTRIES:
        countries = safe_countries(LOW_SPREAD_LANGUAGE_TO_COUNTRIES[token])
        result.update({
            "language_signal_countries": countries,
            "country_attribution_method": "inferred_from_language_low_spread",
            "should_auto_expand": False,
            "confidence": "low-medium",
            "note": "Preserved as a soft regional language signal across a small number of countries; not used for strict country assignment.",
            "mapping_bucket": "low_spread_signal_only",
        })
        return result

    # 6. Candidate-only single-country soft signal
    if token in CANDIDATE_ONLY_LANGUAGE_TO_COUNTRY:
        country = safe_country(CANDIDATE_ONLY_LANGUAGE_TO_COUNTRY[token])
        if country:
            result.update({
                "candidate_country": country,
                "country_attribution_method": "candidate_from_language_signal",
                "should_auto_expand": False,
                "confidence": "low-medium",
                "note": "Single-country language signal retained for analysis but not used for strict country assignment.",
                "mapping_bucket": "candidate_only_signal",
            })
            return result

    # 7. Broad/global languages: no mapping
    if token in GLOBAL_OR_BROAD_LANGUAGES:
        result.update({
            "country_attribution_method": "ambiguous_global_language",
            "should_auto_expand": False,
            "confidence": "none",
            "note": "Globally or broadly distributed language; no country inferred from language alone.",
            "mapping_bucket": "global_no_map",
        })
        return result

    # 8. Generic canonical / alias / iso languages not otherwise handled
    if entry_type in {"canonical_language", "alias_mapped", "iso_mapped"}:
        result.update({
            "country_attribution_method": "ambiguous_language",
            "should_auto_expand": False,
            "confidence": "low",
            "note": "No explicit locale or approved strict/soft language mapping available.",
            "mapping_bucket": "ambiguous_no_map",
        })
        return result

    # 9. Variants without explicit mapping
    if entry_type == "dialect_or_variant":
        result.update({
            "country_attribution_method": "variant_needs_manual_or_llm_review",
            "should_auto_expand": False,
            "confidence": "low",
            "note": "Variant form retained, but not mapped automatically because country signal is not explicit enough.",
            "mapping_bucket": "variant_candidate_only",
        })
        return result

    # 10. Fallback
    result.update({
        "country_attribution_method": "unmapped",
        "should_auto_expand": False,
        "confidence": "low",
        "note": "No mapping rule applied.",
        "mapping_bucket": "unmapped",
    })
    return result


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to normalized_token_frequency.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--token_col", default="normalized_token", help="Token column name")
    parser.add_argument("--entry_type_col", default="entry_type", help="Entry type column name")
    parser.add_argument("--frequency_col", default="frequency", help="Frequency column name")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    required = {args.token_col, args.entry_type_col, args.frequency_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df[[args.token_col, args.entry_type_col, args.frequency_col]].copy()
    df[args.token_col] = df[args.token_col].astype(str)
    df[args.entry_type_col] = df[args.entry_type_col].astype(str)
    df[args.frequency_col] = pd.to_numeric(df[args.frequency_col], errors="coerce").fillna(0).astype(int)

    mapped_rows = []
    for _, row in df.iterrows():
        token = row[args.token_col]
        entry_type = row[args.entry_type_col]
        frequency = row[args.frequency_col]

        mapped = map_token_to_country(token, entry_type)

        mapped_rows.append({
            "normalized_token": token,
            "entry_type": entry_type,
            "frequency": frequency,
            "candidate_country": mapped["candidate_country"],
            "language_signal_countries": "; ".join(mapped["language_signal_countries"]) if mapped["language_signal_countries"] else "",
            "country_attribution_method": mapped["country_attribution_method"],
            "should_auto_expand": mapped["should_auto_expand"],
            "confidence": mapped["confidence"],
            "mapping_bucket": mapped["mapping_bucket"],
            "note": mapped["note"],
        })

    out = pd.DataFrame(mapped_rows).sort_values(
        ["should_auto_expand", "frequency", "normalized_token"],
        ascending=[False, False, True]
    )

    auto_expand = out[out["should_auto_expand"]].copy()
    candidate_only = out[
        out["candidate_country"].notna() & (~out["should_auto_expand"])
    ].copy()
    low_spread_signal = out[
        out["mapping_bucket"] == "low_spread_signal_only"
    ].copy()
    ambiguous = out[
        (~out["should_auto_expand"]) &
        (out["candidate_country"].isna()) &
        (out["mapping_bucket"] != "low_spread_signal_only")
    ].copy()

    out.to_csv(outdir / "language_country_mapping_table.csv", index=False)
    auto_expand.to_csv(outdir / "auto_expand_language_tokens.csv", index=False)
    candidate_only.to_csv(outdir / "candidate_only_language_tokens.csv", index=False)
    low_spread_signal.to_csv(outdir / "low_spread_language_signal_tokens.csv", index=False)
    ambiguous.to_csv(outdir / "ambiguous_language_tokens.csv", index=False)

    summary = {
        "input_tokens": int(out.shape[0]),
        "weighted_frequency_total": int(out["frequency"].sum()),

        "auto_expand_tokens": int(auto_expand.shape[0]),
        "auto_expand_weighted_frequency": int(auto_expand["frequency"].sum()),

        "candidate_only_tokens": int(candidate_only.shape[0]),
        "candidate_only_weighted_frequency": int(candidate_only["frequency"].sum()),

        "low_spread_signal_tokens": int(low_spread_signal.shape[0]),
        "low_spread_signal_weighted_frequency": int(low_spread_signal["frequency"].sum()),

        "ambiguous_or_unmapped_tokens": int(ambiguous.shape[0]),
        "ambiguous_or_unmapped_weighted_frequency": int(ambiguous["frequency"].sum()),

        "auto_locale_tokens": int((out["mapping_bucket"] == "auto_locale").sum()),
        "auto_high_concentration_tokens": int((out["mapping_bucket"] == "auto_high_concentration").sum()),
        "auto_variant_tokens": int((out["mapping_bucket"] == "auto_variant").sum()),
        "candidate_only_signal_tokens": int((out["mapping_bucket"] == "candidate_only_signal").sum()),
        "low_spread_signal_only_tokens": int((out["mapping_bucket"] == "low_spread_signal_only").sum()),
        "global_no_map_tokens": int((out["mapping_bucket"] == "global_no_map").sum()),
        "script_no_map_tokens": int((out["mapping_bucket"] == "script_no_map").sum()),
        "ambiguous_no_map_tokens": int((out["mapping_bucket"] == "ambiguous_no_map").sum()),
        "variant_candidate_only_tokens": int((out["mapping_bucket"] == "variant_candidate_only").sum()),
        "unresolved_locale_tokens": int((out["mapping_bucket"] == "unresolved_locale").sum()),
    }

    summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    summary_df.to_csv(outdir / "mapping_summary.csv", index=False)

    print("\nLanguage-to-country mapping complete.")
    print(f"Outputs written to: {outdir.resolve()}")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()