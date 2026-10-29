#!/usr/bin/env python3
"""
run_author_affiliation_llm_fallback_openai.py

Run LLM fallback on targeted author affiliation rows using the OpenAI API
with the Responses API and Structured Outputs.

Design:
- Model extracts institutions, countries, affiliation type
- Code normalizes and validates countries against the approved list

python run_author_affiliation_llm_fallback.py --input author_affiliation_llm_fallback_input_v2.csv --output author_affiliation_llm_fallback_output_full.csv --model gpt-4.1-mini --save-every 50 --resume

Default model:
    gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


STANDARD_COUNTRIES = [
    "Nigeria", "South Africa", "Angola", "Burkina Faso", "Cabo Verde", "Morocco",
    "Egypt", "Tunisia", "Benin", "Malawi", "South Sudan", "Democratic Republic of the Congo",
    "Kenya", "Algeria", "Mozambique", "Zambia", "Guinea", "Senegal", "Madagascar",
    "Cameroon", "Cote d'Ivoire", "Guinea-Bissau", "São Tomé and Príncipe", "Libya",
    "Ethiopia", "Uganda", "Rwanda", "Tanzania", "Togo", "Burundi", "Ghana",
    "Eritrea", "Niger", "Somalia", "Zimbabwe", "Djibouti", "Mali",
    "Central African Republic", "Namibia", "Sierra Leone", "Comoros", "Mauritius",
    "Seychelles", "Gambia", "Liberia", "Equatorial Guinea", "Sudan", "Gabon",
    "Botswana", "Chad", "Congo (Congo-Brazzaville)", "Eswatini", "Lesotho",
    "Mauritania", "United States", "Canada", "Mexico", "Brazil", "Paraguay",
    "Peru", "Bolivia", "Colombia", "Chile", "Guatemala", "Ecuador", "Argentina",
    "Costa Rica", "Cuba", "Dominican Republic", "El Salvador", "Honduras",
    "Nicaragua", "Panama", "Uruguay", "Venezuela", "Haiti", "Jamaica",
    "Suriname", "Antigua and Barbuda", "Bahamas", "Barbados", "Belize",
    "Dominica", "Grenada", "Guyana", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Trinidad and Tobago", "India", "Philippines",
    "Singapore", "China", "Japan", "South Korea", "Myanmar", "Indonesia",
    "Vietnam", "Turkey", "Iraq", "Saudi Arabia", "Malaysia", "Iran",
    "Bangladesh", "Pakistan", "Taiwan", "Thailand", "Kazakhstan", "Kyrgyzstan",
    "Timor-Leste", "Lebanon", "Syria", "Yemen", "Bahrain", "Jordan", "Kuwait",
    "Oman", "Qatar", "United Arab Emirates", "Sri Lanka", "Tajikistan",
    "Afghanistan", "Israel", "Azerbaijan", "Uzbekistan", "Nepal", "Brunei",
    "Georgia", "Cambodia", "Armenia", "Mongolia", "Laos", "Maldives",
    "Turkmenistan", "Bhutan", "Cyprus", "North Korea", "Palestine",
    "United Kingdom", "Ireland", "Spain", "Italy", "Germany", "Russia",
    "France", "Switzerland", "Poland", "Netherlands", "Latvia", "Portugal",
    "Romania", "Greece", "Slovenia", "Serbia", "Slovakia", "Luxembourg",
    "Belgium", "Ukraine", "Belarus", "Austria", "Czechia", "Denmark",
    "Sweden", "Norway", "Finland", "Hungary", "Bulgaria", "Estonia",
    "Iceland", "Albania", "Croatia", "Lithuania", "Malta", "North Macedonia",
    "Vatican City", "Bosnia and Herzegovina", "Moldova", "Montenegro", "Andorra",
    "Kosovo", "Liechtenstein", "Monaco", "San Marino", "Australia",
    "New Zealand", "Papua New Guinea", "Samoa", "Fiji", "Tonga", "Vanuatu",
    "Marshall Islands", "Micronesia", "Nauru", "Solomon Islands", "Palau",
    "Kiribati", "Tuvalu",
]

ALLOWED_COUNTRIES = set(STANDARD_COUNTRIES)
ALLOWED_COUNTRIES_WITH_UNKNOWN = ALLOWED_COUNTRIES | {"unknown"}

COUNTRY_ALIAS_MAP = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "republic of korea": "South Korea",
    "korea, south": "South Korea",
    "south korea": "South Korea",
    "north korea": "North Korea",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "dr congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "democratic republic of congo": "Democratic Republic of the Congo",
    "congo-brazzaville": "Congo (Congo-Brazzaville)",
    "republic of the congo": "Congo (Congo-Brazzaville)",
    "russian federation": "Russia",
    "czech republic": "Czechia",
    "ivory coast": "Cote d'Ivoire",
    "côte d'ivoire": "Cote d'Ivoire",
    "cape verde": "Cabo Verde",
    "eswatini": "Eswatini",
    "swaziland": "Eswatini",
    "timor leste": "Timor-Leste",
    "vatican": "Vatican City",
    "palestinian territories": "Palestine",
    "state of palestine": "Palestine",
    "micronesia, federated states of": "Micronesia",
    "federated states of micronesia": "Micronesia",
}

SYSTEM_PROMPT = """You extract structured author affiliation metadata from paper header text.

Rules:
1. Use only the provided text.
2. Do not infer country from personal names.
3. You may infer country from institution names only when the institution-country mapping is obvious and standard from the text itself or from a highly standard institutional name.
4. You may infer country from widely known standard institutions.
5. Return standard modern country names when possible.
6. If uncertain, return "unknown" for countries or leave the countries list empty.
7. Do not explain your reasoning outside the JSON fields.
8. If the regex institution extraction appears partially correct, refine it rather than discarding it.
9. When multiple institutions are present, keep them separate and do not collapse them into one location.
"""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "institutions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "countries": {
            "type": "array",
            "items": {"type": "string"},
        },
        "affiliation_type": {
            "type": "string",
            "enum": ["academia", "industry", "mixed", "unknown"],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "notes": {
            "type": "string",
        },
    },
    "required": ["institutions", "countries", "affiliation_type", "confidence", "notes"],
}


def build_user_prompt(row: pd.Series) -> str:
    return f"""Extract author affiliation metadata from the following paper header fields.

paper_url: {row.get('paper_url', '')}

header_text:
{row.get('header_text', '')}

author_affiliations_raw:
{row.get('author_affiliations_raw', '')}

emails:
{row.get('emails', '')}

regex_country_guess:
{row.get('author_all_inferred_countries', '')}

regex_affiliation_type:
{row.get('author_affiliation_type', '')}

regex_institutions_clean:
{row.get('author_institutions_regex_clean', '')}
"""


def normalize_country_name(country: str) -> str:
    c = str(country).strip()
    if not c:
        return ""

    c = re.sub(r"\s+", " ", c)
    c_lower = c.lower()

    if c_lower == "unknown":
        return "unknown"

    if c in ALLOWED_COUNTRIES:
        return c

    if c_lower in COUNTRY_ALIAS_MAP:
        mapped = COUNTRY_ALIAS_MAP[c_lower]
        if mapped in ALLOWED_COUNTRIES:
            return mapped

    # Title-case fallback only for exact allowed match after normalization
    title_c = c.title()
    if title_c in ALLOWED_COUNTRIES:
        return title_c

    return ""


def normalize_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    institutions = obj.get("institutions", [])
    countries = obj.get("countries", [])
    affiliation_type = obj.get("affiliation_type", "unknown")
    confidence = obj.get("confidence", "low")
    notes = obj.get("notes", "")

    if not isinstance(institutions, list):
        institutions = []
    if not isinstance(countries, list):
        countries = []

    institutions = [str(x).strip() for x in institutions if str(x).strip()]

    normalized_countries: List[str] = []
    for x in countries:
        norm = normalize_country_name(x)
        if norm:
            normalized_countries.append(norm)

    # dedupe preserve order
    deduped_countries: List[str] = []
    seen = set()
    for c in normalized_countries:
        if c not in seen:
            seen.add(c)
            deduped_countries.append(c)

    real_countries = [c for c in deduped_countries if c != "unknown"]
    if real_countries:
        countries = real_countries
    elif "unknown" in deduped_countries:
        countries = ["unknown"]
    else:
        countries = []

    if affiliation_type not in {"academia", "industry", "mixed", "unknown"}:
        affiliation_type = "unknown"

    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "institutions": institutions,
        "countries": countries,
        "affiliation_type": affiliation_type,
        "confidence": confidence,
        "notes": str(notes),
    }


def call_model(client: OpenAI, model: str, user_prompt: str) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "author_affiliation_extraction",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        store=False,
    )

    text = response.output_text
    parsed = json.loads(text)
    return normalize_output(parsed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fallback input CSV")
    parser.add_argument("--output", required=True, help="Output CSV with LLM results")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint frequency")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional pause between requests")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    completed_urls = set()
    existing_rows = []

    if args.resume and output_path.exists():
        prev = pd.read_csv(output_path)
        if "paper_url" in prev.columns:
            completed_urls = set(prev["paper_url"].dropna().astype(str))
            existing_rows = prev.to_dict(orient="records")

    if completed_urls:
        df = df[~df["paper_url"].astype(str).isin(completed_urls)].copy()

    client = OpenAI(api_key=api_key)

    results = existing_rows[:]
    total = len(df)

    print(f"Rows to process: {total}")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        paper_url = str(row.get("paper_url", ""))

        try:
            parsed = call_model(
                client=client,
                model=args.model,
                user_prompt=build_user_prompt(row),
            )

            results.append({
                "paper_url": paper_url,
                "llm_institutions": "; ".join(parsed["institutions"]),
                "llm_countries": "; ".join(parsed["countries"]),
                "llm_affiliation_type": parsed["affiliation_type"],
                "llm_confidence": parsed["confidence"],
                "llm_notes": parsed["notes"],
                "llm_status": "ok",
            })

        except Exception as e:
            results.append({
                "paper_url": paper_url,
                "llm_institutions": "",
                "llm_countries": "",
                "llm_affiliation_type": "",
                "llm_confidence": "",
                "llm_notes": "",
                "llm_status": f"error:{type(e).__name__}",
            })

        if idx % args.save_every == 0 or idx == total:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"[{idx}/{total}] saved -> {output_path}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()