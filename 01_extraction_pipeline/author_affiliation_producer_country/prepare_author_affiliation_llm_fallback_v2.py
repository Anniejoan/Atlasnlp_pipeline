#!/usr/bin/env python3
"""
prepare_author_affiliation_llm_fallback_v2.py

Build targeted LLM fallback input from the cleaned regex output.

Criteria:
- missing author country
- unknown affiliation type
- missing cleaned institution
- no affiliation extracted
- suspicious cleaned institution string (still looks noisy)

Usage:
    python prepare_author_affiliation_llm_fallback_v2.py --input author_affiliation_all_regex_with_institutions_clean.csv --output author_affiliation_llm_fallback_input_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cleaned regex output CSV")
    parser.add_argument("--output", required=True, help="Fallback candidate CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    required_cols = [
        "paper_url",
        "header_text",
        "author_affiliations_raw",
        "emails",
        "author_all_inferred_countries",
        "author_affiliation_type",
        "affiliation_extracted",
        "author_institutions_regex",
        "author_institutions_regex_clean",
        "author_institutions_regex_clean_count",
        "author_institutions_regex_was_modified",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    missing_country = df["author_all_inferred_countries"].fillna("").astype(str).str.strip() == ""
    unknown_type = df["author_affiliation_type"].fillna("").astype(str).str.strip().eq("unknown")
    no_affiliation = ~df["affiliation_extracted"].fillna(False).astype(bool)
    missing_institution = df["author_institutions_regex_clean"].fillna("").astype(str).str.strip() == ""

    # Target only institution strings that still look suspicious after cleaning.
    # This is much narrower than sending all modified rows.
    suspicious_institution = (
        df["author_institutions_regex_clean"]
        .fillna("")
        .astype(str)
        .str.contains(
            r"proceedings|annual meeting|association for computational linguistics|pages\s+\d|"
            r"\b[A-Z][a-zA-Z]+[0-9]+\b|"
            r"\d+[A-Z][a-zA-Z]|"
            r"\{.*@.*\}|"
            r"@|"
            r"correspondence|"
            r"\b[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+[A-Z][a-z]+\b",
            case=False,
            regex=True,
        )
    )

    fallback_mask = (
        missing_country
        | unknown_type
        | no_affiliation
        | missing_institution
        | suspicious_institution
    )

    fallback_df = df.loc[fallback_mask].copy()

    fallback_df["llm_reason_missing_country"] = missing_country.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_unknown_type"] = unknown_type.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_no_affiliation_extracted"] = no_affiliation.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_missing_institution"] = missing_institution.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_suspicious_institution"] = suspicious_institution.loc[fallback_mask].astype(int)

    keep_cols = [
        "paper_url",
        "header_text",
        "author_affiliations_raw",
        "emails",
        "author_all_inferred_countries",
        "author_country_inference_method",
        "author_affiliation_type",
        "affiliation_extracted",
        "author_institutions_regex",
        "author_institutions_regex_clean",
        "author_institutions_regex_clean_count",
        "author_institutions_regex_was_modified",
        "llm_reason_missing_country",
        "llm_reason_unknown_type",
        "llm_reason_no_affiliation_extracted",
        "llm_reason_missing_institution",
        "llm_reason_suspicious_institution",
    ]

    existing_cols = [c for c in keep_cols if c in fallback_df.columns]
    fallback_df = fallback_df[existing_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Fallback rows: {len(fallback_df)}")
    print(f"Missing country rows in fallback: {int(fallback_df['llm_reason_missing_country'].sum())}")
    print(f"Unknown type rows in fallback: {int(fallback_df['llm_reason_unknown_type'].sum())}")
    print(f"No affiliation rows in fallback: {int(fallback_df['llm_reason_no_affiliation_extracted'].sum())}")
    print(f"Missing institution rows in fallback: {int(fallback_df['llm_reason_missing_institution'].sum())}")
    print(f"Suspicious institution rows in fallback: {int(fallback_df['llm_reason_suspicious_institution'].sum())}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()