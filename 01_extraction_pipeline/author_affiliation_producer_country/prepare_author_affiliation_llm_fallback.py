#!/usr/bin/env python3
"""
prepare_author_affiliation_llm_fallback.py

Create a targeted fallback CSV for LLM-based author affiliation recovery.

Input:
- combined regex output CSV

Output:
- CSV containing only rows that need LLM fallback

python prepare_author_affiliation_llm_fallback.py --input author_affiliation_all_regex.csv --output author_affiliation_llm_fallback_input.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Combined regex output CSV")
    parser.add_argument("--output", required=True, help="Fallback candidate CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    for col in [
        "paper_url",
        "header_text",
        "author_affiliations_raw",
        "emails",
        "author_all_inferred_countries",
        "author_affiliation_type",
        "affiliation_extracted",
    ]:
        if col not in df.columns:
            df[col] = ""

    countries_empty = df["author_all_inferred_countries"].fillna("").astype(str).str.strip() == ""
    unknown_type = df["author_affiliation_type"].fillna("").astype(str).str.strip().eq("unknown")
    no_affiliation = ~df["affiliation_extracted"].fillna(False).astype(bool)

    raw_aff = df["author_affiliations_raw"].fillna("").astype(str)
    suspicious_raw = (
        raw_aff.str.contains(
            r"proceedings of|association for computational linguistics|pages \d|workshop on|joint workshop",
            case=False,
            regex=True,
        )
        | raw_aff.str.len().lt(8)
    )

    fallback_mask = countries_empty | unknown_type | no_affiliation | suspicious_raw
    fallback_df = df.loc[fallback_mask].copy()

    fallback_df["llm_reason_missing_country"] = countries_empty.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_unknown_type"] = unknown_type.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_no_affiliation_extracted"] = no_affiliation.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_suspicious_raw"] = suspicious_raw.loc[fallback_mask].astype(int)

    keep_cols = [
        "paper_url",
        "header_text",
        "author_affiliations_raw",
        "emails",
        "author_all_inferred_countries",
        "author_country_inference_method",
        "author_affiliation_type",
        "affiliation_extracted",
        "llm_reason_missing_country",
        "llm_reason_unknown_type",
        "llm_reason_no_affiliation_extracted",
        "llm_reason_suspicious_raw",
    ]

    existing = [c for c in keep_cols if c in fallback_df.columns]
    fallback_df = fallback_df[existing]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Fallback rows: {len(fallback_df)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()