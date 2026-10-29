#!/usr/bin/env python3
"""
merge_author_affiliation_regex_llm.py

Merge full regex baseline with LLM fallback output.

Inputs:
- cleaned regex baseline CSV
- LLM fallback output CSV

Outputs:
- merged CSV with final institution / country / affiliation type fields
- optional summary printed to console

Usage:
    python merge_author_affiliation_regex_llm.py --regex-input author_affiliation_all_regex_with_institutions_clean.csv --llm-input author_affiliation_llm_fallback_output_full.csv --output author_affiliation_all_merged.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regex-input", required=True, help="Cleaned regex baseline CSV")
    parser.add_argument("--llm-input", required=True, help="LLM fallback output CSV")
    parser.add_argument("--output", required=True, help="Merged output CSV")
    return parser.parse_args()


def normalize_cell(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_semicolon(x) -> List[str]:
    s = normalize_cell(x)
    if not s:
        return []
    return [part.strip() for part in s.split(";") if part.strip()]


def join_semicolon(items: List[str]) -> str:
    seen = set()
    out = []
    for item in items:
        item = normalize_cell(item)
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return "; ".join(out)


def has_value(x) -> bool:
    return normalize_cell(x) != ""


def choose_final_institutions(regex_clean: str, llm_inst: str) -> tuple[str, str]:
    regex_ok = has_value(regex_clean)
    llm_ok = has_value(llm_inst)

    if regex_ok and not llm_ok:
        return regex_clean, "regex_clean"
    if not regex_ok and llm_ok:
        return llm_inst, "llm_fallback"
    if regex_ok and llm_ok:
        if regex_clean == llm_inst:
            return regex_clean, "regex_clean"
        # prefer llm on fallback rows when it provided a non-empty repair
        return llm_inst, "llm_fallback"
    return "", ""


def choose_final_countries(regex_countries: str, llm_countries: str) -> tuple[str, str]:
    regex_ok = has_value(regex_countries)
    llm_ok = has_value(llm_countries)

    if regex_ok and not llm_ok:
        return regex_countries, "regex"
    if not regex_ok and llm_ok:
        return llm_countries, "llm_fallback"
    if regex_ok and llm_ok:
        # keep regex when already present unless llm adds clearly more coverage
        regex_list = split_semicolon(regex_countries)
        llm_list = split_semicolon(llm_countries)

        if set(llm_list) == set(regex_list):
            return regex_countries, "regex"
        if len(llm_list) > len(regex_list):
            return llm_countries, "llm_fallback"
        return regex_countries, "regex"
    return "", ""


def choose_final_affiliation_type(regex_type: str, llm_type: str) -> tuple[str, str]:
    regex_type = normalize_cell(regex_type)
    llm_type = normalize_cell(llm_type)

    regex_ok = regex_type not in {"", "unknown"}
    llm_ok = llm_type not in {"", "unknown"}

    if regex_ok and not llm_ok:
        return regex_type, "regex"
    if not regex_ok and llm_ok:
        return llm_type, "llm_fallback"
    if regex_ok and llm_ok:
        if regex_type == llm_type:
            return regex_type, "regex"
        # prefer llm only when regex was weak; otherwise keep regex
        return regex_type, "regex"
    # both weak
    if regex_type == "unknown":
        return "unknown", "regex"
    if llm_type == "unknown":
        return "unknown", "llm_fallback"
    return "", ""


def main():
    args = parse_args()

    regex_path = Path(args.regex_input)
    llm_path = Path(args.llm_input)
    output_path = Path(args.output)

    regex_df = pd.read_csv(regex_path)
    llm_df = pd.read_csv(llm_path)

    # Keep only relevant llm columns
    llm_keep = [
        "paper_url",
        "llm_institutions",
        "llm_countries",
        "llm_affiliation_type",
        "llm_confidence",
        "llm_notes",
        "llm_status",
    ]
    llm_df = llm_df[[c for c in llm_keep if c in llm_df.columns]].copy()

    merged = regex_df.merge(llm_df, on="paper_url", how="left")

    # create regex convenience columns
    merged["author_institutions_regex_final_candidate"] = merged.get("author_institutions_regex_clean", "").fillna("").astype(str).str.strip()
    merged["author_countries_regex_final_candidate"] = merged.get("author_all_inferred_countries", "").fillna("").astype(str).str.strip()
    merged["author_affiliation_type_regex_final_candidate"] = merged.get("author_affiliation_type", "").fillna("").astype(str).str.strip()

    final_inst = []
    final_inst_source = []
    final_country = []
    final_country_source = []
    final_type = []
    final_type_source = []

    for _, row in merged.iterrows():
        inst, inst_src = choose_final_institutions(
            regex_clean=row.get("author_institutions_regex_final_candidate", ""),
            llm_inst=row.get("llm_institutions", ""),
        )
        ctry, ctry_src = choose_final_countries(
            regex_countries=row.get("author_countries_regex_final_candidate", ""),
            llm_countries=row.get("llm_countries", ""),
        )
        afft, afft_src = choose_final_affiliation_type(
            regex_type=row.get("author_affiliation_type_regex_final_candidate", ""),
            llm_type=row.get("llm_affiliation_type", ""),
        )

        final_inst.append(inst)
        final_inst_source.append(inst_src)
        final_country.append(ctry)
        final_country_source.append(ctry_src)
        final_type.append(afft)
        final_type_source.append(afft_src)

    merged["author_institutions_final"] = final_inst
    merged["author_institutions_final_source"] = final_inst_source

    merged["author_countries_final"] = final_country
    merged["author_countries_final_source"] = final_country_source

    merged["author_affiliation_type_final"] = final_type
    merged["author_affiliation_type_final_source"] = final_type_source

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    # console summary
    total = len(merged)
    inst_cov = (merged["author_institutions_final"].fillna("").astype(str).str.strip() != "").sum()
    ctry_cov = (merged["author_countries_final"].fillna("").astype(str).str.strip() != "").sum()
    type_cov = (merged["author_affiliation_type_final"].fillna("").astype(str).str.strip() != "").sum()

    print(f"Total rows: {total}")
    print(f"Rows with final institutions: {inst_cov}")
    print(f"Rows with final countries: {ctry_cov}")
    print(f"Rows with final affiliation type: {type_cov}")
    print("\nInstitution source counts:")
    print(merged["author_institutions_final_source"].fillna("").replace("", pd.NA).value_counts(dropna=False).to_string())
    print("\nCountry source counts:")
    print(merged["author_countries_final_source"].fillna("").replace("", pd.NA).value_counts(dropna=False).to_string())
    print("\nAffiliation type source counts:")
    print(merged["author_affiliation_type_final_source"].fillna("").replace("", pd.NA).value_counts(dropna=False).to_string())
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()