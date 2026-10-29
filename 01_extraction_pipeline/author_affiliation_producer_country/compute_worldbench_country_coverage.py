#!/usr/bin/env python3
"""
compute_worldbench_country_coverage.py — Compute WorldBench country coverage statistics

Paper section: §4 "WorldBench Overview" (Table 2 summary statistics)
               §5 "Dataset Geography Analysis"
Role in pipeline: Step 9b (final step of author affiliation sub-pipeline) —
  combines content-side country attribution (from updated_master_with_normalized_languages.csv)
  with producer-side country attribution (from author_affiliation_all_merged.csv)
  to compute the coverage statistics reported in Table 2 and used in Section 5.

  Note: master_with_langmap.csv is an older version of updated_master_with_language_mapping_v2.csv
  that does not include the Portuguese policy update; always use updated_master_with_normalized_languages.csv.

Inputs:
  - Master dataset with content-side country attribution columns
  - Merged author-affiliation CSV (producer-side countries)

Outputs:
  - coverage_summary.json  — machine-readable summary
  - coverage_summary.txt   — human-readable summary
  - coverage_flags.csv     — row-level flags (which rows have content/producer coverage)

Usage:
    python compute_worldbench_country_coverage.py \\
      --master-input /home/jnwatu/Worldbench/compiled_ACL/updated_master_with_normalized_languages.csv \\
      --author-input author_affiliation_all_merged.csv \\
      --output-json coverage_summary.json \\
      --output-txt coverage_summary.txt \\
      --output-flags coverage_flags.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-input", required=True, help="Master/content-side CSV")
    parser.add_argument("--author-input", required=True, help="Merged author-affiliation CSV")
    parser.add_argument("--output-json", required=True, help="Coverage summary JSON")
    parser.add_argument("--output-txt", required=True, help="Coverage summary TXT")
    parser.add_argument("--output-flags", required=True, help="Row-level coverage flags CSV")
    return parser.parse_args()


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def has_value(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() != ""


def summarize_counts(series: pd.Series) -> Dict[str, int]:
    vc = series.value_counts(dropna=False)
    out = {}
    for k, v in vc.items():
        key = "NaN" if pd.isna(k) else str(k)
        out[key] = int(v)
    return out


def main():
    args = parse_args()

    master_path = Path(args.master_input)
    author_path = Path(args.author_input)

    master_df = pd.read_csv(master_path)
    author_df = pd.read_csv(author_path)

    # Join key
    if "paper_url" in author_df.columns and "Paper Link (URL)" in master_df.columns:
        merged = master_df.merge(
            author_df,
            left_on="Paper Link (URL)",
            right_on="paper_url",
            how="left",
            suffixes=("", "_author"),
        )
    elif "paper_url" in master_df.columns and "paper_url" in author_df.columns:
        merged = master_df.merge(author_df, on="paper_url", how="left", suffixes=("", "_author"))
    else:
        raise ValueError("Could not find a compatible join key between master-input and author-input.")

    total_rows = len(merged)

    # -------------------------
    # Content-side country fields
    # -------------------------
    content_country_col = None
    for candidate in [
        "all_countries_covered",
        "All Countries Covered (you can select multiple countries)",
        "Normalized Countries",
        "Assigned country",
    ]:
        if candidate in merged.columns:
            content_country_col = candidate
            break

    if content_country_col is None:
        raise ValueError("Could not find a content-side country column in master-input.")

    content_method_col = None
    for candidate in [
        "Country Attribution Method",
        "country_attribution_method",
    ]:
        if candidate in merged.columns:
            content_method_col = candidate
            break

    content_country_present = has_value(merged[content_country_col])

    if content_method_col is not None:
        method_series = merged[content_method_col].fillna("").astype(str).str.strip().str.lower()

        language_based = (
            method_series.str.contains("language", regex=False)
            | method_series.str.contains("lang", regex=False)
        )

        explicit_or_non_language = content_country_present & (~language_based)
        language_inferred = content_country_present & language_based
    else:
        language_based = pd.Series([False] * total_rows, index=merged.index)
        explicit_or_non_language = content_country_present
        language_inferred = pd.Series([False] * total_rows, index=merged.index)

    content_missing = ~content_country_present

    # -------------------------
    # Author-side country fields
    # -------------------------
    if "author_countries_final" not in merged.columns:
        raise ValueError("author_countries_final not found in author-input.")
    if "author_countries_final_source" not in merged.columns:
        raise ValueError("author_countries_final_source not found in author-input.")

    author_country_present = has_value(merged["author_countries_final"])
    author_country_missing = ~author_country_present

    # -------------------------
    # Overlap / combined coverage
    # -------------------------
    both_present = content_country_present & author_country_present
    content_only = content_country_present & (~author_country_present)
    author_only = (~content_country_present) & author_country_present
    neither_present = (~content_country_present) & (~author_country_present)

    flags_df = pd.DataFrame({
        "content_country_present": content_country_present.astype(int),
        "content_country_explicit_or_non_language": explicit_or_non_language.astype(int),
        "content_country_language_inferred": language_inferred.astype(int),
        "content_country_missing": content_missing.astype(int),
        "author_country_present": author_country_present.astype(int),
        "author_country_missing": author_country_missing.astype(int),
        "both_content_and_author_country": both_present.astype(int),
        "content_only_country": content_only.astype(int),
        "author_only_country": author_only.astype(int),
        "neither_content_nor_author_country": neither_present.astype(int),
    })

    # keep useful identifiers if available
    id_cols = []
    for c in ["Paper Link (URL)", "paper_url", "Dataset name"]:
        if c in merged.columns:
            id_cols.append(c)

    flags_out = pd.concat([merged[id_cols].copy(), flags_df], axis=1)

    summary: Dict[str, Any] = {
        "total_rows": int(total_rows),

        "content_side": {
            "country_column_used": content_country_col,
            "country_present": int(content_country_present.sum()),
            "country_present_rate": round(float(content_country_present.mean()), 6),
            "explicit_or_non_language_country_present": int(explicit_or_non_language.sum()),
            "explicit_or_non_language_country_present_rate": round(float(explicit_or_non_language.mean()), 6),
            "language_inferred_country_present": int(language_inferred.sum()),
            "language_inferred_country_present_rate": round(float(language_inferred.mean()), 6),
            "country_missing": int(content_missing.sum()),
            "country_missing_rate": round(float(content_missing.mean()), 6),
        },

        "author_side": {
            "country_present": int(author_country_present.sum()),
            "country_present_rate": round(float(author_country_present.mean()), 6),
            "country_missing": int(author_country_missing.sum()),
            "country_missing_rate": round(float(author_country_missing.mean()), 6),
            "country_source_counts": summarize_counts(
                merged["author_countries_final_source"].replace("", pd.NA)
            ),
            "institution_present": int(has_value(merged["author_institutions_final"]).sum()) if "author_institutions_final" in merged.columns else None,
            "institution_present_rate": round(float(has_value(merged["author_institutions_final"]).mean()), 6) if "author_institutions_final" in merged.columns else None,
            "affiliation_type_counts": summarize_counts(
                merged["author_affiliation_type_final"].replace("", pd.NA)
            ) if "author_affiliation_type_final" in merged.columns else {},
        },

        "overlap": {
            "both_content_and_author_country": int(both_present.sum()),
            "both_content_and_author_country_rate": round(float(both_present.mean()), 6),
            "content_only_country": int(content_only.sum()),
            "content_only_country_rate": round(float(content_only.mean()), 6),
            "author_only_country": int(author_only.sum()),
            "author_only_country_rate": round(float(author_only.mean()), 6),
            "neither_content_nor_author_country": int(neither_present.sum()),
            "neither_content_nor_author_country_rate": round(float(neither_present.mean()), 6),
        },
    }

    # write outputs
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_flags = Path(args.output_flags)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_flags.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("WorldBench Country Coverage Summary")
    lines.append("=" * 36)
    lines.append(f"Total rows: {summary['total_rows']}")
    lines.append("")
    lines.append("A. Content-side country coverage")
    lines.append(f"  Country present: {summary['content_side']['country_present']} / {summary['total_rows']} ({summary['content_side']['country_present_rate']:.4f})")
    lines.append(f"  Explicit/non-language present: {summary['content_side']['explicit_or_non_language_country_present']} / {summary['total_rows']} ({summary['content_side']['explicit_or_non_language_country_present_rate']:.4f})")
    lines.append(f"  Language-inferred present: {summary['content_side']['language_inferred_country_present']} / {summary['total_rows']} ({summary['content_side']['language_inferred_country_present_rate']:.4f})")
    lines.append(f"  Missing: {summary['content_side']['country_missing']} / {summary['total_rows']} ({summary['content_side']['country_missing_rate']:.4f})")
    lines.append("")
    lines.append("B. Author-affiliation country coverage")
    lines.append(f"  Country present: {summary['author_side']['country_present']} / {summary['total_rows']} ({summary['author_side']['country_present_rate']:.4f})")
    lines.append(f"  Missing: {summary['author_side']['country_missing']} / {summary['total_rows']} ({summary['author_side']['country_missing_rate']:.4f})")
    lines.append("  Country source counts:")
    for k, v in summary["author_side"]["country_source_counts"].items():
        lines.append(f"    {k}: {v}")
    if summary["author_side"]["institution_present"] is not None:
        lines.append(f"  Institution present: {summary['author_side']['institution_present']} / {summary['total_rows']} ({summary['author_side']['institution_present_rate']:.4f})")
    if summary["author_side"]["affiliation_type_counts"]:
        lines.append("  Affiliation type counts:")
        for k, v in summary["author_side"]["affiliation_type_counts"].items():
            lines.append(f"    {k}: {v}")
    lines.append("")
    lines.append("C. Overlap")
    lines.append(f"  Both content and author country: {summary['overlap']['both_content_and_author_country']} / {summary['total_rows']} ({summary['overlap']['both_content_and_author_country_rate']:.4f})")
    lines.append(f"  Content only: {summary['overlap']['content_only_country']} / {summary['total_rows']} ({summary['overlap']['content_only_country_rate']:.4f})")
    lines.append(f"  Author only: {summary['overlap']['author_only_country']} / {summary['total_rows']} ({summary['overlap']['author_only_country_rate']:.4f})")
    lines.append(f"  Neither: {summary['overlap']['neither_content_nor_author_country']} / {summary['total_rows']} ({summary['overlap']['neither_content_nor_author_country_rate']:.4f})")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    flags_out.to_csv(output_flags, index=False)

    print("\n".join(lines))
    print(f"\nWrote JSON: {output_json}")
    print(f"Wrote TXT: {output_txt}")
    print(f"Wrote flags CSV: {output_flags}")


if __name__ == "__main__":
    main()