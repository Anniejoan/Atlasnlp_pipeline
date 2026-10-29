#!/usr/bin/env python3
"""
aggregate_author_affiliation_diagnostics.py

Aggregate shard-level diagnostics JSON files into one overall report.

Optional:
- also scan shard CSV outputs and create a fallback-candidate CSV for rows that
  likely need LLM extraction.

Expected shard files:
- author_affiliation_shard_001.csv
- author_affiliation_shard_001.diagnostics.json
- ...

Usage:
    python aggregate_author_affiliation_diagnostics.py \
      --diagnostics-dir /path/to/author_affiliation_outputs \
      --output-json /path/to/author_affiliation_outputs/aggregate_diagnostics.json \
      --output-txt /path/to/author_affiliation_outputs/aggregate_summary.txt

With fallback candidate export:
    python aggregate_author_affiliation_diagnostics.py --diagnostics-dir /scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_dataset_pipeline/author_affiliation_outputs --csv-dir /scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_dataset_pipeline/author_affiliation_outputs --output-json aggregate_diagnostics.json --output-txt aggregate_summary.txt --fallback-csv author_affiliation_llm_fallback_candidates.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-dir",
        required=True,
        help="Directory containing shard .diagnostics.json files",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Directory containing shard CSVs. Needed only if you want fallback CSV generation.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to aggregated diagnostics JSON output",
    )
    parser.add_argument(
        "--output-txt",
        required=True,
        help="Path to aggregated diagnostics text summary output",
    )
    parser.add_argument(
        "--fallback-csv",
        default=None,
        help="Optional path to write candidate rows for LLM fallback",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def counter_from_dict(d: Dict[str, Any]) -> Counter:
    c = Counter()
    for k, v in (d or {}).items():
        c[str(k)] += int(v)
    return c


def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def aggregate_diagnostics(diag_files: List[Path]) -> Dict[str, Any]:
    total_rows = 0
    rows_with_any_country = 0
    rows_missing_any_country = 0
    affiliation_extracted_true = 0

    pdf_download_status_counts = Counter()
    country_inference_method_counts = Counter()
    affiliation_type_counts = Counter()
    missing_text_extraction_method_counts = Counter()
    missing_affiliation_type_counts = Counter()
    top_missing_email_domains = Counter()

    shard_summaries = []

    for diag_file in sorted(diag_files):
        d = load_json(diag_file)

        total_rows += safe_int(d.get("total_rows", 0))
        rows_with_any_country += safe_int(d.get("rows_with_any_inferred_author_country", 0))
        rows_missing_any_country += safe_int(d.get("rows_missing_any_author_country", 0))
        affiliation_extracted_true += safe_int(d.get("affiliation_extracted_true", 0))

        pdf_download_status_counts += counter_from_dict(d.get("pdf_download_status_counts", {}))
        country_inference_method_counts += counter_from_dict(d.get("country_inference_method_counts", {}))
        affiliation_type_counts += counter_from_dict(d.get("affiliation_type_counts", {}))
        missing_text_extraction_method_counts += counter_from_dict(d.get("missing_text_extraction_method_counts", {}))
        missing_affiliation_type_counts += counter_from_dict(d.get("missing_affiliation_type_counts", {}))
        top_missing_email_domains += counter_from_dict(d.get("top_missing_email_domains", {}))

        shard_summaries.append({
            "file": diag_file.name,
            "total_rows": safe_int(d.get("total_rows", 0)),
            "rows_with_any_inferred_author_country": safe_int(d.get("rows_with_any_inferred_author_country", 0)),
            "coverage_rate": d.get("coverage_rate", None),
        })

    coverage_rate = (rows_with_any_country / total_rows) if total_rows else 0.0
    missing_rate = (rows_missing_any_country / total_rows) if total_rows else 0.0
    affiliation_extracted_rate = (affiliation_extracted_true / total_rows) if total_rows else 0.0

    aggregate = {
        "num_shards": len(diag_files),
        "total_rows": total_rows,
        "rows_with_any_inferred_author_country": rows_with_any_country,
        "rows_missing_any_author_country": rows_missing_any_country,
        "coverage_rate": round(coverage_rate, 6),
        "missing_rate": round(missing_rate, 6),
        "affiliation_extracted_true": affiliation_extracted_true,
        "affiliation_extracted_rate": round(affiliation_extracted_rate, 6),
        "pdf_download_status_counts": dict(pdf_download_status_counts),
        "country_inference_method_counts": dict(country_inference_method_counts),
        "affiliation_type_counts": dict(affiliation_type_counts),
        "missing_text_extraction_method_counts": dict(missing_text_extraction_method_counts),
        "missing_affiliation_type_counts": dict(missing_affiliation_type_counts),
        "top_missing_email_domains": dict(top_missing_email_domains.most_common(50)),
        "shards": shard_summaries,
    }
    return aggregate


def write_text_summary(report: Dict[str, Any], output_txt: Path) -> None:
    lines = []
    lines.append("Aggregated Author Affiliation Diagnostics")
    lines.append("=" * 44)
    lines.append(f"Num shards: {report.get('num_shards', 0)}")
    lines.append(f"Total rows: {report.get('total_rows', 0)}")
    lines.append(
        f"Rows with any inferred author country: "
        f"{report.get('rows_with_any_inferred_author_country', 0)} / {report.get('total_rows', 0)}"
    )
    lines.append(f"Coverage rate: {report.get('coverage_rate', 0.0):.4f}")
    lines.append(f"Missing rate: {report.get('missing_rate', 0.0):.4f}")
    lines.append("")
    lines.append(
        f"Affiliation extracted: {report.get('affiliation_extracted_true', 0)} / {report.get('total_rows', 0)} "
        f"({report.get('affiliation_extracted_rate', 0.0):.4f})"
    )
    lines.append("")

    def add_counter_block(title: str, key: str):
        lines.append(title)
        for k, v in sorted(report.get(key, {}).items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {k}: {v}")
        lines.append("")

    add_counter_block("PDF download status counts:", "pdf_download_status_counts")
    add_counter_block("Country inference method counts:", "country_inference_method_counts")
    add_counter_block("Affiliation type counts:", "affiliation_type_counts")
    add_counter_block("Missing text extraction method counts:", "missing_text_extraction_method_counts")
    add_counter_block("Missing affiliation type counts:", "missing_affiliation_type_counts")

    lines.append("Top missing email domains:")
    for k, v in sorted(report.get("top_missing_email_domains", {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("Per-shard summary:")
    for shard in report.get("shards", []):
        lines.append(
            f"  {shard['file']}: "
            f"{shard['rows_with_any_inferred_author_country']} / {shard['total_rows']} "
            f"(coverage={shard['coverage_rate']})"
        )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_fallback_candidates(csv_files: List[Path]) -> pd.DataFrame:
    dfs = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = csv_file.name
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)

    # Normalize missing columns if needed
    for col in [
        "author_all_inferred_countries",
        "author_affiliation_type",
        "affiliation_extracted",
        "author_affiliations_raw",
        "header_text",
        "emails",
        "paper_url",
    ]:
        if col not in all_df.columns:
            all_df[col] = ""

    countries_empty = all_df["author_all_inferred_countries"].fillna("").astype(str).str.strip() == ""
    unknown_type = all_df["author_affiliation_type"].fillna("").astype(str).str.strip().eq("unknown")
    no_affiliation = ~all_df["affiliation_extracted"].fillna(False).astype(bool)

    # suspiciously noisy or too-thin affiliation block
    raw_aff = all_df["author_affiliations_raw"].fillna("").astype(str)
    suspicious_raw = (
        raw_aff.str.contains(r"proceedings of|association for computational linguistics|pages \d", case=False, regex=True)
        | raw_aff.str.len().lt(8)
    )

    fallback_mask = countries_empty | unknown_type | no_affiliation | suspicious_raw

    fallback_df = all_df.loc[fallback_mask].copy()

    # add explicit reason columns
    fallback_df["llm_reason_missing_country"] = countries_empty.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_unknown_type"] = unknown_type.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_no_affiliation_extracted"] = no_affiliation.loc[fallback_mask].astype(int)
    fallback_df["llm_reason_suspicious_raw"] = suspicious_raw.loc[fallback_mask].astype(int)

    # keep useful columns first
    preferred_cols = [
        "paper_url",
        "source_file",
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
    existing_preferred = [c for c in preferred_cols if c in fallback_df.columns]
    remaining = [c for c in fallback_df.columns if c not in existing_preferred]
    fallback_df = fallback_df[existing_preferred + remaining]

    return fallback_df


def main():
    args = parse_args()

    diagnostics_dir = Path(args.diagnostics_dir)
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)

    diag_files = sorted(diagnostics_dir.glob("*.diagnostics.json"))
    if not diag_files:
        raise FileNotFoundError(f"No .diagnostics.json files found in {diagnostics_dir}")

    report = aggregate_diagnostics(diag_files)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_text_summary(report, output_txt)

    print(f"Aggregated {len(diag_files)} diagnostics files")
    print(f"Total rows: {report['total_rows']}")
    print(
        f"Rows with any inferred author country: "
        f"{report['rows_with_any_inferred_author_country']} / {report['total_rows']}"
    )
    print(f"Coverage rate: {report['coverage_rate']:.4f}")
    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote text summary: {output_txt}")

    if args.fallback_csv:
        if not args.csv_dir:
            raise ValueError("--csv-dir is required when using --fallback-csv")

        csv_dir = Path(args.csv_dir)
        csv_files = sorted(csv_dir.glob("author_affiliation_shard_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No shard CSVs found in {csv_dir}")

        fallback_df = build_fallback_candidates(csv_files)
        fallback_path = Path(args.fallback_csv)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_df.to_csv(fallback_path, index=False)

        print(f"Wrote fallback candidate CSV: {fallback_path}")
        print(f"Fallback candidate rows: {len(fallback_df)}")


if __name__ == "__main__":
    main()