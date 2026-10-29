#!/usr/bin/env python3
"""
Merge second-pass shard outputs and safely reintegrate them into the full master CSV,
using normalized paper URLs as the stable merge key.

Built-in key handling:
- Master key column: 'Paper Link (URL)'
- Shard key column: 'url'
- Unified internal key: 'paper_url'

Shard layout expected:
    shard_dir/
        shard_001/
            country_second_pass_summary.csv
        shard_002/
            country_second_pass_summary.csv
        ...

Features:
- Recursively finds shard CSVs inside shard_* folders
- Aligns schemas across shards
- Concatenates shard outputs
- Normalizes URLs before matching
- Detects duplicate normalized keys
- Loads the full master CSV
- Safely updates only selected second-pass columns
- Preserves original master values when second-pass values are blank/missing
- Writes:
    - merged_second_pass.csv
    - updated_master.csv
    - validation_report.json
    - validation_report.txt
    - duplicate_second_pass_keys.csv (if duplicates exist)
    - duplicate_master_keys.csv (if duplicates exist)
    - second_pass_keys_missing_in_master.csv
- Supports optional collapsing of duplicate keys by keeping the last occurrence

Example:
python merge_second_pass.py \
  --master-csv /path/to/master.csv \
  --shard-dir /path/to/shards \
  --update-columns all_countries_covered country_attribution_method language_without_explicit_country_flag \
  --output-dir /path/to/output \
  --prefer-latest-duplicate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd


MASTER_KEY_COL = "Paper Link (URL)"
SHARD_KEY_COL = "url"
UNIFIED_KEY_COL = "paper_url"
DEFAULT_SHARD_FILENAME = "country_second_pass_summary.csv"

DEFAULT_NULL_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "none",
    "None",
    "null",
    "Null",
    "NULL",
    "n/a",
    "N/A",
    "na",
    "NA",
    "<NA>",
}


@dataclass
class ValidationReport:
    shard_files_found: int
    shard_directories_found: int
    shard_directories_missing_file: int
    shard_rows_total_raw: int
    shard_rows_missing_raw_key: int

    merged_second_pass_rows: int
    merged_second_pass_unique_keys: int
    merged_second_pass_duplicate_key_rows: int
    duplicate_keys_sample_second_pass: List[str]

    master_rows_before: int
    master_rows_missing_raw_key: int
    master_unique_keys_before: int
    master_duplicate_key_rows: int
    duplicate_keys_sample_master: List[str]

    second_pass_keys_missing_in_master: int
    second_pass_keys_missing_in_master_sample: List[str]
    master_keys_without_second_pass_match: int

    rows_updated_any_column: int
    per_column_updates: Dict[str, int]

    master_rows_after: int
    master_unique_keys_after: int

    update_columns_requested: List[str]
    update_columns_applied: List[str]
    shard_columns_union_count: int
    shard_columns_union: List[str]
    notes: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely merge second-pass shard outputs back into a master CSV using normalized paper URLs."
    )
    parser.add_argument(
        "--master-csv",
        required=True,
        help="Path to the full master CSV (~18k rows).",
    )
    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Directory containing shard_*/country_second_pass_summary.csv outputs.",
    )
    parser.add_argument(
        "--update-columns",
        nargs="+",
        required=True,
        help="Columns to safely update from second-pass outputs into master.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--shard-filename",
        default=DEFAULT_SHARD_FILENAME,
        help=f"Expected CSV filename inside each shard folder (default: {DEFAULT_SHARD_FILENAME}).",
    )
    parser.add_argument(
        "--prefer-latest-duplicate",
        action="store_true",
        help="If duplicate normalized paper URLs exist, keep the last occurrence instead of failing.",
    )
    parser.add_argument(
        "--master-encoding",
        default="utf-8",
        help="Encoding for master CSV (default: utf-8).",
    )
    parser.add_argument(
        "--shard-encoding",
        default="utf-8",
        help="Encoding for shard CSVs (default: utf-8).",
    )
    return parser.parse_args()


def normalize_column_name(col: str) -> str:
    return str(col).strip()


def normalize_scalar(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in DEFAULT_NULL_LIKE_VALUES:
        return None
    return text


def is_nonempty_value(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text not in DEFAULT_NULL_LIKE_VALUES


def normalize_url(url: object) -> Optional[str]:
    text = normalize_scalar(url)
    if text is None:
        return None

    try:
        parsed = urlparse(text)
    except Exception:
        return text

    scheme = "https"
    netloc = (parsed.netloc or "").lower().strip()
    path = (parsed.path or "").strip()

    if not netloc and parsed.path:
        maybe = urlparse("https://" + text)
        netloc = (maybe.netloc or "").lower().strip()
        path = (maybe.path or "").strip()

    path = path.rstrip("/")

    normalized = urlunparse((scheme, netloc, path, "", "", ""))

    if normalized == "https://":
        return None
    return normalized


def load_csv(path: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding=encoding,
    )


def prepare_master_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    if MASTER_KEY_COL not in df.columns:
        raise ValueError(
            f"Master CSV is missing required key column '{MASTER_KEY_COL}'. "
            f"Available columns: {list(df.columns)}"
        )

    raw_missing = int(df[MASTER_KEY_COL].map(normalize_scalar).isna().sum())
    df[UNIFIED_KEY_COL] = df[MASTER_KEY_COL].map(normalize_url)
    return df, raw_missing


def prepare_shard_df(df: pd.DataFrame, shard_name: str) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    if SHARD_KEY_COL not in df.columns:
        raise ValueError(
            f"Shard file '{shard_name}' is missing required key column '{SHARD_KEY_COL}'. "
            f"Available columns: {list(df.columns)}"
        )

    raw_missing = int(df[SHARD_KEY_COL].map(normalize_scalar).isna().sum())
    df[UNIFIED_KEY_COL] = df[SHARD_KEY_COL].map(normalize_url)
    df["_source_shard_file"] = shard_name
    return df, raw_missing


def align_schemas(dfs: Sequence[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str]]:
    all_columns = sorted({col for df in dfs for col in df.columns})
    aligned = [df.reindex(columns=all_columns) for df in dfs]
    return aligned, all_columns


def get_duplicate_rows(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    dup_mask = df[key_col].notna() & df[key_col].duplicated(keep=False)
    return df.loc[dup_mask].copy()


def duplicate_stats(df: pd.DataFrame, key_col: str) -> Tuple[int, List[str]]:
    dup_df = get_duplicate_rows(df, key_col)
    dup_keys = dup_df[key_col].dropna().astype(str).unique().tolist()
    return int(len(dup_df)), dup_keys[:20]


def find_shard_csvs(shard_dir: Path, shard_filename: str) -> Tuple[List[Path], int, int]:
    shard_pattern = re.compile(r"^shard_\d+$")
    shard_dirs = sorted(
        [p for p in shard_dir.iterdir() if p.is_dir() and shard_pattern.match(p.name)],
        key=lambda p: p.name,
    )

    shard_files: List[Path] = []
    missing_count = 0

    for sdir in shard_dirs:
        expected = sdir / shard_filename
        if expected.exists() and expected.is_file():
            shard_files.append(expected)
        else:
            missing_count += 1

    return shard_files, len(shard_dirs), missing_count


def choose_existing_update_columns(
    shard_df: pd.DataFrame,
    master_df: pd.DataFrame,
    requested_columns: Sequence[str],
) -> Tuple[List[str], List[str], pd.DataFrame]:
    notes: List[str] = []
    applied: List[str] = []
    master_df = master_df.copy()

    for col in requested_columns:
        in_shard = col in shard_df.columns
        in_master = col in master_df.columns

        if not in_shard and not in_master:
            notes.append(f"Skipped '{col}': not found in shard outputs or master CSV.")
            continue

        if in_shard and not in_master:
            master_df[col] = pd.NA
            notes.append(f"Column '{col}' existed in shard outputs but not in master CSV; added it to master.")

        if not in_shard and in_master:
            notes.append(f"Skipped '{col}': exists in master CSV but not in shard outputs.")
            continue

        applied.append(col)

    return applied, notes, master_df


def safely_merge_second_pass(
    master_df: pd.DataFrame,
    second_pass_df: pd.DataFrame,
    update_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    master = master_df.copy()
    second_pass = second_pass_df.copy()

    master["_original_row_order"] = range(len(master))

    sp_renamed = second_pass[[UNIFIED_KEY_COL] + list(update_columns)].copy()
    rename_map = {col: f"{col}__second_pass" for col in update_columns}
    sp_renamed = sp_renamed.rename(columns=rename_map)

    merged = master.merge(
        sp_renamed,
        on=UNIFIED_KEY_COL,
        how="left",
        validate="m:1",
    )

    per_column_updates: Dict[str, int] = {}
    updated_any_mask = pd.Series(False, index=merged.index)

    for col in update_columns:
        sp_col = f"{col}__second_pass"

        if col not in merged.columns:
            merged[col] = pd.NA

        new_values = merged[sp_col]
        update_mask = new_values.map(is_nonempty_value)

        merged.loc[update_mask, col] = new_values[update_mask]

        per_column_updates[col] = int(update_mask.sum())
        updated_any_mask = updated_any_mask | update_mask.fillna(False)

        merged = merged.drop(columns=[sp_col])

    rows_updated_any_column = int(updated_any_mask.sum())
    merged = merged.sort_values("_original_row_order").drop(columns=["_original_row_order"])
    return merged, per_column_updates, rows_updated_any_column


def write_report(report: ValidationReport, output_dir: Path) -> None:
    json_path = output_dir / "validation_report.json"
    txt_path = output_dir / "validation_report.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    lines = [
        "WORLD BENCH SECOND-PASS MERGE VALIDATION REPORT",
        "=" * 52,
        "",
        f"Master raw key column: {MASTER_KEY_COL}",
        f"Shard raw key column: {SHARD_KEY_COL}",
        f"Unified normalized key column: {UNIFIED_KEY_COL}",
        "",
        f"Shard directories found: {report.shard_directories_found}",
        f"Shard files found: {report.shard_files_found}",
        f"Shard directories missing expected file: {report.shard_directories_missing_file}",
        f"Raw shard rows total: {report.shard_rows_total_raw}",
        f"Raw shard rows missing raw key: {report.shard_rows_missing_raw_key}",
        f"Merged second-pass rows: {report.merged_second_pass_rows}",
        f"Merged second-pass unique keys: {report.merged_second_pass_unique_keys}",
        f"Merged second-pass duplicate key rows: {report.merged_second_pass_duplicate_key_rows}",
        f"Duplicate key sample (second pass): {report.duplicate_keys_sample_second_pass}",
        "",
        f"Master rows before: {report.master_rows_before}",
        f"Master rows missing raw key: {report.master_rows_missing_raw_key}",
        f"Master unique keys before: {report.master_unique_keys_before}",
        f"Master duplicate key rows: {report.master_duplicate_key_rows}",
        f"Duplicate key sample (master): {report.duplicate_keys_sample_master}",
        "",
        f"Second-pass keys missing in master: {report.second_pass_keys_missing_in_master}",
        f"Missing key sample: {report.second_pass_keys_missing_in_master_sample}",
        f"Master keys without second-pass match: {report.master_keys_without_second_pass_match}",
        "",
        f"Rows updated in any tracked column: {report.rows_updated_any_column}",
        f"Per-column updates: {report.per_column_updates}",
        "",
        f"Master rows after: {report.master_rows_after}",
        f"Master unique keys after: {report.master_unique_keys_after}",
        "",
        f"Requested update columns: {report.update_columns_requested}",
        f"Applied update columns: {report.update_columns_applied}",
        "",
        f"Union shard column count: {report.shard_columns_union_count}",
        "",
        "Notes:",
    ]
    lines.extend([f"- {note}" for note in report.notes] if report.notes else ["- None"])

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    master_csv = Path(args.master_csv)
    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    shard_paths, shard_dir_count, shard_missing_count = find_shard_csvs(
        shard_dir=shard_dir,
        shard_filename=args.shard_filename,
    )

    if not shard_paths:
        raise FileNotFoundError(
            f"No shard files found under {shard_dir} in shard_* folders with filename '{args.shard_filename}'."
        )

    shard_dfs: List[pd.DataFrame] = []
    raw_shard_row_count = 0
    raw_shard_missing_key_count = 0

    for path in shard_paths:
        df = load_csv(path, encoding=args.shard_encoding)
        raw_shard_row_count += len(df)
        df, missing_raw = prepare_shard_df(df, str(path.relative_to(shard_dir)))
        raw_shard_missing_key_count += missing_raw
        shard_dfs.append(df)

    aligned_shards, shard_columns_union = align_schemas(shard_dfs)
    merged_second_pass = pd.concat(aligned_shards, ignore_index=True)

    missing_normalized_key_rows = int(merged_second_pass[UNIFIED_KEY_COL].isna().sum())
    if missing_normalized_key_rows > 0:
        merged_second_pass = merged_second_pass[merged_second_pass[UNIFIED_KEY_COL].notna()].copy()

    sp_dup_rows, sp_dup_sample = duplicate_stats(merged_second_pass, UNIFIED_KEY_COL)
    if sp_dup_rows > 0:
        get_duplicate_rows(merged_second_pass, UNIFIED_KEY_COL).to_csv(
            output_dir / "duplicate_second_pass_keys.csv",
            index=False,
            encoding="utf-8",
        )

    if sp_dup_rows > 0 and not args.prefer-latest-duplicate:
        raise ValueError(
            f"Duplicate normalized paper URLs detected in second-pass data ({sp_dup_rows} rows). "
            f"Sample keys: {sp_dup_sample}. "
            f"Inspect duplicate_second_pass_keys.csv or rerun with --prefer-latest-duplicate."
        )

    if sp_dup_rows > 0 and args.prefer_latest_duplicate:
        merged_second_pass = (
            merged_second_pass.drop_duplicates(subset=[UNIFIED_KEY_COL], keep="last")
            .copy()
            .reset_index(drop=True)
        )

    master_df = load_csv(master_csv, encoding=args.master_encoding)
    master_df, raw_master_missing_key_count = prepare_master_df(master_df)

    missing_master_normalized_key_rows = int(master_df[UNIFIED_KEY_COL].isna().sum())

    master_dup_rows, master_dup_sample = duplicate_stats(master_df, UNIFIED_KEY_COL)
    if master_dup_rows > 0:
        get_duplicate_rows(master_df, UNIFIED_KEY_COL).to_csv(
            output_dir / "duplicate_master_keys.csv",
            index=False,
            encoding="utf-8",
        )
        raise ValueError(
            f"Duplicate normalized paper URLs detected in master CSV ({master_dup_rows} rows). "
            f"Sample keys: {master_dup_sample}. "
            f"Inspect duplicate_master_keys.csv before merging."
        )

    update_columns_applied, notes, master_df = choose_existing_update_columns(
        shard_df=merged_second_pass,
        master_df=master_df,
        requested_columns=args.update_columns,
    )

    if not update_columns_applied:
        raise ValueError(
            "None of the requested update columns could be applied. "
            "Check column names in shards and master CSV."
        )

    second_pass_keys = set(merged_second_pass[UNIFIED_KEY_COL].dropna().astype(str))
    master_keys = set(master_df[UNIFIED_KEY_COL].dropna().astype(str))

    second_pass_keys_missing_in_master = sorted(second_pass_keys - master_keys)
    if second_pass_keys_missing_in_master:
        missing_df = merged_second_pass[
            merged_second_pass[UNIFIED_KEY_COL].isin(second_pass_keys_missing_in_master)
        ].copy()
        missing_df.to_csv(
            output_dir / "second_pass_keys_missing_in_master.csv",
            index=False,
            encoding="utf-8",
        )

    master_keys_without_second_pass_match = len(master_keys - second_pass_keys)

    updated_master, per_column_updates, rows_updated_any_column = safely_merge_second_pass(
        master_df=master_df,
        second_pass_df=merged_second_pass,
        update_columns=update_columns_applied,
    )

    if len(updated_master) != len(master_df):
        raise RuntimeError(
            f"Row count changed after merge: before={len(master_df)}, after={len(updated_master)}"
        )

    updated_master_dup_rows, updated_master_dup_sample = duplicate_stats(updated_master, UNIFIED_KEY_COL)
    if updated_master_dup_rows > 0:
        raise RuntimeError(
            f"Updated master has duplicate normalized paper URLs ({updated_master_dup_rows} rows). "
            f"Sample: {updated_master_dup_sample}"
        )

    merged_second_pass_out = merged_second_pass.sort_values(UNIFIED_KEY_COL).reset_index(drop=True)
    updated_master_out = updated_master.copy()

    merged_second_pass_out.to_csv(
        output_dir / "merged_second_pass.csv",
        index=False,
        encoding="utf-8",
    )
    updated_master_out.to_csv(
        output_dir / "updated_master.csv",
        index=False,
        encoding="utf-8",
    )

    notes.extend([
        f"Rows with missing normalized shard key dropped before merge: {missing_normalized_key_rows}",
        f"Rows with missing normalized master key retained but unmatched: {missing_master_normalized_key_rows}",
        "URL normalization applied: scheme forced to https, host lowercased, trailing slash removed, query and fragment removed.",
        f"Master raw key column '{MASTER_KEY_COL}' mapped internally to '{UNIFIED_KEY_COL}'.",
        f"Shard raw key column '{SHARD_KEY_COL}' mapped internally to '{UNIFIED_KEY_COL}'.",
        f"Shard discovery used folder pattern 'shard_*' and filename '{args.shard_filename}'.",
    ])

    if sp_dup_rows > 0 and args.prefer_latest_duplicate:
        notes.append(
            "Duplicate second-pass normalized keys were collapsed by keeping the last occurrence."
        )

    report = ValidationReport(
        shard_files_found=len(shard_paths),
        shard_directories_found=shard_dir_count,
        shard_directories_missing_file=shard_missing_count,
        shard_rows_total_raw=raw_shard_row_count,
        shard_rows_missing_raw_key=raw_shard_missing_key_count,

        merged_second_pass_rows=len(merged_second_pass),
        merged_second_pass_unique_keys=merged_second_pass[UNIFIED_KEY_COL].nunique(dropna=True),
        merged_second_pass_duplicate_key_rows=0 if args.prefer_latest_duplicate else sp_dup_rows,
        duplicate_keys_sample_second_pass=[] if args.prefer_latest_duplicate else sp_dup_sample,

        master_rows_before=len(master_df),
        master_rows_missing_raw_key=raw_master_missing_key_count,
        master_unique_keys_before=master_df[UNIFIED_KEY_COL].nunique(dropna=True),
        master_duplicate_key_rows=master_dup_rows,
        duplicate_keys_sample_master=master_dup_sample,

        second_pass_keys_missing_in_master=len(second_pass_keys_missing_in_master),
        second_pass_keys_missing_in_master_sample=second_pass_keys_missing_in_master[:20],
        master_keys_without_second_pass_match=master_keys_without_second_pass_match,

        rows_updated_any_column=rows_updated_any_column,
        per_column_updates=per_column_updates,

        master_rows_after=len(updated_master_out),
        master_unique_keys_after=updated_master_out[UNIFIED_KEY_COL].nunique(dropna=True),

        update_columns_requested=list(args.update_columns),
        update_columns_applied=update_columns_applied,
        shard_columns_union_count=len(shard_columns_union),
        shard_columns_union=shard_columns_union,
        notes=notes,
    )

    write_report(report, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)