#!/usr/bin/env python3
"""
Merge AtlasNLP content-country v2 shard outputs safely.

Expected shard layout
---------------------
parent/
  run_shard_00/
    summary.csv
    run_summary.json
    failures.jsonl          # optional
    items/*.v2.json
  run_shard_01/
    ...

What this script does
---------------------
1. Finds shard directories.
2. Loads every shard summary.csv.
3. Checks duplicate record_ids across shards.
4. Optionally compares against the original input CSV and reports missing/unexpected rows.
5. Merges all summary rows into one CSV.
6. Aggregates token usage, verifier-call counts, estimated spend, and elapsed-time stats.
7. Concatenates failures.jsonl files.
8. Writes a machine-readable merge report and a human-readable text report.

It does NOT silently discard duplicates or missing rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def find_shard_dirs(parent: Path, pattern: str) -> List[Path]:
    shards = sorted([p for p in parent.glob(pattern) if p.is_dir()])
    if not shards:
        raise FileNotFoundError(
            f"No shard directories found under {parent} matching pattern {pattern!r}"
        )
    return shards


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_input_record_ids(input_csv: Path) -> pd.Series:
    df = pd.read_csv(input_csv)

    for col in ["audit_record_id", "record_id", "source_row_index"]:
        if col in df.columns:
            return df[col].astype(str)

    # Fallback to deterministic row IDs if no explicit identifier exists.
    return pd.Series([f"row_{i}" for i in range(len(df))], name="record_id")


def numeric_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parent",
        required=True,
        help="Directory containing shard output folders",
    )
    p.add_argument(
        "--pattern",
        default="run_shard_*",
        help="Glob pattern for shard directories (default: run_shard_*)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory for merged outputs",
    )
    p.add_argument(
        "--input_csv",
        default=None,
        help=(
            "Optional original full input CSV. If supplied, the merger checks "
            "for missing and unexpected record IDs and restores original row order."
        ),
    )
    p.add_argument(
        "--require_complete",
        action="store_true",
        help=(
            "Exit non-zero if expected records are missing/unexpected/duplicated "
            "or if any shard failures exist."
        ),
    )
    args = p.parse_args()

    parent = Path(args.parent)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = find_shard_dirs(parent, args.pattern)

    summary_frames: List[pd.DataFrame] = []
    shard_meta: List[Dict[str, Any]] = []
    all_failures: List[str] = []

    for shard in shard_dirs:
        summary_path = shard / "summary.csv"
        run_summary_path = shard / "run_summary.json"
        failures_path = shard / "failures.jsonl"

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.csv in shard: {shard}")

        sdf = pd.read_csv(summary_path)

        if "record_id" not in sdf.columns:
            raise ValueError(f"{summary_path} does not contain record_id")

        sdf.insert(0, "source_shard", shard.name)
        summary_frames.append(sdf)

        if run_summary_path.exists():
            meta = load_json(run_summary_path)
        else:
            meta = {
                "records_completed": len(sdf),
                "warning": "run_summary.json missing; counts inferred from summary.csv",
            }

        meta["source_shard"] = shard.name
        meta["summary_rows"] = len(sdf)
        shard_meta.append(meta)

        if failures_path.exists():
            with open(failures_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_failures.append(line)

    merged = pd.concat(summary_frames, ignore_index=True)
    merged["record_id"] = merged["record_id"].astype(str)

    # -----------------------------
    # Duplicate checks
    # -----------------------------
    duplicate_mask = merged["record_id"].duplicated(keep=False)
    duplicates = (
        merged.loc[duplicate_mask]
        .sort_values(["record_id", "source_shard"])
        .copy()
    )
    duplicate_ids = sorted(duplicates["record_id"].unique().tolist())

    # -----------------------------
    # Optional completeness check
    # -----------------------------
    missing_ids: List[str] = []
    unexpected_ids: List[str] = []
    expected_n: Optional[int] = None
    expected_ids_ordered: Optional[pd.Series] = None

    if args.input_csv:
        input_csv = Path(args.input_csv)
        expected_ids_ordered = get_input_record_ids(input_csv).astype(str)

        if expected_ids_ordered.duplicated().any():
            dup_expected = sorted(
                expected_ids_ordered[expected_ids_ordered.duplicated(keep=False)]
                .unique()
                .tolist()
            )
            raise ValueError(
                "Input CSV itself contains duplicate record IDs. "
                f"Examples: {dup_expected[:10]}"
            )

        expected_ids = set(expected_ids_ordered.tolist())
        observed_ids = set(merged["record_id"].tolist())

        expected_n = len(expected_ids)
        missing_ids = sorted(expected_ids - observed_ids)
        unexpected_ids = sorted(observed_ids - expected_ids)

    # -----------------------------
    # Aggregate costs / usage
    # -----------------------------
    total_records = len(merged)

    total_estimated_cost = numeric_sum(merged, "estimated_cost_usd")

    total_extractor_input = int(numeric_sum(merged, "extractor_input_tokens"))
    total_extractor_output = int(numeric_sum(merged, "extractor_output_tokens"))
    total_extractor_reasoning = int(numeric_sum(merged, "extractor_reasoning_tokens"))

    total_verifier_input = int(numeric_sum(merged, "verifier_input_tokens"))
    total_verifier_output = int(numeric_sum(merged, "verifier_output_tokens"))
    total_verifier_reasoning = int(numeric_sum(merged, "verifier_reasoning_tokens"))

    if "verifier_input_tokens" in merged.columns:
        verifier_calls = int(
            (pd.to_numeric(merged["verifier_input_tokens"], errors="coerce").fillna(0) > 0).sum()
        )
    else:
        verifier_calls = 0

    if "elapsed_seconds" in merged.columns:
        elapsed_series = pd.to_numeric(
            merged["elapsed_seconds"], errors="coerce"
        ).dropna()
    else:
        elapsed_series = pd.Series(dtype=float)

    # -----------------------------
    # Status / relationship counts
    # -----------------------------
    status_counts: Dict[str, int] = {}
    if "content_country_status" in merged.columns:
        status_counts = {
            str(k): int(v)
            for k, v in merged["content_country_status"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }

    relationship_counts: Dict[str, int] = {}
    if "dataset_relationship" in merged.columns:
        relationship_counts = {
            str(k): int(v)
            for k, v in merged["dataset_relationship"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }

    # -----------------------------
    # Restore input order if possible
    # -----------------------------
    if expected_ids_ordered is not None:
        order_map = {rid: i for i, rid in enumerate(expected_ids_ordered)}
        merged["_merge_order"] = merged["record_id"].map(order_map)
        merged = (
            merged.sort_values(
                ["_merge_order", "record_id", "source_shard"],
                na_position="last",
            )
            .drop(columns=["_merge_order"])
        )
    else:
        merged = merged.sort_values(["record_id", "source_shard"])

    # -----------------------------
    # Write outputs
    # -----------------------------
    merged_path = out_dir / "content_country_v2_merged.csv"
    merged.to_csv(merged_path, index=False)

    duplicates_path = out_dir / "duplicate_records.csv"
    duplicates.to_csv(duplicates_path, index=False)

    (out_dir / "missing_record_ids.txt").write_text(
        "\n".join(missing_ids), encoding="utf-8"
    )
    (out_dir / "unexpected_record_ids.txt").write_text(
        "\n".join(unexpected_ids), encoding="utf-8"
    )
    (out_dir / "failures_merged.jsonl").write_text(
        ("\n".join(all_failures) + "\n") if all_failures else "",
        encoding="utf-8",
    )

    with open(out_dir / "shard_run_summaries.json", "w", encoding="utf-8") as f:
        json.dump(shard_meta, f, indent=2, ensure_ascii=False)

    complete = (
        len(duplicate_ids) == 0
        and len(missing_ids) == 0
        and len(unexpected_ids) == 0
        and len(all_failures) == 0
    )

    merge_report = {
        "shards_found": len(shard_dirs),
        "shard_names": [p.name for p in shard_dirs],
        "merged_rows": total_records,
        "unique_record_ids": int(merged["record_id"].nunique()),
        "duplicate_record_ids": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "input_csv_supplied": bool(args.input_csv),
        "expected_unique_records": expected_n,
        "missing_record_ids": len(missing_ids),
        "unexpected_record_ids": len(unexpected_ids),
        "failure_lines": len(all_failures),
        "verifier_calls": verifier_calls,
        "verifier_call_rate": (
            verifier_calls / total_records if total_records else None
        ),
        "estimated_total_cost_usd": total_estimated_cost,
        "estimated_cost_per_merged_record_usd": (
            total_estimated_cost / total_records if total_records else None
        ),
        "total_extractor_input_tokens": total_extractor_input,
        "total_extractor_output_tokens": total_extractor_output,
        "total_extractor_reasoning_tokens": total_extractor_reasoning,
        "total_verifier_input_tokens": total_verifier_input,
        "total_verifier_output_tokens": total_verifier_output,
        "total_verifier_reasoning_tokens": total_verifier_reasoning,
        "mean_per_record_elapsed_seconds": (
            float(elapsed_series.mean()) if len(elapsed_series) else None
        ),
        "median_per_record_elapsed_seconds": (
            float(elapsed_series.median()) if len(elapsed_series) else None
        ),
        "content_country_status_counts": status_counts,
        "dataset_relationship_counts": relationship_counts,
        "complete": complete,
    }

    with open(out_dir / "merge_report.json", "w", encoding="utf-8") as f:
        json.dump(merge_report, f, indent=2, ensure_ascii=False)

    report_lines = [
        "AtlasNLP content-country v2 shard merge",
        "=========================================",
        f"Shards found: {len(shard_dirs)}",
        f"Merged rows: {total_records}",
        f"Unique record IDs: {merged['record_id'].nunique()}",
        f"Duplicate record IDs: {len(duplicate_ids)}",
    ]

    if expected_n is not None:
        report_lines.extend([
            f"Expected unique records: {expected_n}",
            f"Missing record IDs: {len(missing_ids)}",
            f"Unexpected record IDs: {len(unexpected_ids)}",
        ])

    report_lines.extend([
        f"Failure lines: {len(all_failures)}",
        f"Verifier calls: {verifier_calls}",
        (
            f"Verifier call rate: {verifier_calls / total_records:.3f}"
            if total_records
            else "Verifier call rate: n/a"
        ),
        f"Estimated total API cost: ${total_estimated_cost:.4f}",
        (
            f"Estimated cost per merged record: "
            f"${total_estimated_cost / total_records:.6f}"
            if total_records
            else "Estimated cost per merged record: n/a"
        ),
        "",
        "Content-country status counts:",
    ])

    for k, v in status_counts.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.append("")
    report_lines.append("Dataset relationship counts:")
    for k, v in relationship_counts.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.extend([
        "",
        f"COMPLETE: {complete}",
        "",
        f"Merged CSV: {merged_path}",
        f"JSON report: {out_dir / 'merge_report.json'}",
    ])

    report_txt = "\n".join(report_lines)
    (out_dir / "merge_report.txt").write_text(report_txt, encoding="utf-8")
    print(report_txt)

    if args.require_complete and not complete:
        raise SystemExit(
            "Merge is not complete. Inspect duplicate_records.csv, "
            "missing_record_ids.txt, unexpected_record_ids.txt, "
            "and failures_merged.jsonl."
        )


if __name__ == "__main__":
    main()
