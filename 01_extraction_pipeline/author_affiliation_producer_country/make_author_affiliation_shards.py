#!/usr/bin/env python3
"""
make_author_affiliation_shards.py

Create a shard manifest for author affiliation extraction.

Example:
    python make_author_affiliation_shards.py \
        --input master.csv \
        --output-manifest shard_manifest.csv \
        --shard-size 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to master CSV")
    parser.add_argument("--output-manifest", required=True, help="Path to shard manifest CSV")
    parser.add_argument("--shard-size", type=int, default=1000, help="Rows per shard")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_manifest = Path(args.output_manifest)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    total_rows = len(df)
    shard_size = args.shard_size

    num_shards = math.ceil(total_rows / shard_size)

    rows = []
    for shard_idx in range(num_shards):
        start_row = shard_idx * shard_size
        end_row = min((shard_idx + 1) * shard_size - 1, total_rows - 1)

        rows.append({
            "shard_id": f"{shard_idx + 1:03d}",
            "start_row": start_row,
            "end_row": end_row,
            "row_count": end_row - start_row + 1,
        })

    manifest_df = pd.DataFrame(rows)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_manifest, index=False)

    print(f"Total rows: {total_rows}")
    print(f"Shard size: {shard_size}")
    print(f"Num shards: {num_shards}")
    print(f"Manifest written to: {output_manifest}")


if __name__ == "__main__":
    main()