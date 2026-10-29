#!/usr/bin/env python3
"""
combine_author_affiliation_shards.py


python combine_author_affiliation_shards.py --input-dir /scratch/<PI_ACCOUNT_ROOT>/<PI_ACCOUNT>/jnwatu/worldbench_dataset_pipeline/author_affiliation_outputs --output author_affiliation_all_regex.csv
Combine shard CSV outputs into one file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with shard CSVs")
    parser.add_argument("--output", required=True, help="Combined output CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    files = sorted(input_dir.glob("author_affiliation_shard_*.csv"))
    if not files:
        raise FileNotFoundError(f"No shard CSVs found in {input_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Combined {len(files)} files")
    print(f"Total rows: {len(combined)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()