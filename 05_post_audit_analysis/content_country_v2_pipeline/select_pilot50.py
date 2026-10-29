#!/usr/bin/env python3
"""Select a reproducible random 50-record development pilot from the completed 400."""

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260826)
    args = p.parse_args()

    df = pd.read_csv(args.audit_csv)
    if len(df) < args.n:
        raise ValueError(f"Input has only {len(df)} rows")

    pilot = df.sample(n=args.n, random_state=args.seed).copy()
    pilot.insert(0, "pilot_order", range(1, len(pilot) + 1))
    pilot.to_csv(args.out, index=False)

    print(f"Wrote {args.out}")
    print("\nHuman verdict composition:")
    print(pilot["human_content_country_verdict"].value_counts(dropna=False).to_string())

    if "multi_country_content" in pilot:
        print("\nMulti-country composition:")
        print(pilot["multi_country_content"].value_counts(dropna=False).to_string())

    if "human_failure_code" in pilot:
        affil = pilot["human_failure_code"].fillna("").str.contains("AFFIL_AS_CONTENT").sum()
        print(f"\nAFFIL_AS_CONTENT rows: {affil}/{len(pilot)}")


if __name__ == "__main__":
    main()
