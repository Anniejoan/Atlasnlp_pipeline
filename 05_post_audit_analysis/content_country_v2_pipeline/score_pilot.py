#!/usr/bin/env python3
"""
Score a v2 pilot against the reviewed Stage-1 audit reference.

Important:
- This is a DEVELOPMENT score, not the final independent validation.
- `ai_manual_content_countries` came from the assisted audit workflow and should
  be treated as the reviewed reference set for development, not as a blind gold set.
"""

import argparse
import ast
import json
from pathlib import Path
import pandas as pd

ALIASES = {
    "USA": "United States",
    "US": "United States",
    "U.S.": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Czech Republic": "Czechia",
    "Russian Federation": "Russia",
    "Republic of Korea": "South Korea",
    "Korea": "South Korea",
    "UAE": "United Arab Emirates",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d’Ivoire": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "State of Palestine": "Palestine",
    "Türkiye": "Turkey",
}

def norm_country(x):
    x = " ".join(str(x).strip().split())
    return ALIASES.get(x, x)


def parse_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [norm_country(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s or s in {"[]", "None", "nan", "Not stated"}:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [norm_country(z) for z in v if str(z).strip()]
    except Exception:
        pass
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [norm_country(z) for z in v if str(z).strip()]
    except Exception:
        pass
    return [norm_country(s)]


def as_set(x):
    return set(parse_list(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot_csv", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    pilot = pd.read_csv(args.pilot_csv)
    pred = pd.read_csv(args.summary_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "audit_record_id" not in pilot.columns:
        raise ValueError("Pilot CSV needs audit_record_id")
    if "record_id" not in pred.columns:
        raise ValueError("Prediction summary needs record_id")

    merged = pilot.merge(
        pred,
        left_on="audit_record_id",
        right_on="record_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_v2"),
    )

    merged["reference_set"] = merged["ai_manual_content_countries"].apply(as_set)
    merged["v2_explicit_set"] = merged["explicit_content_countries"].apply(as_set)
    merged["v2_inferred_set"] = merged["inferred_content_countries"].apply(as_set)
    merged["v2_combined_set"] = merged.apply(
        lambda r: r["v2_explicit_set"] | r["v2_inferred_set"], axis=1
    )

    # Assertion-level explicit precision against reviewed reference country set.
    explicit_pred = 0
    explicit_correct = 0
    for _, r in merged.iterrows():
        for c in r["v2_explicit_set"]:
            explicit_pred += 1
            if c in r["reference_set"]:
                explicit_correct += 1

    explicit_precision = (
        explicit_correct / explicit_pred if explicit_pred else float("nan")
    )

    merged["explicit_has_false_country"] = merged.apply(
        lambda r: not r["v2_explicit_set"].issubset(r["reference_set"]), axis=1
    )
    merged["combined_exact_reference"] = merged.apply(
        lambda r: r["v2_combined_set"] == r["reference_set"], axis=1
    )
    merged["reference_empty"] = merged["reference_set"].apply(lambda x: len(x) == 0)
    merged["v2_any_country"] = merged["v2_combined_set"].apply(lambda x: len(x) > 0)

    # Especially important: does v2 hallucinate explicit countries on rows whose
    # reviewed reference set is empty?
    empty_rows = merged[merged["reference_empty"]]
    explicit_on_empty = empty_rows["v2_explicit_set"].apply(lambda x: len(x) > 0).sum()

    # Human Inferred rows should not casually become explicit under v2.
    inferred_rows = merged[
        merged["human_content_country_verdict"].astype(str).str.lower() == "inferred"
    ]
    explicit_on_inferred_rows = (
        inferred_rows["v2_explicit_set"].apply(lambda x: len(x) > 0).sum()
    )

    # Human Incorrect + empty reference should ideally become unattributed/excluded.
    incorrect_empty = merged[
        (merged["human_content_country_verdict"].astype(str).str.lower() == "incorrect")
        & merged["reference_empty"]
    ]
    clean_abstain = (
        ~incorrect_empty["v2_any_country"]
        | incorrect_empty["content_country_status"].astype(str).str.startswith("EXCLUDED")
    ).sum()

    metrics = {
        "records_with_predictions": int(merged["content_country_status"].notna().sum()),
        "explicit_assertions_predicted": int(explicit_pred),
        "explicit_assertions_matching_reviewed_reference": int(explicit_correct),
        "explicit_assertion_precision_vs_reviewed_reference": (
            None if pd.isna(explicit_precision) else explicit_precision
        ),
        "rows_with_any_false_explicit_country": int(
            merged["explicit_has_false_country"].sum()
        ),
        "combined_country_set_exact_match_rate": float(
            merged["combined_exact_reference"].mean()
        ),
        "reference_empty_rows": int(len(empty_rows)),
        "reference_empty_rows_with_v2_explicit_country": int(explicit_on_empty),
        "human_inferred_rows": int(len(inferred_rows)),
        "human_inferred_rows_with_v2_explicit_country": int(explicit_on_inferred_rows),
        "human_incorrect_reference_empty_rows": int(len(incorrect_empty)),
        "human_incorrect_reference_empty_clean_abstain_or_excluded": int(clean_abstain),
        "human_incorrect_reference_empty_clean_rate": (
            float(clean_abstain / len(incorrect_empty)) if len(incorrect_empty) else None
        ),
    }

    # Review only the important disagreements.
    merged["needs_pilot_manual_check"] = (
        merged["explicit_has_false_country"]
        | (~merged["combined_exact_reference"])
        | (
            (merged["human_content_country_verdict"].astype(str).str.lower() == "inferred")
            & merged["v2_explicit_set"].apply(lambda x: len(x) > 0)
        )
    )

    printable = merged.copy()
    for col in [
        "reference_set",
        "v2_explicit_set",
        "v2_inferred_set",
        "v2_combined_set",
    ]:
        printable[col] = printable[col].apply(lambda x: json.dumps(sorted(x), ensure_ascii=False))

    printable.to_csv(out_dir / "pilot_scored_all.csv", index=False)
    printable[printable["needs_pilot_manual_check"]].to_csv(
        out_dir / "pilot_disagreements_to_check.csv", index=False
    )

    with open(out_dir / "pilot_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nManual-check rows: {int(merged['needs_pilot_manual_check'].sum())}/{len(merged)}")
    print(f"Wrote {out_dir/'pilot_disagreements_to_check.csv'}")


if __name__ == "__main__":
    main()
