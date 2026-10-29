#!/usr/bin/env python3
"""
extra_lang_normalize_gold.py — Gold-side counterpart to extra_lang_normalize.py

Role: applies the exact same second-stage language cleanup (DIRECT_MAP,
  ISO_CODE_MAP, sign-language/script/historical/variant classification, etc.
  from extra_lang_normalize.py, imported not copied) to WorldBench-Gold's
  Tier A, so Gold's `language_cleaned` / `language_excluded_from_country_inference`
  are directly comparable to Core's.

Mirrors final_merge.py's own tierA_expanded step (explode countries, explode
languages) before applying the single-language cleanup, since Core's
`language_cleaned` is only computed in "single" (exploded) mode.

Input:  unified_worldbench_views/language_cleanup_v2/worldbench_gold_lang_tierA.csv
Output: unified_worldbench_views/language_cleanup_v2/worldbench_gold_lang_tierA_expanded_second_stage_cleaned.csv

Run from compiled_ACL/task_matrix_construction/:
  python extra_lang_normalize_gold.py
"""
import ast
from pathlib import Path

import pandas as pd

import extra_lang_normalize as eln  # reuse exact maps/classification logic, not reimplemented

INPUT_CSV = (
    "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/"
    "unified_worldbench_views/language_cleanup_v2/worldbench_gold_lang_tierA.csv"
)
OUTPUT_CSV = (
    "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/"
    "unified_worldbench_views/language_cleanup_v2/worldbench_gold_lang_tierA_expanded_second_stage_cleaned.csv"
)


def parse_list_cell(x):
    """best_country_list_valid / language_list were written by final_merge_gold.py
    as Python-list reprs (e.g. "['Kenya', 'Uganda']"); parse them back."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


df = pd.read_csv(INPUT_CSV)
print(f"Gold Tier A datasets (before explode): {len(df)}")

df["best_country_list_valid"] = df["best_country_list_valid"].apply(parse_list_cell)
df["language_list"] = df["language_list"].apply(parse_list_cell)

# Mirror final_merge.py's tierA_expanded: explode country, then explode language
# -> one row per (dataset, language, country) combination.
df_expanded = (
    df.explode("best_country_list_valid")
    .explode("language_list")
    .rename(columns={"best_country_list_valid": "content_country", "language_list": "language"})
    .copy()
)
print(f"Gold Tier A expanded (dataset, language, country) rows: {len(df_expanded)}")

# Apply the exact same single-mode cleanup extra_lang_normalize.py uses for
# worldbench_lang_tierA_expanded (mode="single"): parse_single_language_cell +
# classify_and_normalize_token, take the first (only) token's classification.
parsed = df_expanded["language"].apply(eln.parse_single_language_cell)
processed = parsed.apply(eln.process_language_tokens)

df_expanded["parsed_language_tokens"] = parsed
df_expanded["canonical_language_list_cleaned"] = processed.apply(lambda x: x["canonical_language_list"])
df_expanded["language_cleaned"] = processed.apply(
    lambda x: x["canonical_language_list"][0] if x["canonical_language_list"] else None
)
df_expanded["language_cleaned_type"] = processed.apply(
    lambda x: x["language_type_records"][0]["normalized_language_type"] if x["language_type_records"] else None
)
df_expanded["language_excluded_from_country_inference"] = processed.apply(
    lambda x: x["language_type_records"][0]["exclude_from_country_inference"] if x["language_type_records"] else True
)

df_expanded.to_csv(OUTPUT_CSV, index=False)
print(f"Written: {OUTPUT_CSV}")

usable = df_expanded[
    df_expanded["language_cleaned"].notna() & (df_expanded["language_excluded_from_country_inference"] == False)
]
print(f"\nGold Tier A rows with a country-inference-safe language_cleaned: {len(usable)}")
print(f"Gold Tier A unique language_cleaned (country-inference-safe, comparable to Core's 283): "
      f"{usable['language_cleaned'].nunique()}")
