#!/usr/bin/env python3
"""
merge_core_gold_tierA.py — Merge Core + Gold Tier A (exploded) into one population

Merge policy: "Gold overrides Core." If the same dataset (matched on the
combined key: normalized paper_url + lowercased Dataset name) appears in
BOTH Core's and Gold's full base populations, Core's copy is dropped
entirely (whether or not it made Tier A) and Gold's judgment is used
instead -- Gold is human-reviewed (98.6% Validation Status == Done),
Core is automated LLM extraction. Everything else is a simple union.

Input:
  worldbench_lang_tierA_expanded_second_stage_cleaned.csv       (Core Tier A, exploded)
  worldbench_gold_lang_tierA_expanded_second_stage_cleaned.csv  (Gold Tier A, exploded)
  worldbench_base_ground_truth_second_stage_cleaned.csv         (Core full base, for dedup keys)
  worldbench_gold_base_ground_truth.csv                          (Gold full base, for dedup keys)

Output:
  worldbench_merged_lang_tierA_expanded.csv
"""
import re

import pandas as pd

DIR = "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2"


def normalize_url(url):
    if pd.isna(url):
        return None
    s = str(url).strip()
    s = re.sub(r"[?#].*$", "", s)
    return s.rstrip("/")


def make_key(df, url_col, name_col):
    url = df[url_col].apply(normalize_url).astype(str).str.strip()
    name = df[name_col].astype(str).str.strip().str.lower()
    return url + "||" + name


# --- Load full base populations (for identity/dedup keys) ---
core_base = pd.read_csv(f"{DIR}/worldbench_base_ground_truth_second_stage_cleaned.csv",
                         usecols=["Dataset name", "paper_url_key"])
gold_base = pd.read_csv(f"{DIR}/worldbench_gold_base_ground_truth.csv",
                         usecols=["Dataset name", "Paper Link (URL)"])

core_base["key"] = core_base["paper_url_key"].astype(str).str.strip() + "||" + core_base["Dataset name"].astype(str).str.strip().str.lower()
gold_base["key"] = make_key(gold_base, "Paper Link (URL)", "Dataset name")

overlap_keys = set(core_base["key"]) & set(gold_base["key"])
print(f"Datasets present in BOTH Core and Gold (exact paper_url + name match): {len(overlap_keys)}")

# --- Load Tier A expanded (exploded) populations ---
core_exp = pd.read_csv(f"{DIR}/worldbench_lang_tierA_expanded_second_stage_cleaned.csv")
gold_exp = pd.read_csv(f"{DIR}/worldbench_gold_lang_tierA_expanded_second_stage_cleaned.csv")

core_exp["key"] = core_exp["paper_url_key"].astype(str).str.strip() + "||" + core_exp["Dataset name"].astype(str).str.strip().str.lower()
gold_exp["key"] = make_key(gold_exp, "Paper Link (URL)", "Dataset name")

n_core_before = core_exp["key"].nunique()
core_exp_deduped = core_exp[~core_exp["key"].isin(overlap_keys)].copy()
n_core_after = core_exp_deduped["key"].nunique()
print(f"Core Tier A unique datasets before dedup: {n_core_before}; "
      f"dropped in favor of Gold: {n_core_before - n_core_after}; remaining: {n_core_after}")
print(f"Gold Tier A unique datasets (all kept): {gold_exp['key'].nunique()}")

# --- Union on the shared columns both files actually have ---
SHARED_COLS = [
    "Dataset name", "Task Category", "normalized_task_category", "Country Attribution Method",
    "Is dataset in multiple choice format?", "content_country", "language", "language_cleaned",
    "language_excluded_from_country_inference", "num_languages", "num_content_countries",
    "Dataset Link (URL)", "Paper Link (URL)", "key",
]
merged = pd.concat([core_exp_deduped[SHARED_COLS], gold_exp[SHARED_COLS]], ignore_index=True)
merged["source"] = ["Core"] * len(core_exp_deduped) + ["Gold"] * len(gold_exp)

merged.to_csv(f"{DIR}/worldbench_merged_lang_tierA_expanded.csv", index=False)

print(f"\nMerged Tier A expanded rows: {len(merged)}")
print(f"Merged Tier A unique datasets: {merged['key'].nunique()}")
merged_langs = merged[
    merged["language_cleaned"].notna() & (merged["language_excluded_from_country_inference"] == False)
]["language_cleaned"].nunique()
print(f"Merged Tier A unique country-inference-safe languages: {merged_langs}")
