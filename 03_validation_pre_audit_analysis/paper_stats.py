# =============================================================================
# paper_stats.py — Compute WorldBench summary statistics (Table 2 in paper)
#
# Paper section: §4 "WorldBench Overview" → Table 2
# Produces: Table 2 — "Summary statistics for WorldBench"
#   Reports counts for WorldBench-Core, WorldBench-Gold entries/unique datasets,
#   Core+Gold combined, with counts of multilingual/multimodal/synthetic/MCQ datasets,
#   countries represented, task categories, audited languages, and producer metadata.
#
# Run from compiled_ACL/task_matrix_construction/worldbench_matrix_outputs/:
#   python paper_stats.py
#
# Input:
#   unified_worldbench_views/language_cleanup_v2/human_worldbench.csv
#   unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv
#
# Output:
#   worldbench_core_gold_stats_comparison.csv — per-set summary table
#   worldbench_expanded_coverage_stats.csv    — Core+Gold combined coverage
# =============================================================================
import pandas as pd
import numpy as np
import re
import ast

# ==================================================
# Load data
# ==================================================

# df_auto = pd.read_csv("automated_full_set.csv")
# df_gold = pd.read_csv("human_validated_set.csv")

df_gold = pd.read_csv('/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/human_worldbench.csv')
df_auto = pd.read_csv("/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv")

# ==================================================
# Helpers
# ==================================================

MISSING_VALUES = {"", "nan", "none", "not stated", "unknown", "n/a", "na"}

def is_missing(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in MISSING_VALUES

def yes_count(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"])
        .sum()
    )

def contains_multilingual(series):
    return (
        series.astype(str)
        .str.contains("multilingual", case=False, na=False)
        .sum()
    )

def split_items(x):
    if is_missing(x):
        return []

    if isinstance(x, list):
        return x

    x = str(x).strip()

    # Python-list-like strings: ['Kenya', 'Uganda']
    if x.startswith("[") and x.endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return re.split(r";|,", x)

def clean_item(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x.lower() in MISSING_VALUES:
        return None
    return x

def to_set(x):
    return {clean_item(i) for i in split_items(x) if clean_item(i)}

COUNTRY_ALIASES = {
    "USA": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Viet Nam": "Vietnam",
}

def normalize_country(c):
    if c is None:
        return None
    c = c.strip()
    return COUNTRY_ALIASES.get(c, c)

def country_set_from_gold_row(row):
    assigned = to_set(row.get("Assigned country"))
    covered = to_set(row.get("All Countries Covered (you can select multiple countries)"))
    return {normalize_country(c) for c in assigned.union(covered) if normalize_country(c)}

def country_set_from_auto_row(row):
    # Prefer validated/cleaned country list if available
    for col in ["best_country_list_valid", "Normalized Countries", "all_countries_covered"]:
        if col in row.index:
            s = {normalize_country(c) for c in to_set(row.get(col)) if normalize_country(c)}
            if s:
                return s
    return set()

def language_set_from_row(row):
    for col in [
        "canonical_language_string_cleaned",
        "languages_normalized_joined",
        "Normalized Languages",
        "Languages covered (separate using semicolon)",
    ]:
        if col in row.index:
            s = to_set(row.get(col))
            if s:
                return s
    return set()

def task_value_from_row(row):
    for col in ["normalized_task_category", "Task Category"]:
        if col in row.index and not is_missing(row.get(col)):
            return str(row.get(col)).strip()
    return None

def dataset_name_key(x):
    if is_missing(x):
        return None
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def paper_url_key(x):
    if is_missing(x):
        return None
    return str(x).strip().lower().rstrip("/")

# ==================================================
# Prepare clean versions
# ==================================================

auto = df_auto.copy()
gold_entries = df_gold.copy()

auto["dataset_key"] = auto["Dataset name"].apply(dataset_name_key)
gold_entries["dataset_key"] = gold_entries["Dataset name"].apply(dataset_name_key)

auto["paper_key"] = auto.get("paper_url", auto.get("Paper Link (URL)", pd.Series(index=auto.index))).apply(paper_url_key)
gold_entries["paper_key"] = gold_entries.get("Paper Link (URL)", pd.Series(index=gold_entries.index)).apply(paper_url_key)

auto["country_set"] = auto.apply(country_set_from_auto_row, axis=1)
gold_entries["country_set"] = gold_entries.apply(country_set_from_gold_row, axis=1)

auto["language_set"] = auto.apply(language_set_from_row, axis=1)
gold_entries["language_set"] = gold_entries.apply(language_set_from_row, axis=1)

auto["task_clean"] = auto.apply(task_value_from_row, axis=1)
gold_entries["task_clean"] = gold_entries.apply(task_value_from_row, axis=1)

# Deduplicate gold to dataset-level
gold_unique = (
    gold_entries
    .sort_values(["dataset_key", "paper_key"])
    .groupby("dataset_key", dropna=True)
    .agg({
        "Dataset name": "first",
        "paper_key": "first",
        "country_set": lambda sets: set().union(*sets),
        "language_set": lambda sets: set().union(*sets),
        "task_clean": lambda x: x.dropna().iloc[0] if len(x.dropna()) else None,
        "Language coverage type": "first",
        "Is the dataset multimodal?": "first",
        "Synthetic?": "first",
        "Is dataset in multiple choice format?": "first",
    })
    .reset_index()
)

# ==================================================
# Generic stats function
# ==================================================

def summarize_dataset(df, name, is_gold_unique=False):
    n = len(df)

    country_missing = df["country_set"].apply(lambda s: len(s) == 0).sum()
    country_present = n - country_missing

    countries = sorted(set().union(*df["country_set"])) if n else []
    languages = sorted(set().union(*df["language_set"])) if n else []

    task_count = df["task_clean"].dropna().nunique()

    multilingual = contains_multilingual(df["Language coverage type"]) if "Language coverage type" in df.columns else np.nan
    multimodal = yes_count(df["Is the dataset multimodal?"]) if "Is the dataset multimodal?" in df.columns else np.nan
    synthetic = yes_count(df["Synthetic?"]) if "Synthetic?" in df.columns else np.nan
    mcq = yes_count(df["Is dataset in multiple choice format?"]) if "Is dataset in multiple choice format?" in df.columns else np.nan

    multilingual_missing_country = (
        df["Language coverage type"].astype(str).str.contains("multilingual", case=False, na=False)
        & df["country_set"].apply(lambda s: len(s) == 0)
    ).sum() if "Language coverage type" in df.columns else np.nan

    rows = {
        "Set": name,
        "Total rows / unique datasets": n,
        "Datasets with country info": country_present,
        "Datasets missing country info": country_missing,
        "% missing country info": round(100 * country_missing / n, 1) if n else np.nan,
        "Countries represented": len(countries),
        "Normalized task categories": task_count,
        "Unique languages represented": len(languages),
        "Multilingual datasets": multilingual,
        "Multilingual datasets missing country": multilingual_missing_country,
        "Multimodal datasets": multimodal,
        "Synthetic datasets": synthetic,
        "Multiple-choice datasets": mcq,
    }

    return rows

summary_rows = [
    summarize_dataset(auto, "WorldBench-Core / Auto"),
    summarize_dataset(gold_entries, "WorldBench-Gold entries"),
    summarize_dataset(gold_unique, "WorldBench-Gold unique datasets"),
]

summary_df = pd.DataFrame(summary_rows)
print(summary_df.T)

# ==================================================
# Optional: expanded coverage view
# Auto + Gold, deduplicated by paper URL when possible, then dataset name
# Use this ONLY for coverage statistics, not producer geography.
# ==================================================

auto_for_expand = auto[[
    "Dataset name", "dataset_key", "paper_key", "country_set", "language_set", "task_clean"
]].copy()
auto_for_expand["source"] = "auto"

gold_for_expand = gold_unique[[
    "Dataset name", "dataset_key", "paper_key", "country_set", "language_set", "task_clean"
]].copy()
gold_for_expand["source"] = "gold"

expanded = pd.concat([auto_for_expand, gold_for_expand], ignore_index=True)

# Prefer paper key when available, otherwise dataset key
expanded["dedupe_key"] = expanded["paper_key"].where(
    expanded["paper_key"].notna(),
    expanded["dataset_key"]
)

expanded_unique = (
    expanded
    .sort_values("source")  # auto first alphabetically; not important
    .groupby("dedupe_key", dropna=True)
    .agg({
        "Dataset name": "first",
        "country_set": lambda sets: set().union(*sets),
        "language_set": lambda sets: set().union(*sets),
        "task_clean": lambda x: x.dropna().iloc[0] if len(x.dropna()) else None,
        "source": lambda x: ";".join(sorted(set(x))),
    })
    .reset_index()
)

expanded_stats = {
    "Set": "WorldBench expanded coverage view",
    "Total unique datasets": len(expanded_unique),
    "Countries represented": len(set().union(*expanded_unique["country_set"])),
    "Datasets with country info": expanded_unique["country_set"].apply(lambda s: len(s) > 0).sum(),
    "Datasets missing country info": expanded_unique["country_set"].apply(lambda s: len(s) == 0).sum(),
    "Normalized task categories": expanded_unique["task_clean"].dropna().nunique(),
    "Unique languages represented": len(set().union(*expanded_unique["language_set"])),
}

expanded_summary_df = pd.DataFrame([expanded_stats])
print("\nExpanded coverage view:")
print(expanded_summary_df.T)

# ==================================================
# Save outputs
# ==================================================

summary_df.to_csv("worldbench_core_gold_stats_comparison.csv", index=False)
expanded_summary_df.to_csv("worldbench_expanded_coverage_stats.csv", index=False)

print("\nSaved:")
print("- worldbench_core_gold_stats_comparison.csv")
print("- worldbench_expanded_coverage_stats.csv")


# ### checks

# from collections import Counter

# lang_counter = Counter()

# for s in auto["language_set"]:
#     for l in s:
#         if l:
#             lang_counter[l] += 1

# # top 50 most common
# print(pd.DataFrame(lang_counter.most_common(50), columns=["Language", "Count"]))

# # languages that appear only once (often noise)
# rare = [k for k, v in lang_counter.items() if v == 1]

# print("Rare examples:")
# print(rare[:50])