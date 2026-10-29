# =============================================================================
# validation_0.py — Compute automated extraction validation metrics (Table 1)
#
# Paper section: §3.4 "Validation of Automated Extraction" → Table 1
# Produces: Table 1 — validation results on 460 aligned Gold/Auto dataset pairs
#   Metrics reported: main-country recovery (87.0%), at-least-one-correct (91.5%),
#   exact match (65.2%), task label exact match (89.1%).
#
# Run from compiled_ACL/task_matrix_construction/worldbench_matrix_outputs/:
#   python validation_0.py
#
# Input:
#   unified_worldbench_views/language_cleanup_v2/human_worldbench.csv     — WorldBench-Gold
#   unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv  — WorldBench-Core
#
# Output:
#   human_auto_intersection.csv  — the 460-entry aligned evaluation set
#   Metrics printed to stdout
# =============================================================================
import pandas as pd
import re
import ast

human = pd.read_csv('/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/human_worldbench.csv')
auto = pd.read_csv("/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv")

# normalize urls (important)
human["paper_url"] = human["Paper Link (URL)"].str.strip().str.lower()
auto["paper_url"] = auto["Paper Link (URL)"].str.strip().str.lower()

# inner join → intersection
merged = human.merge(auto, on="paper_url", suffixes=("_human", "_auto"))

merged.to_csv('human_auto_intersection.csv')

# print("Overlap size:", len(merged))
# print("Human size", len(human))
# print("Auto size", len(auto))


print("Human size:", len(human))
print("Auto size:", len(auto))
print("Overlap size:", len(merged))
print("Percent of human matched:", round(len(merged) / len(human) * 100, 2))
print("Percent of auto matched:", round(len(merged) / len(auto) * 100, 2))

# ----------------------------
# 4. Helpers
# ----------------------------
def clean_country_name(x):
    if pd.isna(x):
        return None

    x = str(x).strip()
    if not x or x.lower() in {"nan", "none", "not stated", "unknown"}:
        return None

    aliases = {
        "usa": "United States",
        "u.s.": "United States",
        "u.s.a.": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "uae": "United Arab Emirates",
        "viet nam": "Vietnam",
    }

    return aliases.get(x.lower(), x)


def to_country_set(x):
    """
    Handles:
    - semicolon strings
    - comma strings
    - Python-list-like strings: ['Kenya', 'Uganda']
    - NaN / Not stated
    """
    if pd.isna(x):
        return set()

    if isinstance(x, list):
        items = x
    else:
        x = str(x).strip()

        if not x or x.lower() in {"nan", "none", "not stated", "unknown"}:
            return set()

        # Handle list-like strings
        if x.startswith("[") and x.endswith("]"):
            try:
                items = ast.literal_eval(x)
            except Exception:
                items = re.split(r";|,", x)
        else:
            items = re.split(r";|,", x)

    cleaned = set()
    for item in items:
        c = clean_country_name(item)
        if c:
            cleaned.add(c)

    return cleaned


def choose_human_coverage(row):
    coverage_col = "All Countries Covered (you can select multiple countries)_human"
    assigned_col = "Assigned country_human"

    coverage = to_country_set(row.get(coverage_col))
    assigned = to_country_set(row.get(assigned_col))

    return coverage.union(assigned)


# ----------------------------
# 5. Build country sets
# ----------------------------
merged["human_primary_set"] = merged["Assigned country_human"].apply(to_country_set)
merged["human_coverage_set"] = merged.apply(choose_human_coverage, axis=1)

# Auto country field: use best valid list if available, otherwise Normalized Countries
if "best_country_list_valid" in merged.columns:
    merged["auto_country_set"] = merged["best_country_list_valid"].apply(to_country_set)
else:
    merged["auto_country_set"] = merged["Normalized Countries"].apply(to_country_set)

# ----------------------------
# 6. Country validation metrics
# ----------------------------
merged["auto_has_country"] = merged["auto_country_set"].apply(lambda s: len(s) > 0)

merged["primary_country_overlap"] = merged.apply(
    lambda row: len(row["human_primary_set"] & row["auto_country_set"]) > 0,
    axis=1
)

merged["coverage_country_overlap"] = merged.apply(
    lambda row: len(row["human_coverage_set"] & row["auto_country_set"]) > 0,
    axis=1
)

merged["coverage_exact_match"] = merged.apply(
    lambda row: row["human_coverage_set"] == row["auto_country_set"],
    axis=1
)

merged["auto_expands_human_coverage"] = merged.apply(
    lambda row: (
        len(row["auto_country_set"]) > len(row["human_coverage_set"])
        and len(row["human_coverage_set"] & row["auto_country_set"]) > 0
    ),
    axis=1
)

merged["auto_missing_country"] = merged["auto_country_set"].apply(lambda s: len(s) == 0)

# ----------------------------
# 7. Task validation metrics
# ----------------------------
def normalize_task(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = x.replace(" and ", " & ")
    return x

merged["task_human_norm"] = merged["Task Category_human"].apply(normalize_task)

if "normalized_task_category" in merged.columns:
    merged["task_auto_norm"] = merged["normalized_task_category"].apply(normalize_task)
else:
    merged["task_auto_norm"] = merged["Task Category_auto"].apply(normalize_task)

merged["task_exact_match"] = merged["task_human_norm"] == merged["task_auto_norm"]

# ----------------------------
# 8. Summary table
# ----------------------------
summary = {
    "human_size": len(human),
    "auto_size": len(auto),
    "overlap_size": len(merged),
    "percent_human_matched": len(merged) / len(human) * 100,
    "percent_auto_matched": len(merged) / len(auto) * 100,
    "auto_has_country_rate": merged["auto_has_country"].mean() * 100,
    "primary_country_overlap_rate": merged["primary_country_overlap"].mean() * 100,
    "coverage_country_overlap_rate": merged["coverage_country_overlap"].mean() * 100,
    "coverage_exact_match_rate": merged["coverage_exact_match"].mean() * 100,
    "auto_expands_human_coverage_rate": merged["auto_expands_human_coverage"].mean() * 100,
    "auto_missing_country_rate": merged["auto_missing_country"].mean() * 100,
    "task_exact_match_rate": merged["task_exact_match"].mean() * 100,
}

summary_df = pd.DataFrame([summary]).round(2)
print(summary_df.T)

# ----------------------------
# 9. Error / disagreement inspection files
# ----------------------------
country_disagreements = merged[
    ~merged["coverage_country_overlap"]
][[
    "Dataset name_human",
    "Dataset name_auto",
    "Assigned country_human",
    "All Countries Covered (you can select multiple countries)_human",
    "Normalized Countries",
    "best_country_list_valid",
    "Task Category_human",
    "normalized_task_category",
    "paper_url_key",
]]

task_disagreements = merged[
    ~merged["task_exact_match"]
][[
    "Dataset name_human",
    "Dataset name_auto",
    "Task Category_human",
    "normalized_task_category",
    "Assigned country_human",
    "All Countries Covered (you can select multiple countries)_human",
    "Normalized Countries",
    "paper_url_key",
]]

expansion_cases = merged[
    merged["auto_expands_human_coverage"]
][[
    "Dataset name_human",
    "Dataset name_auto",
    "Assigned country_human",
    "All Countries Covered (you can select multiple countries)_human",
    "Normalized Countries",
    "best_country_list_valid",
    "paper_url_key",
]]

# ----------------------------
# 10. Attribution method analysis
# ----------------------------

# Normalize attribution method (auto side)
def normalize_method(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).lower()

    if "explicit" in x:
        return "explicit"
    elif "language" in x:
        return "language_inferred"
    elif "inferred" in x:
        return "language_inferred"
    elif "not stated" in x or "unknown" in x:
        return "unknown"
    else:
        return "other"
merged["attribution_method_group"] = merged["country_attribution_method"].apply(normalize_method)

# Compute grouped metrics
attribution_analysis = merged.groupby("attribution_method_group")[[
    "coverage_country_overlap",
    "coverage_exact_match",
    "auto_expands_human_coverage",
    "auto_missing_country"
]].mean().round(3) * 100

print(merged["attribution_method_group"].value_counts())
print("\nAttribution Method Breakdown (%):")
print(attribution_analysis)

# Save it
attribution_analysis.to_csv("attribution_method_analysis.csv")

country_disagreements.to_csv("country_disagreements_for_review.csv", index=False)
task_disagreements.to_csv("task_disagreements_for_review.csv", index=False)
expansion_cases.to_csv("auto_expands_human_coverage_cases.csv", index=False)
summary_df.to_csv("validation_summary_metrics.csv", index=False)

print("Saved review files.")