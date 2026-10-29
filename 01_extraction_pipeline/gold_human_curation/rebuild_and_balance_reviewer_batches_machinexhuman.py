# =============================================================================
# rebuild_and_balance_reviewer_batches_machinexhuman.py — Re-balance review batches
#
# Paper section: §3.2 "Human Curation and Validation"
# Role in pipeline: Step 5d (supplementary) — rebuilds human review batches
#   to better balance machine-assisted vs. pure human reviewer assignments.
#   Run if the initial human_review_splitting.py produced imbalanced batches.
#
# Run from compiled_ACL/:
#   python rebuild_and_balance_reviewer_batches_machinexhuman.py
#
# Input:  human_compiled.csv
# Output: Updated machine_batches/ reviewer CSV files
# =============================================================================
import pandas as pd
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# Update these paths before running
HUMAN_FILE = Path("human_compiled.csv")
MACHINE_BATCH_DIR = Path("machine_batches")   # folder containing the 18 machine reviewer files
OUTPUT_DIR = Path("reviewer_batches")

NUM_REVIEWERS = 18
REVIEWERS = [f"reviewer_{i:02d}" for i in range(1, NUM_REVIEWERS + 1)]

# If True, keep machine row reviewer assignment from filename only as reference;
# final reviewer assignment is recomputed from the merged pool.
KEEP_ORIGINAL_MACHINE_REVIEWER_FIELD = True

# ============================================================
# IO HELPERS
# ============================================================

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")

def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output type: {path}")

# ============================================================
# NORMALIZATION
# ============================================================

def is_blank(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}

def norm_text(x) -> str:
    if is_blank(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_url(x) -> str:
    if is_blank(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.rstrip("/")
    s = re.sub(r"\s+", "", s)
    return s

def normalize_semicolon_list(x) -> str:
    if is_blank(x):
        return ""
    raw = str(x)
    parts = re.split(r"[;,|]", raw)
    parts = [norm_text(p) for p in parts if norm_text(p)]
    parts = sorted(set(parts))
    return "; ".join(parts)

def parse_to_sorted_set_string(values) -> str:
    parts = []
    for value in values:
        if is_blank(value):
            continue
        for piece in re.split(r"[;,|]", str(value)):
            piece = norm_text(piece)
            if piece:
                parts.append(piece)
    return "; ".join(sorted(set(parts)))

def yes_no_flag(x) -> str:
    s = norm_text(x)
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return ""

# ============================================================
# COUNTRY / LANGUAGE KEY BUILDING
# ============================================================

def normalize_country_set(row: pd.Series) -> str:
    """
    Build a country-set key with a priority order instead of concatenating everything.

    Priority:
      1. Reviewer Corrected Countries
      2. Normalized Countries
      3. Assigned country
      4. All Countries Covered

    Important:
    - If a more authoritative field exists, do not mix in weaker fields.
    - This prevents accidental collapse of rows that represent different country instances.
    """
    priority_columns = [
        "Reviewer Corrected Countries",
        "Normalized Countries",
        "Assigned country",
        "All Countries Covered (you can select multiple countries)",
    ]

    for col in priority_columns:
        if col in row.index and not is_blank(row.get(col)):
            return parse_to_sorted_set_string([row.get(col)])

    return ""

def normalize_language_set(row: pd.Series) -> str:
    """
    Build a language-set key with a priority order.

    Priority:
      1. Normalized Languages
      2. Languages covered
    """
    priority_columns = [
        "Normalized Languages",
        "Languages covered (separate using semicolon)",
    ]

    for col in priority_columns:
        if col in row.index and not is_blank(row.get(col)):
            return parse_to_sorted_set_string([row.get(col)])

    return ""

# ============================================================
# SCHEMA + ENRICHMENT
# ============================================================

STANDARD_REVIEW_COLUMNS = [
    "Validation Flags",
    "Primary Issue",
    "Fields To Check",
    "Normalized Countries",
    "Normalized Languages",
    "Primary Modality",
    "Candidate Country From Language",
    "Review Tier",
    "Needs Machine Followup",
    "Needs Human Review",
    "Multi Country",
    "Review Source",
    "Reviewer Corrected Dataset name",
    "Reviewer Corrected Countries",
    "Reviewer Corrected Country Attribution",
    "Reviewer Corrected Origin Type",
    "Reviewer Corrected Dataset Link",
    "Reviewer Corrected License",
    "Reviewer Notes",
    "Reviewer Decision",
]

BASE_COLUMNS_ORDER = [
    "Dataset name",
    "Task Category",
    "Language coverage type",
    "Country Attribution Method",
    "Origin Type",
    "License",
    "Assigned country",
    "Country resource level",
    "Dataset already in the matrix?",
    "Year created",
    "Languages covered (separate using semicolon)",
    "Dataset Link (URL)",
    "Paper Link (URL)",
    "Is the dataset multimodal?",
    "Is dataset in multiple choice format?",
    "Short description about dataset",
    "Synthetic?",
    "Synthetic notes (e.g. machine translated?)",
    "Multi Country?",
    "All Countries Covered (you can select multiple countries)",
    "Have you added your entry to the matrix? (if it wasn't there before)",
    "Notes ",
    "Validation Status (reviewers only)",
    "Verified By  (reviewers only)",
    "Supplementary Link (from import)",
]

APPROVED_ATTRIBUTION_MAP = {
    "language-based inclusion (german)": "Inferred (by language)",
    "unspecified/ambigous": "Unspecified/Ambiguous",
    "unspecified/ambiguous": "Unspecified/Ambiguous",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean import artifact column
    if "Unnamed: 24" in df.columns:
        if "Supplementary Link (from import)" not in df.columns:
            df = df.rename(columns={"Unnamed: 24": "Supplementary Link (from import)"})
        else:
            mask = df["Supplementary Link (from import)"].isna() | (
                df["Supplementary Link (from import)"].astype(str).str.strip() == ""
            )
            df.loc[mask, "Supplementary Link (from import)"] = df.loc[mask, "Unnamed: 24"]
            df = df.drop(columns=["Unnamed: 24"])

    for col in STANDARD_REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    if "Supplementary Link (from import)" not in df.columns:
        df["Supplementary Link (from import)"] = ""

    return df

def classify_human_row(row: pd.Series):
    reasons = []

    notes_val = row.get("Notes ", "")
    supp_val = row.get("Supplementary Link (from import)", "")
    origin_type = row.get("Origin Type", "")
    attr_method = row.get("Country Attribution Method", "")
    resource_level = row.get("Country resource level", "")

    machine_hint = False
    if not is_blank(notes_val) and "pilot-automated" in norm_text(notes_val):
        machine_hint = True
        reasons.append("machine_added_hint")
    if not is_blank(supp_val):
        machine_hint = True
        reasons.append("supplementary_link_from_import")

    if is_blank(resource_level):
        reasons.append("missing_resource_level")
    if norm_text(origin_type) == "not stated":
        reasons.append("origin_not_stated")
    if norm_text(attr_method) in {"not stated", "unspecified/ambigous", "unspecified/ambiguous"}:
        reasons.append("country_attribution_unspecified")
    if norm_text(attr_method) == "language-based inclusion (german)":
        reasons.append("nonstandard_attribution_label")

    row_source = "human_compiled_machine_added" if machine_hint else "human_compiled"
    needs_attention = "Yes" if reasons else "No"

    return row_source, needs_attention, "; ".join(sorted(set(reasons)))

def enrich_human_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = standardize_columns(df)

    row_sources = []
    attn_flags = []
    attn_reasons = []

    for _, row in df.iterrows():
        row_source, needs_attention, reason = classify_human_row(row)
        row_sources.append(row_source)
        attn_flags.append(needs_attention)
        attn_reasons.append(reason)

    df["Original Pool"] = "human_file"
    df["Row Source"] = row_sources
    df["Needs Extra Attention"] = attn_flags
    df["Attention Reason"] = attn_reasons
    df["Original Reviewer"] = ""

    if "Country Attribution Method" in df.columns:
        df["Country Attribution Method"] = df["Country Attribution Method"].apply(
            lambda x: APPROVED_ATTRIBUTION_MAP.get(norm_text(x), x)
        )

    if "Dataset Link (URL)" in df.columns and "Supplementary Link (from import)" in df.columns:
        mask = df["Dataset Link (URL)"].isna() | (df["Dataset Link (URL)"].astype(str).str.strip() == "")
        df.loc[mask, "Dataset Link (URL)"] = df.loc[mask, "Supplementary Link (from import)"]

    return df

def infer_machine_row_source(row: pd.Series) -> str:
    review_source = norm_text(row.get("Review Source", ""))
    review_tier = norm_text(row.get("Review Tier", ""))

    if "human_review" in review_source or "human review" in review_source:
        return "machine_pipeline_human_review"
    if "contradiction" in review_source:
        return "machine_pipeline_contradiction"
    if "followup" in review_source:
        return "machine_pipeline_followup"
    if "human" in review_tier:
        return "machine_pipeline_human_review"
    if "contradiction" in review_tier:
        return "machine_pipeline_contradiction"
    return "machine_pipeline"

def compute_attention_for_machine(row: pd.Series):
    reasons = []

    if is_blank(row.get("Country resource level", "")):
        reasons.append("missing_resource_level")
    if norm_text(row.get("Origin Type", "")) == "not stated":
        reasons.append("origin_not_stated")
    if norm_text(row.get("Country Attribution Method", "")) in {"not stated", "unspecified/ambigous", "unspecified/ambiguous"}:
        reasons.append("country_attribution_unspecified")
    if not is_blank(row.get("Validation Flags", "")):
        reasons.append("validation_flags_present")
    if not is_blank(row.get("Primary Issue", "")):
        reasons.append("primary_issue_present")

    needs_attention = "Yes" if reasons else "No"
    return needs_attention, "; ".join(sorted(set(reasons)))

def enrich_machine_df(df: pd.DataFrame, original_reviewer: str) -> pd.DataFrame:
    df = df.copy()
    df = standardize_columns(df)

    row_sources = []
    attn_flags = []
    attn_reasons = []

    for _, row in df.iterrows():
        row_sources.append(infer_machine_row_source(row))
        attn, reason = compute_attention_for_machine(row)
        attn_flags.append(attn)
        attn_reasons.append(reason)

    df["Original Pool"] = "machine_batches"
    df["Row Source"] = row_sources
    df["Needs Extra Attention"] = attn_flags
    df["Attention Reason"] = attn_reasons
    df["Original Reviewer"] = original_reviewer if KEEP_ORIGINAL_MACHINE_REVIEWER_FIELD else ""

    if "Country Attribution Method" in df.columns:
        df["Country Attribution Method"] = df["Country Attribution Method"].apply(
            lambda x: APPROVED_ATTRIBUTION_MAP.get(norm_text(x), x)
        )

    return df

# ============================================================
# DEDUP + COMPLEXITY
# ============================================================

def row_nonempty_count(row: pd.Series, exclude_prefix="__") -> int:
    count = 0
    for col, val in row.items():
        if str(col).startswith(exclude_prefix):
            continue
        if not is_blank(val):
            count += 1
    return count

def canonical_dataset_identity(row: pd.Series) -> str:
    """
    Identify the dataset itself, independent of country attribution.

    Priority:
      1. Reviewer Corrected Dataset Link
      2. Dataset URL
      3. Paper URL + reviewer corrected dataset name / dataset name
      4. reviewer corrected dataset name / dataset name
      5. paper URL
    """
    corrected_url = norm_url(row.get("Reviewer Corrected Dataset Link", ""))
    dataset_url = norm_url(row.get("Dataset Link (URL)", ""))
    paper_key = norm_url(row.get("Paper Link (URL)", ""))

    corrected_name = norm_text(row.get("Reviewer Corrected Dataset name", ""))
    dataset_name = norm_text(row.get("Dataset name", ""))
    chosen_name = corrected_name if corrected_name else dataset_name

    if corrected_url:
        return f"url::{corrected_url}"
    if dataset_url:
        return f"url::{dataset_url}"
    if paper_key and chosen_name:
        return f"paper+dataset::{paper_key} || {chosen_name}"
    if chosen_name:
        return f"dataset::{chosen_name}"
    if paper_key:
        return f"paper::{paper_key}"
    return "unknown_dataset"

def add_matching_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["__dataset_identity__"] = df.apply(canonical_dataset_identity, axis=1)
    df["__dataset_name_norm__"] = df.apply(
        lambda row: norm_text(row.get("Reviewer Corrected Dataset name", "")) or norm_text(row.get("Dataset name", "")),
        axis=1
    )
    df["__task_norm__"] = df.get("Task Category", "").apply(norm_text)
    df["__country_norm__"] = df.apply(normalize_country_set, axis=1)
    df["__language_norm__"] = df.apply(normalize_language_set, axis=1)

    def normalized_multi_country(row):
        corrected_countries = normalize_country_set(row)
        raw_multi = yes_no_flag(row.get("Multi Country?", ""))
        if corrected_countries and ";" in corrected_countries:
            return "yes"
        return raw_multi

    df["__multi_country_norm__"] = df.apply(normalized_multi_country, axis=1)
    df["__origin_type_norm__"] = df.apply(
        lambda row: norm_text(row.get("Reviewer Corrected Origin Type", "")) or norm_text(row.get("Origin Type", "")),
        axis=1
    )
    df["__attrib_method_norm__"] = df.apply(
        lambda row: norm_text(row.get("Reviewer Corrected Country Attribution", "")) or norm_text(row.get("Country Attribution Method", "")),
        axis=1
    )

    # Dedup definition:
    # duplicate only if same dataset identity AND same country set AND same task AND same language set AND same multi-country status
    df["__dedupe_key__"] = (
        df["__dataset_identity__"].fillna("") + " || " +
        "country=" + df["__country_norm__"].fillna("") + " || " +
        "task=" + df["__task_norm__"].fillna("") + " || " +
        "lang=" + df["__language_norm__"].fillna("") + " || " +
        "multi=" + df["__multi_country_norm__"].fillna("")
    )

    # looser grouping key for debug / audit only
    df["__dataset_group_key__"] = df["__dataset_identity__"]

    return df

def source_priority(row_source: str) -> int:
    s = norm_text(row_source)
    if "contradiction" in s:
        return 4
    if "human_review" in s:
        return 3
    if "machine_pipeline" in s:
        return 2
    if "human_compiled_machine_added" in s:
        return 1
    return 0

def merge_text_values(values) -> str:
    pieces = []
    for v in values:
        if is_blank(v):
            continue
        pieces.extend([p.strip() for p in str(v).split(";") if p.strip()])
    return "; ".join(sorted(set(pieces)))

def merge_duplicate_group(group: pd.DataFrame) -> pd.Series:
    """
    Merge rows that truly represent the same dataset-country instance.
    Keep the richest row as base, then fill blanks from others.
    Preserve provenance and union selected metadata.
    """
    group = group.copy()
    group["__info_score__"] = group.apply(row_nonempty_count, axis=1)
    group["__source_priority__"] = group["Row Source"].apply(source_priority)

    group = group.sort_values(
        by=["__source_priority__", "__info_score__"],
        ascending=[False, False],
        kind="stable"
    )

    base = group.iloc[0].copy()

    # Fill blanks in base from other rows
    for _, row in group.iloc[1:].iterrows():
        for col in group.columns:
            if str(col).startswith("__"):
                continue
            if is_blank(base.get(col)) and not is_blank(row.get(col)):
                base[col] = row[col]

    # Union provenance-like fields
    union_text_columns = [
        "Original Pool",
        "Row Source",
        "Attention Reason",
        "Original Reviewer",
        "Validation Flags",
        "Primary Issue",
        "Fields To Check",
        "Review Source",
        "Review Tier",
        "Reviewer Notes",
    ]

    for col in union_text_columns:
        if col in group.columns:
            base[col] = merge_text_values(group[col].tolist())

    if "Needs Extra Attention" in group.columns:
        any_yes = any(str(v).strip().lower() == "yes" for v in group["Needs Extra Attention"].tolist())
        base["Needs Extra Attention"] = "Yes" if any_yes else "No"

    # Prefer reviewer-corrected fields if any row has them
    preferred_carryover_columns = [
        "Reviewer Corrected Dataset name",
        "Reviewer Corrected Countries",
        "Reviewer Corrected Country Attribution",
        "Reviewer Corrected Origin Type",
        "Reviewer Corrected Dataset Link",
        "Reviewer Corrected License",
        "Normalized Countries",
        "Normalized Languages",
    ]

    for col in preferred_carryover_columns:
        if col in group.columns:
            nonblank = [v for v in group[col].tolist() if not is_blank(v)]
            if nonblank:
                if col in {"Normalized Countries", "Reviewer Corrected Countries"}:
                    base[col] = parse_to_sorted_set_string(nonblank)
                elif col in {"Normalized Languages"}:
                    base[col] = parse_to_sorted_set_string(nonblank)
                else:
                    base[col] = nonblank[0]

    base["Merged Duplicate Count"] = len(group)

    return base.drop(labels=[c for c in base.index if str(c).startswith("__")], errors="ignore")

def global_deduplicate(df: pd.DataFrame):
    df = add_matching_keys(df)
    before = len(df)

    deduped_rows = []
    duplicate_log_rows = []

    for dedupe_key, group in df.groupby("__dedupe_key__", dropna=False, sort=False):
        if len(group) > 1:
            duplicate_log_rows.append({
                "Dedupe Key": dedupe_key,
                "Rows Merged": len(group),
                "Dataset Identity": group["__dataset_identity__"].iloc[0] if "__dataset_identity__" in group.columns else "",
                "Dataset Name": group["Dataset name"].iloc[0] if "Dataset name" in group.columns else "",
                "Country Set": group["__country_norm__"].iloc[0] if "__country_norm__" in group.columns else "",
                "Task": group["__task_norm__"].iloc[0] if "__task_norm__" in group.columns else "",
                "Languages": group["__language_norm__"].iloc[0] if "__language_norm__" in group.columns else "",
                "Multi Country": group["__multi_country_norm__"].iloc[0] if "__multi_country_norm__" in group.columns else "",
            })

        deduped_rows.append(merge_duplicate_group(group))

    deduped = pd.DataFrame(deduped_rows).reset_index(drop=True)
    duplicate_log_df = pd.DataFrame(duplicate_log_rows)
    after = len(deduped)

    return deduped, duplicate_log_df, before, after

def compute_complexity(row: pd.Series) -> int:
    score = 1
    reasons = norm_text(row.get("Attention Reason", ""))

    if "missing_resource_level" in reasons:
        score += 2
    if "origin_not_stated" in reasons:
        score += 2
    if "country_attribution_unspecified" in reasons:
        score += 2
    if "nonstandard_attribution_label" in reasons:
        score += 1
    if "validation_flags_present" in reasons or "primary_issue_present" in reasons:
        score += 1
    if "machine_added_hint" in reasons or "supplementary_link_from_import" in reasons:
        score += 1

    country_norm = normalize_country_set(row)
    lang_norm = normalize_language_set(row)

    if not is_blank(country_norm) and ";" in country_norm:
        score += 1
    if not is_blank(lang_norm) and ";" in lang_norm:
        score += 1
    if yes_no_flag(row.get("Multi Country?", "")) == "yes":
        score += 1
    if is_blank(row.get("Dataset Link (URL)", "")) and is_blank(row.get("Reviewer Corrected Dataset Link", "")):
        score += 1

    return score

# ============================================================
# BALANCED SPLIT
# ============================================================

def build_stratum(row: pd.Series) -> str:
    source = row.get("Row Source", "")
    attention = row.get("Needs Extra Attention", "")
    multi = yes_no_flag(row.get("Multi Country?", ""))

    # if normalized country key itself shows multiple countries, also treat as multi
    country_key = normalize_country_set(row)
    if country_key and ";" in country_key:
        multi = "yes"

    return f"{source}__{attention}__multi_{multi}"

def assign_balanced_batches(df: pd.DataFrame, reviewers):
    df = df.copy()
    df["Complexity Score"] = df.apply(compute_complexity, axis=1)
    df["__stratum__"] = df.apply(build_stratum, axis=1)

    assignments = {r: [] for r in reviewers}
    reviewer_load = {r: 0 for r in reviewers}
    reviewer_rows = {r: 0 for r in reviewers}

    for _, group in df.groupby("__stratum__", sort=False):
        group = group.sample(frac=1, random_state=42).copy()
        group = group.sort_values(by="Complexity Score", ascending=False, kind="stable")

        for _, row in group.iterrows():
            chosen = sorted(
                reviewers,
                key=lambda r: (reviewer_load[r], reviewer_rows[r], r)
            )[0]
            assignments[chosen].append(row.drop(labels=["__stratum__"]))
            reviewer_load[chosen] += int(row["Complexity Score"])
            reviewer_rows[chosen] += 1

    out = {}
    for reviewer in reviewers:
        if assignments[reviewer]:
            out[reviewer] = pd.DataFrame(assignments[reviewer]).reset_index(drop=True)
        else:
            out[reviewer] = pd.DataFrame(columns=[c for c in df.columns if c != "__stratum__"])

    return out, reviewer_load, reviewer_rows

# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    human_df = read_table(HUMAN_FILE)
    human_df = enrich_human_df(human_df)

    machine_files = sorted([
        p for p in MACHINE_BATCH_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ])

    if len(machine_files) != NUM_REVIEWERS:
        raise ValueError(
            f"Expected {NUM_REVIEWERS} machine reviewer files in {MACHINE_BATCH_DIR}, found {len(machine_files)}."
        )

    machine_parts = []
    for idx, file_path in enumerate(machine_files, start=1):
        reviewer_name = f"reviewer_{idx:02d}"
        mdf = read_table(file_path)
        mdf = enrich_machine_df(mdf, original_reviewer=reviewer_name)
        machine_parts.append(mdf)

    machine_df = pd.concat(machine_parts, ignore_index=True)

    write_table(human_df, OUTPUT_DIR / "01_human_tagged.xlsx")
    write_table(machine_df, OUTPUT_DIR / "02_machine_collapsed_tagged.xlsx")

    union_cols = sorted(set(human_df.columns).union(set(machine_df.columns)))
    human_df = human_df.reindex(columns=union_cols)
    machine_df = machine_df.reindex(columns=union_cols)

    combined_raw = pd.concat([machine_df, human_df], ignore_index=True)
    write_table(combined_raw, OUTPUT_DIR / "03_combined_raw_before_dedup.xlsx")

    deduped_df, duplicate_log_df, before, after = global_deduplicate(combined_raw)
    write_table(deduped_df, OUTPUT_DIR / "04_combined_deduplicated.xlsx")
    write_table(duplicate_log_df, OUTPUT_DIR / "04b_duplicate_merge_log.xlsx")

    splits, reviewer_load, reviewer_rows = assign_balanced_batches(deduped_df, REVIEWERS)

    summary_rows = []
    for reviewer in REVIEWERS:
        batch = splits[reviewer].copy()

        batch["Reviewer"] = reviewer
        batch["Reviewer Row ID"] = range(1, len(batch) + 1)

        preferred_front = [
            "Reviewer Row ID",
            "Reviewer",
            "Original Reviewer",
            "Original Pool",
            "Row Source",
            "Needs Extra Attention",
            "Attention Reason",
            "Complexity Score",
            "Merged Duplicate Count",
        ]

        ordered_cols = (
            preferred_front
            + [c for c in BASE_COLUMNS_ORDER if c in batch.columns]
            + [c for c in STANDARD_REVIEW_COLUMNS if c in batch.columns]
            + [c for c in batch.columns if c not in set(preferred_front + BASE_COLUMNS_ORDER + STANDARD_REVIEW_COLUMNS)]
        )

        seen = set()
        ordered_cols_unique = []
        for c in ordered_cols:
            if c in batch.columns and c not in seen:
                ordered_cols_unique.append(c)
                seen.add(c)

        batch = batch[ordered_cols_unique]

        write_table(batch, OUTPUT_DIR / "reviewer_batches" / f"{reviewer}_balanced_merged_batch.xlsx")

        summary = {
            "Reviewer": reviewer,
            "Rows": len(batch),
            "Total Complexity": reviewer_load[reviewer],
            "Human Compiled": int(batch["Row Source"].astype(str).str.contains("human_compiled", case=False, na=False).sum()),
            "Machine Pipeline": int(batch["Row Source"].astype(str).str.contains("machine_pipeline", case=False, na=False).sum()),
            "Needs Extra Attention": int((batch["Needs Extra Attention"] == "Yes").sum()),
            "Multi Country Rows": int(
                batch.apply(lambda r: 1 if normalize_country_set(r) and ";" in normalize_country_set(r) else 0, axis=1).sum()
            ),
        }
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    write_table(summary_df, OUTPUT_DIR / "05_reviewer_batch_summary.xlsx")

    overall_summary = pd.DataFrame([
        {"Metric": "Machine raw rows", "Value": len(machine_df)},
        {"Metric": "Human raw rows", "Value": len(human_df)},
        {"Metric": "Combined raw rows", "Value": before},
        {"Metric": "Combined deduplicated rows", "Value": after},
        {"Metric": "Duplicates removed globally", "Value": before - after},
    ])
    write_table(overall_summary, OUTPUT_DIR / "06_overall_summary.xlsx")

    print("Done.")
    print(overall_summary)
    print(summary_df)

if __name__ == "__main__":
    main()