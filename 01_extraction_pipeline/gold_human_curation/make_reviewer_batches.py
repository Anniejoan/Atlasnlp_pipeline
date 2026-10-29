# =============================================================================
# make_reviewer_batches.py — Prepare machine-assisted human review batches
#
# Paper section: §3.2 "Human Curation and Validation" / §3.4 "Validation"
# Role in pipeline: Step 5b — takes the outputs of validation_triage.py
#   (contradictions and human_review cases) and assembles review batches
#   for human annotators in the machine_batches/ folder.
#
# Run from compiled_ACL/:
#   python make_reviewer_batches.py
#
# Input:  contra_fixed_contradictions.csv, contra_fixedhuman_review.csv
# Output: auto_collaborator_pool.csv, machine_batches/reviewer_NN.csv
# =============================================================================
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------- INPUTS ----------
# Update these paths if needed
CONTRADICTIONS_CSV = "contra_fixed_contradictions.csv"
HUMAN_REVIEW_CSV = "contra_fixedhuman_review.csv"

OUTPUT_POOL = "auto_collaborator_pool.csv"
OUTPUT_DIR = "reviewer_batches"
N_REVIEWERS = 18

# ---------- FLAG MAPPINGS ----------
FLAG_TO_FIELDS = {
    "COUNTRY_ATTRIBUTION_CONTRADICTION": ["All Countries Covered", "Country Attribution Method"],
    "MULTI_COUNTRY_CONTRADICTION": ["All Countries Covered", "Multi Country"],
    "LANGUAGE_COUNT_CONTRADICTION": ["Languages covered", "Language coverage type"],
    "MODALITY_CONTRADICTION": ["Is the dataset multimodal?", "Short description"],
    "LINK_MISSING": ["Dataset Link"],
    "DATASET_LINK_NOT_URL": ["Dataset Link"],
    "LICENSE_MISSING": ["License"],
    "LICENSE_SUSPICIOUS": ["License"],
    "LICENSE_MAY_BE_LINK": ["License"],
    "COUNTRY_INFERABLE_FROM_LANGUAGE": ["Languages covered", "All Countries Covered"],
    "COUNTRY_MISSING_BUT_LANGUAGE_PRESENT": ["Languages covered", "All Countries Covered"],
    "COUNTRIES_MISSING": ["All Countries Covered", "Languages covered"],
    "MULTIPLE_DATASETS_IN_NAME": ["Dataset name", "Short description"],
    "ORIGIN_TYPE_MISSING": ["Origin Type"],
    "COUNTRY_ATTRIBUTION_MISSING": ["Country Attribution Method"],
}

PRIMARY_ISSUE_PRIORITY = [
    ("MULTIPLE_DATASETS_IN_NAME", "Possible benchmark/compilation paper"),
    ("COUNTRY_ATTRIBUTION_CONTRADICTION", "Country metadata contradiction"),
    ("MULTI_COUNTRY_CONTRADICTION", "Country metadata contradiction"),
    ("LANGUAGE_COUNT_CONTRADICTION", "Language coverage contradiction"),
    ("MODALITY_CONTRADICTION", "Modality contradiction"),
    ("DATASET_LINK_NOT_URL", "Dataset link issue"),
    ("LINK_MISSING", "Missing dataset link"),
    ("LICENSE_SUSPICIOUS", "License issue"),
    ("LICENSE_MAY_BE_LINK", "License issue"),
    ("LICENSE_MISSING", "Missing license"),
    ("COUNTRY_INFERABLE_FROM_LANGUAGE", "Country inference needed"),
    ("COUNTRY_MISSING_BUT_LANGUAGE_PRESENT", "Country inference needed"),
    ("COUNTRIES_MISSING", "Missing country metadata"),
    ("ORIGIN_TYPE_MISSING", "Origin type review needed"),
    ("COUNTRY_ATTRIBUTION_MISSING", "Country attribution review needed"),
]

DEFAULT_PRIMARY_ISSUE = "General metadata review"

# Reviewer-facing columns to include if present
PREFERRED_COLUMNS = [
    "Dataset name",
    "Paper Link",
    "Task Category",
    "Languages covered",
    "Language coverage type",
    "All Countries Covered",
    "Country Attribution Method",
    "Origin Type",
    "Dataset Link",
    "License",
    "Short description",
    "Validation Flags",
    "Primary Issue",
    "Fields To Check",
]

REVIEWER_COLUMNS_TO_ADD = [
    "Reviewer Corrected Dataset name",
    "Reviewer Corrected Countries",
    "Reviewer Corrected Country Attribution",
    "Reviewer Corrected Origin Type",
    "Reviewer Corrected Dataset Link",
    "Reviewer Corrected License",
    "Reviewer Notes",
    "Reviewer Decision",
]


def parse_flags(flag_string: str):
    if pd.isna(flag_string) or not str(flag_string).strip():
        return []
    return [f.strip() for f in str(flag_string).split(";") if f.strip()]


def derive_primary_issue(flags):
    flag_set = set(flags)
    for flag, label in PRIMARY_ISSUE_PRIORITY:
        if flag in flag_set:
            return label
    return DEFAULT_PRIMARY_ISSUE


def derive_fields_to_check(flags):
    fields = []
    seen = set()
    for flag in flags:
        for field in FLAG_TO_FIELDS.get(flag, []):
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return "; ".join(fields)


def prepare_pool(df, source_name):
    df = df.copy()

    if "Validation Flags" not in df.columns:
        df["Validation Flags"] = ""

    df["_parsed_flags"] = df["Validation Flags"].apply(parse_flags)
    df["Primary Issue"] = df["_parsed_flags"].apply(derive_primary_issue)
    df["Fields To Check"] = df["_parsed_flags"].apply(derive_fields_to_check)
    df["Review Source"] = source_name

    for col in REVIEWER_COLUMNS_TO_ADD:
        if col not in df.columns:
            df[col] = ""

    return df


def balanced_split(df, n_reviewers):
    # round-robin by issue type to balance difficult cases
    grouped = defaultdict(list)
    for idx, row in df.iterrows():
        grouped[row["Primary Issue"]].append(idx)

    reviewer_indices = [[] for _ in range(n_reviewers)]
    cursor = 0

    for issue, idxs in grouped.items():
        for idx in idxs:
            reviewer_indices[cursor % n_reviewers].append(idx)
            cursor += 1

    return reviewer_indices


def main():
    contradictions = pd.read_csv(CONTRADICTIONS_CSV)
    human_review = pd.read_csv(HUMAN_REVIEW_CSV)

    contradictions = prepare_pool(contradictions, "contradictions")
    human_review = prepare_pool(human_review, "human_review")

    pool = pd.concat([contradictions, human_review], ignore_index=True)

    # Keep useful columns first, then any remaining columns, then reviewer columns
    existing_preferred = [c for c in PREFERRED_COLUMNS if c in pool.columns]
    existing_reviewer = [c for c in REVIEWER_COLUMNS_TO_ADD if c in pool.columns]
    remaining = [c for c in pool.columns if c not in existing_preferred + existing_reviewer + ["_parsed_flags"]]

    final_columns = existing_preferred + remaining + existing_reviewer
    pool = pool[final_columns]

    pool.to_csv(OUTPUT_POOL, index=False)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    reviewer_indices = balanced_split(pool, N_REVIEWERS)

    for i, idxs in enumerate(reviewer_indices, start=1):
        reviewer_df = pool.loc[idxs].copy()
        out_path = Path(OUTPUT_DIR) / f"reviewer_{i:02d}.csv"
        reviewer_df.to_csv(out_path, index=False)

    print(f"Wrote {OUTPUT_POOL}")
    print(f"Wrote {OUTPUT_DIR}/reviewer_01.csv ... reviewer_{N_REVIEWERS:02d}.csv")
    print(f"Total rows in automated review pool: {len(pool)}")
    print("\nRows per reviewer:")
    for i, idxs in enumerate(reviewer_indices, start=1):
        print(f"reviewer_{i:02d}: {len(idxs)}")

    print("\nPrimary issue counts:")
    print(pool["Primary Issue"].value_counts().to_string())


if __name__ == "__main__":
    main()