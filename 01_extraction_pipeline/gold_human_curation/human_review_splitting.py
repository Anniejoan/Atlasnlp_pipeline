# =============================================================================
# human_review_splitting.py — Assign entries to human reviewers for WorldBench-Gold
#
# Paper section: §3.2 "Human Curation and Validation"
# Role in pipeline: Step 6 — splits human_compiled.csv (raw Gold contributions)
#   into per-reviewer Excel files for the cross-validation stage.
#   16 pure human reviewers + 2 machine-assisted reviewers (reviewer_17/18).
#
# Run from compiled_ACL/:
#   python human_review_splitting.py
#
# Input:  human_compiled.csv (raw entries submitted by contributors)
# Output: human_review/reviewer_01.xlsx ... reviewer_18.xlsx
#         human_review/split_summary.xlsx
#
# After reviewers complete annotation, the filled Excel files are collected back
# and merged into the master Gold dataset.
# =============================================================================
import pandas as pd
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("human_compiled.csv")
OUTPUT_DIR = Path("human_review")

PURE_HUMAN_REVIEWERS = [f"reviewer_{i:02d}" for i in range(1, 17)]   # reviewer_01 ... reviewer_16
MACHINE_ASSISTED_REVIEWERS = ["reviewer_17", "reviewer_18"]

RANDOM_SEED = 42

# ============================================================
# HELPERS
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

def is_not_stated(x) -> bool:
    return norm_text(x) == "not stated"

def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path}")

# ============================================================
# MACHINE-ASSISTED DETECTION
# ============================================================

def detect_machine_assisted_row(row: pd.Series) -> bool:
    """
    Classify as machine-assisted if:

    Strong hints:
    - Notes contains 'pilot-automated'
    - OR Unnamed: 24 is non-empty
    - OR Supplementary Link (from import) is non-empty

    Weak hints:
    - Origin Type == 'Not stated'
    - Country Attribution Method == 'Not stated'
    - License == 'Not stated'
    - Country resource level == 'Not stated'

    Rule:
    - machine-assisted if any strong hint is present
    - OR if at least 2 weak hints are present
    """
    notes_val = row.get("Notes ", "")
    unnamed_24 = row.get("Unnamed: 24", "")
    supp_link = row.get("Supplementary Link (from import)", "")

    # strong hints
    if "pilot-automated" in norm_text(notes_val):
        return True
    if not is_blank(unnamed_24):
        return True
    if not is_blank(supp_link):
        return True

    weak_score = 0

    if is_not_stated(row.get("Origin Type", "")):
        weak_score += 1

    if is_not_stated(row.get("All Countries Covered (you can select multiple countries)", "")):
        weak_score += 1

    if is_not_stated(row.get("Country Attribution Method", "")):
        weak_score += 1

    if is_not_stated(row.get("License", "")):
        weak_score += 1

    if is_not_stated(row.get("Country resource level", "")):
        weak_score += 1

    return weak_score > 0

def print_detection_breakdown(df: pd.DataFrame) -> None:
    """
    Print useful counts so you can sanity-check the split.
    """
    notes_hits = (
        df["Notes "].fillna("").astype(str).str.lower().str.contains("pilot-automated").sum()
        if "Notes " in df.columns else 0
    )

    unnamed_hits = (
        (~df["Unnamed: 24"].apply(is_blank)).sum()
        if "Unnamed: 24" in df.columns else 0
    )

    supp_hits = (
        (~df["Supplementary Link (from import)"].apply(is_blank)).sum()
        if "Supplementary Link (from import)" in df.columns else 0
    )

    origin_not_stated = (
        df["Origin Type"].fillna("").astype(str).str.strip().str.lower().eq("not stated").sum()
        if "Origin Type" in df.columns else 0
    )

    attr_not_stated = (
        df["Country Attribution Method"].fillna("").astype(str).str.strip().str.lower().eq("not stated").sum()
        if "Country Attribution Method" in df.columns else 0
    )

    license_not_stated = (
        df["License"].fillna("").astype(str).str.strip().str.lower().eq("not stated").sum()
        if "License" in df.columns else 0
    )

    resource_not_stated = (
        df["Country resource level"].fillna("").astype(str).str.strip().str.lower().eq("not stated").sum()
        if "Country resource level" in df.columns else 0
    )
    

    countries_not_stated = (
        df["All Countries Covered (you can select multiple countries)"].fillna("").astype(str).str.strip().str.lower().eq("not stated").sum()
        if "All Countries Covered (you can select multiple countries)" in df.columns else 0
    )

    print("\nDetection breakdown:")
    print(f"Notes contains 'pilot-automated': {notes_hits}")
    print(f"Unnamed: 24 non-empty: {unnamed_hits}")
    print(f"Supplementary Link (from import) non-empty: {supp_hits}")
    print(f"Origin Type = 'Not stated': {origin_not_stated}")
    print(f"Country Attribution Method = 'Not stated': {attr_not_stated}")
    print(f"License = 'Not stated': {license_not_stated}")
    print(f"Country resource level = 'Not stated': {resource_not_stated}")
    print(f"All Countries = 'Not stated': {countries_not_stated}")

# ============================================================
# COLUMN SELECTION
# ============================================================

def build_reviewer_view(df: pd.DataFrame, row_source_label: str) -> pd.DataFrame:
    """
    Keep only the columns reviewers need.
    """
    df = df.copy()

    # If import artifact exists, rename it consistently
    if "Unnamed: 24" in df.columns and "Supplementary Link (from import)" not in df.columns:
        df = df.rename(columns={"Unnamed: 24": "Supplementary Link (from import)"})

    wanted_columns = [
        "Assigned country",
        "Country resource level",
        "Dataset name",
        "Task Category",
        "Country Attribution Method",
        "Year created",
        "Languages covered (separate using semicolon)",
        "Language coverage type",
        "Dataset Link (URL)",
        "Paper Link (URL)",
        "Is the dataset multimodal?",
        "Is dataset in multiple choice format?",
        "Short description about dataset",
        "License",
        "Origin Type",
        "Synthetic?",
        "Synthetic notes (e.g. machine translated?)",
        "Multi Country?",
        "All Countries Covered (you can select multiple countries)",
        "Notes" ,
        "Validation Status (reviewers only)",
        "Verified By  (reviewers only)",
        "Supplementary Link (from import)",
    ]

    existing = [c for c in wanted_columns if c in df.columns]
    out = df[existing].copy()

    out.insert(0, "Row Source", row_source_label)
    out.insert(1, "Validation Priority", "Light" if row_source_label == "human_compiled" else "Deeper")

    # Minimal reviewer-facing columns for now
    out["Validator Notes"] = ""
    out["Changes Made"] = ""

    return out

# ============================================================
# SPLITTING
# ============================================================

def round_robin_split(df: pd.DataFrame, reviewers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Shuffle first, then assign rows in round-robin order.
    """
    if len(df) == 0:
        return {r: pd.DataFrame(columns=df.columns) for r in reviewers}

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    buckets = {r: [] for r in reviewers}

    for i, (_, row) in enumerate(df.iterrows()):
        reviewer = reviewers[i % len(reviewers)]
        buckets[reviewer].append(row)

    out = {}
    for reviewer, rows in buckets.items():
        if rows:
            reviewer_df = pd.DataFrame(rows).reset_index(drop=True)
        else:
            reviewer_df = pd.DataFrame(columns=df.columns)

        reviewer_df.insert(0, "Reviewer Row ID", range(1, len(reviewer_df) + 1))
        reviewer_df.insert(1, "Reviewer", reviewer)
        out[reviewer] = reviewer_df

    return out

# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read input
    df = pd.read_csv(INPUT_FILE)

    print_detection_breakdown(df)

    # Tag rows
    machine_mask = df.apply(detect_machine_assisted_row, axis=1)

    pure_human_raw = df[~machine_mask].copy().reset_index(drop=True)
    machine_assisted_raw = df[machine_mask].copy().reset_index(drop=True)

    print(f"\nTotal rows in input: {len(df)}")
    print(f"Pure human rows: {len(pure_human_raw)}")
    print(f"Machine-assisted human rows: {len(machine_assisted_raw)}")

    # Build reviewer-facing views
    pure_human = build_reviewer_view(pure_human_raw, "human_compiled")
    machine_assisted = build_reviewer_view(machine_assisted_raw, "human_compiled_machine_assisted")

    # Save the two master subsets
    write_table(pure_human, OUTPUT_DIR / "pure_human_master.xlsx")
    write_table(machine_assisted, OUTPUT_DIR / "machine_assisted_human_master.xlsx")

    # Split pure human among reviewer_01 ... reviewer_16
    pure_splits = round_robin_split(pure_human, PURE_HUMAN_REVIEWERS)

    # Split machine-assisted among reviewer_17 and reviewer_18
    machine_splits = round_robin_split(machine_assisted, MACHINE_ASSISTED_REVIEWERS)

    # Write outputs
    summary_rows = []

    for reviewer, reviewer_df in pure_splits.items():
        out_path = OUTPUT_DIR / f"{reviewer}.xlsx"
        write_table(reviewer_df, out_path)
        summary_rows.append({
            "Reviewer": reviewer,
            "Split Type": "pure_human",
            "Rows": len(reviewer_df),
        })

    for reviewer, reviewer_df in machine_splits.items():
        out_path = OUTPUT_DIR / f"{reviewer}.xlsx"
        write_table(reviewer_df, out_path)
        summary_rows.append({
            "Reviewer": reviewer,
            "Split Type": "machine_assisted_human",
            "Rows": len(reviewer_df),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Reviewer").reset_index(drop=True)
    write_table(summary_df, OUTPUT_DIR / "split_summary.xlsx")

    print("\nReviewer split summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()