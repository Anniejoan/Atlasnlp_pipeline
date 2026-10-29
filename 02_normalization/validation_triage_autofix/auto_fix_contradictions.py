# =============================================================================
# auto_fix_contradictions.py — Deterministically fix contradictions in raw results
#
# Paper section: §3.4 "Validation of Automated Extraction"
# Role in pipeline: Step 5a — runs BEFORE validation_triage.py.
#   Applies rule-based fixes for common LLM inconsistencies (e.g. country name
#   abbreviation normalization, language-country mismatches, modality flags).
#
# Run from compiled_ACL/:
#   python auto_fix_contradictions.py
#
# Input:  validation_triage.csv (merged raw LLM results, renamed from acl_raw_results.csv)
# Output: validation_triage_autofixed.csv (fed into validation_triage.py)
#         autofix_changed_rows.csv (rows that were modified, for audit)
# =============================================================================
import pandas as pd
import re

INPUT_CSV = "validation_triage.csv"
OUTPUT_CSV = "validation_triage_autofixed.csv"
OUTPUT_FIXED_ONLY = "autofix_changed_rows.csv"


def split_multi_value_cell(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value or value.lower() == "not stated":
        return []
    parts = re.split(r"\s*,\s*|\s*;\s*", value)
    return [p.strip() for p in parts if p.strip()]


def normalize_country_list(value):
    country_map = {
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "US": "United States",
        "U.S.": "United States",
        "USA": "United States",
    }
    items = split_multi_value_cell(value)
    out = []
    seen = set()
    for x in items:
        x = country_map.get(x, x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_language_list(value):
    lang_map = {
        "EN": "English",
        "eng": "English",
        "ASL": "American Sign Language",
        "BSL": "British Sign Language",
        "ISL": "Indian Sign Language",
        "GSL": "Greek Sign Language",
        "LSA": "Argentine Sign Language",
    }
    items = split_multi_value_cell(value)
    out = []
    seen = set()
    for x in items:
        x = lang_map.get(x, x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def remove_flag(flag_string, flags_to_remove):
    if pd.isna(flag_string) or not str(flag_string).strip():
        return ""
    flags = [f.strip() for f in str(flag_string).split(";") if f.strip()]
    flags = [f for f in flags if f not in flags_to_remove]
    return "; ".join(flags)


def assign_review_tier_from_flags(flag_string):
    flags = set()
    if pd.notna(flag_string) and str(flag_string).strip():
        flags = {f.strip() for f in str(flag_string).split(";") if f.strip()}

    contradiction_markers = {
        "COUNTRY_ATTRIBUTION_CONTRADICTION",
        "MULTI_COUNTRY_CONTRADICTION",
        "LANGUAGE_COUNT_CONTRADICTION",
        "MODALITY_CONTRADICTION",
        "TASK_CATEGORY_COLUMN_ERROR",
        "DATASET_NAME_COLUMN_ERROR",
        "DATASET_LINK_NOT_URL",
    }

    human_markers = {
        "LICENSE_SUSPICIOUS",
        "MULTIPLE_DATASETS_IN_NAME",
        "COUNTRY_MISSING_BUT_LANGUAGE_PRESENT",
    }

    machine_markers = {
        "LINK_MISSING",
        "LICENSE_MISSING",
        "COUNTRIES_MISSING",
        "COUNTRY_INFERABLE_FROM_LANGUAGE",
        "DESCRIPTION_CONTAINS_URL",
        "LICENSE_MAY_BE_LINK",
    }

    deferred_markers = {
        "ORIGIN_TYPE_MISSING",
        "COUNTRY_ATTRIBUTION_MISSING",
        "MULTIPLE_CHOICE_MISSING",
    }

    if any(f in contradiction_markers for f in flags):
        return "contradiction"
    if any(f in human_markers for f in flags):
        return "human_review"
    if any(f in machine_markers for f in flags):
        return "machine_followup"
    if any(f in deferred_markers for f in flags):
        return "deferred_review"
    return "auto_validated"


def main():
    df = pd.read_csv(INPUT_CSV)
    changed_rows = []

    for idx, row in df.iterrows():
        changed = False

        countries = normalize_country_list(row.get("All Countries Covered (you can select multiple countries)", ""))
        langs = normalize_language_list(row.get("Languages covered (separate using semicolon)", ""))

        current_attr = str(row.get("Country Attribution Method", "Not stated")).strip()
        current_multi = str(row.get("Multi Country", "Not stated")).strip()
        current_lang_cov = str(row.get("Language coverage type", "Not stated")).strip()

        old_flags = str(row.get("Validation Flags", "")).strip()

        # 1) If countries are explicitly listed but attribution says Not stated, set to Explicit
        if countries and current_attr == "Not stated":
            df.at[idx, "Country Attribution Method"] = "Explicit (metadata or creator affiliation))"
            changed = True

        # 2) Recompute Multi Country from country count
        if len(countries) > 1 and current_multi != "Yes":
            df.at[idx, "Multi Country"] = "Yes"
            changed = True
        elif len(countries) == 1 and current_multi != "No":
            df.at[idx, "Multi Country"] = "No"
            changed = True

        # 3) Recompute Language coverage type from language count
        if len(langs) > 1 and current_lang_cov != "Multilingual":
            df.at[idx, "Language coverage type"] = "Multilingual"
            changed = True
        elif len(langs) == 1 and current_lang_cov != "Monolingual":
            df.at[idx, "Language coverage type"] = "Monolingual"
            changed = True

        if changed:
            # Remove contradiction flags that should now be resolved
            new_flags = remove_flag(old_flags, {
                "COUNTRY_ATTRIBUTION_CONTRADICTION",
                "MULTI_COUNTRY_CONTRADICTION",
                "LANGUAGE_COUNT_CONTRADICTION",
                "COUNTRY_ATTRIBUTION_MISSING",
            })

            df.at[idx, "Validation Flags"] = new_flags
            new_tier = assign_review_tier_from_flags(new_flags)
            df.at[idx, "Review Tier"] = new_tier
            df.at[idx, "Needs Machine Followup"] = "Yes" if new_tier == "machine_followup" else "No"
            df.at[idx, "Needs Human Review"] = "Yes" if new_tier in {"human_review", "contradiction"} else "No"

            if "Needs Deferred Review" in df.columns:
                df.at[idx, "Needs Deferred Review"] = "Yes" if new_tier == "deferred_review" else "No"

            changed_rows.append(df.loc[idx].to_dict())

    df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(changed_rows).to_csv(OUTPUT_FIXED_ONLY, index=False)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_FIXED_ONLY}")
    print(f"Changed rows: {len(changed_rows)}")


if __name__ == "__main__":
    main()