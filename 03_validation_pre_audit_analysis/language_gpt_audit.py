# =============================================================================
# language_gpt_audit.py — GPT-based audit of language labels in WorldBench
#
# Paper section: §3.1 "Annotation Framework and Schema Design" (language normalization)
#               Appendix / language quality analysis
# Role in pipeline: Post-processing audit — uses GPT-4o-mini to validate and
#   normalize language labels in both WorldBench-Gold and WorldBench-Core,
#   flagging labels that are non-standard, ambiguous, or should be merged.
#   Outputs feed into the language_cleanup_v2/ audit folder for human review.
#
# Run from compiled_ACL/task_matrix_construction/worldbench_matrix_outputs/:
#   export OPENAI_API_KEY="sk-..."
#   python language_gpt_audit.py
#
# Input:
#   unified_worldbench_views/language_cleanup_v2/human_worldbench.csv
#   unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv
#
# Output (all in language_audit_outputs/):
#   language_audit_final.csv, language_mapping_dictionary.csv,
#   language_audit_needs_review.csv, etc.
# =============================================================================
import os
import re
import ast
import json
import time
import pandas as pd
from openai import OpenAI

# ==================================================
# CONFIG
# ==================================================


GOLD_PATH = '/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/human_worldbench.csv'
AUTO_PATH = "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv"


MODEL = "gpt-4o-mini"
BATCH_SIZE = 20
SLEEP_BETWEEN_CALLS = 0.5

OUT_DIR = "language_audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==================================================
# LOAD DATA
# ==================================================

df_auto = pd.read_csv(AUTO_PATH)
df_gold = pd.read_csv(GOLD_PATH)

print("Auto rows:", len(df_auto))
print("Gold rows:", len(df_gold))

# ==================================================
# HELPERS
# ==================================================

MISSING_VALUES = {
    "", "nan", "none", "not stated", "unknown", "n/a", "na",
    "not specified", "unspecified"
}

LANGUAGE_COLS = [
    "canonical_language_string_cleaned",
    "languages_normalized_joined",
    "Normalized Languages",
    "Languages covered (separate using semicolon)",
]

def is_missing(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in MISSING_VALUES

def split_language_items(x):
    if is_missing(x):
        return []

    x = str(x).strip()

    # Handle Python-list-like strings
    if x.startswith("[") and x.endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    # Split common separators
    return re.split(r";|,", x)

def basic_clean_language(x):
    if is_missing(x):
        return None

    x = str(x).strip()
    x = re.sub(r"\s+", " ", x).strip()

    if is_missing(x):
        return None

    return x

def collect_languages(df):
    langs = set()

    for col in LANGUAGE_COLS:
        if col not in df.columns:
            continue

        for val in df[col]:
            for item in split_language_items(val):
                cleaned = basic_clean_language(item)
                if cleaned:
                    langs.add(cleaned)

    return langs

# ==================================================
# EXTRACT UNIQUE LANGUAGE LABELS
# ==================================================

auto_langs = collect_languages(df_auto)
gold_langs = collect_languages(df_gold)

all_langs = sorted(auto_langs | gold_langs)

lang_df = pd.DataFrame({"raw_language": all_langs})
lang_df.to_csv(f"{OUT_DIR}/unique_language_labels_for_audit.csv", index=False)

print("Unique raw language labels:", len(all_langs))
print(lang_df.head(30))

# ==================================================
# GPT AUDIT FUNCTION
# ==================================================

SYSTEM_PROMPT = """
You are auditing language labels for an NLP dataset catalog.

For each raw label, determine whether it refers to a human language, dialect, named language variety,
macrolanguage, script variant, or locale variant.

Return valid JSON only. Do not include markdown.
"""

def audit_language_batch(batch):
    user_prompt = f"""
For each raw language label below, return a JSON list of objects with exactly these fields:

- raw_language: original label exactly as provided
- is_valid_language: true or false
- canonical_language: standard English name if valid, otherwise null
- label_type: one of ["language", "dialect_or_variety", "script_variant", "locale_variant", "macrolanguage", "not_language", "artifact", "uncertain"]
- notes: brief explanation

Rules:
- Preserve real low-resource languages. Do not reject a language just because it is rare.
- Dialects and named varieties should be treated as valid, but map to the most appropriate canonical language when obvious.
- Script variants like "pa-Arab", "sat-Latn", or "ks-Deva" should map to the base language if identifiable.
- Locale variants like "pt-BR", "en-US", or "zh-CN" should map to the base language unless the label clearly names a distinct variety.
- Nationalities, countries, regions, datasets, tasks, or vague descriptors such as "Indian", "Argentinian", "African", "European", "Multilingual", or "Code-mixed" are not languages.
- If uncertain, mark label_type as "uncertain"; set is_valid_language=true only if it plausibly refers to a human language or named variety.
- Output valid JSON only.

Labels:
{json.dumps(batch, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    # Allow either {"results": [...]} or a direct list if model returns it
    if isinstance(parsed, dict):
        if "results" in parsed:
            return parsed["results"]
        # fallback: first list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
    elif isinstance(parsed, list):
        return parsed

    raise ValueError(f"Unexpected JSON structure: {parsed}")

# ==================================================
# RUN AUDIT IN BATCHES
# ==================================================

batches = [
    all_langs[i:i + BATCH_SIZE]
    for i in range(0, len(all_langs), BATCH_SIZE)
]

all_results = []

for i, batch in enumerate(batches, start=1):
    print(f"Auditing batch {i}/{len(batches)}...")

    try:
        results = audit_language_batch(batch)
        all_results.extend(results)

        # Save checkpoint after every batch
        pd.DataFrame(all_results).to_csv(
            f"{OUT_DIR}/language_audit_checkpoint.csv",
            index=False
        )

    except Exception as e:
        print(f"Batch {i} failed:", e)
        pd.DataFrame({"raw_language": batch}).to_csv(
            f"{OUT_DIR}/failed_batch_{i:04d}.csv",
            index=False
        )

    time.sleep(SLEEP_BETWEEN_CALLS)

audit_df = pd.DataFrame(all_results)

# Ensure expected columns exist
expected_cols = [
    "raw_language",
    "is_valid_language",
    "canonical_language",
    "label_type",
    "notes",
]
for col in expected_cols:
    if col not in audit_df.columns:
        audit_df[col] = None

audit_df = audit_df[expected_cols]
audit_df.to_csv(f"{OUT_DIR}/language_audit_final.csv", index=False)

print("Audit complete.")
print("Audited labels:", len(audit_df))
print(audit_df["label_type"].value_counts(dropna=False))

# ==================================================
# BUILD RAW → CANONICAL MAP
# ==================================================

def parse_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}

def clean_canonical(x):
    if is_missing(x):
        return None
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x if x else None

language_map = {}

for _, row in audit_df.iterrows():
    raw = row["raw_language"]
    valid = parse_bool(row["is_valid_language"])
    canonical = clean_canonical(row["canonical_language"])

    if valid and canonical:
        language_map[raw] = canonical
    else:
        language_map[raw] = None

# Save mapping table
map_df = pd.DataFrame([
    {"raw_language": raw, "mapped_language": mapped}
    for raw, mapped in language_map.items()
])
map_df.to_csv(f"{OUT_DIR}/language_mapping_dictionary.csv", index=False)

# ==================================================
# APPLY MAP TO DATASETS
# ==================================================

def normalize_language_list_from_row(row):
    output = set()

    for col in LANGUAGE_COLS:
        if col not in row.index:
            continue

        raw_val = row.get(col)

        for item in split_language_items(raw_val):
            raw = basic_clean_language(item)
            if not raw:
                continue

            canonical = language_map.get(raw)

            if canonical:
                output.add(canonical)

    return sorted(output)

def language_string(lang_list):
    return "; ".join(lang_list)

df_auto["audited_language_list"] = df_auto.apply(normalize_language_list_from_row, axis=1)
df_auto["audited_languages"] = df_auto["audited_language_list"].apply(language_string)
df_auto["num_audited_languages"] = df_auto["audited_language_list"].apply(len)

df_gold["audited_language_list"] = df_gold.apply(normalize_language_list_from_row, axis=1)
df_gold["audited_languages"] = df_gold["audited_language_list"].apply(language_string)
df_gold["num_audited_languages"] = df_gold["audited_language_list"].apply(len)

# ==================================================
# SUMMARY STATS
# ==================================================

auto_audited_languages = set().union(*df_auto["audited_language_list"])
gold_audited_languages = set().union(*df_gold["audited_language_list"])
combined_audited_languages = auto_audited_languages | gold_audited_languages

print("\nLanguage counts after audit:")
print("Auto audited languages:", len(auto_audited_languages))
print("Gold audited languages:", len(gold_audited_languages))
print("Combined audited languages:", len(combined_audited_languages))

pd.DataFrame({"language": sorted(auto_audited_languages)}).to_csv(
    f"{OUT_DIR}/worldbench_auto_audited_languages.csv",
    index=False
)

pd.DataFrame({"language": sorted(gold_audited_languages)}).to_csv(
    f"{OUT_DIR}/worldbench_gold_audited_languages.csv",
    index=False
)

pd.DataFrame({"language": sorted(combined_audited_languages)}).to_csv(
    f"{OUT_DIR}/worldbench_combined_audited_languages.csv",
    index=False
)

# Useful review files
audit_df[audit_df["label_type"].isin(["not_language", "artifact", "uncertain"])].to_csv(
    f"{OUT_DIR}/language_audit_needs_review.csv",
    index=False
)

audit_df[audit_df["is_valid_language"].apply(parse_bool) == False].to_csv(
    f"{OUT_DIR}/language_audit_removed_labels.csv",
    index=False
)

# Save updated datasets
df_auto.to_csv(f"{OUT_DIR}/automated_full_set_with_audited_languages.csv", index=False)
df_gold.to_csv(f"{OUT_DIR}/human_validated_set_with_audited_languages.csv", index=False)

print("\nSaved outputs to:", OUT_DIR)