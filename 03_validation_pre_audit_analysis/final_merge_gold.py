# =============================================================================
# final_merge_gold.py — Build the WorldBench-Gold equivalent of Tier A
#
# Paper section: §4 "WorldBench Overview"; §5 "Dataset Geography Analysis"
# Role in pipeline: Gold-side counterpart to final_merge.py's Tier A construction.
#   Gold (human_worldbench.csv) doesn't have the pre-derived num_languages /
#   num_content_countries / matrix_compatible columns that Core's pipeline
#   computes -- it only has the raw "Languages covered" / "All Countries
#   Covered" / "Task Category" fields. This script reuses final_merge.py's
#   exact parsing, country-normalization, and task-normalization logic
#   (copied verbatim, not reimplemented) so Gold's Tier A is directly
#   comparable to Core's rather than built with subtly different rules.
#
# Tier A rule (identical to Core): matrix_compatible (valid country + valid
#   task) AND ((num_languages==1 & num_countries>=1) OR (num_languages>1 &
#   num_countries==1)).
#
# Run from compiled_ACL/task_matrix_construction/:
#   python final_merge_gold.py
#
# Input:
#   unified_worldbench_views/language_cleanup_v2/human_worldbench.csv
#   valid_countries.csv
#
# Output (in unified_worldbench_views/language_cleanup_v2/):
#   worldbench_gold_base_ground_truth.csv
#   worldbench_gold_lang_tierA.csv
# =============================================================================
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = (
    "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/"
    "unified_worldbench_views/language_cleanup_v2/human_worldbench.csv"
)
COUNTRY_LIST_FILE = "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/valid_countries.csv"
COUNTRY_LIST_COLUMN = "country"

OUTDIR = Path(
    "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/"
    "unified_worldbench_views/language_cleanup_v2"
)
OUTPUT_BASE = OUTDIR / "worldbench_gold_base_ground_truth.csv"
OUTPUT_TIER_A = OUTDIR / "worldbench_gold_lang_tierA.csv"

# --- Gold's raw column names (no "Normalized Countries"/"Normalized Languages"
#     enrichment exists for Gold, so priority lists are shorter than Core's,
#     but pull from the same underlying fields.) ---
COUNTRY_PRIORITY_COLS = [
    "All Countries Covered (you can select multiple countries)",
    "Assigned country",
]
LANGUAGE_PRIORITY_COLS = [
    "Languages covered (separate using semicolon)",
]

# =============================================================================
# Everything below is copied verbatim from final_merge.py so Gold's Tier A
# uses identical parsing/validation rules to Core's.
# =============================================================================

ALLOWED_TASK_CATEGORIES = [
    "Raw Corpus",
    "AI/LLM Agents",
    "Code Models",
    "Computational Social Science, Cultural Analytics, NLP for social good",
    "Dialogue and Interactive Systems",
    "Discourse, Pragmatics and Reasoning",
    "Efficient/Low-Resource Methods for NLP, LLM Efficiency",
    "Ethics, Bias, and Fairness",
    "Generalizability and Transfer",
    "Hierarchical Structure Prediction, Syntax, and Parsing",
    "Human-AI Interaction/Cooperation, Human Centered NLP",
    "Information Extraction, Retrieval, and Text Mining",
    "Interpretability, Model Analysis, Transparency, and Explainability",
    "Language Modeling",
    "Linguistic Theories, Cognitive Modeling, and Psycholinguistics",
    "Machine Learning for NLP",
    "Machine Translation",
    "Mathematical, Symbolic, and Logical Reasoning in NLP",
    "Multilinguality and Language Diversity",
    "Multimodality and Language Grounding to Vision, Robotics and Beyond",
    "Natural Language Generation",
    "Neurosymbolic approaches to NLP",
    "NLP Applications",
    "Phonology, morphology, and word segmentation",
    "Question Answering",
    "Retrieval Augmented Language Models",
    "Safety and Alignment in LLMs",
    "Semantics: Lexical and Sentence Level, Textual Inference",
    "Sentiment Analysis, Stylistic Analysis, and Argument Mining",
    "Speech recognition, text-to-speech, and spoken language understanding",
    "Summarization",
    "Syntax: Tagging, Chunking, and Parsing",
]
ALLOWED_TASK_SET = set(ALLOWED_TASK_CATEGORIES)

NON_COUNTRY_TERMS = {
    "english", "arabic", "french", "spanish", "german",
    "portuguese", "chinese", "korean", "japanese",
    "africa", "asia", "eastern europe", "latin america",
    "south america", "europe",
    "basque country", "catalonia",
    "swahili-speaking countries",
}

MISSING_VALUES = {
    "", "nan", "none", "null", "not stated", "not specified", "unknown", "n/a", "na"
}


def load_valid_countries(filepath, column="country"):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Country list file not found: {filepath}")
    if path.suffix.lower() == ".csv":
        country_df = pd.read_csv(path, encoding="latin-1")
        if column not in country_df.columns:
            raise ValueError(f"Country list CSV must contain column '{column}'. Found columns: {list(country_df.columns)}")
        countries = country_df[column].dropna().astype(str).str.strip()
        countries = countries[countries != ""]
        return set(countries)
    if path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    raise ValueError(f"Unsupported country list format: {path.suffix}. Use .csv or .txt")


VALID_COUNTRIES = load_valid_countries(COUNTRY_LIST_FILE, COUNTRY_LIST_COLUMN)


def is_missing(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in MISSING_VALUES


def clean_text(x):
    if is_missing(x):
        return np.nan
    return re.sub(r"\s+", " ", str(x).strip())


def canonicalize_for_matching(s):
    if s is None:
        return None
    s = s.lower().strip()
    s = s.replace("&", "and")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[,:()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


TASK_CANONICAL_LOOKUP = {canonicalize_for_matching(task): task for task in ALLOWED_TASK_CATEGORIES}
TASK_ALIASES = {
    "human ai interaction cooperation human centered nlp": "Human-AI Interaction/Cooperation, Human Centered NLP",
    "information extraction retrieval and text mining": "Information Extraction, Retrieval, and Text Mining",
    "computational social science cultural analytics nlp for social good": "Computational Social Science, Cultural Analytics, NLP for social good",
    "efficient low resource methods for nlp llm efficiency": "Efficient/Low-Resource Methods for NLP, LLM Efficiency",
    "speech recognition text to speech and spoken language understanding": "Speech recognition, text-to-speech, and spoken language understanding",
    "semantics lexical and sentence level textual inference": "Semantics: Lexical and Sentence Level, Textual Inference",
    "multimodality and language grounding to vision robotics and beyond": "Multimodality and Language Grounding to Vision, Robotics and Beyond",
}


def normalize_task(raw_task):
    raw = clean_text(raw_task)
    if pd.isna(raw):
        return np.nan, "missing"
    if raw in ALLOWED_TASK_SET:
        return raw, "exact"
    key = canonicalize_for_matching(raw)
    if key in TASK_ALIASES:
        return TASK_ALIASES[key], "alias"
    if key in TASK_CANONICAL_LOOKUP:
        return TASK_CANONICAL_LOOKUP[key], "normalized"
    return raw, "unmapped"


def dedupe_preserve_order(items):
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_country_token(token):
    if token is None:
        return None
    s = str(token).strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s).strip()
    alias_map = {
        "USA": "United States", "US": "United States", "U.S.": "United States", "U.S.A.": "United States",
        "UK": "United Kingdom", "U.K.": "United Kingdom", "UAE": "United Arab Emirates",
        "United States of America": "United States", "KSA": "Saudi Arabia", "Slovak Republic": "Slovakia",
        "FYR Macedonia": "North Macedonia", "The Netherlands": "Netherlands", "Bosnia": "Bosnia and Herzegovina",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina", "Quebec": "Canada", "Alaska": "United States",
        "Florida": "United States", "Scotland": "United Kingdom", "Macao": "China", "Macau": "China",
        "P.R. China": "China", "Russian Federation": "Russia", "Galicia": "Spain", "Greenland": "Denmark",
        "Republic of Korea": "South Korea", "Korea, South": "South Korea", "Korea, North": "North Korea",
        "Congo": "Congo (Congo-Brazzaville)", "Republic of the Congo": "Congo (Congo-Brazzaville)",
        "Czech Republic": "Czechia", "Sao Tome and Principe": "São Tomé and Príncipe",
        "São Tomé & Príncipe": "São Tomé and Príncipe", "Ivory Coast": "Cote d'Ivoire",
        "Côte d’Ivoire": "Cote d'Ivoire", "Cote d Ivoire": "Cote d'Ivoire", "Holy See": "Vatican City",
        "Federated States of Micronesia": "Micronesia", "Mainland China": "China", "Hong Kong": "China",
        "Hawaii": "United States", "Puerto Rico": "United States", "America": "United States",
        "People’s Republic of China": "China", "England": "United Kingdom", "Kurdish area": "Turkey",
        "Great Britain": "United Kingdom", "California": "United States", "Wales": "United Kingdom",
        "Republic of Moldova": "Moldova", "Lao": "Laos", "Hong Kong S.A.R.": "China", "Delaware": "United States",
        "Faroe Islands": "Denmark", "Türkiye": "Turkey", "Brunei Darussalam": "Brunei",
        "Syrian Arab Republic": "Syria", "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "Viet Nam": "Vietnam", "Sultanate of Oman": "Oman", "Republic of Guyana": "Guyana",
        "State of Kuwait": "Kuwait", "Jersey": "United Kingdom", "Montserrat": "United Kingdom",
        "Réunion": "France",
    }
    return alias_map.get(s, s)


def split_country_field(value):
    if is_missing(value):
        return []
    s = str(value).strip()
    items = []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            items = [str(x).strip() for x in parsed] if isinstance(parsed, list) else [s]
        except Exception:
            items = [s]
    else:
        items = [s]
    parts = []
    for item in items:
        item = item.replace("|", ",").replace(" / ", ",").replace(";", ",")
        parts.extend(p.strip() for p in item.split(",") if p.strip())
    countries = []
    for part in parts:
        part = normalize_country_token(part)
        if not part:
            continue
        if part.lower() in NON_COUNTRY_TERMS:
            continue
        if part.lower() not in MISSING_VALUES:
            countries.append(part)
    return dedupe_preserve_order(countries)


def split_list_field(value):
    if is_missing(value):
        return []
    s = str(value).strip()
    items = []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            items = [str(x).strip() for x in parsed] if isinstance(parsed, list) else [s]
        except Exception:
            items = [s]
    else:
        items = [s]
    parts = []
    for item in items:
        item = item.replace("|", ";").replace(",", ";").replace(" / ", ";")
        parts.extend(p.strip() for p in item.split(";") if p.strip())
    cleaned = []
    for p in parts:
        p = clean_text(p)
        if pd.isna(p):
            continue
        cleaned.append(p)
    return dedupe_preserve_order(cleaned)


def choose_best_country(row):
    for col in COUNTRY_PRIORITY_COLS:
        if col in row and not is_missing(row[col]):
            countries = split_country_field(row[col])
            if countries:
                return col, countries
    return np.nan, []


def choose_language_list(row):
    for col in LANGUAGE_PRIORITY_COLS:
        if col in row and not is_missing(row[col]):
            vals = split_list_field(row[col])
            if vals:
                return vals, col
    return [], np.nan


# =============================================================================
# Main
# =============================================================================

df = pd.read_csv(INPUT_CSV).copy()
df["source_row_index"] = df.index

field_used_list, raw_country_list, valid_country_list, invalid_country_list = [], [], [], []
for _, row in df.iterrows():
    field_used, countries = choose_best_country(row)
    valid = [c for c in countries if c in VALID_COUNTRIES]
    invalid = [c for c in countries if c not in VALID_COUNTRIES]
    field_used_list.append(field_used)
    raw_country_list.append(countries)
    valid_country_list.append(valid)
    invalid_country_list.append(invalid)

df["best_country_field_used"] = field_used_list
df["best_country_list_raw"] = raw_country_list
df["best_country_list_valid"] = valid_country_list
df["invalid_countries_dropped"] = invalid_country_list
df["has_valid_content_country"] = df["best_country_list_valid"].apply(lambda x: len(x) > 0)

norm_tasks, task_statuses = [], []
for val in df["Task Category"]:
    task, status = normalize_task(val)
    norm_tasks.append(task)
    task_statuses.append(status)

df["normalized_task_category"] = norm_tasks
df["task_normalization_status"] = task_statuses
df["has_valid_task"] = df["normalized_task_category"].isin(ALLOWED_TASK_SET)

language_lists, language_sources = [], []
for _, row in df.iterrows():
    vals, source = choose_language_list(row)
    language_lists.append(vals)
    language_sources.append(source)

df["language_list"] = language_lists
df["language_source_used"] = language_sources
df["languages_normalized_joined"] = df["language_list"].apply(lambda x: "; ".join(x) if isinstance(x, list) and x else np.nan)
df["num_languages"] = df["language_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["num_content_countries"] = df["best_country_list_valid"].apply(lambda x: len(x) if isinstance(x, list) else 0)

df["matrix_compatible"] = df["has_valid_content_country"] & df["has_valid_task"]
df.to_csv(OUTPUT_BASE, index=False)

df_tierA = df[
    df["matrix_compatible"]
    & (
        ((df["num_languages"] == 1) & (df["num_content_countries"] >= 1))
        | ((df["num_languages"] > 1) & (df["num_content_countries"] == 1))
    )
].copy()
df_tierA.to_csv(OUTPUT_TIER_A, index=False)

print(f"Gold rows total: {len(df)}")
print(f"Gold matrix_compatible (valid country + valid task): {df['matrix_compatible'].sum()}")
print(f"Gold Tier A (controlled) datasets: {len(df_tierA)}")

gold_tierA_langs = set()
df_tierA["language_list"].apply(lambda x: gold_tierA_langs.update(x))
print(f"Gold Tier A unique languages: {len(gold_tierA_langs)}")
