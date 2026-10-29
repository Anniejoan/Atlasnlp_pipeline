# =============================================================================
# final_merge.py — Build unified WorldBench dataset views for analysis
#
# Paper section: §4 "WorldBench Overview"; §5 "Dataset Geography Analysis"
# Role in pipeline: Step 10b — assembles the final unified dataset views that
#   all analysis scripts and notebooks consume.
#   Joins the master metadata with author-affiliation (producer-country) data,
#   applies task normalization, and produces tiered views based on how country
#   attribution was obtained:
#     Tier A: explicit country mentions or locale-specific language signals
#     Tier B: excluded (globally distributed languages without supporting context)
#     Tier C: naive expansion (all language-associated countries, for comparison)
#   The "base ground truth" view (Tier A) is what the paper's analyses use.
#
# Run from compiled_ACL/task_matrix_construction/:
#   python final_merge.py
#
# Input:
#   compiled_ACL/updated_master_with_normalized_languages.csv  — enriched master
#   compiled_ACL/author_affiliation/author_affiliation_all_merged.csv — producer data
#
# Output (all in unified_worldbench_views/):
#   worldbench_base_ground_truth.csv  — main analysis dataset (Tier A)
#   worldbench_lang_tierA.csv         — datasets with strict language signals
#   worldbench_lang_tierB_excluded.csv — excluded broad-language datasets
#   worldbench_lang_tierC_naive.csv   — naive expansion for comparison (Figure 5)
#   worldbench_lang_tierA_expanded.csv, worldbench_lang_tierC_expanded.csv
#   worldbench_country_only_expanded.csv
#   worldbench_country_task_expanded.csv — used by Analysis Exploration.ipynb
#   worldbench_dataset_views_summary.csv — summary of each view
# =============================================================================
import ast
import pandas as pd
import numpy as np
import re
from pathlib import Path

INPUT_CSV = "/home/jnwatu/Worldbench/compiled_ACL/updated_master_with_normalized_languages.csv"
AUTHOR_CSV = "/home/jnwatu/Worldbench/compiled_ACL/author_affiliation/author_affiliation_all_merged_corrected.csv"
COUNTRY_LIST_FILE = "/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/valid_countries.csv"
COUNTRY_LIST_COLUMN = "country"

OUTDIR = Path("/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = OUTDIR / "worldbench_base_ground_truth.csv"
OUTPUT_TIER_A = OUTDIR / "worldbench_lang_tierA.csv"
OUTPUT_TIER_B = OUTDIR / "worldbench_lang_tierB_excluded.csv"
OUTPUT_TIER_C = OUTDIR / "worldbench_lang_tierC_naive.csv"
OUTPUT_TIER_A_EXPANDED = OUTDIR / "worldbench_lang_tierA_expanded.csv"
OUTPUT_TIER_C_EXPANDED = OUTDIR / "worldbench_lang_tierC_expanded.csv"
OUTPUT_COUNTRY_ONLY = OUTDIR / "worldbench_country_only_expanded.csv"
OUTPUT_COUNTRY_TASK = OUTDIR / "worldbench_country_task_expanded.csv"
OUTPUT_SUMMARY = OUTDIR / "worldbench_dataset_views_summary.csv"

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

COUNTRY_PRIORITY_COLS = [
    "all_countries_covered",
    "All Countries Covered (you can select multiple countries)",
    "Normalized Countries",
    "Assigned country",
]

LANGUAGE_PRIORITY_COLS = [
    "Normalized Languages",
    "Languages covered (separate using semicolon)",
]

PRODUCER_CANDIDATE_COLS = [
    "author_countries_final",
    "author_all_inferred_countries",
    "author_affiliation_countries",
    "author_institution_countries",
    "author_email_countries",
]

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

def normalize_url(url):
    if is_missing(url):
        return np.nan
    s = str(url).strip()
    s = re.sub(r"[?#].*$", "", s)
    s = s.rstrip("/")
    return s

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
        "USA": "United States",
        "US": "United States",
        "U.S.": "United States",
        "U.S.A.": "United States",
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "UAE": "United Arab Emirates",
        "United States of America": "United States",
        "KSA": "Saudi Arabia",
        "Slovak Republic": "Slovakia",
        "FYR Macedonia": "North Macedonia",
        "The Netherlands": "Netherlands",
        "Bosnia": "Bosnia and Herzegovina",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Quebec": "Canada",
        "Alaska": "United States",
        "Florida": "United States",
        "Scotland": "United Kingdom",
        "Macao": "China",
        "Macau": "China",
        "P.R. China": "China",
        "Russian Federation": "Russia",
        "Galicia": "Spain",
        "Greenland": "Denmark",
        "Republic of Korea": "South Korea",
        "Korea, South": "South Korea",
        "Korea, North": "North Korea",
        "Congo": "Congo (Congo-Brazzaville)",
        "Republic of the Congo": "Congo (Congo-Brazzaville)",
        "Czech Republic": "Czechia",
        "Sao Tome and Principe": "São Tomé and Príncipe",
        "São Tomé & Príncipe": "São Tomé and Príncipe",
        "Ivory Coast": "Cote d'Ivoire",
        "Côte d’Ivoire": "Cote d'Ivoire",
        "Cote d Ivoire": "Cote d'Ivoire",
        "Holy See": "Vatican City",
        "Federated States of Micronesia": "Micronesia",
        "Mainland China": "China",
        "Hong Kong": "China",
        "Hawaii": "United States",
        "Puerto Rico": "United States",
        "America": "United States",
        "People’s Republic of China": "China",
        "England": "United Kingdom",
        "Kurdish area": "Turkey",
        "Great Britain": "United Kingdom",
        "California": "United States",
        "Wales": "United Kingdom",
        "Republic of Moldova": "Moldova",
        "Lao": "Laos",
        "Hong Kong S.A.R.": "China",
        "Delaware": "United States",
        "Faroe Islands": "Denmark",
        "Türkiye": "Turkey",
        "Brunei Darussalam": "Brunei",
        "Syrian Arab Republic": "Syria",
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "Viet Nam": "Vietnam",
        "Sultanate of Oman": "Oman",
        "Republic of Guyana": "Guyana",
        "State of Kuwait": "Kuwait",
        "Jersey": "United Kingdom",
        "Montserrat": "United Kingdom",
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
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                items = [s]
        except Exception:
            items = [s]
    else:
        items = [s]
    parts = []
    for item in items:
        item = item.replace("|", ",").replace(" / ", ",").replace(";", ",")
        subparts = [p.strip() for p in item.split(",") if p.strip()]
        parts.extend(subparts)
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
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                items = [s]
        except Exception:
            items = [s]
    else:
        items = [s]
    parts = []
    for item in items:
        item = item.replace("|", ";").replace(",", ";").replace(" / ", ";")
        subparts = [p.strip() for p in item.split(";") if p.strip()]
        parts.extend(subparts)
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

def aggregate_country_lists(series):
    merged = []
    for item in series:
        if isinstance(item, list):
            merged.extend(item)
    return dedupe_preserve_order(merged)

def aggregate_first_nonmissing(series):
    for x in series:
        if not is_missing(x):
            return x
    return np.nan

def aggregate_sources(series):
    vals = []
    for x in series:
        if not is_missing(x):
            vals.append(str(x).strip())
    vals = dedupe_preserve_order(vals)
    return "; ".join(vals) if vals else np.nan

def choose_producer_country_source(row):
    for col in PRODUCER_CANDIDATE_COLS:
        if col in row and not is_missing(row[col]):
            vals = split_country_field(row[col])
            if vals:
                return vals, col
    return [], np.nan

df = pd.read_csv(INPUT_CSV).copy()
author_df = pd.read_csv(AUTHOR_CSV).copy()
df["source_row_index"] = df.index

if "paper_url" in df.columns:
    df["paper_url_key"] = df["paper_url"].apply(normalize_url)
else:
    df["paper_url_key"] = df["Paper Link (URL)"].apply(normalize_url)
author_df["paper_url_key"] = author_df["paper_url"].apply(normalize_url)

field_used_list = []
raw_country_list = []
valid_country_list = []
invalid_country_list = []

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

norm_tasks = []
task_statuses = []
for val in df["Task Category"]:
    task, status = normalize_task(val)
    norm_tasks.append(task)
    task_statuses.append(status)

df["normalized_task_category"] = norm_tasks
df["task_normalization_status"] = task_statuses
df["has_valid_task"] = df["normalized_task_category"].isin(ALLOWED_TASK_SET)

language_lists = []
language_sources = []
for _, row in df.iterrows():
    vals, source = choose_language_list(row)
    language_lists.append(vals)
    language_sources.append(source)

df["language_list"] = language_lists
df["language_source_used"] = language_sources
df["languages_normalized_joined"] = df["language_list"].apply(lambda x: "; ".join(x) if isinstance(x, list) and x else np.nan)
df["num_languages"] = df["language_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["num_content_countries"] = df["best_country_list_valid"].apply(lambda x: len(x) if isinstance(x, list) else 0)

producer_info = author_df.apply(choose_producer_country_source, axis=1)
author_df["producer_country_list"] = [x[0] for x in producer_info]
author_df["producer_country_source_used"] = [x[1] for x in producer_info]

author_with_producers = author_df[author_df["paper_url_key"].notna() & author_df["producer_country_list"].map(len).gt(0)].copy()

agg_dict = {
    "producer_country_list": aggregate_country_lists,
    "producer_country_source_used": aggregate_sources,
}
if "author_affiliation_type_final" in author_with_producers.columns:
    agg_dict["author_affiliation_type_final"] = aggregate_first_nonmissing
if "author_affiliation_type_final_source" in author_with_producers.columns:
    agg_dict["author_affiliation_type_final_source"] = aggregate_first_nonmissing

producer_map = author_with_producers.groupby("paper_url_key", as_index=False).agg(agg_dict)
producer_map["producer_countries"] = producer_map["producer_country_list"].apply(lambda x: "; ".join(x) if isinstance(x, list) and x else np.nan)
producer_map["num_producer_countries"] = producer_map["producer_country_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)

df_base = df.merge(producer_map, on="paper_url_key", how="left")
df_base["has_producer_country"] = df_base["producer_country_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
df_base["matrix_compatible"] = df_base["has_valid_content_country"] & df_base["has_valid_task"]
df_base.to_csv(OUTPUT_BASE, index=False)

df_tierA = df_base[
    df_base["matrix_compatible"] &
    (
        ((df_base["num_languages"] == 1) & (df_base["num_content_countries"] >= 1)) |
        ((df_base["num_languages"] > 1) & (df_base["num_content_countries"] == 1))
    )
].copy()

df_tierB = df_base[
    df_base["matrix_compatible"] &
    (df_base["num_languages"] > 1) &
    (df_base["num_content_countries"] > 1)
].copy()

df_tierC = df_base[
    df_base["matrix_compatible"] &
    (df_base["num_languages"] >= 1) &
    (df_base["num_content_countries"] >= 1)
].copy()

df_country_only = (
    df_base[df_base["matrix_compatible"]]
    .explode("best_country_list_valid")
    .rename(columns={"best_country_list_valid": "content_country"})
    .copy()
)
df_country_only["languages_in_dataset"] = df_country_only["language_list"].apply(lambda x: "; ".join(x) if isinstance(x, list) and x else np.nan)
df_country_only["content_country_exploded"] = True
df_country_only["language_exploded"] = False

df_country_task = df_country_only.copy()
df_country_task = df_country_task.rename(columns={"normalized_task_category": "Task Category"})

df_tierA_expanded = (
    df_tierA
    .explode("best_country_list_valid")
    .explode("language_list")
    .rename(columns={"best_country_list_valid": "content_country", "language_list": "language"})
    .copy()
)
df_tierA_expanded["language_country_analysis_tier"] = "A_reliable"

df_tierC_expanded = (
    df_tierC
    .explode("best_country_list_valid")
    .explode("language_list")
    .rename(columns={"best_country_list_valid": "content_country", "language_list": "language"})
    .copy()
)
df_tierC_expanded["language_country_analysis_tier"] = "C_naive"

df_tierA.to_csv(OUTPUT_TIER_A, index=False)
df_tierB.to_csv(OUTPUT_TIER_B, index=False)
df_tierC.to_csv(OUTPUT_TIER_C, index=False)
df_tierA_expanded.to_csv(OUTPUT_TIER_A_EXPANDED, index=False)
df_tierC_expanded.to_csv(OUTPUT_TIER_C_EXPANDED, index=False)
df_country_only.to_csv(OUTPUT_COUNTRY_ONLY, index=False)
df_country_task.to_csv(OUTPUT_COUNTRY_TASK, index=False)

summary_rows = [
    {
        "dataset_view_name": "worldbench_base_ground_truth",
        "path": str(OUTPUT_BASE),
        "unit_of_analysis": "1 row = 1 dataset",
        "row_count": len(df_base),
        "parent_source": "updated_master_with_normalized_languages + author_affiliation_all_merged",
        "matrix_compatible_only": False,
        "content_country_exploded": False,
        "language_exploded": False,
        "producer_country_exploded": False,
        "content_country_column": "best_country_list_valid (list)",
        "language_column": "language_list (list)",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "No",
        "notes_on_inclusion": "Single ground-truth base with normalized task, validated content country, and producer metadata attached."
    },
    {
        "dataset_view_name": "worldbench_lang_tierA",
        "path": str(OUTPUT_TIER_A),
        "unit_of_analysis": "1 row = 1 dataset",
        "row_count": len(df_tierA),
        "parent_source": "worldbench_base_ground_truth",
        "matrix_compatible_only": True,
        "content_country_exploded": False,
        "language_exploded": False,
        "producer_country_exploded": False,
        "content_country_column": "best_country_list_valid (list)",
        "language_column": "language_list (list)",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "Yes",
        "notes_on_inclusion": "Monolingual datasets and multilingual single-country datasets only."
    },
    {
        "dataset_view_name": "worldbench_lang_tierB_excluded",
        "path": str(OUTPUT_TIER_B),
        "unit_of_analysis": "1 row = 1 dataset",
        "row_count": len(df_tierB),
        "parent_source": "worldbench_base_ground_truth",
        "matrix_compatible_only": True,
        "content_country_exploded": False,
        "language_exploded": False,
        "producer_country_exploded": False,
        "content_country_column": "best_country_list_valid (list)",
        "language_column": "language_list (list)",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "No",
        "notes_on_inclusion": "Multilingual multi-country datasets excluded from direct language-country analysis."
    },
    {
        "dataset_view_name": "worldbench_lang_tierC_naive",
        "path": str(OUTPUT_TIER_C),
        "unit_of_analysis": "1 row = 1 dataset",
        "row_count": len(df_tierC),
        "parent_source": "worldbench_base_ground_truth",
        "matrix_compatible_only": True,
        "content_country_exploded": False,
        "language_exploded": False,
        "producer_country_exploded": False,
        "content_country_column": "best_country_list_valid (list)",
        "language_column": "language_list (list)",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "No",
        "notes_on_inclusion": "Naive comparison parent view for showing inflation from multilingual multi-country expansion."
    },
    {
        "dataset_view_name": "worldbench_lang_tierA_expanded",
        "path": str(OUTPUT_TIER_A_EXPANDED),
        "unit_of_analysis": "1 row = 1 dataset-language-country combination",
        "row_count": len(df_tierA_expanded),
        "parent_source": "worldbench_lang_tierA",
        "matrix_compatible_only": True,
        "content_country_exploded": True,
        "language_exploded": True,
        "producer_country_exploded": False,
        "content_country_column": "content_country",
        "language_column": "language",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "Yes",
        "notes_on_inclusion": "Reliable exploded view for language-country analysis."
    },
    {
        "dataset_view_name": "worldbench_lang_tierC_expanded",
        "path": str(OUTPUT_TIER_C_EXPANDED),
        "unit_of_analysis": "1 row = 1 dataset-language-country combination",
        "row_count": len(df_tierC_expanded),
        "parent_source": "worldbench_lang_tierC_naive",
        "matrix_compatible_only": True,
        "content_country_exploded": True,
        "language_exploded": True,
        "producer_country_exploded": False,
        "content_country_column": "content_country",
        "language_column": "language",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "No",
        "notes_on_inclusion": "Naively exploded language-country baseline used to demonstrate inflation."
    },
    {
        "dataset_view_name": "worldbench_country_only_expanded",
        "path": str(OUTPUT_COUNTRY_ONLY),
        "unit_of_analysis": "1 row = 1 dataset-country combination",
        "row_count": len(df_country_only),
        "parent_source": "worldbench_base_ground_truth",
        "matrix_compatible_only": True,
        "content_country_exploded": True,
        "language_exploded": False,
        "producer_country_exploded": False,
        "content_country_column": "content_country",
        "language_column": "language_list / languages_in_dataset",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "No",
        "notes_on_inclusion": "Matrix-compatible country-expanded view with languages preserved as dataset-level metadata."
    },
    {
        "dataset_view_name": "worldbench_country_task_expanded",
        "path": str(OUTPUT_COUNTRY_TASK),
        "unit_of_analysis": "1 row = 1 dataset-country combination",
        "row_count": len(df_country_task),
        "parent_source": "worldbench_base_ground_truth",
        "matrix_compatible_only": True,
        "content_country_exploded": True,
        "language_exploded": False,
        "producer_country_exploded": False,
        "content_country_column": "content_country",
        "language_column": "language_list / languages_in_dataset",
        "producer_country_column": "producer_country_list (list)",
        "safe_for_language_country_inference": "No",
        "notes_on_inclusion": "Same matrix-compatible population used for country-task and producer-vs-content analyses."
    },
]

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_SUMMARY, index=False)

print("Wrote unified build script outputs to:")
for p in [OUTPUT_BASE, OUTPUT_TIER_A, OUTPUT_TIER_B, OUTPUT_TIER_C, OUTPUT_TIER_A_EXPANDED, OUTPUT_TIER_C_EXPANDED, OUTPUT_COUNTRY_ONLY, OUTPUT_COUNTRY_TASK, OUTPUT_SUMMARY]:
    print(" -", p)
