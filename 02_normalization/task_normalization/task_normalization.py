

# =============================================================================
# task_normalization.py — Build the country × task matrix for WorldBench-Core
#
# Paper section: §5.1 "Dataset Coverage is Highly Uneven Across Countries and Tasks"
#                §5.4 "Task Coverage is Fragmented Across Countries"
# Role in pipeline: Step 10a — takes the fully enriched master dataset and produces
#   the country × task coverage matrix used in Figure 2 (heatmap) and Table 4.
#   Filters to 197 valid countries and 30 normalized task categories.
#   Each dataset-country-task triple is exploded to one row for matrix building.
#
# Run from compiled_ACL/task_matrix_construction/:
#   python task_normalization.py
#
# Input:  compiled_ACL/updated_master_with_normalized_languages.csv
# Output (all in worldbench_matrix_outputs/):
#   dataset_country_task_expanded.csv — one row per dataset-country-task triple
#   country_task_long.csv             — long-format country-task counts
#   country_task_matrix.csv           — pivot matrix (countries × tasks)
#   country_totals.csv                — per-country dataset counts
#   task_totals.csv                   — per-task dataset counts
#   coverage_stats.txt                — coverage statistics (59.7% empty, etc.)
#   task_normalization_review.csv     — unmapped task labels for inspection
#   invalid_countries_review.csv      — country values not in the allowed list
# =============================================================================

import pandas as pd
import re
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = '/home/jnwatu/Worldbench/compiled_ACL/updated_master_with_normalized_languages.csv'
COUNTRY_LIST_FILE = "valid_countries.csv"
COUNTRY_LIST_COLUMN = "country"

OUTDIR = Path("worldbench_matrix_outputs")
OUTDIR.mkdir(exist_ok=True)

OUTPUT_EXPANDED = OUTDIR / "dataset_country_task_expanded.csv"
OUTPUT_LONG = OUTDIR / "country_task_long.csv"
OUTPUT_MATRIX = OUTDIR / "country_task_matrix.csv"
OUTPUT_COUNTRY_TOTALS = OUTDIR / "country_totals.csv"
OUTPUT_TASK_TOTALS = OUTDIR / "task_totals.csv"
OUTPUT_TASK_REVIEW = OUTDIR / "task_normalization_review.csv"
OUTPUT_INVALID_COUNTRIES = OUTDIR / "invalid_countries_review.csv"
OUTPUT_STATS = OUTDIR / "coverage_stats.txt"
OUTPUT_SAMPLE = OUTDIR / "expanded_sample_rows.csv"

# =========================================================
# FIXED TASK LIST
# =========================================================

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

NON_COUNTRY_TERMS = {
    "english", "arabic", "french", "spanish", "german",
    "portuguese", "chinese", "korean", "japanese",
    "africa", "asia", "eastern europe", "latin america",
    "south america", "europe",
    "basque country", "catalonia",
    "swahili-speaking countries",
}

SUBNATIONAL_MAPPINGS = {
    "Quebec", "Alaska", "Florida", "Scotland", "Macao", "Galicia", "Faroe Islands", "Hong Kong"
}

# =========================================================
# HELPERS
# =========================================================

MISSING_VALUES = {
    "", "nan", "none", "null", "not stated", "not specified", "unknown", "n/a", "na"
}

def load_valid_countries(filepath, column="country"):
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Country list file not found: {filepath}")

    if path.suffix.lower() == ".csv":
        country_df = pd.read_csv(path, encoding = 'latin-1')

        if column not in country_df.columns:
            raise ValueError(
                f"Country list CSV must contain column '{column}'. "
                f"Found columns: {list(country_df.columns)}"
            )

        countries = (
            country_df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        countries = countries[countries != ""]
        return set(countries)

    if path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    raise ValueError(f"Unsupported country list format: {path.suffix}. Use .csv or .txt")

VALID_COUNTRIES = load_valid_countries(COUNTRY_LIST_FILE, COUNTRY_LIST_COLUMN)
print(f"Loaded {len(VALID_COUNTRIES)} valid countries from {COUNTRY_LIST_FILE}")

if len(VALID_COUNTRIES) != 197:
    raise ValueError(
        f"Expected 197 countries, but loaded {len(VALID_COUNTRIES)} from {COUNTRY_LIST_FILE}"
    )

def is_missing(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s.lower() in MISSING_VALUES

def clean_text(x):
    if is_missing(x):
        return None
    return re.sub(r"\s+", " ", str(x).strip())
    

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
        "UAE" : "United Arab Emirates",

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

        # map TO your canonical 197-country list
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
        "Hawaii" : "United States",
        "Puerto Rico": "United States",
        "America": "United States",
        
        "People’s Republic of China": "China",
        "England": "United Kingdom",
        'Kurdish area': "Turkey",
        "Great Britain":"United Kingdom",
        "California": "United States",
        "Wales": "United Kingdom",
        "England": "United Kingdom",
        "Republic of Moldova": "Moldova",
        "Lao": "Laos",
        "Hong Kong S.A.R.": "China",
        "Delaware": "United States",
        "Faroe Islands": "Denmark",
        "Türkiye": "Turkey",
        "Brunei Darussalam": "Brunei",
        "Syrian Arab Republic": "Syria",
        "Democratic Republic of Congo": "Democratic Republic of the Congo"

    }

    return alias_map.get(s, s)

# def normalize_country_token(token):
#     if token is None:
#         return None
#     s = str(token).strip().strip('"').strip("'")
#     s = re.sub(r"\s+", " ", s).strip()

#     alias_map = {
#         "USA": "United States",
#         "US": "United States",
#         "U.S.": "United States",
#         "U.S.A.": "United States",
#         "UK": "United Kingdom",
#         "U.K.": "United Kingdom",
#         "Russian Federation": "Russia",
#         "Czechia": "Czech Republic",
#         "Viet Nam": "Vietnam",
#         "Türkiye": "Turkey",
#         "Korea, South": "South Korea",
#         "Republic of Korea": "South Korea",
#         "Korea, North": "North Korea",
#         "Czechia": "Czech Republic",
#         "Congo (Congo-Brazzaville)": "Congo",
#         "Republic of the Congo": "Congo",
#         "São Tomé and Príncipe": "Sao Tome and Principe",
#         "Cote d Ivoire": "Cote d'Ivoire",
#         "Ivory Coast": "Cote d'Ivoire",
#     }
#     return alias_map.get(s, s)

import ast

def split_country_field(value):
    if is_missing(value):
        return []

    s = str(value).strip()

    items = []

    # Try to parse Python-list-like strings safely
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
        # normalize all separators before splitting
        item = item.replace("|", ",")
        item = item.replace(" / ", ",")
        item = item.replace(";", ",")

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

    seen = set()
    deduped = []
    for c in countries:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped

def choose_best_country(row):

    for col in COUNTRY_PRIORITY_COLS:
        if col in row and not is_missing(row[col]):
            countries = split_country_field(row[col])
            if countries:
                return col, countries
    return None, []

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

TASK_CANONICAL_LOOKUP = {
    canonicalize_for_matching(task): task
    for task in ALLOWED_TASK_CATEGORIES
}

TASK_ALIASES = {
    "human ai interaction cooperation human centered nlp":
        "Human-AI Interaction/Cooperation, Human Centered NLP",
    "information extraction retrieval and text mining":
        "Information Extraction, Retrieval, and Text Mining",
    "computational social science cultural analytics nlp for social good":
        "Computational Social Science, Cultural Analytics, NLP for social good",
    "efficient low resource methods for nlp llm efficiency":
        "Efficient/Low-Resource Methods for NLP, LLM Efficiency",
    "speech recognition text to speech and spoken language understanding":
        "Speech recognition, text-to-speech, and spoken language understanding",
    "semantics lexical and sentence level textual inference":
        "Semantics: Lexical and Sentence Level, Textual Inference",
    "multimodality and language grounding to vision robotics and beyond":
        "Multimodality and Language Grounding to Vision, Robotics and Beyond",
}

def normalize_task(raw_task):
    raw = clean_text(raw_task)
    if raw is None:
        return None, "missing"

    if raw in ALLOWED_TASK_SET:
        return raw, "exact"

    key = canonicalize_for_matching(raw)

    if key in TASK_ALIASES:
        return TASK_ALIASES[key], "alias"

    if key in TASK_CANONICAL_LOOKUP:
        return TASK_CANONICAL_LOOKUP[key], "normalized"

    return raw, "unmapped"

# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(INPUT_CSV).copy()
df["source_row_index"] = df.index

# =========================================================
# BEST COUNTRY SELECTION
# =========================================================

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

# =========================================================
# TASK NORMALIZATION
# =========================================================

norm_tasks = []
task_statuses = []

for val in df["Task Category"]:
    task, status = normalize_task(val)
    norm_tasks.append(task)
    task_statuses.append(status)

df["normalized_task_category"] = norm_tasks
df["task_normalization_status"] = task_statuses


# ==========================================
# DEBUG: TASK FILTER DROPS
# ==========================================

problem_rows = df[
    (df["has_valid_content_country"]) &
    (~df["normalized_task_category"].isin(ALLOWED_TASK_SET))
][["Dataset name", "Task Category", "normalized_task_category"]]

print(f"Rows with valid country but invalid task: {len(problem_rows)}")

print(problem_rows.head(10))

problem_rows.to_csv(OUTDIR / "invalid_task_rows.csv", index=False)

# =========================================================
# EXPAND TO DATASET-COUNTRY-TASK
# =========================================================

expanded_rows = []

for _, row in df.iterrows():
    countries = row["best_country_list_valid"]
    task = row["normalized_task_category"]
    dataset_name = clean_text(row.get("Dataset name"))

    if not countries:
        continue
    if task is None:
        continue
    if task not in ALLOWED_TASK_SET:
        continue
    if dataset_name is None:
        dataset_name = "Not stated"

    for country in countries:
        expanded_rows.append({
            "source_row_index": row["source_row_index"],
            "Dataset name": dataset_name,
            "Task Category": task,
            "country": country,
            "best_country_field_used": row["best_country_field_used"],
            "Country Attribution Method": clean_text(row.get("Country Attribution Method")),
            "Origin Type": clean_text(row.get("Origin Type")),
            "Multi Country?": clean_text(row.get("Multi Country?")),
            "paper_url": clean_text(row.get("paper_url")) or clean_text(row.get("Paper Link (URL)")),
        })

expanded_df = pd.DataFrame(expanded_rows)

if len(expanded_df) > 0:
    expanded_df = expanded_df.drop_duplicates(
        subset=["source_row_index", "Dataset name", "Task Category", "country"]
    ).reset_index(drop=True)

# =========================================================
# AGGREGATION
# =========================================================

country_task_long = (
    expanded_df
    .groupby(["country", "Task Category"], as_index=False)
    .agg(dataset_count=("source_row_index", "nunique"))
    .sort_values(["country", "Task Category"])
)

country_task_matrix = (
    country_task_long
    .pivot(index="country", columns="Task Category", values="dataset_count")
    .fillna(0)
    .astype(int)
)

country_totals = (
    expanded_df
    .groupby("country", as_index=False)
    .agg(total_datasets=("source_row_index", "nunique"))
    .sort_values("total_datasets", ascending=False)
)

task_totals = (
    expanded_df
    .groupby("Task Category", as_index=False)
    .agg(total_datasets=("source_row_index", "nunique"))
    .sort_values("total_datasets", ascending=False)
)

task_review = (
    df[["Task Category", "normalized_task_category", "task_normalization_status"]]
    .drop_duplicates()
    .sort_values(["task_normalization_status", "Task Category"])
)

invalid_country_review = (
    df.loc[
        df["invalid_countries_dropped"].apply(len) > 0,
        ["Dataset name", "paper_url", "best_country_field_used", "best_country_list_raw", "invalid_countries_dropped"]
    ]
    .copy()
)

sample_rows = expanded_df.head(10)

# =========================================================
# STATS
# =========================================================

total_rows = len(df)
rows_with_valid_country = int(df["has_valid_content_country"].sum())
rows_missing_valid_country = total_rows - rows_with_valid_country
rows_with_unmapped_tasks = int((df["task_normalization_status"] == "unmapped").sum())
rows_with_valid_country_and_valid_task = int(
    ((df["has_valid_content_country"]) & (df["normalized_task_category"].isin(ALLOWED_TASK_SET))).sum()
)

stats_text = f"""
WorldBench country x task matrix stats
=====================================

Input rows: {total_rows}

Rows with valid content-side country: {rows_with_valid_country}
Rows missing valid content-side country: {rows_missing_valid_country}

Rows with unmapped task labels: {rows_with_unmapped_tasks}
Rows contributing to matrix: {rows_with_valid_country_and_valid_task}

Expanded dataset-country-task rows: {len(expanded_df)}
Unique countries in matrix: {expanded_df['country'].nunique() if len(expanded_df) else 0}
Unique tasks in matrix: {expanded_df['Task Category'].nunique() if len(expanded_df) else 0}

Top 10 countries by dataset count:
{country_totals.head(10).to_string(index=False) if len(country_totals) else 'None'}

Top 10 tasks by dataset count:
{task_totals.head(10).to_string(index=False) if len(task_totals) else 'None'}
""".strip()


# ==========================================
# MISSING COUNTRIES
# ==========================================

observed_countries = set(expanded_df["country"].unique())
missing_countries = VALID_COUNTRIES - observed_countries

missing_countries_df = pd.DataFrame({
    "country": sorted(missing_countries)
})

missing_countries_df.to_csv(OUTDIR / "missing_countries.csv", index=False)

print(f"Missing countries: {len(missing_countries)}")
print(sorted(missing_countries))



# ==========================================
# LOW COVERAGE COUNTRIES
# ==========================================

low_coverage = country_totals[country_totals["total_datasets"] <= 5]

low_coverage.to_csv(OUTDIR / "low_coverage_countries.csv", index=False)

print(f"Countries with ≤5 datasets: {len(low_coverage)}")



# ==========================================
# DROP REASON ANALYSIS
# ==========================================

df["has_any_country"] = df["best_country_list_raw"].apply(lambda x: len(x) > 0)
df["has_valid_country"] = df["best_country_list_valid"].apply(lambda x: len(x) > 0)
df["has_valid_task"] = df["normalized_task_category"].isin(ALLOWED_TASK_SET)
df["has_dataset_name"] = df["Dataset name"].notna()

drop_analysis = pd.DataFrame({
    "total_rows": [len(df)],
    "has_any_country": [df["has_any_country"].sum()],
    "has_valid_country": [df["has_valid_country"].sum()],
    "has_valid_task": [df["has_valid_task"].sum()],
    "final_rows_used": [len(expanded_df["source_row_index"].unique())],
})

print(drop_analysis)

# More detailed breakdown
df["drop_reason"] = "kept"

df.loc[~df["has_any_country"], "drop_reason"] = "missing_country"
df.loc[(df["has_any_country"]) & (~df["has_valid_country"]), "drop_reason"] = "invalid_country"
df.loc[(df["has_valid_country"]) & (~df["has_valid_task"]), "drop_reason"] = "invalid_task"

drop_counts = df["drop_reason"].value_counts()

print(drop_counts)

drop_counts.to_csv(OUTDIR / "drop_reason_counts.csv")

# ===================================================
# Missing Tasks
# ===================================================

missing_tasks = set(ALLOWED_TASK_CATEGORIES) - set(expanded_df["Task Category"].unique())

print("Missing tasks:", missing_tasks)

pd.DataFrame({"missing_tasks": list(missing_tasks)}).to_csv(
    OUTDIR / "missing_tasks.csv", index=False
)


invalid_examples = df.loc[
    df["invalid_countries_dropped"].apply(len) > 0,
    ["Dataset name", "best_country_list_raw", "invalid_countries_dropped"]
].head(30)

#print(invalid_examples.to_string(index=False))


# =========================================================
# SAVE
# =========================================================

expanded_df.to_csv(OUTPUT_EXPANDED, index=False)
country_task_long.to_csv(OUTPUT_LONG, index=False)
country_task_matrix.to_csv(OUTPUT_MATRIX)
country_totals.to_csv(OUTPUT_COUNTRY_TOTALS, index=False)
task_totals.to_csv(OUTPUT_TASK_TOTALS, index=False)
task_review.to_csv(OUTPUT_TASK_REVIEW, index=False)
invalid_country_review.to_csv(OUTPUT_INVALID_COUNTRIES, index=False)
sample_rows.to_csv(OUTPUT_SAMPLE, index=False)

with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
    f.write(stats_text)

print(stats_text)
print("\nSaved outputs to:", OUTDIR.resolve())
