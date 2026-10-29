"""
metadata_extractor.py  (originally end2end_extract.py)

End-to-end two-stage GPT-4o-mini dataset-metadata extraction pipeline.

Paper section: §3.3 "Automated Dataset Expansion"
Role in pipeline: Step 3 — the core LLM extraction engine run on each SLURM shard.

Per-paper processing:
  1) Download PDF from ACL Anthology
  2) Extract text page-by-page
  3) Stage 1: LLM extracts structured metadata from full text (GPT-4o-mini)
  4) Schema validation + triage (flags issues, sets needs_stage2=True when needed)
  5) Stage 2 (conditional): LLM re-runs on relevant snippets to fix flagged issues
  6) Write per-paper cache (.pdf, .pages.json, .stage1.json, .final.json)
     and append to shard-level results.jsonl + results.tsv

Output schema (pipeline/schema.json) captures:
  - dataset name, task_category, languages, countries_covered (content side)
  - country_attribution_method, origin_type
  - multimodal, multiple_choice_format, synthetic
  - dataset link, license, short description

Run a local pilot (20 papers):
  export OPENAI_API_KEY="sk-..."
  python pipeline/metadata_extractor.py \\
    --manifest data/manifest_test.jsonl \\
    --schema pipeline/schema.json \\
    --out pilot_test_run \\
    --max_items 20

Run in production via SLURM:
  See pipeline/run_shard.sh
"""

from __future__ import annotations

import os
import re
import io
import csv
import json
import time
import hashlib
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import requests
from pypdf import PdfReader
from jsonschema import validate as jsonschema_validate


# -----------------------------
# Defaults / Config
# -----------------------------

MODEL_STAGE1_DEFAULT = "gpt-4o-mini"
MODEL_STAGE2_DEFAULT = "gpt-4o-mini"
PROMPT_VERSION_DEFAULT = "v1.0"

USER_AGENT = "paper-metadata-extractor/1.0"
REQUEST_TIMEOUT = 60

ANCHORS = {
    "dataset": re.compile(r"\b(dataset|benchmark|corpus|we (introduce|present|release)|data( set)? collection)\b", re.I),
    "link": re.compile(r"(github\.com|huggingface\.co|zenodo\.org|doi\.org|arxiv\.org)", re.I),
    "license": re.compile(r"\b(license|licence|CC BY|Creative Commons|MIT License|Apache License|GPL|ODC)\b", re.I),
    "appendix": re.compile(r"\bappendix\b", re.I),
}

LINK_HINT = re.compile(r"(github\.com|huggingface\.co|zenodo\.org|doi\.org)", re.I)
LICENSE_HINT = re.compile(r"\b(license|licence|CC BY|Creative Commons|MIT License|Apache License|GPL|ODC)\b", re.I)
DATASET_HINT = re.compile(r"\b(we introduce|we present|we release)\b.*\b(dataset|benchmark|corpus)\b", re.I)

ALLOWED_TASK_CATEGORIES = [
    "Code Models",
    "Computational Social Science, Cultural Analytics, NLP for Social Good",
    "Dialogue and Interactive Systems",
    "Discourse, Pragmatics, and Reasoning",
    "Efficient / Low-Resource Methods for NLP, LLM Efficiency",
    "Ethics, Bias, and Fairness",
    "Generalizability and Transfer",
    "Hierarchical Structure Prediction, Syntax, and Parsing",
    "Human-AI Interaction / Cooperation, Human-Centered NLP",
    "Information Extraction, Retrieval, and Text Mining",
    "Interpretability, Model Analysis, Transparency, and Explainability",
    "Language Modeling",
    "LLM Agents",
    "Linguistic Theories, Cognitive Modeling, and Psycholinguistics",
    "Machine Learning for NLP",
    "Machine Translation",
    "Mathematical, Symbolic, and Logical Reasoning in NLP",
    "Multilinguality and Language Diversity",
    "Multimodality and Language Grounding to Vision, Robotics, and Beyond",
    "Natural Language Generation",
    "Neurosymbolic Approaches to NLP",
    "NLP Applications",
    "Phonology, Morphology, and Word Segmentation",
    "Question Answering",
    "Retrieval-Augmented Language Models",
    "Safety and Alignment in LLMs",
    "Semantics: Lexical and Sentence Level, Textual Inference",
    "Sentiment Analysis, Stylistic Analysis, and Argument Mining",
    "Speech Recognition, Text-to-Speech, and Spoken Language Understanding",
    "Summarization",
    "Syntax: Tagging, Chunking, and Parsing",
]
ALLOWED_TASK_CATEGORIES_SET = set(ALLOWED_TASK_CATEGORIES)

CSV_FIELDS = [
    "Dataset name",
    "Task Category",
    "Country Attribution Method",
    "Year",
    "Languages covered",
    "Is dataset multilingual",
    "Dataset link",
    "Paper link",
    "Multimodal",
    "Multiple Choice Format?",
    "Short description",
    "License",
    "Origin Type",
    "Synthetic?",
    "Synthetic Notes",
    "Multi-Country?",
    "Countries covered",
]

NEW_DATASET_HINT = re.compile(
    r"\b("
    r"we introduce (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we present (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we release (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we construct (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we create (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we build (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we collect(ed)? (a|an|the)?\s*(new )?(dataset|corpus)|"
    r"we annotate(d)? (a|an|the)?\s*(new )?(dataset|corpus)|"
    r"our (new )?(dataset|benchmark|corpus)|"
    r"this paper introduces (a|an|the)?\s*(new )?(dataset|benchmark|corpus)|"
    r"we propose (a|an|the)?\s*(new )?(dataset|benchmark|corpus)"
    r")\b",
    re.I
)

EXISTING_DATASET_HINT = re.compile(
    r"\b("
    r"we evaluate on|"
    r"we experiment on|"
    r"we use the following datasets|"
    r"we use publicly available datasets|"
    r"existing datasets|"
    r"public datasets|"
    r"standard benchmarks|"
    r"benchmarks include|"
    r"datasets used|"
    r"we report results on"
    r")\b",
    re.I
)

COMPILATION_HINT = re.compile(
    r"\b("
    r"we compile|"
    r"we aggregate|"
    r"we collect existing datasets|"
    r"we unify existing datasets|"
    r"we combine datasets|"
    r"we benchmark on|"
    r"across multiple datasets|"
    r"across existing datasets"
    r")\b",
    re.I
)

# Extra safety guard:
# catches patterns like "we introduce XYZ, a new dataset"
NEW_DATASET_NAME_PATTERN = re.compile(
    r"\b("
    r"introduce|present|release|propose"
    r")\b.{0,120}\b([A-Z][A-Za-z0-9\-\_]+)\b.{0,40}\b(dataset|benchmark|corpus)\b",
    re.I | re.S
)


# -----------------------------
# Basic IO helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def join_pages_with_markers(pages: List[str], page_numbers: Optional[List[int]] = None) -> str:
    out = []
    if page_numbers is None:
        page_numbers = list(range(1, len(pages) + 1))
    for pno, txt in zip(page_numbers, pages):
        out.append(f"=== PAGE {pno} ===\n{txt}")
    return "\n\n".join(out)


# -----------------------------
# Manifest loading
# -----------------------------

def load_manifest(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                paper_id = obj.get("paper_id") or obj.get("ID")
                url = obj.get("url")

                if not paper_id or not url:
                    raise ValueError(f"JSONL line {line_no} missing ID/paper_id or url")

                items.append({
                    "paper_id": str(paper_id),
                    "url": str(url),
                    # keep extras if you want
                    "year": str(obj.get("year")) if obj.get("year") is not None else "",
                    "title": str(obj.get("title")) if obj.get("title") is not None else "",
                    "abstract": str(obj.get("abstract")) if obj.get("abstract") is not None else "",
                    "text": str(obj.get("text")) if obj.get("text") is not None else "",
                })
        return items

    # CSV loading with columns paper_id,url (case-insensitive)
    ...


    if path.lower().endswith(".csv"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "paper_id" not in reader.fieldnames or "url" not in reader.fieldnames:
                raise ValueError("CSV must have columns: paper_id,url")
            for row in reader:
                if not row.get("paper_id") or not row.get("url"):
                    continue
                items.append({"paper_id": str(row["paper_id"]), "url": str(row["url"])})
        return items

    raise ValueError("Manifest must be .jsonl or .csv")


# -----------------------------
# PDF fetching & extraction
# -----------------------------

def fetch_pdf(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

def extract_text_pages_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        pages.append(t.strip())
    return pages


# -----------------------------
# Snippet selection
# -----------------------------

def select_relevant_pages(
    pages: List[str],
    window: int = 1,
    max_pages: int = 8
) -> Tuple[List[int], str]:
    hits = set()
    for idx, text in enumerate(pages, start=1):
        if any(rx.search(text) for rx in ANCHORS.values()):
            for j in range(max(1, idx - window), min(len(pages), idx + window) + 1):
                hits.add(j)

    if not hits:
        hits = set(range(1, min(len(pages), 4) + 1))

    selected = sorted(hits)[:max_pages]
    snippet_text = join_pages_with_markers([pages[i - 1] for i in selected], page_numbers=selected)
    return selected, snippet_text


# -----------------------------
# Validation & triage
# -----------------------------

@dataclass
class ValidationResult:
    ok: bool
    flags: List[str]
    needs_stage2: bool

def validate_extraction(obj: Dict[str, Any], schema: Dict[str, Any], full_text: str) -> ValidationResult:
    flags: List[str] = []
    needs_stage2 = False

    FIELDS_REQUIRE_EVIDENCE_ALWAYS = {
        "dataset_name",
        "task_category",
        "short_description",
        "dataset_link",
        "license",
        "country_attribution_method",
        "origin_type",
    }

    FIELDS_REQUIRE_EVIDENCE_ONLY_IF_YES = {
        "multimodal",
        "multiple_choice_format",
        "synthetic",
        "multi_country",
    }

    try:
        jsonschema_validate(instance=obj, schema=schema)
    except Exception:
        return ValidationResult(ok=False, flags=["SCHEMA_FAIL"], needs_stage2=True)

    dataset = obj.get("dataset", {})
    evidence = obj.get("evidence", [])

    evidence_fields = {
        e.get("field")
        for e in evidence
        if isinstance(e, dict) and e.get("field")
    }

    # Evidence checks:
    # 1) Always require evidence for key scalar fields
    # 2) Require evidence for Yes/positive classification fields only when value == "Yes"
    # 3) Do NOT require evidence for list fields here (languages_covered, countries_covered),
    #    because the model may support them indirectly and we handle consistency separately.
    for k, v in dataset.items():
        if isinstance(v, str):
            if v == "Not stated":
                continue

            if k in FIELDS_REQUIRE_EVIDENCE_ALWAYS and k not in evidence_fields:
                flags.append(f"EVIDENCE_MISSING:{k}")

            if k in FIELDS_REQUIRE_EVIDENCE_ONLY_IF_YES and v == "Yes" and k not in evidence_fields:
                flags.append(f"EVIDENCE_MISSING:{k}")

    if any(f.startswith("EVIDENCE_MISSING:") for f in flags):
        needs_stage2 = True

    # Task category
    tc = dataset.get("task_category", "Not stated")
    if tc == "Not stated":
        flags.append("MISSING_TASK_CATEGORY")
        needs_stage2 = True
    elif tc not in ALLOWED_TASK_CATEGORIES_SET:
        flags.append("TASK_CATEGORY_NOT_ALLOWED")
        needs_stage2 = True

    # Dataset name with hint
    dn = dataset.get("dataset_name", "Not stated")
    if dn == "Not stated" and DATASET_HINT.search(full_text):
        flags.append("MISSING_DATASET_NAME")
        needs_stage2 = True

    # Link/license hints
    # Link hint: keep as a flag, but do NOT automatically trigger Stage 2.
    # We already have regex fallback link extraction in process_manifest().
    if dataset.get("dataset_link", "Not stated") == "Not stated" and LINK_HINT.search(full_text):
        flags.append("LINK_PRESENT_BUT_NOT_EXTRACTED")

    # License hint: usually keep as a flag only.
    # Many papers mention licensing generally without giving a usable dataset license.
    if dataset.get("license", "Not stated") == "Not stated" and LICENSE_HINT.search(full_text):
        flags.append("LICENSE_KEYWORDS_PRESENT")

    # Language consistency
    langs = dataset.get("languages_covered", [])
    ml = dataset.get("is_dataset_multilingual", "Not stated")
    if isinstance(langs, list):
        langs = [x for x in langs if isinstance(x, str) and x.strip()]

    if ml == "Monolingual" and isinstance(langs, list) and len(langs) > 1:
        flags.append("INCONSISTENT_LANGUAGES")
        needs_stage2 = True

    if ml == "Multilingual" and isinstance(langs, list) and len(langs) == 1:
        flags.append("INCONSISTENT_LANGUAGES")
        needs_stage2 = True

    # Country consistency
    countries = dataset.get("countries_covered", [])
    mc = dataset.get("multi_country", "Not stated")
    if isinstance(countries, list):
        countries = [x for x in countries if isinstance(x, str) and x.strip()]

    if mc == "No" and isinstance(countries, list) and len(countries) > 1:
        flags.append("INCONSISTENT_COUNTRIES")
        needs_stage2 = True

    if mc == "Yes" and isinstance(countries, list) and len(countries) == 1:
        flags.append("INCONSISTENT_COUNTRIES")
        needs_stage2 = True

    return ValidationResult(
        ok=(len(flags) == 0),
        flags=sorted(set(flags)),
        needs_stage2=needs_stage2
    )

# -----------------------------
# OpenAI calls (JSON mode)
# -----------------------------

def call_openai_structured(model: str, system: str, user: str, schema: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from openai import OpenAI
    import os, json

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "DatasetMetadataExtraction",
                "schema": schema,
                "strict": True
            }
        },
        temperature=0
    )

    content = resp.choices[0].message.content
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", None),
        "output_tokens": getattr(resp.usage, "completion_tokens", None),
        "total_tokens": getattr(resp.usage, "total_tokens", None),
    }
    return json.loads(content), usage



# -----------------------------
# Prompt builders
# -----------------------------

def build_stage1_prompts(paper_id: str, paper_url: str, full_text: str, prompt_version: str) -> Tuple[str, str]:
    system = (
        "You extract dataset metadata from the provided paper text conservatively and audibly.\n"
        "Rules:\n"
        "- Do not guess. Do not infer beyond explicit statements.\n"
        "- If a field is not explicitly stated in the paper text, output \"Not stated\".\n"
        "- Every non-\"Not stated\" field MUST have an evidence item with page number and a short quote (<= 20 words).\n"
        "- Output MUST be valid JSON only.\n"
        "- If multiple datasets are described, choose the primary dataset introduced by the paper; otherwise flag ambiguity.\n"
        "- All required fields must be present. If a required field is not explicitly stated, output \"Not stated\".\n"
    )

    user = (
        "Return JSON with this structure:\n"
        "{paper:{...}, dataset:{...}, evidence:[...], quality:{flags:[], needs_stage2:boolean}, run:{...}}\n\n"
        "Allowed task categories (choose exactly one):\n"
        + "; ".join(ALLOWED_TASK_CATEGORIES)
        + "\n\n"
        f"Paper ID: {paper_id}\n"
        f"Paper URL: {paper_url}\n"
        f"Prompt version: {prompt_version}\n\n"
        "Paper text with page markers:\n"
        f"{full_text}\n"
    )
    return system, user

def build_stage2_prompts(
    stage1_json: Dict[str, Any],
    snippet_text: str
) -> Tuple[str, str]:

    system = (
        "You correct and complete a previous dataset metadata extraction.\n"
        "Rules:\n"
        "- You MUST output the FULL JSON object matching the required schema.\n"
        "- Do NOT output a patch.\n"
        "- Do not guess. If a field is not explicitly supported by the snippets, keep it as \"Not stated\".\n"
        "- Every non-\"Not stated\" dataset field MUST have an evidence item (page + <=20 word quote).\n"
        "- Do not remove existing correct evidence unless it is wrong.\n"
        "- The final output MUST match the schema exactly.\n"
    )

    user = (
        "Previous JSON (Stage 1 result):\n"
        f"{json.dumps(stage1_json, ensure_ascii=False)}\n\n"
        "Relevant snippets with page markers:\n"
        f"{snippet_text}\n\n"
        "Return the FULL corrected JSON object matching the schema."
    )

    return system, user


def merge_stage2(stage1: Dict[str, Any], stage2: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(stage1))  # deep copy
    upd = stage2.get("dataset_updates", {}) or {}
    if isinstance(upd, dict):
        out["dataset"].update(upd)

    ev_add = stage2.get("evidence_additions", []) or []
    if isinstance(ev_add, list):
        out["evidence"].extend(ev_add)

    q_upd = stage2.get("quality_updates", {}) or {}
    out.setdefault("quality", {})
    if isinstance(q_upd, dict):
        if "flags" in q_upd and isinstance(q_upd["flags"], list):
            out["quality"]["flags"] = q_upd["flags"]
        out["quality"]["needs_stage2"] = bool(q_upd.get("needs_stage2", False))
    else:
        out["quality"]["needs_stage2"] = False

    return out

def classify_dataset_role(full_text: str) -> Tuple[str, str]:
    has_new = bool(NEW_DATASET_HINT.search(full_text))
    has_existing = bool(EXISTING_DATASET_HINT.search(full_text))
    has_compilation = bool(COMPILATION_HINT.search(full_text))
    has_named_new_dataset = bool(NEW_DATASET_NAME_PATTERN.search(full_text))

    # Safety guard: if there is any strong sign of a newly introduced dataset, do NOT skip
    if has_new or has_named_new_dataset:
        return "New dataset proposed", "matched_new_dataset_signal"

    # Only skip when there is evidence of existing/compiled datasets
    # and no evidence of a new dataset
    if has_compilation:
        return "Existing datasets used", "matched_compilation_hint"

    if has_existing:
        return "Existing datasets used", "matched_existing_dataset_hint"

    return "Unclear", "no_clear_signal"


# -----------------------------
# CSV conversion
# -----------------------------

def json_to_csv_row(obj: Dict[str, Any]) -> Dict[str, str]:
    d = obj.get("dataset", {})
    p = obj.get("paper", {})

    def join_list(x):
        if isinstance(x, list):
            vals = [str(v).strip() for v in x if str(v).strip()]
            return ", ".join(vals) if vals else "Not stated"
        if isinstance(x, str) and x.strip():
            return x.strip()
        return "Not stated"

    return {
        "Dataset name": d.get("dataset_name", "Not stated"),
        "Task Category": d.get("task_category", "Not stated"),
        "Country Attribution Method": d.get("country_attribution_method", "Not stated"),
        "Year": p.get("year", "Not stated"),
        "Languages covered": join_list(d.get("languages_covered", [])),
        "Is dataset multilingual": d.get("is_dataset_multilingual", "Not stated"),
        "Dataset link": d.get("dataset_link", "Not stated"),
        "Paper link": p.get("paper_link_in_text", p.get("paper_url", "Not stated")),
        "Multimodal": d.get("multimodal", "Not stated"),
        "Multiple Choice Format?": d.get("multiple_choice_format", "Not stated"),
        "Short description": d.get("short_description", "Not stated"),
        "License": d.get("license", "Not stated"),
        "Origin Type": d.get("origin_type", "Not stated"),
        "Synthetic?": d.get("synthetic", "Not stated"),
        "Synthetic Notes": d.get("synthetic_notes", "Not stated"),
        "Multi-Country?": d.get("multi_country", "Not stated"),
        "Countries covered": join_list(d.get("countries_covered", [])),
    }


# -----------------------------
# Main processing
# -----------------------------

def resolve_pdf_url(url: str) -> str:
    # ACL Anthology landing -> PDF
    if url.startswith("https://aclanthology.org/") and not url.lower().endswith(".pdf"):
        if url.endswith("/"):
            url = url[:-1]
        return url + ".pdf"
    return url


URL_RE = re.compile(r'https?://[^\s\]\)\}>,"]+')

LANGUAGE_MAP = {
    "ASL": "American Sign Language",
    "BSL": "British Sign Language",
    "ISL": "Indian Sign Language",
    "GSL": "Greek Sign Language",
    "LSA": "Argentine Sign Language",
}

COUNTRY_MAP = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "U.S.": "United States",
    "USA": "United States",
    "US": "United States",
}

def normalize_dataset_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj.setdefault("dataset", {})
    d = obj["dataset"]

    # Normalize languages
    langs = d.get("languages_covered", [])
    if isinstance(langs, list):
        cleaned_langs = []
        for x in langs:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x:
                continue
            x = LANGUAGE_MAP.get(x, x)
            cleaned_langs.append(x)

        # deduplicate while preserving order
        seen = set()
        deduped_langs = []
        for x in cleaned_langs:
            if x not in seen:
                seen.add(x)
                deduped_langs.append(x)

        d["languages_covered"] = deduped_langs

        if d.get("is_dataset_multilingual") == "Not stated":
            if len(deduped_langs) == 1:
                d["is_dataset_multilingual"] = "Monolingual"
            elif len(deduped_langs) > 1:
                d["is_dataset_multilingual"] = "Multilingual"

    # Normalize countries
    countries = d.get("countries_covered", [])
    if isinstance(countries, list):
        cleaned_countries = []
        for c in countries:
            if not isinstance(c, str):
                continue
            c = c.strip()
            if not c:
                continue
            c = COUNTRY_MAP.get(c, c)
            cleaned_countries.append(c)

        seen = set()
        deduped_countries = []
        for c in cleaned_countries:
            if c not in seen:
                seen.add(c)
                deduped_countries.append(c)

        d["countries_covered"] = deduped_countries

        if d.get("multi_country") == "Not stated":
            if len(deduped_countries) == 1:
                d["multi_country"] = "No"
            elif len(deduped_countries) > 1:
                d["multi_country"] = "Yes"

    obj["dataset"] = d
    return obj

def extract_candidate_dataset_link(text: str) -> str:
    urls = URL_RE.findall(text)
    preferred = []

    for u in urls:
        ul = u.lower().rstrip(".,);:")
        if "huggingface.co/datasets" in ul:
            preferred.append(u.rstrip(".,);:"))
        elif "zenodo.org" in ul:
            preferred.append(u.rstrip(".,);:"))
        elif "github.com" in ul:
            preferred.append(u.rstrip(".,);:"))
        elif "drive.google.com" in ul:
            preferred.append(u.rstrip(".,);:"))

    return preferred[0] if preferred else "Not stated"

def get_role_evidence_snippet(full_text: str) -> str:
    for pattern in [COMPILATION_HINT, EXISTING_DATASET_HINT, NEW_DATASET_HINT]:
        m = pattern.search(full_text)
        if m:
            start = max(0, m.start() - 80)
            end = min(len(full_text), m.end() + 80)
            return full_text[start:end].replace("\n", " ")
    return ""

def process_manifest(
    manifest: List[Dict[str, str]],
    out_dir: str,
    schema: Dict[str, Any],
    model_stage1: str,
    model_stage2: str,
    prompt_version: str,
    max_items: Optional[int] = None,
    sleep_s: float = 0.0,
) -> None:
    ensure_dir(out_dir)
    items_dir = os.path.join(out_dir, "items")
    ensure_dir(items_dir)

    results_jsonl = os.path.join(out_dir, "results.jsonl")
    failures_jsonl = os.path.join(out_dir, "failures.jsonl")
    results_csv = os.path.join(out_dir, "results.tsv")
    usage_jsonl = os.path.join(out_dir, "usage.jsonl")
    compilations_jsonl = os.path.join(out_dir, "dataset_compilation_papers.jsonl")

    csv_exists = os.path.exists(results_csv)
    csv_f = open(results_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    if not csv_exists:
        writer.writeheader()

    n_total = len(manifest) if max_items is None else min(len(manifest), max_items)
    shard_start_time = time.time()

    for idx, item in enumerate(manifest[:n_total], start=1):
        paper_start_time = time.time()
        paper_id = str(item["paper_id"])
        landing_url = str(item["url"])
        pdf_url = resolve_pdf_url(landing_url)

        manifest_year = str(item.get("year", "Not stated")) if item.get("year") else "Not stated"
        manifest_title = str(item.get("title", "Not stated")) if item.get("title") else "Not stated"

        base = os.path.join(items_dir, paper_id)
        pdf_path = base + ".pdf"
        pages_path = base + ".pages.json"
        stage1_path = base + ".stage1.json"
        final_path = base + ".final.json"

        skipped_compilations = 0

        try:
            if os.path.exists(final_path):
                final_obj = load_json(final_path)
                save_jsonl_line(results_jsonl, final_obj)
                writer.writerow(json_to_csv_row(final_obj))
                print(f"[{idx}/{n_total}] {paper_id}: already finalized", flush=True)
                continue

            # 1) Download PDF
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
            else:
                pdf_bytes = fetch_pdf(pdf_url)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

            # 2) Extract text pages
            if os.path.exists(pages_path):
                pages = load_json(pages_path).get("pages", [])
                if not isinstance(pages, list):
                    pages = []
            else:
                pages = extract_text_pages_from_pdf(pdf_bytes)
                save_json(pages_path, {"pages": pages})

            full_text = join_pages_with_markers(pages)

            # 2.5) Check dataset role in paper based on text cues (for triage)
            dataset_role, dataset_role_reason = classify_dataset_role(full_text)

            if dataset_role == "Existing datasets used":
                skipped_compilations += 1
                evidence_snippet = get_role_evidence_snippet(full_text[:5000])

                save_jsonl_line(compilations_jsonl, {
                    "paper_id": paper_id,
                    "paper_url": pdf_url,
                    "paper_link_in_text": landing_url,
                    "year": manifest_year,
                    "title": manifest_title,
                    "dataset_role": dataset_role,
                    "reason": dataset_role_reason,
                    "evidence_snippet": evidence_snippet
                })

                print(f"[{idx}/{n_total}] {paper_id}: skipped (existing datasets)", flush=True)
                continue

            # 3) Stage 1
            if os.path.exists(stage1_path):
                stage1 = load_json(stage1_path)
                usage1 = {}
            else:
                s1_sys, s1_user = build_stage1_prompts(paper_id, pdf_url, full_text, prompt_version)
                stage1, usage1 = call_openai_structured(model_stage1, s1_sys, s1_user, schema)

                save_jsonl_line(usage_jsonl, {
                    "paper_id": paper_id,
                    "stage": 1,
                    **usage1
                })

                stage1.setdefault("paper", {})
                stage1.setdefault("dataset", {})
                stage1.setdefault("evidence", [])
                stage1.setdefault("quality", {})
                stage1.setdefault("run", {})

                # Inject canonical fields required by schema
                stage1["paper"]["paper_id"] = paper_id
                stage1["paper"]["paper_url"] = pdf_url
                stage1["paper"]["paper_link_in_text"] = landing_url
                stage1["paper"]["year"] = manifest_year
                stage1["paper"]["title"] = manifest_title

                stage1["run"] = {
                    "model": model_stage1,
                    "prompt_version": prompt_version
                }

                stage1 = normalize_dataset_fields(stage1)
                save_json(stage1_path, stage1)

            # normalize cached stage1 too
            stage1 = normalize_dataset_fields(stage1)

            # Fallback link extraction after Stage 1
            if stage1["dataset"].get("dataset_link") == "Not stated":
                guessed_link = extract_candidate_dataset_link(full_text)
                if guessed_link != "Not stated":
                    stage1["dataset"]["dataset_link"] = guessed_link

            # 4) Validate + triage
            v1 = validate_extraction(stage1, schema, full_text)
            stage1.setdefault("quality", {})
            stage1["quality"]["flags"] = v1.flags
            stage1["quality"]["needs_stage2"] = v1.needs_stage2
            stage1["quality"]["dataset_role"] = dataset_role

            final_obj = stage1

            # 5) Stage 2 if needed
            if v1.needs_stage2:
                selected_pages, snippet_text = select_relevant_pages(pages, window=1, max_pages=15)

                s2_sys, s2_user = build_stage2_prompts(stage1, snippet_text)

                stage2, usage2 = call_openai_structured(
                    model_stage2,
                    s2_sys,
                    s2_user,
                    schema
                )

                save_jsonl_line(usage_jsonl, {
                    "paper_id": paper_id,
                    "stage": 2,
                    **usage2
                })

                stage2.setdefault("paper", {})
                stage2.setdefault("dataset", {})
                stage2.setdefault("evidence", [])
                stage2.setdefault("quality", {})
                stage2.setdefault("run", {})

                stage2["paper"]["paper_id"] = paper_id
                stage2["paper"]["paper_url"] = pdf_url
                stage2["paper"]["paper_link_in_text"] = landing_url
                stage2["paper"]["year"] = manifest_year
                stage2["paper"]["title"] = manifest_title

                stage2["run"] = {
                    "model": model_stage2,
                    "prompt_version": prompt_version
                }

                stage2 = normalize_dataset_fields(stage2)

                # Fallback link extraction after Stage 2
                if stage2["dataset"].get("dataset_link") == "Not stated":
                    guessed_link = extract_candidate_dataset_link(snippet_text)
                    if guessed_link != "Not stated":
                        stage2["dataset"]["dataset_link"] = guessed_link

                final_obj = stage2

                v2 = validate_extraction(final_obj, schema, full_text)
                final_obj.setdefault("quality", {})
                final_obj["quality"]["flags"] = v2.flags
                final_obj["quality"]["needs_stage2"] = v2.needs_stage2

            if "SCHEMA_FAIL" in final_obj.get("quality", {}).get("flags", []):
                save_jsonl_line(failures_jsonl, {
                    "paper": final_obj.get("paper", {"paper_id": paper_id, "paper_url": pdf_url}),
                    "error": "Schema fail after Stage 2",
                    "quality": final_obj.get("quality", {}),
                    "raw": final_obj
                })
                continue

            # 6) Save final + emit
            save_json(final_path, final_obj)
            save_jsonl_line(results_jsonl, final_obj)
            writer.writerow(json_to_csv_row(final_obj))
            paper_elapsed = time.time() - paper_start_time
            shard_elapsed = time.time() - shard_start_time
            avg_per_paper = shard_elapsed / idx
            remaining = avg_per_paper * (n_total - idx)

            print(
                f"[{idx}/{n_total}] {paper_id}: done "
                f"(needs_stage2={final_obj.get('quality', {}).get('needs_stage2', False)}) "
                f"| paper={paper_elapsed:.1f}s "
                f"| shard_elapsed={shard_elapsed/60:.1f}m "
                f"| avg={avg_per_paper:.1f}s/paper "
                f"| eta={remaining/60:.1f}m", flush=True
            )

            if sleep_s > 0:
                time.sleep(sleep_s)
        
        except Exception as e:
            fail = {
                "paper": {"paper_id": paper_id, "paper_url": pdf_url},
                "error": str(e),
                "run": {"model": model_stage1, "prompt_version": prompt_version},
            }
            save_jsonl_line(failures_jsonl, fail)
            paper_elapsed = time.time() - paper_start_time
            shard_elapsed = time.time() - shard_start_time
            avg_per_paper = shard_elapsed / idx
            remaining = avg_per_paper * (n_total - idx)

            print(
                f"[{idx}/{n_total}] {paper_id}: FAILED: {e} "
                f"| paper={paper_elapsed:.1f}s "
                f"| shard_elapsed={shard_elapsed/60:.1f}m "
                f"| avg={avg_per_paper:.1f}s/paper "
                f"| eta={remaining/60:.1f}m"
            , flush=True)
    if idx % 50 == 0:
        processed_papers = n_total - skipped_compilations
        print(f"Progress: {idx}/{n_total} | processed_papers={processed_papers} | skipped_compilations={skipped_compilations}", flush=True)
    total_elapsed = time.time() - shard_start_time
    print(f"Shard complete: {n_total} papers in {total_elapsed/60:.1f} minutes ({total_elapsed/n_total:.1f}s per paper)", flush=True)
    
    csv_f.close()
    


# -----------------------------
# CLI entrypoint
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage dataset metadata extractor (GPT-4o-mini).")
    parser.add_argument("--manifest", required=True, help="Path to manifest (.jsonl or .csv) with paper_id,url")
    parser.add_argument("--schema", required=True, help="Path to JSON schema file (schema.json)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model_stage1", default=MODEL_STAGE1_DEFAULT)
    parser.add_argument("--model_stage2", default=MODEL_STAGE2_DEFAULT)
    parser.add_argument("--prompt_version", default=PROMPT_VERSION_DEFAULT)
    parser.add_argument("--max_items", type=int, default=None, help="Limit number of items (pilot)")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="Sleep between papers (rate limiting)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set.\n"
            "macOS/Linux: export OPENAI_API_KEY='sk-...'\n"
            "Windows PowerShell: setx OPENAI_API_KEY 'sk-...' (then open new terminal)"
        )

    ensure_dir(args.out)

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    manifest = load_manifest(args.manifest)
    if not manifest:
        raise RuntimeError("Manifest is empty.")

    process_manifest(
        manifest=manifest,
        out_dir=args.out,
        schema=schema,
        model_stage1=args.model_stage1,
        model_stage2=args.model_stage2,
        prompt_version=args.prompt_version,
        max_items=args.max_items,
        sleep_s=args.sleep_s,
    )
    

if __name__ == "__main__":
    main()
