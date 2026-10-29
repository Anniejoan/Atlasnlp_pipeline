# =============================================================================
# country_second_pass.py — LLM second pass to recover missing country metadata
#
# Paper section: §3.3 "Automated Dataset Expansion"
#   "To improve coverage, we perform a second-pass extraction for entries with
#    incomplete or missing geographic metadata."
# Role in pipeline: Step 6 — re-processes papers that have no country attribution
#   after the first LLM pass. Downloads PDFs again and prompts GPT-4o-mini with
#   a schema focused specifically on geographic attribution.
#
# Sub-pipeline within country_second_pass/:
#   1. fix_input.py              — prepare and fix the input CSV
#   2. second_pass_sharding.py   — split into shards for parallel runs
#   3. run_country_second_pass.sh — SLURM/local runner for each shard
#   4. THIS SCRIPT               — the actual LLM second-pass extractor
#   5. merge_second_pass.py      — merge shard outputs into updated_master.csv
#
# Run from country_second_pass/:
#   python country_second_pass.py --input <shard.csv> --out <run_dir/> ...
#
# Input:  Rows with missing country metadata from the master CSV
# Output: Updated country assignments in country_second_pass_merged/updated_master.csv
# =============================================================================
import os
import io
import re
import csv
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader

# =============================
# Config
# =============================

USER_AGENT = "Mozilla/5.0"
REQUEST_TIMEOUT = 60
MODEL_DEFAULT = "gpt-4o-mini"
PROMPT_VERSION_DEFAULT = "country_second_pass_v1"

# =============================
# Helpers
# =============================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}

def is_missing_or_not_stated(x: Any) -> bool:
    if is_blank(x):
        return True
    return str(x).strip().lower() == "not stated"

def join_pages_with_markers(pages: List[str], page_numbers: List[int] = None) -> str:
    out = []
    if page_numbers is None:
        page_numbers = list(range(1, len(pages) + 1))
    for pnum, text in zip(page_numbers, pages):
        out.append(f"\n\n=== PAGE {pnum} ===\n{text}")
    return "".join(out).strip()

# =============================
# Manifest loading
# =============================

def load_manifest(path: str) -> List[Dict[str, Any]]:
    p = Path(path)

    if p.suffix.lower() == ".csv":
        items = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"paper_id", "url"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV must have columns: {sorted(required)}")
            for row in reader:
                if not row.get("paper_id") or not row.get("url"):
                    continue
                items.append(dict(row))
        return items

    if p.suffix.lower() == ".jsonl":
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("paper_id") and obj.get("url"):
                    items.append(obj)
        return items

    raise ValueError("Manifest must be .csv or .jsonl")

# =============================
# PDF fetching & extraction
# =============================
def looks_like_pdf_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False


def to_pdf_url(url: str) -> str:
    if not url:
        return url

    url = url.strip()

    if "aclanthology.org" in url and not url.lower().endswith(".pdf"):
        return url.rstrip("/") + ".pdf"

    return url


def fetch_pdf(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    pdf_url = to_pdf_url(url)

    r = requests.get(pdf_url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    r.raise_for_status()

    content = r.content

    if not content.startswith(b"%PDF-"):
        preview = content[:80]
        raise ValueError(
            f"Downloaded content is not a PDF. Header starts with: {preview!r}. URL used: {pdf_url}"
        )

    return content

def extract_text_pages_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        pages.append(t.strip())
    return pages

# =============================
# OpenAI structured call
# =============================

def call_openai_structured(model: str, system: str, user: str, schema: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from openai import OpenAI

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
                "name": "CountrySecondPass",
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

# =============================
# Prompt builder
# =============================

def build_country_second_pass_prompt(
    row: Dict[str, Any],
    full_text: str
) -> Tuple[str, str]:
    system = (
        "You are helping clean a global NLP dataset registry.\n"
        "Your task is ONLY to recover COUNTRY coverage metadata conservatively.\n\n"
        "Rules:\n"
        "- Do not guess.\n"
        "- Prefer explicit evidence from the paper text.\n"
        "- Return ALL COUNTRIES COVERED as a STRING containing COUNTRY NAMES ONLY.\n"
        "- If one country is identified, return that country name.\n"
        "- If multiple countries are identified, return them as a semicolon-separated string, e.g. 'Nigeria; Ghana; Kenya'.\n"
        "- If the paper mentions only languages and does not explicitly identify countries, return 'Not stated'.\n"
        "- Do NOT put language names into all_countries_covered.\n"
        "- Do NOT infer countries from widely spoken languages such as English, Arabic, Chinese, Spanish, Portuguese, or French.\n"
        "- For country_attribution_method, use exactly one of:\n"
        "  * Explicit country focus\n"
        "  * Multi-country coverage\n"
        "  * Inferred by language\n"
        "  * Not stated\n"
        "- Use 'Inferred by language' only when the language-country tie is strong and unambiguous, and there is no better explicit evidence.\n"
        "- If the evidence is only language-level and no defensible country can be assigned, return 'Not stated' for both fields.\n"
        "- Every non-'Not stated' field must have supporting evidence with page number and short quote.\n"
        "- Output valid JSON only.\n"
    )

    user = (
        "Return JSON with this structure:\n"
        "{paper:{...}, country_recovery:{...}, evidence:[...], run:{...}}\n\n"
        f"Paper ID: {row['paper_id']}\n"
        f"Paper URL: {row['url']}\n"
        f"Dataset name: {row.get('Dataset name', '')}\n"
        f"Short description: {row.get('Short description about dataset', '')}\n"
        f"Languages: {row.get('Languages covered (separate using semicolon)', '')}\n"
        f"Candidate country from language: {row.get('Candidate Country From Language', '')}\n"
        f"Current all countries: {row.get('All Countries Covered (you can select multiple countries)', '')}\n"
        f"Current attribution method: {row.get('Country Attribution Method', '')}\n"
        f"Validation flags: {row.get('Validation Flags', '')}\n"
        f"Review tier: {row.get('Review Tier', '')}\n\n"
        "Important:\n"
        "- all_countries_covered must contain country names only, not languages.\n"
        "- If only languages are mentioned, return 'Not stated'.\n\n"
        "Paper text with page markers:\n"
        f"{full_text}\n"
    )

    return system, user

# =============================
# Validation for second pass
# =============================

def normalize_country_recovery(obj: Dict[str, Any]) -> Dict[str, Any]:
    cr = obj.setdefault("country_recovery", {})

    countries = cr.get("all_countries_covered", "Not stated")
    method = cr.get("country_attribution_method", "Not stated")

    if is_blank(countries):
        countries = "Not stated"
    else:
        countries = str(countries).strip()

    # If the model put languages in the country field, force to Not stated
    if countries != "Not stated" and looks_like_language_list(countries):
        countries = "Not stated"
        method = "Not stated"

    cr["all_countries_covered"] = countries
    cr["country_attribution_method"] = method if not is_blank(method) else "Not stated"

    if is_blank(cr.get("rationale")):
        cr["rationale"] = ""

    obj.setdefault("paper", {})
    obj.setdefault("evidence", [])
    obj.setdefault("run", {})
    return obj

LANGUAGE_NAME_BLOCKLIST = {
    "english", "arabic", "chinese", "indonesian", "french", "spanish",
    "portuguese", "german", "swahili", "yoruba", "igbo", "hausa",
    "hindi", "tamil", "bengali", "urdu", "russian", "japanese",
    "korean", "turkish", "persian", "farsi", "dutch", "italian",
    "polish", "ukrainian", "thai", "vietnamese", "malay", "indonesian",
    "amharic", "somali", "zulu", "xhosa", "catalan", "greek",
    "hebrew", "latin", "multilingual"
}

AMBIGUOUS_LANGUAGE_NAMES = {
    "english", "arabic", "chinese", "spanish", "portuguese", "french", "german"
}

def split_semicolon_values(s: str) -> list[str]:
    if is_blank(s):
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]

def looks_like_language_list(s: str) -> bool:
    parts = split_semicolon_values(s)
    if not parts:
        return False

    lowered = [p.lower() for p in parts]
    hits = sum(1 for p in lowered if p in LANGUAGE_NAME_BLOCKLIST)
    return hits > 0

def contains_ambiguous_language_name(s: str) -> bool:
    parts = split_semicolon_values(s)
    lowered = [p.lower() for p in parts]
    return any(p in AMBIGUOUS_LANGUAGE_NAMES for p in lowered)

def validate_country_recovery(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    flags = []
    cr = obj.get("country_recovery", {})

    countries = cr.get("all_countries_covered", "Not stated")
    method = cr.get("country_attribution_method", "Not stated")

    if is_blank(countries):
        countries = "Not stated"

    if countries == "Not stated" and method != "Not stated":
        flags.append("METHOD_WITHOUT_COUNTRY")

    country_parts = split_semicolon_values(countries) if countries != "Not stated" else []

    if len(country_parts) > 1 and method not in {"Multi-country coverage", "Inferred by language"}:
        flags.append("MULTI_COUNTRY_METHOD_MISMATCH")

    if len(country_parts) == 1 and method == "Multi-country coverage":
        flags.append("SINGLE_COUNTRY_WITH_MULTI_METHOD")

    # New checks: reject obvious language outputs
    if countries != "Not stated" and looks_like_language_list(countries):
        flags.append("LANGUAGE_NAMES_IN_COUNTRY_FIELD")

    # Stronger check: ambiguous languages should not be used as countries
    if countries != "Not stated" and contains_ambiguous_language_name(countries):
        flags.append("AMBIGUOUS_LANGUAGE_USED_AS_COUNTRY")

    return (len(flags) == 0), sorted(set(flags))

# =============================
# Main processing
# =============================

def process_manifest(
    manifest: List[Dict[str, Any]],
    out_dir: str,
    schema: Dict[str, Any],
    model: str,
    prompt_version: str,
    max_items: int = None,
    sleep_s: float = 0.0
) -> None:
    ensure_dir(out_dir)
    pdf_dir = os.path.join(out_dir, "pdfs")
    pages_dir = os.path.join(out_dir, "pages")
    json_dir = os.path.join(out_dir, "json")
    ensure_dir(pdf_dir)
    ensure_dir(pages_dir)
    ensure_dir(json_dir)

    results_jsonl = os.path.join(out_dir, "country_second_pass_results.jsonl")
    usage_jsonl = os.path.join(out_dir, "country_second_pass_usage.jsonl")
    summary_csv = os.path.join(out_dir, "country_second_pass_summary.csv")

    fieldnames = [
        "paper_id",
        "url",
        "dataset_name",
        "all_countries_covered",
        "country_attribution_method",
        "rationale",
        "ok",
        "flags"
    ]

    n_total = len(manifest if max_items is None else manifest[:max_items])

    with open(summary_csv, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(manifest[:max_items] if max_items else manifest, start=1):
            paper_id = str(row["paper_id"])
            url = str(row["url"])

            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
            pdf_path = os.path.join(pdf_dir, f"{safe_id}.pdf")
            pages_path = os.path.join(pages_dir, f"{safe_id}.pages.json")
            out_json_path = os.path.join(json_dir, f"{safe_id}.country_second_pass.json")

            try:
                if os.path.exists(out_json_path):
                    obj = load_json(out_json_path)
                    ok, flags = validate_country_recovery(obj)
                    cr = obj["country_recovery"]
                    writer.writerow({
                        "paper_id": paper_id,
                        "url": url,
                        "dataset_name": row.get("Dataset name", ""),
                        "all_countries_covered": json.dumps(cr.get("all_countries_covered", "Not stated"), ensure_ascii=False),
                        "country_attribution_method": cr.get("country_attribution_method", "Not stated"),
                        "rationale": cr.get("rationale", ""),
                        "ok": ok,
                        "flags": "; ".join(flags)
                    })
                    print(f"[{idx}/{n_total}] {paper_id}: already processed", flush=True)
                    continue

                # PDF
                if os.path.exists(pdf_path) and looks_like_pdf_file(pdf_path):
                    print(f"Using cached PDF: {pdf_path}")
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                else:
                    if os.path.exists(pdf_path):
                        print(f"Cached file is invalid, deleting and re-downloading: {pdf_path}")
                        try:
                            os.remove(pdf_path)
                        except OSError:
                            pass

                    print(f"Downloading PDF for {paper_id} from {to_pdf_url(url)}")
                    pdf_bytes = fetch_pdf(url)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_bytes)



                # Pages
                if os.path.exists(pages_path):
                    pages = load_json(pages_path).get("pages", [])
                else:
                    pages = extract_text_pages_from_pdf(pdf_bytes)
                    save_json(pages_path, {"pages": pages})

                full_text = join_pages_with_markers(pages)

                system, user = build_country_second_pass_prompt(row, full_text)
                obj, usage = call_openai_structured(model, system, user, schema)

                obj = normalize_country_recovery(obj)
                obj["paper"]["paper_id"] = paper_id
                obj["paper"]["paper_url"] = url
                obj["run"] = {
                    "model": model,
                    "prompt_version": prompt_version
                }

                ok, flags = validate_country_recovery(obj)
                obj["validation"] = {"ok": ok, "flags": flags}

                save_json(out_json_path, obj)
                save_jsonl_line(results_jsonl, obj)
                save_jsonl_line(usage_jsonl, {
                    "paper_id": paper_id,
                    **usage
                })

                cr = obj["country_recovery"]
                writer.writerow({
                    "paper_id": paper_id,
                    "url": url,
                    "dataset_name": row.get("Dataset name", ""),
                    "all_countries_covered": json.dumps(cr.get("all_countries_covered", "Not stated"), ensure_ascii=False),
                    "country_attribution_method": cr.get("country_attribution_method", "Not stated"),
                    "rationale": cr.get("rationale", ""),
                    "ok": ok,
                    "flags": "; ".join(flags)
                })

                print(f"[{idx}/{n_total}] {paper_id}: done", flush=True)

                if sleep_s > 0:
                    time.sleep(sleep_s)

            except Exception as e:
                writer.writerow({
                    "paper_id": paper_id,
                    "url": url,
                    "dataset_name": row.get("Dataset name", ""),
                    "all_countries_covered": "",
                    "country_attribution_method": "",
                    "rationale": f"ERROR: {e}",
                    "ok": False,
                    "flags": "EXCEPTION"
                })
                print(f"[{idx}/{n_total}] {paper_id}: ERROR -> {e}", flush=True)

# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser(description="Focused country second-pass recovery pipeline.")
    parser.add_argument("--manifest", required=True, help="CSV/JSONL with paper_id,url and supporting row fields")
    parser.add_argument("--schema", required=True, help="Path to country second-pass schema")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--prompt_version", default=PROMPT_VERSION_DEFAULT)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    manifest = load_manifest(args.manifest)
    if not manifest:
        raise RuntimeError("Manifest is empty")

    process_manifest(
        manifest=manifest,
        out_dir=args.out,
        schema=schema,
        model=args.model,
        prompt_version=args.prompt_version,
        max_items=args.max_items,
        sleep_s=args.sleep_s
    )

if __name__ == "__main__":
    main()