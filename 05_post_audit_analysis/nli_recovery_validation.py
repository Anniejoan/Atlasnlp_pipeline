# =============================================================================
# nli_recovery_validation.py — Reproduce the reviewer-response NLI-stage /
# end-to-end recovery numbers (rebuttal report, Issue 5 / W3).
#
# This is a PAPER-LEVEL recall/coverage check, distinct from Table 1
# (validation_0.py), which is an ENTRY-LEVEL (dataset-pair) accuracy check.
# See README note at the bottom of this file for why the two differ.
#
# Reproduces:
#   425 unique AtlasNLP-Gold papers with ACL Anthology URLs
#   377 (88.7%) of those pass the NLI threshold (appear in data/manifest.jsonl)
#   372 (87.5%) of those appear in AtlasNLP-Core (18,035 rows)
#
# Run from anywhere:
#   python nli_recovery_validation.py
#
# Inputs:
#   compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/human_worldbench.csv
#   data/manifest.jsonl   — 20,277 NLI-passing candidates (0.5 entailment threshold)
#   compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv
#
# Output:
#   nli_recovery_validation_summary.json
#   nli_recovery_gold_papers_in_core.csv      — the 372 Gold ACL papers that
#     DO appear in Core (the overlap set), one row per unique paper.
#   nli_recovery_gold_papers_not_in_core.csv  — the 53 Gold ACL papers that
#     did not make it into Core (5 didn't pass NLI, 48 passed NLI but were
#     later rejected/unmatched during extraction), for manual inspection.
# =============================================================================
import csv
import json
from pathlib import Path

REPO_ROOT = Path("/home/jnwatu/Worldbench")
GOLD_PATH = REPO_ROOT / "compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/human_worldbench.csv"
MANIFEST_PATH = REPO_ROOT / "data/manifest.jsonl"
CORE_PATH = REPO_ROOT / "compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv"
OUT_DIR = Path(__file__).parent


def normalize_url(x):
    """Lowercase, strip whitespace, strip trailing '/', strip trailing '.pdf'.

    Gold mixes both 'https://aclanthology.org/X/' and
    'https://aclanthology.org/X.pdf' styles for the same paper; without
    stripping both suffixes, naive matching undercounts unique papers and
    misses cross-style matches against Core/manifest.
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s == "nan":
        return None
    s = s.rstrip("/")
    if s.endswith(".pdf"):
        s = s[:-4]
    return s or None


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    gold_rows = load_csv(GOLD_PATH)
    core_rows = load_csv(CORE_PATH)

    # ---- Gold: unique ACL-Anthology papers ----
    gold_acl_urls = set()
    gold_url_to_rows = {}
    for r in gold_rows:
        u = normalize_url(r.get("Paper Link (URL)", ""))
        if u and "aclanthology.org" in u:
            gold_acl_urls.add(u)
            gold_url_to_rows.setdefault(u, []).append(r)

    # ---- NLI manifest: 20,277 entailment-passing candidates ----
    manifest_urls = set()
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            u = normalize_url(d.get("url"))
            if u:
                manifest_urls.add(u)

    # ---- Core: 18,035 rows ----
    core_urls = set()
    core_url_to_rows = {}
    for r in core_rows:
        u = normalize_url(r.get("paper_url") or r.get("Paper Link (URL)", ""))
        if u:
            core_urls.add(u)
            core_url_to_rows.setdefault(u, []).append(r)

    pass_nli = gold_acl_urls & manifest_urls
    in_core = gold_acl_urls & core_urls
    not_in_core = gold_acl_urls - core_urls

    n_gold = len(gold_acl_urls)
    n_pass_nli = len(pass_nli)
    n_in_core = len(in_core)

    summary = {
        "unique_gold_acl_papers": n_gold,
        "pass_nli_threshold": n_pass_nli,
        "pass_nli_threshold_pct": round(n_pass_nli / n_gold * 100, 1),
        "appear_in_core": n_in_core,
        "appear_in_core_pct": round(n_in_core / n_gold * 100, 1),
        "nli_passing_but_absent_from_core": n_pass_nli - len(pass_nli & in_core),
        "not_recovered_total": n_gold - n_in_core,
    }

    print(json.dumps(summary, indent=2))

    with open(OUT_DIR / "nli_recovery_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save the not-recovered papers (failed NLI, or passed NLI but dropped
    # during extraction/dedup) for manual review.
    not_recovered_fields = ["paper_url_normalized", "passed_nli", "dataset_name", "paper_link_raw"]
    with open(OUT_DIR / "nli_recovery_gold_papers_not_in_core.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=not_recovered_fields)
        writer.writeheader()
        for u in sorted(not_in_core):
            for r in gold_url_to_rows[u]:
                writer.writerow({
                    "paper_url_normalized": u,
                    "passed_nli": u in manifest_urls,
                    "dataset_name": r.get("Dataset name", ""),
                    "paper_link_raw": r.get("Paper Link (URL)", ""),
                })

    # Save the 372 Gold ACL papers that DO appear in Core (the overlap set),
    # one row per unique paper, with Gold- and Core-side dataset names listed
    # so multi-dataset papers are still visible without exploding to entry level.
    overlap_fields = [
        "paper_url_normalized",
        "passed_nli",
        "gold_dataset_names",
        "gold_dataset_count",
        "core_dataset_names",
        "core_source_row_indices",
        "core_dataset_count",
        "paper_link_raw",
    ]
    with open(OUT_DIR / "nli_recovery_gold_papers_in_core.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overlap_fields)
        writer.writeheader()
        for u in sorted(in_core):
            gold_side = gold_url_to_rows[u]
            core_side = core_url_to_rows.get(u, [])
            writer.writerow({
                "paper_url_normalized": u,
                "passed_nli": u in manifest_urls,
                "gold_dataset_names": "; ".join(sorted({r.get("Dataset name", "") for r in gold_side})),
                "gold_dataset_count": len(gold_side),
                "core_dataset_names": "; ".join(sorted({r.get("Dataset name", "") for r in core_side})),
                "core_source_row_indices": "; ".join(str(r.get("source_row_index", "")) for r in core_side),
                "core_dataset_count": len(core_side),
                "paper_link_raw": gold_side[0].get("Paper Link (URL)", ""),
            })

    print(f"\nSaved: {OUT_DIR / 'nli_recovery_validation_summary.json'}")
    print(f"Saved: {OUT_DIR / 'nli_recovery_gold_papers_in_core.csv'}")
    print(f"Saved: {OUT_DIR / 'nli_recovery_gold_papers_not_in_core.csv'}")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Why this differs from Table 1 (validation_0.py, 460-entry aligned set):
#
# Table 1 is an ENTRY-level (dataset-pair) accuracy check: it inner-joins every
# Gold row against every Core row on paper_url, keeping one row per matching
# (Gold-dataset, Core-dataset) pair. A paper with multiple Gold-listed datasets
# and multiple Core-listed datasets contributes multiple rows for that one
# paper, so the 460 rows span only 286 unique papers. It answers: "given a
# dataset was found by both a human and the pipeline for the same paper, how
# well does the pipeline's extracted country/task label match the human's?"
#
# This script is a PAPER-level recall/coverage check: Gold is deduplicated to
# 425 unique papers with ACL Anthology URLs first, then each paper is checked
# for membership in the NLI manifest and in Core, with no regard for how many
# datasets that paper contributes on either side. It answers: "of the papers a
# human confirmed contain a real dataset, what fraction survived NLI screening
# and ended up in Core at all?" — a question about pipeline recall, not
# post-recovery metadata accuracy.
#
# A secondary, smaller contributor to the count difference: validation_0.py's
# join only does .str.strip().lower() on URLs, without stripping the trailing
# '/' or '.pdf' that Gold mixes across rows for the same paper — this script's
# stricter normalize_url() catches a few extra cross-style matches that
# validation_0.py's naive join misses.
# -----------------------------------------------------------------------------
