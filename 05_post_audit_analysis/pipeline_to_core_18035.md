# From the ACL Anthology to WorldBench-Core: how the 18,035 records were built

This document covers only the **beginning-to-18,035-records phase** — from the raw ACL
Anthology export down to the finished WorldBench-Core collection. It does not cover
anything downstream of Core (language cleanup, Tier A eligibility, Gold merging, the
post-audit content-country re-extraction, etc.) — those are separate, later phases.

Two pipelines chain together here, living in two different places:

1. **Candidate filtering** (produces `data/manifest.jsonl`) — runs in a separate repo,
   `acl_anthology_paper_finder`, not present in this repo or on this machine. Reconstructed
   from a provenance report generated against that repo on 2026-08-26.
2. **Metadata extraction** (`data/manifest.jsonl` → Core) — runs entirely in this repo,
   under `pipeline/`, and is fully inspectable here.

---

## Phase 1 — Candidate filtering (`acl_anthology_paper_finder`, external repo)

Six stages narrow the full ACL Anthology down to a set of papers likely to introduce a
dataset. Counts below are exact, read directly from the files in that repo.

| # | Script | Output file | Rows | What happens |
|---|--------|-------------|------|--------------|
| 1 | `anthology_bib_to_jsonl.py` | `anthology_full.jsonl` | 119,963 | Parses `anthology.bib` (149MB ACL BibTeX export) with `bibtexparser`; one JSON object per entry, every BibTeX field kept as-is (including whole-volume proceedings entries). |
| 2 | `anthology_trimmer.py` | `anthology_slim.jsonl` | 119,963 | Keeps 6 fields (`ID`, `title`, `abstract`, `year`, `booktitle`, `url`), adds a `has_abstract` bool, concatenates `title + abstract` into a `text` field. |
| 3 | (filter step) | `has_abstract.jsonl` | 71,847 | Keeps only entries with a non-empty abstract. The other 43,058 (no abstract, not a conference placeholder) are set aside in `no_abstract_not_conference.jsonl` and drop out of this funnel entirely. |
| 4 | `anthology_dataset_or_not.py` | `has_abstract_dataset_status.jsonl` | 71,847 (scored) | Every abstract is run through `tasksource/ModernBERT-base-nli` as zero-shot entailment: premise = title+abstract, hypothesis = *"The paper proposes a dataset."* Each row gets `has_dataset_status` (entailment / neutral / contradiction) and `has_dataset_score`. |
| 5a | (same script) | `may_have_dataset.jsonl` | 23,674 | Any row labeled `entailment`, regardless of score — the loose pool. |
| 5b | (same script) | `has_dataset.jsonl` | 20,277 | `entailment` **and** `has_dataset_score > 0.5` — the 0.5-threshold cut the rest of the pipeline treats as "papers likely describing a dataset." |
| 6 | (copy) | `manifest.jsonl` | 20,277 | Byte-for-byte identical to `has_dataset.jsonl` (verified via sorted diff), just renamed to mark the handoff into stage 2. Every row's `has_dataset_status` is `"entailment"` by construction — the field carries no further filtering signal once you're reading `manifest.jsonl` itself. |

`manifest.jsonl` was then copied into this repo as `/home/jnwatu/Worldbench/data/manifest.jsonl`,
which is where Phase 2 picks it up.

**Note on run history:** an earlier extraction script in that same external repo,
`pilot_metadata_extractor.py` (internally named `end2end_extract.py`), had only been run on
3 pilot batches of 20 papers each (60 papers, 2 failures) at the time `manifest.jsonl` was
handed off — the full 20,277-row manifest was never processed end-to-end *there*. That
script is the same lineage as the one that actually did the full run (see Phase 2) — it was
carried into this repo and matured/renamed, not abandoned.

---

## Phase 2 — LLM metadata extraction (this repo, `pipeline/`)

This is the stage that actually produced WorldBench-Core. Paper section: §3.3, "Automated
Dataset Expansion."

### Step 1 — Split the manifest into shards
`pipeline/split_manifest.py data/manifest.jsonl <shards_dir> 600` — chunks the 20,277-row
manifest into `manifest_000.jsonl`, `manifest_001.jsonl`, ... for parallel processing.

### Step 2 — Parallel extraction via SLURM job array
`pipeline/run_shard.sh` (`sbatch`, `--array=1-20`, account `<PI_ACCOUNT>`, 16h walltime,
16GB/4cpu per task) launches one array task per shard. Each task runs:

```
python pipeline/metadata_extractor.py \
  --manifest manifest_NNN.jsonl \
  --schema pipeline/schema.json \
  --out shard_NNN/
```

`pipeline/metadata_extractor.py` (docstring: *"originally end2end_extract.py"*) does this
**per paper**:

1. **Resolve & fetch** — rewrite the ACL Anthology landing URL to its direct `.pdf` link
   (`resolve_pdf_url`) and download it.
2. **Extract text** page-by-page with `pypdf`.
3. **Role triage** (`classify_dataset_role`) — regex cues over the full text decide whether
   the paper *introduces* a new dataset, *reuses/compiles* existing ones, or is unclear.
   A strong "new dataset" signal always wins (safety guard against false negatives); absent
   that, a compilation/existing-dataset signal routes the paper to be **skipped** and logged
   to `dataset_compilation_papers.jsonl` — never sent to the LLM at all. This is the
   pipeline's second, sharper filter on top of the upstream NLI cut.
4. **Stage 1 extraction** — the full paper text goes to **GPT-4o-mini** with a strict JSON
   schema (`pipeline/schema.json`): dataset name, task category (31-way enum), languages,
   `is_dataset_multilingual`, license, `origin_type`, synthetic status, `multi_country`,
   `countries_covered`, `country_attribution_method`, and more. Every field requires a
   page + quote evidence citation (the `evidence` array in the schema).
5. **Validate & triage** (`validate_extraction`) — checks schema compliance and internal
   consistency (e.g. "Monolingual" with two listed languages); sets `needs_stage2 = true`
   on anything that fails.
6. **Stage 2 repair** (conditional) — flagged records go back to **GPT-4o-mini** with just
   the relevant page snippets, returning a corrected full record (`merge_stage2`).
7. **Emit** — final JSON cached per paper (`items/<paper_id>.final.json`), appended to the
   shard's `results.jsonl`, flattened into `results.tsv` (17 columns via `json_to_csv_row`),
   with token usage to `usage.jsonl` and schema failures to `failures.jsonl`.

### Step 3 — Merge shards
After all 20 array tasks complete:
- `pipeline/merge_jsonl_shards.py` → concatenates every shard's `results.jsonl`,
  `failures.jsonl`, `usage.jsonl`, `dataset_compilation_papers.jsonl` into
  `compiled_ACL/merged_*.jsonl`.
- `pipeline/merge_tsv_shards.py` → concatenates every shard's `results.tsv` into
  `compiled_ACL/merged_results.tsv` (header written once), the base for downstream
  validation/triage work.

### The final accounting (verified directly from `compiled_ACL/merged_*` files)

| Outcome | File | Rows |
|---|---|---|
| Successfully extracted → **WorldBench-Core** | `merged_results.jsonl` | **18,035** |
| Failed extraction (schema/fetch/other failure) | `merged_failures.jsonl` | 1,337 |
| Skipped as reuse/compilation (role triage, never sent to LLM) | `merged_dataset_compilation_papers.jsonl` | 905 |
| **Total (must equal manifest size)** | | 1,337 + 905 + 18,035 = **20,277** ✓ |

This 1,337 + 905 = **2,242** matches the reviewer-response report's stated figure exactly:
*"20,277 to 18,035: ... 2,242 NLI-passing candidates did not yield an identifiable new
dataset during extraction; 1,244 are LREC 2006–2014 papers."*
(`merged_usage.jsonl` has 24,946 rows, not 20,277 — it logs one entry per LLM call, and
records that needed stage-2 repair contribute a second call.)

---

## Summary funnel (source → Core)

```
anthology.bib (149MB)              119,963
  → anthology_full.jsonl           119,963   (bib parsed to JSON)
  → anthology_slim.jsonl           119,963   (trimmed to 6 fields)
  → has_abstract.jsonl              71,847   (has a non-empty abstract)
  → may_have_dataset.jsonl          23,674   (NLI: any entailment label)
  → has_dataset.jsonl / manifest    20,277   (NLI: entailment & score > 0.5)
      ├─ dataset_compilation_papers    905   (regex role-triage: reuse/compilation, skipped)
      ├─ failures                    1,337   (schema fail / fetch fail / other)
      └─ WorldBench-Core            18,035   (successfully extracted, GPT-4o-mini 2-stage)
```

## Sources
- Pipeline-report PDF for the `acl_anthology_paper_finder` repo, generated 2026-08-26
  (counts read directly from files in that repo; not re-verifiable from this machine since
  that repo isn't present here).
- `pipeline/split_manifest.py`, `pipeline/run_shard.sh`, `pipeline/metadata_extractor.py`,
  `pipeline/merge_jsonl_shards.py`, `pipeline/merge_tsv_shards.py`, `pipeline/schema.json`
  (all in this repo, read directly).
- `compiled_ACL/merged_results.jsonl`, `merged_failures.jsonl`,
  `merged_dataset_compilation_papers.jsonl`, `merged_usage.jsonl` row counts (verified
  directly, `wc -l`, on 2026-08-26).
- `atlasNLP_reviewer_response_report_v9_corrected.pdf`, Appendix A.5 / Issue 5, for the
  2,242 / 1,244-LREC cross-check.
