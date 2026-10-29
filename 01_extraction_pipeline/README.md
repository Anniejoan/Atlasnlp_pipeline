# Stage 1 — Extraction Pipeline

Turns the 20,277 NLI-filtered candidate papers into structured dataset
records, three parallel tracks: automated Core extraction, human-curated
Gold, and author-affiliation-based producer country. See
`../PIPELINE_REPORT.md` for the full narrative and numbers — this file is
just the "what's here and how to run it" reference.

Original repo location: `/home/jnwatu/Worldbench/pipeline/` (core
extraction, country second pass) and `/home/jnwatu/Worldbench/compiled_ACL/`
and `author_affiliation/` subtrees (producer country), plus the Gold
collection scripts in the repo root. Not copied here: the input
`data/manifest.jsonl` (53MB — the 20,277-row candidate list; original path
`/home/jnwatu/Worldbench/data/manifest.jsonl`) and all raw per-shard LLM
outputs/caches.

## `core_extraction/`

The main two-stage GPT-4o-mini extraction that turned 20,277 candidates
into 18,035 initial Core records (905 role-triaged out, 1,337 failed).

1. `split_manifest.py` — splits `manifest.jsonl` into N shards for a SLURM
   array.
2. `run_shard.sh` — SLURM array submission script; each task runs
   `metadata_extractor.py` over one shard.
3. `metadata_extractor.py` — the extractor itself: downloads each paper's
   PDF, extracts text, runs a regex role-triage (introduces / extends /
   reuses a dataset), sends eligible papers to GPT-4o-mini for structured
   extraction against `schema.json`, and retries schema-invalid responses
   with a second, more targeted prompt.
4. `merge_jsonl_shards.py` / `merge_tsv_shards.py` — combine the per-shard
   outputs into the single merged Core file.

`schema.json` defines the required fields and types for a valid extraction
record — read this to understand exactly what fields Core carries at this
stage (note: the content-country field here is the one later found to be
unreliable in stage 4 and replaced in stage 5).

## `gold_human_curation/`

The parallel human-curated reference set (original 1,661 entries).

- `human_review_splitting.py` / `make_reviewer_batches.py` — assign papers
  to reviewers by country of expertise.
- `rebuild_and_balance_reviewer_batches_machinexhuman.py` — the
  cross-validation stage: shuffles and rebalances batches so a second
  reviewer checks each entry.
- `human_compiled.csv` — raw contributor submissions feeding into this
  process (not the final Gold set — that's assembled in stage 3).

## `country_second_pass/`

A targeted follow-up LLM pass for Core records that came out of
`core_extraction/` missing country attribution. `second_pass_sharding.py`
/ `run_country_second_pass.sh` shard and submit the job;
`country_second_pass.py` does the extraction against
`country_second_pass_schema.json`; `fix_input.py` prepares inputs;
`merge_second_pass.py` folds results back into the main Core file.

## `author_affiliation_producer_country/`

Independent of content-country: extracts each paper's author institutional
affiliations and maps them to producer countries.

- `extract_author_institutions_regex.py` / `clean_author_institutions_regex.py`
  — regex-based affiliation parsing (fast path, most papers).
- `prepare_author_affiliation_llm_fallback.py` /
  `prepare_author_affiliation_llm_fallback_v2.py` /
  `run_author_affiliation_llm_fallback.py` — LLM fallback for affiliations
  the regex pass can't resolve.
- `make_author_affiliation_shards.py` / `combine_author_affiliation_shards.py`
  — SLURM sharding/merging for the LLM fallback.
- `merge_author_affiliation_regex_llm.py` — combines regex + LLM-fallback
  results into one table.
- `apply_country_extraction_fix.py` — a targeted correction pass (see the
  original repo's commit history / `pipeline_tracker.md` in stage 2 for
  what this fixed).
- `aggregate_author_affiliation_diagnostics.py` /
  `compute_worldbench_country_coverage.py` — diagnostic/coverage summaries.
- `author_affiliation_all_merged_corrected.csv` — the final merged,
  corrected producer-country table carried forward into later stages.

Not copied: the intermediate regex-only and LLM-fallback-only candidate
CSVs (each 10-40MB) — only the final merged/corrected output is kept here.
They're still in the original repo's `author_affiliation/` directory if
needed.

## Where this hands off

Stage 1's outputs (initial 18,035-row Core, 1,661-row Gold candidate pool,
producer-country table) feed directly into stage 2
(`../02_normalization/`) for language/task/country cleanup before any
paper figures are built.
