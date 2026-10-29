# Stage 4 — Audit

A 400-record human audit of the initial (pre-audit) Core collection that
found the content-country field was correct only 32% of the time and
outright incorrect 46.8% of the time. This is the finding that triggered
the evidence-gated re-extraction in stage 5. See `../PIPELINE_REPORT.md`
for the full write-up. Original repo location:
`/home/jnwatu/Worldbench/Audit/`.

This folder is intentionally flat (no run order) — it's a record of what
was found, not a pipeline to re-run.

## Files

- `atlasnlp_audit_sampling_frame.csv` / `atlasnlp_audit_stratum_counts.csv`
  — how the 400-record stratified sample was drawn from the full Core
  collection.
- `atlasnlp_stage1_random400_preaudit.csv` — the 400 sampled records as
  they looked *before* audit (i.e., the original extraction's claims).
- `"Audit book.ipynb"` — the manual review notebook: each record's PDF was
  pulled up and the claimed content-country checked against it.
- `atlasnlp_stage1_random400_ai_audit_completed_400.csv` — the completed
  audit results, one row per sampled record, with the reviewer's
  correct/incorrect judgment and notes.
- `audit_log.txt` — the narrative findings log. **Read this first.** It
  documents the four recurring failure modes found (affiliation mistaken
  for provenance, language-only inference, reused/compiled datasets
  misattributed, incidental mentions mistaken for evidence) and the fix
  design that became stage 5's re-extraction spec: hide old
  country/producer labels from the model, strip affiliation headers,
  require a specific evidence-type citation per claim, add an independent
  verifier call, and build final country sets deterministically in Python
  rather than trusting the LLM's own aggregation.

Not copied: `Audit/Core_audit_start.csv` (30MB — the full pre-audit pool
the 400-record sample was drawn from, not just the sample itself). Still
at its original path if you need the full pool rather than the sample.

## Where this hands off

The fix design in `audit_log.txt` is implemented directly in stage 5's
`content_country_v2.py` (`../05_post_audit_analysis/content_country_v2_pipeline/`)
— every design choice named there traces back to a specific failure mode
documented here.
