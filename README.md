# AtlasNLP Pipeline — Organized Code Repository

This is a **curated, organized copy** of the scripts and key outputs behind
AtlasNLP, a country-aware atlas of NLP dataset
representation. It exists to make the pipeline's five-stage flow easy to
follow end to end. 
This folder pulls out the scripts that actually matter at each stage, plus
the key input/output CSV at each handoff point, so a new reader can
understand *what ran, in what order, producing what* without wading through
the full repo.

**If you need a file that isn't here** (raw LLM outputs, intermediate
per-shard caches, full author-affiliation regex/LLM-fallback candidate
files, etc.), it's still in the original repo - generate yours or contact the repo owner.

**Read `PIPELINE_REPORT.md` first.** It's the narrative: what each stage did,
the actual numbers at every handoff, and — importantly — the corrections
made along the way (several real bugs were found and fixed after the
original extraction, which is why the final published numbers differ from
the first "18,035 records" headline). The per-stage READMEs below are the
mechanical "what's in this folder and how to run it" reference.

## The five stages

```
1. EXTRACTION PIPELINE          2. NORMALIZATION           3. VALIDATION &
   ACL papers -> structured         Language/country/task      PRE-AUDIT ANALYSIS
   dataset records                  cleanup, Tier A rule       Funnel numbers,
   (Core: LLM extraction)                                      Table 1/2/3,
   (Gold: human curation)                                      pre-audit paper figures
          |                              |                           |
          v                              v                           v
   01_extraction_pipeline/       02_normalization/          03_validation_pre_audit_analysis/
                                                                       |
                                                                       v
   5. POST-AUDIT ANALYSIS   <----------------------------   4. AUDIT
      Re-extraction with         A 400-record human audit       Found the original
      evidence-gated prompts,    found the original content-    content-country field
      final Core/Gold numbers    country field was wrong         was only 32% correct
      used in the camera-ready   46.8% of the time -- this
      paper (13,462 / 1,480)     stage documents that finding
          |
          v
   05_post_audit_analysis/   <-------------------------   04_audit/
```

| Folder | Stage | What happens here |
|---|---|---|
| [`01_extraction_pipeline/`](01_extraction_pipeline/) | 1 | ACL Anthology papers -> structured dataset-metadata records (Core, via two-stage GPT-4o-mini extraction) and the parallel human-curated Gold set |
| [`02_normalization/`](02_normalization/) | 2 | Language token cleanup, language-to-country mapping policy, task-category normalization, contradiction/validation triage |
| [`03_validation_pre_audit_analysis/`](03_validation_pre_audit_analysis/) | 3 | Core+Gold merge, Tier A ("controlled") eligibility rule, the country-disaggregability funnel, and the pre-audit versions of the paper's validation table and figures |
| [`04_audit/`](04_audit/) | 4 | The 400-record human audit that found the original content-country field was wrong 46.8% of the time — the finding that triggered stage 5 |
| [`05_post_audit_analysis/`](05_post_audit_analysis/) | 5 | The evidence-gated re-extraction pipeline and the final normalization that produced the numbers actually used in the camera-ready paper (13,462 Core records, 1,480 Gold entries) |



## Quick orientation for a new reader

1. Read `PIPELINE_REPORT.md` for the full narrative and numbers.
2. Walk the five stage folders in order (01 -> 05); each has its own
   `README.md` explaining the scripts in it, the order to run them in, and
   where its inputs/outputs live in the original repo.
3. The single most important fact to hold onto: **the "18,035 records"
   number you'll see in early-stage files is not the final Core size.**
   That number is the *initial* automated extraction, before the audit
   (stage 4) found it needed tightening. The number actually reported in
   the camera-ready paper is **13,462** Core records and **1,480** Gold
   entries — both only exist after stage 5. See `PIPELINE_REPORT.md` for
   exactly how one became the other.
