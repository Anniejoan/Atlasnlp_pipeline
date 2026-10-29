# AtlasNLP Pipeline Report: From the ACL Anthology to the Camera-Ready Numbers

This is the narrative version of what the five stage folders in this
repository do. It exists so a new reader (or collaborator, or reviewer) can
understand *why* the numbers move the way they do across the pipeline,
without having to reconstruct that history from scattered scripts and
reports. Every number below was verified directly against the files in this
repo (or their originals) rather than carried
over from memory.

---

## The one-paragraph version

AtlasNLP started from 119,963 ACL Anthology entries, narrowed to 20,277
candidate papers by an NLI dataset-detection filter, then automatically
extracted structured dataset metadata (task, language, represented country,
producer country) from each one with a two-stage GPT-4o-mini pipeline. That
produced an **initial** 18,035-record Core collection. A 400-record human
audit of that initial collection then found a serious problem: the
represented-country field was correct only 32% of the time, and 46.8% of
audited records had the wrong country entirely — mostly from conflating
author affiliation, language, or incidental mentions with actual dataset
provenance. Rather than patch the extraction, a second, stricter,
evidence-gated extraction pipeline was built and re-run over all 20,277
candidates from scratch, with every content-country claim required to cite
page-level evidence and pass an independent verifier call. Applying the
resulting post-audit eligibility and attribution criteria to the full
collection produced the numbers actually reported in the camera-ready
paper: **13,462** Core dataset-contribution records and a final **1,480**-entry
Gold reference set.

---

## Stage 1 — Extraction pipeline (`01_extraction_pipeline/`)

**Candidate filtering** (upstream of this repo, in a separate
`acl_anthology_paper_finder` repo not included here): `anthology.bib`
(119,963 ACL Anthology entries) was filtered down through a structural
pass (has an abstract: 71,847) and a ModernBERT-base-NLI zero-shot
entailment check against the hypothesis *"The paper proposes a dataset"*
(entailment, score > 0.5) to **20,277 candidate papers**. This is
`data/manifest.jsonl` in the original repo.

**Core extraction** (`core_extraction/`): `metadata_extractor.py` runs a
two-stage GPT-4o-mini pipeline per paper — download the PDF, extract text
per page, run a regex role-triage to decide whether the paper *introduces*,
*extends*, or merely *reuses* a dataset, extract structured metadata with
page+quote evidence citations, validate against `schema.json`, and send
anything that fails validation back for a second, more targeted LLM pass.
Run as a 20-way SLURM array (`split_manifest.py` / `run_shard.sh`), merged
with `merge_jsonl_shards.py` / `merge_tsv_shards.py`.

Final accounting (verified directly against the merged shard outputs):

| Outcome | Count |
|---|---|
| Role-triage excluded (reuse/compilation, never sent to the LLM) | 905 |
| Failed extraction (schema/fetch/other failure) | 1,337 |
| **Successfully extracted -> initial Core** | **18,035** |
| Total (= candidate papers) | 20,277 |

**Gold human curation** (`gold_human_curation/`): in parallel, contributors
from diverse geographic backgrounds independently collected and annotated
datasets for assigned countries (`human_review_splitting.py` /
`make_reviewer_batches.py`), with a second cross-validation stage where
assignments were shuffled and re-reviewed
(`rebuild_and_balance_reviewer_batches_machinexhuman.py`). This produced the
original **1,661-entry** Gold collection (`human_compiled.csv` here is the
raw contributor-submission input to that process).

**Producer-country extraction** (`author_affiliation_producer_country/`):
a separate pipeline regex-extracts author affiliations from each paper's
PDF, falls back to an LLM pass for ambiguous cases, and normalizes
institution names to producer countries — independent of, and not fed by,
the represented-country extraction above.

**Country second pass** (`country_second_pass/`): a follow-up LLM pass
specifically for records that came out of the main extraction still missing
country attribution.

---

## Stage 2 — Normalization (`02_normalization/`)

Three largely independent cleanup layers applied to the initial 18,035-row
Core collection (and, where noted, mirrored onto Gold):

- **Language normalization** (`language_normalization/`): raw free-text
  language labels are deterministically classified and canonicalized
  (`deterministic_language_cleanup.py`), including flagging sign languages,
  scripts, programming languages, language families, historical languages,
  and dialectal variants as `language_excluded_from_country_inference` so
  they're never used to infer a country. `extra_lang_normalize.py` /
  `extra_lang_normalize_gold.py` apply a second-stage pass (the one that
  introduced, and later needed correcting, the Kru-language bug described
  in stage 3).
- **Language-to-country mapping** (`language_country_mapping/`): a
  deliberately conservative curated table — broadly-distributed languages
  (English, Arabic, Spanish, French, Chinese, Swahili, Portuguese, ...) get
  no inferred country at all; only languages with genuinely high
  geographic concentration get one.
- **Task normalization** (`task_normalization/`): raw task-category free
  text is mapped onto a fixed 30/31-category taxonomy, and
  `valid_countries.csv`/`.txt` define the standardized 197-entity
  geopolitical ontology (195 UN member/observer states + Taiwan + Kosovo)
  used everywhere downstream.
- **Validation/triage/autofix** (`validation_triage_autofix/`): rule-based
  detection and automatic correction of common LLM-extraction
  contradictions (e.g. "Monolingual" with two listed languages) before
  records are routed to human review.

`pipeline_tracker.md` in this folder is the original stage-by-stage status
log kept during this phase — useful for seeing which normalization steps
were rerun and why (e.g. a Portuguese reclassification that forced a
mapping-table rerun).

---

## Stage 3 — Validation & pre-audit analysis (`03_validation_pre_audit_analysis/`)

This is where Core and Gold are merged into the dataset actually used for
early paper figures, **before** the stage-4 audit found problems with it.

**Tier A ("controlled") eligibility**: a dataset only counts as having
unambiguous language<->country evidence when it's either (mono-language,
any number of countries) or (multiple languages, exactly one country) —
anything both multilingual *and* multi-country is dropped rather than
guessed at (`final_merge.py` for Core, `final_merge_gold.py` for Gold,
identical rule). Core cleared this bar at 55.3% of rows, Gold at 68.1%
(consistent with Gold being human-curated rather than automatically
extracted).

**Core + Gold merge** (`merge_core_gold_tierA.py`): for the 263 datasets
present in both Core and Gold (matched by paper URL + dataset name), Gold's
human-reviewed version overrides Core's automated one. Full details,
including two real bugs found and fixed during this merge (a
`"kru": "Kurukh"` language-normalization collision, and a 6-dataset Tier A
mutual-exclusivity violation), are in `core_gold_merge_report.md` — read
this file for the complete, numbered account.

**Country-disaggregability funnel**: the headline result of this stage —
merging in Gold **more than doubles** the number of languages with enough
country-attributed data to support cross-country comparison (34 -> 60
languages with >=2 countries; 16 -> 37 with >=3), even though Gold adds
*zero* new languages to the vocabulary. It does this by supplying
*additional countries* for languages Core already had. See
`cross_country_funnel/Funnel.ipynb` for the full construction, and
`trusted_outputs/worldbench_funnel_trusted_final.csv` for the final numbers.

**Pre-audit validation (Table 1)**: `validation_0.py` computes the paper's
Table 1 by joining Gold against Core on paper URL (460 entry-level aligned
pairs, spanning 286 unique papers) and reporting metrics like main-country
recovery (87.0%) and task exact-match (89.1%). **This table reflects the
initial, pre-audit Core extraction** — it is why the audit in stage 4 was
commissioned in the first place (reviewers flagged that these numbers
looked implausibly strong given spot-checked errors).

`notebooks/` holds the main pre-audit analysis notebooks (`Analysis
Exploration.ipynb`, `Language Analysis.ipynb`, and two earlier exploratory
notebooks) that generated the paper's early-draft figures from this
pre-audit dataset.

---

## Stage 4 — Audit (`04_audit/`)

A **400-record random sample** of the initial 18,035-row Core collection
(stratified; see `atlasnlp_audit_stratum_counts.csv` /
`atlasnlp_audit_sampling_frame.csv`) was manually reviewed against the
source PDFs (`Audit book.ipynb`, results in
`atlasnlp_stage1_random400_ai_audit_completed_400.csv`).

**The finding** (full account in `audit_log.txt`): the existing
content-country field was correct only **32%** of the time and outright
**incorrect 46.8%** of the time. The recurring failure modes were:

1. Country assigned from **author/institutional affiliation**, not from
   evidence about the dataset's actual content or participants.
2. Country assigned from **language alone** (e.g. "this is a Vietnamese
   dataset" treated as evidence of Vietnam, even with no stated population
   or collection evidence).
3. **Reused/compiled datasets** credited to the citing paper's apparent
   country rather than excluded as not a primary contribution.
4. **Incidental mentions** (a country named in passing, e.g. a
   related-work citation) mistaken for provenance evidence.

This is the finding that made the original Table 1 numbers untrustworthy
and triggered the stricter re-extraction in stage 5. The audit log also
contains the proposed fix design: hide old country/producer labels from the
extraction model entirely, strip author-affiliation headers, require a
specific evidence-type citation for every claim, add an independent
verifier call, and construct the final explicit/inferred country sets
deterministically in Python rather than trusting the LLM's own aggregation.

---

## Stage 5 — Post-audit analysis (`05_post_audit_analysis/`)

**Re-extraction** (`content_country_v2_pipeline/`): `content_country_v2.py`
re-extracts represented-country evidence for all 20,277 original candidates
from scratch, implementing the stage-4 fix design exactly: the model never
sees the old country/producer labels, author-affiliation headers are
stripped, every explicit-evidence claim must cite one of a fixed taxonomy
of valid evidence types (participant location, country-specific source,
explicit target population, etc. — author affiliation and language-only
mentions are explicitly *not* valid evidence types), a second independent
LLM call verifies each candidate claim, and the final explicit/inferred
country sets are assembled deterministically in Python, never left to the
model. Iterated through three prompt versions (v2.0 -> v2.2) on a 50-paper
pilot before the full run; see `content_country_v2_pipeline/README.md` for
the pilot methodology and go/no-go checklist.

Run as a 32-shard SLURM array over all 18,035 initial-extraction papers
(31 completed cleanly; 4 recorded as processing failures and retained as
such in the final master rather than silently dropped).

**Final normalization** (`build_atlasnlp_v2_final_csvs.py`, in
`final_outputs/`): joins the re-extracted evidence onto the original
sampling frame, normalizes every country string to the standard 197-entity
ontology (collapsing aliases like "Hong Kong" -> China, dropping ambiguous
tokens like "Congo" rather than guessing), and applies the same dataset
dataset-role eligibility criteria the audit's fix design called for. Final
accounting (verified directly against `normalization_report.txt`):

| Outcome | Count | Share of 18,035 |
|---|---|---|
| Reuse of an existing dataset (not a primary contribution) | 4,071 | 22.6% |
| No confidently identifiable target dataset | 408 | 2.3% |
| Unresolved ambiguity among multiple datasets | 90 | 0.5% |
| Processing failures (retained, never relabeled) | 4 | 0.0% |
| **Final AtlasNLP-Core (primary dataset contributions)** | **13,462** | **74.6%** |

Of those 13,462 final Core records: **2,447 (18.2%)** have explicit
represented-country attribution, a further **1,059 (7.9%)** have inferred-only
attribution, and **9,956 (74.0%)** remain unattributed even combining both —
not because they lack geographic context, but because the stricter
evidence standard couldn't recover it reliably.

**Gold final count**: the post-audit Gold validation pass
(`notebooks/Post-audit Gold Validation.ipynb`) narrowed the working Gold
pool to entries marked `Done` in review, producing the final
**1,480-entry / 989-normalized-dataset-group** Gold reference set used in
the camera-ready paper (`final_outputs/atlasnlp_gold_final_1480.csv`).

**NLI-stage recovery check** (`nli_recovery_validation.py`): as an
independent sanity check unrelated to the content-country re-extraction,
this script verifies how many of the 425 Gold papers with ACL Anthology
URLs survived the original NLI filter and made it into Core at all — 377
(88.7%) passed NLI, and 372 (87.5%) ended up in Core. This is a *pipeline
recall* question (did the paper survive screening), distinct from the
content-country *accuracy* question the rest of this stage addresses.

---

## Numbers actually used in the camera-ready paper

| Quantity | Pre-audit (stage 3) | **Post-audit / final (stage 5)** |
|---|---|---|
| AtlasNLP-Core size | 18,035 (initial extraction) | **13,462** (primary dataset contributions only) |
| AtlasNLP-Gold size | 1,661 | **1,480** (989 normalized dataset-name groups) |
| Core records w/ explicit country | *( — see stage 4)* | **2,447 (18.2%)** |
| Core records w/ explicit+inferred | *11,942 (66.2%)* | **3,506 (26.0%)** |
| Countries represented (explicit) | *(— see stage 4)* | **158 of 197** |

If you're citing a number from this project and it isn't one of the bolded
ones above, double-check which stage it came from — the pre-audit numbers
are real artifacts of real pipeline stages (useful for understanding how
the project evolved) but are not what the published paper reports.
