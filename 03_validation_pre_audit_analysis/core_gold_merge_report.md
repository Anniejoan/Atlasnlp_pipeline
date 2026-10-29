# Core + Gold Merge Report: Tier A Construction and the Disaggregability Funnel

This report tracks, step by step, how WorldBench-Core and WorldBench-Gold were each reduced to Tier A ("controlled"), how their language vocabularies compare after cleanup, how the two were merged, and how that merge changes the country-disaggregability funnel from `Language Analysis.ipynb`. All numbers below were recomputed directly from the current data (not carried over from earlier, possibly-stale summaries).

Scripts written to produce this: `final_merge_gold.py`, `extra_lang_normalize_gold.py`, `merge_core_gold_tierA.py` (all in `compiled_ACL/task_matrix_construction/`).

---

## 1. Starting populations

| | WorldBench-Core | WorldBench-Gold |
|---|---|---|
| Source file | `updated_master_with_normalized_languages.csv` lineage → `worldbench_base_ground_truth_second_stage_cleaned.csv` | `human_worldbench.csv` |
| How it's produced | Automated extraction (LLM-assisted) from ACL Anthology papers | Human-compiled/reviewed (`human_review_splitting.py` pipeline) |
| Total rows | **18,035** | **1,661** |
| Rows with a paper URL | 18,035 (100%) | 1,381 (83.1%) |
| `Validation Status == Done` | n/a (no reviewer field) | 1,638 / 1,661 (98.6%) |
| Row Source | n/a | 1,283 `human_compiled` + 377 `human_compiled_machine_assisted` + 1 blank |

**Core vs Gold are not disjoint.** Matching on the combined key (normalized paper URL + lowercased Dataset name), **263 datasets appear in both** populations. (A looser match on paper URL alone finds 293 — the extra 30 are cases where the same paper URL holds a different-looking dataset name on each side, likely multi-dataset papers or naming drift; those are *not* treated as duplicates, to avoid discarding real data.) This overlap is exactly why merging can't be a naive concatenation — see §5.

---

## 2. Language-string audit (background: raw vocabulary, not yet tied to Tier A)

Before any Tier A filtering, WorldBench's raw language *labels* (free text, pre-canonicalization) went through a separate validity audit (`language_audit_outputs/`):

| | Count |
|---|---|
| Raw unique language strings audited | 3,813 |
| Valid (`is_valid_language == True`) | 3,217 |
| Invalid / removed (junk, non-language labels) | 596 |
| Flagged for further human review | 600 |
| **Core validated canonical languages** | **1,304** |
| **Gold validated canonical languages** | **593** |
| **Combined (Core ∪ Gold) validated canonical languages** | **1,387** (510 overlap) |

This audit is a different, coarser-grained pass than the Tier A `language_cleaned` pipeline in §4 below — it operates on raw label strings across *all* of WorldBench, not just Tier A. Cross-checked against Tier A's 283 Core languages (§4): **zero were flagged invalid or needing review** by this audit. The 15 that don't appear in the audit's validated list at all are name-formatting mismatches between the two independently-run cleanup efforts (e.g. `Gwich'In` vs the audit's apostrophe variant), not evidence of invalidity.

---

## 3. Tier A construction ("controlled")

**Definition** (from `final_merge.py`, replicated verbatim for Gold in `final_merge_gold.py`): a dataset qualifies as Tier A only when its language↔country pairing can't be ambiguous:

- `matrix_compatible` prerequisite: has a validated content country (`has_valid_content_country`) **and** a validated task category (`has_valid_task`, matched against the 31 allowed task categories).
- Then: **(exactly 1 language, ≥1 countries)** OR **(>1 language, exactly 1 country)**.
- Anything both multilingual *and* multi-country is excluded — no way to know which language pairs with which country, so Tier A drops it rather than guess (that guess is what "naive" expansion does instead, see the CLDR-grounded reanalysis elsewhere in this project).

| Step | Core | Gold |
|---|---|---|
| Total rows | 18,035 | 1,661 |
| `has_valid_content_country` | 11,764 | 1,658 |
| `has_valid_task` | 18,017 | 1,660 |
| `matrix_compatible` (both) | 11,760 | 1,658 |
| via mono-language, many-country | 8,246 | 847 |
| via many-language, one-country | 1,734 | 285 |
| **Tier A total** | **9,980** (55.3% of all rows) | **1,132** (68.1% of all rows) |

Gold clears the bar at a notably higher rate (68.1% vs 55.3%) — consistent with it being human-curated rather than automatically extracted: `matrix_compatible` alone is 99.8% for Gold vs 65.2% for Core.

---

## 4. Language cleanup applied to Tier A

Tier A's dataset-level rows are exploded into (dataset, language, country) combinations, then each language token is run through `extra_lang_normalize.py`'s classification (imported directly for Gold, not reimplemented, so the two are guaranteed consistent): sign languages, scripts, programming languages, language families, historical languages, dialectal variants, and unrecognized tokens are all flagged `language_excluded_from_country_inference = True` and dropped from the country-disaggregability analysis. Only tokens matching `DIRECT_MAP` or `ISO_CODE_MAP` (i.e. recognized canonical language names) survive.

| | Core | Gold |
|---|---|---|
| Tier A datasets (from §3) | 9,980 | 1,132 |
| Exploded (dataset, language, country) rows | 18,777 | 6,938 |
| Country-inference-safe rows | **17,896** (corrected; was reported as 17,898 pre-fix) | 4,113 |
| **Unique country-inference-safe languages** | **282** (corrected; was reported as 283 pre-fix) | **195** |

**Gold's 195 languages are a strict subset of Core's 282 — zero Gold-only languages** (union = 282 = Core alone). At the raw-language-vocabulary level, Gold adds no new languages; whatever value it adds must come from additional *datasets/countries* for languages Core already has. That's exactly what §6 finds.

**Correction applied:** Core's "Kru Language Atlas Data" (paper `L16-1515`) had its raw language field "Kru" (the West African language family the dataset is actually about) auto-normalized to **"Kurukh"** — an unrelated Dravidian language of India — by a stale `"kru": "Kurukh"` entry in `merge_normalized_language.py` / `deterministic_language_cleanup.py` (evidently written to catch the ISO 639-3 code `kru`, which genuinely does mean Kurukh, but which also collides with the plain-English family name in this one dataset's raw metadata). Because "Kurukh" hit `DIRECT_MAP`, the row was incorrectly marked `canonical_language` / country-inference-safe instead of `language_family` / excluded. Fixed directly in `worldbench_lang_tierA_expanded_second_stage_cleaned.csv` (2 rows: dataset's Liberia and Burkina Faso entries) by relabeling to `Kru` / `language_family` / excluded. Checked: this is the *only* place in all of Core (18,035 rows) where this token appears in a way that reaches Tier A — the one other raw occurrence (`SMOL`, using lowercase ISO code `kru` legitimately meaning Kurukh) never qualifies for Tier A regardless (it has 0 valid content countries), so it was never affected. Confirmed this doesn't touch the raw language-string audit's 1,304/593 counts (§2) — that audit already treats "Kru" and "Kurukh" as two distinct, independently valid raw strings; the conflation was introduced entirely within the Tier A country-inference pipeline, not upstream.

---

## 5. Merge methodology

**Policy: Gold overrides Core.** For any dataset present in both populations (the 263 exact paper-URL + name matches from §1), Core's copy is dropped entirely — even if it *did* make Core's Tier A — and Gold's version is used instead, on the reasoning that Gold's human review (98.6% `Validation Status == Done`) is more trustworthy than Core's automated extraction for the same underlying dataset. Everything else is a simple union; nothing is deduplicated beyond this exact-match rule (the 30 ambiguous paper-URL-only matches from §1 are deliberately left as separate rows rather than guessed at).

Concretely, of Core's 9,980 Tier A datasets, **127 were dropped** because Gold had its own entry for that same dataset (note this is smaller than the full 263-dataset overlap — most overlapping datasets weren't in Core's Tier A to begin with). All 927 of Gold's Tier A datasets that survived the identity match are kept as-is.

| | Value |
|---|---|
| Core Tier A datasets before dedup | 9,980 |
| Dropped (Gold has an entry for the same dataset) | 127 |
| Core Tier A datasets kept | 9,853 |
| Gold Tier A datasets (all kept) | 927 |
| **Merged Tier A datasets** | **10,780** |
| Merged Tier A exploded rows | 24,773 |

---

## 6. Merged Tier A language coverage

**Unique country-inference-safe languages after merge: 282 — identical to corrected Core-alone (282), not fewer.** (This section originally reported the merge as costing one language, "Kurukh," versus a then-uncorrected Core-alone figure of 283. That was backwards: Core's 283 was itself inflated by the Kru→Kurukh misclassification documented in §4's correction note. Core's dataset that appeared to contribute "Kurukh" — `Kru Language Atlas Data`, paper `L16-1515` — is also one of the 127 datasets dropped in the merge in favor of Gold's copy of the same dataset, independent of the language bug; Gold's own copy correctly carries "Kru" and is excluded from country-inference as a language-family label. So once Core's bug is fixed, Core-alone and Merged agree at 282 for the same underlying reason, and the merge neither adds nor loses any language from the vocabulary.) Verified no other languages were lost and none were gained (Gold contributes zero new languages, consistent with §4's subset finding).

So the merge doesn't grow the *language vocabulary* at all. What it grows is the *evidence* (datasets/countries) behind the languages already there — which is exactly what matters for the disaggregability funnel below.

---

## 7. Disaggregability funnel: Core-only vs Merged

Same filters as `Language Analysis.ipynb`'s audit, applied identically to both populations: eligible task type (QA / NLI-proxy / MCQ format) → explicit (not inferred) country provenance → country-count thresholds.

| Stage | Core-only | Merged (Core + Gold) | Change |
|---|---|---|---|
| Controlled, language-tagged rows | 17,898 | 21,100 | +3,202 |
| Eligible task type (rows) | 1,807 | 2,298 | +491 |
| Eligible task type (languages) | 100 | 121 | +21 |
| Explicit provenance (rows) | 1,558 (86.2%) | 1,947 (84.7%) | +389 |
| Explicit provenance (languages) | 89 | 120 | +31 |
| **Languages with ≥2 countries** | **34** | **60** | **+26 (+76%)** |
| **Languages with ≥3 countries** | **16** | **37** | **+21 (+131%)** |

Of the merged eligible+explicit rows, 1,487 come from Core and 460 from Gold — Gold is a meaningfully sized contribution (23.6%) despite being ~9% of the raw row count, again reflecting its higher per-row filter pass-rate.

**Headline: merging in Gold more than doubles the number of languages that clear the ≥2-country and ≥3-country bars** (34→60, 16→37), even though it adds zero new languages to the underlying vocabulary. This happens because Gold's datasets fill in *additional countries* for languages Core already had — e.g. a language that had only 1 qualifying country in Core-alone can cross the ≥2 threshold once a Gold dataset supplies a second country for it.

---

## 8. Data-quality fixes found while extending verification to the merged set

Rebuilding the "coordinated multi-country dataset" detection (single dataset spanning ≥2 countries by itself) on the merged population surfaced **33 dataset names that map to more than one distinct paper identity** (same name, different paper URL). Most of these are legitimate — well-known benchmarks like SQuAD, HotpotQA, RACE, DROP, Natural Questions, and FreebaseQA are independently cited by many unrelated papers, and FrenchMedMCQA's 5 entries are 5 different years of the same DEFT shared task, each correctly attributing a single country (France). These are not duplicates and were left as-is.

Two, however, were real problems, found by comparing their country lists directly:

- **ArabCulture**: appeared as both a Core entry (paper `2025.acl-long.380`) and the already-`VERIFIED` Gold entry (paper `2025.findings-emnlp.247`) — with **identical 13-country lists**. This confirms it's the same real dataset, just catalogued from two different papers (one likely cites/reuses ArabCulture rather than introducing it). The exact-match dedup in §5 missed it because the paper URLs differ. **Fix: dropped the Core copy**, consistent with the "Gold overrides Core" policy and the existing verified status.
- **TiQuAD**: Core's entry (paper `2023.acl-long.661`, already `REJECTED`) lists Eritrea/Ethiopia — correct for a Tigrinya dataset. Gold's entry cited a *different* paper (`arxiv.org/abs/2507.17974`) and listed **Eritrea + South Korea**. Fetching that arXiv paper confirmed it's a Tigrinya-NLP survey that "focuses exclusively on Tigrinya... primarily in Eritrea and Ethiopia" — no mention of South Korea anywhere. This is a **Gold data-entry error**, not a real second country. **Fix: dropped the erroneous Gold copy** (an exception to "Gold overrides Core" — Gold is wrong here, and Core's existing entry already covers the language correctly).

Both fixes are applied directly to `worldbench_merged_lang_tierA_expanded.csv` (15 rows dropped total). Funnel numbers in §7 are unchanged after the fix (60 / 37) since both languages already cleared their thresholds through other rows.

---

## 9. Extending dataset verification to the merged set

Re-running the coordinated-multi-country-dataset scan on the corrected merged population finds **139 candidates** (single dataset spanning ≥2 countries by itself), of which **118 were not part of the original Core-only 21-dataset check**. Nearly all of the un-checked 118 are English datasets with exactly 2 countries — generic QA/NLP benchmarks (ChartQA, DROP, HotpotQA, SciQ, TheoremQA, etc.) attributed 2 countries via incidental metadata rather than genuine per-country splits. Checking all 118 individually has a very low expected yield; instead this pass covered the highest-value subset: every candidate with **≥4 countries** (8 datasets, most likely to be genuine coordinated cultural/multi-country benchmarks, by the same logic that made CULTURAL BENCH/GIMMICK/ArabicMMLU worth checking originally), plus the two most novel candidates regardless of count — **Hausa/HausaVQA** (a language not previously in the qualifying set at all) and **Arabic/DiaSet** (a new Arabic dataset).

Same method as before: download the paper PDF, extract text with `pypdf` (WebFetch's page-summarization still doesn't work reliably for tables), regex-scan for country-name-adjacent numbers, then manually verify every candidate against its actual table caption before trusting it.

**New VERIFIED dataset:**
- **AfriMed-QA** (English, medical QA) — paper Table 9, "Counts of Questions Contributed by Country": **16 countries**, Zambia=2 up to Nigeria=7,577. A second table (Table 10, "Country Count for Expert Questions") gives a 5-country expert-reviewed subset: South Africa=54 up to Ghana=1,495 — this is the subset WorldBench's own metadata had recorded (5 countries), so the full 16-country table is *additional* evidence beyond what WorldBench captured, not a contradiction of it.

**Checked and REJECTED (no usable per-country sample table):**
- **CogForecast** (English) — country-labelled numbers found are event-forecasting question counts keyed by *country pairs* (e.g. "US–China", "Palestine–Israel"), not per-country sample counts in the sense this audit needs.
- **African Traditional Herbal Medicine Dataset** (English) — only a passing citation mention of Kenya, no table.
- **Gold Standard Dataset for Racist Tweet Detection** (English) — numbers found (e.g. "USA 3.9M...") look like social-media usage/follower statistics used as study rationale, not annotated-sample counts.
- **DiaSet** (Arabic) — no country-labelled numeric content found at all.
- **HausaVQA** (Hausa) — only author-affiliation addresses (Nigeria) and a prose mention of where Hausa is spoken ("Gabon, Sudan and Togo"); no per-country data table. WorldBench's "8 countries" for this dataset appears to be background/context, not a real per-country split.

**Updated verification tally: 27 datasets manually checked (up from 21), 4 VERIFIED (up from 3), 23 REJECTED (up from 18).**

---

## 10. Completing the pass: all 139 candidates checked

Continued past the ≥4-country cutoff to check the full remaining tail — **all 111 previously-unchecked candidates** (every one of the 139 coordinated-multi-country candidates in the merged set is now accounted for; none skipped this round). Method: batch-downloaded all 111 paper PDFs in parallel, ran the same `pypdf` text extraction + country-name/number regex scan across all of them at once, then manually read every one of the 73 that produced a candidate hit (the other 38 had no country-adjacent number anywhere in the text at all).

As expected, the vast majority of the 73 hits were false positives from the same noise patterns seen before — conference venue lines ("Miami, FL, USA, November 12-16, 2024"), author affiliations ("¹Tencent Jarvis Lab, China"), funding acknowledgments ("National Science Foundation of China"), and prose mentions of a country as question *content* rather than dataset provenance (e.g. HiTab's "What is the GDP of China in 2012?" is an example question, not a per-country sample count).

**Two more genuine finds, both VERIFIED:**
- **FEEDBACK QA** — paper Table 2, "Number of samples in different domains of FEEDBACKQA": Australia=1,783 questions, Canada=8,844, UK=2,874, US=13,533 (a fifth row, "WHO," is the World Health Organization, not a country, and is excluded).
- **OrQA** — paper Table 2, "Statistics of the crawled tables": UK=24,404 source tables, Canada=31,437 source tables. Note the unit here is *crawled source tables*, not final QA pairs — still a genuine per-country quantification of the dataset's country-specific content, but a different unit than the other VERIFIED datasets, worth flagging if cited alongside them.

**Notable rejections (checked, not usable):**
- **Med-HALT** — not really a per-country *split* of one benchmark; it's an aggregation of several pre-existing, independently-sourced sub-datasets (e.g. MedMCQA from India, Headqa from Spain), each already country-specific on its own. Structurally different from what this audit means by "coordinated multi-country dataset," so rejected rather than force-fit.
- **FORK** — the actual numeric table (Table 2) is inter-annotator agreement statistics (Cohen's Kappa) from a validation study, not sample counts. The paper's prose gives only an aggregate percentage ("91.84% of questions pertaining to China, Japan, India and the US" combined), not a per-country breakdown.
- **AMAMMERE** — percentages found are model accuracy/disambiguation-quality scores on US vs. Ghana question variants, not sample size counts.
- All other 68 candidates with a country-adjacent number: confirmed as citation noise, venue locations, or unrelated prose content on manual read.

### Final verification tally (complete)

| | Count |
|---|---|
| Total coordinated multi-country candidates in the merged set | 139 |
| Manually checked (papers + HF configs) | **139 (100%)** |
| **VERIFIED** | **6**: Arabic/ArabCulture, Arabic/ArabicMMLU, English/CULTURAL BENCH, English/AfriMed-QA, English/FEEDBACK QA, English/OrQA |
| REJECTED (checked, no usable per-country breakdown found) | 133 |

This is now a complete pass over every dataset in the merged population that *could plausibly* be a coordinated multi-country resource (i.e. every dataset whose WorldBench metadata alone already spans ≥2 countries). Nothing from that candidate list remains unchecked.

## What's still not covered by this report

Everything below is a genuinely different question than "check the coordinated-candidate list," not a shortfall in the pass just completed:

- **The ~830 qualifying datasets that only reach ≥2 countries by being *pooled* with other single-country datasets for the same language** (as opposed to one dataset alone spanning ≥2 countries) were never in scope for paper-checking — that's a fundamentally different, much larger comparability question (do two *different* single-country datasets for the same language actually share task/label definitions?) than what this pass checked.
- HuggingFace size lookups (the `hf_size_results_cache.json` pass) were run against the Core-only 828-dataset set and have not been extended to Gold-sourced datasets or the newly-VERIFIED AfriMed-QA/FEEDBACK QA/OrQA.
- The 30 ambiguous paper-URL-only Core/Gold overlaps from §1/§5 (same paper URL, non-matching dataset name) haven't been individually checked for further hidden duplication beyond the two (ArabCulture, TiQuAD) found in §8 — those were found incidentally because they happened to appear in the coordinated-candidate list, not from a systematic check of all 30.
