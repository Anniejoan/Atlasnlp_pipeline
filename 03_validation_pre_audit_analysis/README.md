# Stage 3 — Validation & Pre-Audit Analysis

Merges the normalized Core and Gold tables, applies the Tier A
("controlled") eligibility rule, builds the country-disaggregability
funnel, and produces the **pre-audit** version of the paper's validation
table and early figures. "Pre-audit" matters here: everything in this
folder reflects the *initial* 18,035-record Core extraction, before the
stage-4 audit found its content-country field was unreliable. See
`../PIPELINE_REPORT.md` for full numbers and context. Original repo
location: `/home/jnwatu/Worldbench/compiled_ACL/` and
`"Cross Country Funnel/"`.

## Core + Gold merge and Tier A rule

1. `final_merge.py` (Core) / `final_merge_gold.py` (Gold) — apply the
   Tier A eligibility rule: a record counts as having unambiguous
   language<->country evidence only if it's (mono-language, any number of
   countries) or (multi-language, exactly one country); anything both
   multilingual *and* multi-country is dropped rather than guessed at.
   Core clears this bar at 55.3% of rows, Gold at 68.1%.
2. `merge_core_gold_tierA.py` — merges the two Tier A-filtered sets: for
   the 263 datasets present in both (matched by paper URL + dataset name),
   Gold's human-reviewed version overrides Core's automated one.
   **Read `core_gold_merge_report.md` for the full account** — it documents
   two real bugs found and fixed during this merge (the `"kru": "Kurukh"`
   language-normalization collision, and a 6-dataset Tier A
   mutual-exclusivity violation) and is the most detailed single
   provenance document in this stage.

## Country-disaggregability funnel

`cross_country_funnel/Funnel.ipynb` (+ `Funnel.xlsx`) builds the headline
result of this stage: merging Gold into Core more than doubles the number
of languages with enough country-attributed data to support cross-country
comparison (34 -> 60 languages with >=2 countries; 16 -> 37 with >=3) —
achieved not by adding new languages, but by Gold supplying *additional
countries* for languages Core already had.

## Validation (pre-audit Table 1) and other analysis scripts

- `validation_0.py` — computes the paper's pre-audit Table 1 by joining
  Gold against Core on paper URL (460 entry-level aligned pairs across 286
  unique papers): main-country recovery (87.0%), task exact-match (89.1%),
  etc. **This table's numbers reflect the pre-audit Core extraction** and
  are superseded by stage 5's post-audit figures — it's kept here because
  it's the exact analysis that motivated commissioning the stage-4 audit.
- `paper_stats.py` — general descriptive statistics used in early-draft
  paper text.
- `heatmap.py` — country x task coverage heatmap (pre-audit version).
- `language_gpt_audit.py` — a targeted GPT-based spot-check of language
  labels, separate from the stage-4 content-country audit.

## `notebooks/`

Main pre-audit analysis notebooks that generated early-draft paper
figures from this stage's merged dataset: `Analysis Exploration.ipynb`,
`Language Analysis.ipynb`, `More Analysis.ipynb`, `Final Dataset
building.ipynb`.

## `trusted_outputs/`

The key intermediate/output CSVs at this stage's handoff points:
`worldbench_base_ground_truth_second_stage_cleaned.csv`,
`human_worldbench.csv`, `worldbench_funnel_trusted_final.csv` (the funnel's
final numbers), `worldbench_60lang_dataset_metadata.csv`,
`worldbench_tier1_single_dataset_multicountry.csv`,
`worldbench_tier2_cross_dataset_same_task.csv`, `country_task_matrix.csv`.

## Where this hands off

This stage's pre-audit Core/Gold merge is what gets sampled for the
400-record human audit in stage 4 (`../04_audit/`) — the audit that found
this stage's content-country field needed a full re-extraction, done in
stage 5.
