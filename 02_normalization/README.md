# Stage 2 — Normalization

Cleans up the raw extraction output from stage 1 before anything is merged
or analyzed: canonicalizes language labels, maps languages to countries
under a conservative policy, standardizes task categories onto a fixed
taxonomy, and auto-fixes common LLM-extraction contradictions. See
`../PIPELINE_REPORT.md` for the narrative. Original repo location:
`/home/jnwatu/Worldbench/compiled_ACL/` (most scripts) and repo root
(`pipeline_tracker.md`).

## `language_normalization/`

Run roughly in this order:

1. `deterministic_language_cleanup.py` — rule-based classification of every
   raw language string: canonical language name vs. sign language, script,
   programming language, language family, historical language, or
   dialectal variant. Anything in the latter categories gets flagged
   `language_excluded_from_country_inference` so it's never used to infer
   a country downstream.
2. `extra_lang_normalize.py` (Core) / `extra_lang_normalize_gold.py` (Gold)
   — second-pass normalization catching edge cases step 1 missed. (This is
   the pass that originally introduced the `"kru": "Kurukh"` collision —
   see `core_gold_merge_report.md` in stage 3 for how that bug was found
   and fixed.)
3. `language_mapping_diagnostics.py` — reports on mapping coverage/gaps to
   guide manual additions to the mapping table.
4. `language_mapping_update_master.py` — applies manual corrections to the
   master language table.
5. `merge_normalized_language.py` — folds the normalized language column
   back into the main record set.
6. `unique_languages.py` — utility to list the distinct language values at
   any point, for spot-checking.

## `language_country_mapping/`

- `language_country_mapping.py` — the curated language-to-country table.
  Deliberately conservative: broadly-distributed languages (English,
  Arabic, Spanish, French, Chinese, Swahili, Portuguese, ...) get **no**
  inferred country at all; only languages with genuinely high geographic
  concentration get one. This conservatism is why "inferred" country
  counts stay modest even after stage 5's re-extraction.
- `link_based_country_inference.py` — a separate inference path using
  paper/dataset links (e.g. GitHub org, hosting domain) rather than
  language, where available.

## `task_normalization/`

- `task_normalization.py` — maps raw free-text task descriptions onto the
  fixed 30/31-category task taxonomy used in every later table and figure.
- `valid_countries.csv` / `valid_countries.txt` — the standardized
  197-entity geopolitical ontology (195 UN member/observer states +
  Taiwan + Kosovo) that every country field in this project is normalized
  against, from stage 2 onward.

## `validation_triage_autofix/`

- `auto_fix_contradictions.py` — rule-based automatic correction of common
  LLM-extraction contradictions (e.g. a record marked "Monolingual" but
  listing two languages).
- `validation_triage.py` — routes records that auto-fix can't resolve
  confidently into a human-review queue.

## `pipeline_tracker.md`

Informal running log of pipeline stages (status: completed / needs rerun /
blocked, etc.) kept during development. Useful for seeing *why* a given
normalization step was rerun (e.g. a language-mapping correction that
forced a downstream rerun) — treat it as a dev diary, not a spec.

## Where this hands off

The normalized Core and Gold tables from this stage feed into stage 3
(`../03_validation_pre_audit_analysis/`), where they're merged together
under the Tier A eligibility rule and used to build the pre-audit paper
figures.
