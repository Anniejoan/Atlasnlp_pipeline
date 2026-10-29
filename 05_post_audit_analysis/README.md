# Stage 5 — Post-Audit Analysis

The evidence-gated re-extraction and final normalization that produced the
numbers actually reported in the camera-ready paper: **13,462** Core
dataset-contribution records and a final **1,480**-entry Gold reference
set. This is the last stage — see `../PIPELINE_REPORT.md` for the full
narrative and every number at every handoff. Original repo location:
`/home/jnwatu/Worldbench/post_audit_pipeline/` and
`/home/jnwatu/Worldbench/Post_audit_analysis/`.

## `content_country_v2_pipeline/`

The re-extraction itself: re-derives represented-country evidence for all
20,277 original candidates from scratch, implementing the stage-4 fix
design exactly (no old labels shown to the model, affiliation headers
stripped, evidence-taxonomy-gated claims, independent verifier call,
deterministic Python assembly of final country sets — not left to the
LLM). Run order:

1. `select_pilot50.py` — picks a 50-paper pilot sample.
2. `content_country_v2.py` — the extractor; run on the pilot first, then
   (after `score_pilot.py` validates pilot quality across prompt
   iterations v2.0 -> v2.2) on the full set via `run_canary.sbatch` (small
   canary run) then `run_full.sbatch` (full 32-shard SLURM array over all
   18,035 initial-extraction papers).
3. `merge_v2_shards.py` — merges the 32 shards (31 completed cleanly; 4
   recorded as processing failures and kept as such, not silently
   dropped).

See this subfolder's own `README.md` for the pilot methodology, go/no-go
checklist, and `requirements.txt` for the exact dependency pins.

## Final normalization

- `build_atlasnlp_v2_final_csvs.py` — joins the re-extracted evidence onto
  the sampling frame, normalizes every country string against the
  197-entity ontology (collapsing aliases, dropping ambiguous tokens
  rather than guessing), and applies the dataset-role eligibility criteria
  to get from 18,035 initial-extraction records down to 13,462 final Core
  records. Produces `normalization_report.json` / `.txt` (the exact
  funnel counts: 4,071 reuse-excluded, 408 no-identifiable-dataset, 90
  unresolved-ambiguity, 4 processing failures, 13,462 remaining) and
  `country_normalization_qc.csv`.
- `nli_recovery_validation.py` — an independent sanity check (not part of
  the content-country fix): verifies how many of the 425 Gold papers with
  ACL Anthology URLs survived the original NLI candidate filter and made
  it into Core at all (377 passed NLI, 372 ended up in Core — a *recall*
  question, distinct from content-country *accuracy*). Outputs
  `nli_recovery_gold_papers_in_core.csv` / `nli_recovery_validation_summary.json`.
- `pipeline_to_core_18035.md` — a provenance document reconciling the
  full "beginning to 18,035 records" funnel (candidate filtering through
  initial extraction) against the external NLI-filtering repo's own
  account of that process.

## `notebooks/`

- `AtlasNLP_Post_Audit_Analysis_v2_2_CAMERA_READY_STYLE.ipynb` — generates
  the camera-ready paper's post-audit figures (including
  `final_outputs/fig22_country_task_explicit_plus_inferred_camera_ready.png/.pdf`).
  Requires `kaleido==0.2.1` specifically (newer kaleido versions are
  incompatible with the plotly version pinned here) — see the site repo's
  notes if this notebook needs rerunning. Must be run from inside this
  `notebooks/` directory (or with its data paths adjusted), since
  `nbconvert` sets cwd to the notebook's own location.
- `"Post-audit Gold Validation.ipynb"` — narrows the Gold candidate pool
  to review-complete entries, producing the final 1,480-entry /
  989-normalized-dataset-group Gold set.

## `final_outputs/`

The camera-ready numbers, as CSVs:

| File | What it is |
|---|---|
| `atlasnlp_core_v2_all_normalized.csv` | All 18,035 initial records post-normalization, with eligibility flags (not yet filtered to the 13,462 final Core) |
| `atlasnlp_core_v2_eligible.csv` | The 13,462 final Core records |
| `atlasnlp_core_v2_explicit.csv` | The 2,447 (18.2%) Core records with explicit country attribution |
| `atlasnlp_core_v2_explicit_plus_inferred.csv` | The 3,506 (26.0%) Core records with explicit or inferred country attribution |
| `atlasnlp_gold_final_1480.csv` | The final 1,480-entry Gold reference set |
| `country_normalization_qc.csv` | Spot-check output for the country-string normalization step |
| `nli_recovery_gold_papers_in_core.csv` | The NLI-recovery sanity check's per-paper detail |

## Where this hands off

This is the end of the main pipeline — these final_outputs/ files are what
the camera-ready paper's tables and figures are built from, and what the
AtlasNLP website (`atlasnlp-site/`, not part of this organized copy) reads
from. There is no stage 6; any further analysis (e.g. the ArabCulture/
ArabicMMLU experiments in `arabic_exp/`) is a separate project that will
get its own organized folder.
