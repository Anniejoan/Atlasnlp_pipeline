# AtlasNLP Content-Country Re-extractor v2

This is a **new corrective pipeline**, not a patch to the original metadata extractor.

## Core design

- The model never receives old AtlasNLP content-country labels.
- The model never receives producer-country labels.
- Page-1 author/affiliation headers are removed when a clear `Abstract` marker exists.
- One extraction call classifies dataset relationship and finds per-country evidence.
- Reuse-only / target-not-found / ambiguous records are excluded.
- Only eligible records with candidate evidence receive a second independent verifier call.
- Python, not the LLM, constructs the final explicit/inferred country sets.
- Invalid evidence types are deterministically blocked.
- Token usage and estimated cost are saved per paper.

### Main analysis
Use `explicit_content_countries` only.

### Sensitivity analysis
Use `explicit_content_countries + inferred_content_countries`.

## Model defaults

- Model: `gpt-5.4-mini-2026-03-17` (snapshot pinned for reproducibility)
- Extractor reasoning: `low`
- Verifier reasoning: `medium`

The verifier gets a short input, so spending somewhat more reasoning there is a
better quality/cost tradeoff than using medium reasoning over every full paper.

## Install

```bash
python -m pip install -U openai pandas pypdf requests
```

Set your key:

```bash
export OPENAI_API_KEY="..."
```

## 1. Select the reproducible 50-paper pilot

```bash
python select_pilot50.py \
  --audit_csv "atlasnlp_stage1_random400_ai_audit - 400answers.csv" \
  --out pilot50.csv \
  --seed 20260826
```

This is a simple random sample of 50 from the completed 400 so its API-cost profile
is useful for rough extrapolation.

## 2. Run v2 on the 50 papers

If you still have the old extractor's `items/` directory containing
`<paper_id>.pages.json`:

```bash
python content_country_v2.py \
  --input pilot50.csv \
  --out pilot50_v2 \
  --cache_items_dir /path/to/OLD_RUN/items
```

If the page caches are unavailable:

```bash
python content_country_v2.py \
  --input pilot50.csv \
  --out pilot50_v2 \
  --allow_download
```

Important outputs:

- `pilot50_v2/items/*.v2.json` — full evidence + verifier record for each paper
- `pilot50_v2/summary.csv` — compact results
- `pilot50_v2/run_summary.json` — actual tokens, estimated spend, cost/paper,
  and naive 18,035-record extrapolation
- `pilot50_v2/failures.jsonl` — only if something failed

## 3. Score against the reviewed audit reference

```bash
python score_pilot.py \
  --pilot_csv pilot50.csv \
  --summary_csv pilot50_v2/summary.csv \
  --out_dir pilot50_v2/scoring
```

Open:

```text
pilot50_v2/scoring/pilot_disagreements_to_check.csv
```

Those are the rows worth quickly inspecting manually.

## Pilot go/no-go

Before a full run, prioritize:

1. **Explicit-country precision >= 90%**, preferably >=95%.
2. No systematic affiliation-only assignments.
3. No systematic multi-country expansion.
4. Reuse-only papers are being excluded sensibly.
5. False explicit assignment on reviewed-reference-empty rows is near zero.

Recall is secondary. Abstention is acceptable.

## 4. Production sharding

The script supports deterministic row sharding:

```bash
python content_country_v2.py \
  --input full_core_18035.csv \
  --out run_shard_00 \
  --cache_items_dir /path/to/OLD_RUN/items \
  --num_shards 32 \
  --shard_index 0
```

Launch shard indices `0..31` using your existing SLURM infrastructure. Give each
shard a separate output directory, then merge `summary.csv` files after completion.

With Tier 4 API limits, start conservatively rather than launching hundreds of
simultaneous workers. The actual pilot will tell you both latency and token cost.

## Input columns

The script accepts these aliases:

Required:
- `paper_url` (or `Paper link` / `paper_link`)
- `Dataset name` (or `dataset_name`)

Optional:
- `Dataset Link (URL)` / `Dataset link` / `dataset_link`
- `audit_record_id` / `record_id` / `source_row_index`

No AtlasNLP content-country or producer fields are read into the model prompt even
if they are present in the input CSV.

## Reproducibility

The per-paper JSON records:
- model snapshot
- prompt version
- reasoning settings
- paper-text hash
- response IDs
- timestamp
- token usage
- estimated cost
- whether page-1 affiliation removal succeeded
- whether the paper text had to be truncated

Do not overwrite the original Core fields. Merge these v2 outputs into new columns.


To merge shards run

python merge_v2_shards.py \
  --parent /path/to/shard_parent \
  --pattern "run_shard_*" \
  --out_dir merged_v2 \
  --input_csv full_core_18035.csv \
  --require_complete