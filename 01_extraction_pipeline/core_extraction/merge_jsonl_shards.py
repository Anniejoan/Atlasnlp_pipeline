# =============================================================================
# merge_jsonl_shards.py — Merge JSONL output files from all SLURM shards
#
# Paper section: §3.3 "Automated Dataset Expansion"
# Role in pipeline: Step 4 — combines per-shard JSONL files into single files
#   written to compiled_ACL/ as the starting point for downstream cleaning.
#
# Run after all run_shard.sh SLURM jobs finish:
#   python pipeline/merge_jsonl_shards.py
#
# Output files in compiled_ACL/:
#   merged_results.jsonl         — all successful extraction results
#   merged_failures.jsonl        — papers that failed extraction
#   merged_usage.jsonl           — token usage per paper/stage
#   merged_dataset_compilation_papers.jsonl — skipped compilation papers
# =============================================================================
from pathlib import Path

base = Path(f"/scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_dataset_pipeline/runs")
outbase = "/home/jnwatu/Worldbench/compiled_ACL"
shards = sorted(base.glob("shard_*"))

for name in ["results.jsonl", "failures.jsonl", "usage.jsonl", "dataset_compilation_papers.jsonl"]:
    out_path = Path(outbase) / f"merged_{name}"
    with open(out_path, "w", encoding="utf-8") as fout:
        for shard in shards:
            p = shard / name
            if p.exists():
                with open(p, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
    print(f"Wrote {out_path}")