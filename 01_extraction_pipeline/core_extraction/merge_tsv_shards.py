# =============================================================================
# merge_tsv_shards.py — Merge TSV output files from all SLURM shards
#
# Paper section: §3.3 "Automated Dataset Expansion"
# Role in pipeline: Step 4 (companion to merge_jsonl_shards.py) — produces a
#   single human-readable spreadsheet of all extracted metadata for inspection.
#
# Run after all run_shard.sh SLURM jobs finish:
#   python pipeline/merge_tsv_shards.py
#
# Output:
#   compiled_ACL/merged_results.tsv — tab-separated table of all extracted records
#   (This becomes the base for validation_triage.py in compiled_ACL/)
# =============================================================================
from pathlib import Path

base = Path(f"/scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_dataset_pipeline/runs")
outbase = "/home/jnwatu/Worldbench/compiled_ACL"
shards = sorted(base.glob("shard_*"))
out_tsv = Path(outbase) / "merged_results.tsv"

with open(out_tsv, "w", encoding="utf-8") as fout:
    wrote_header = False
    for shard in shards:
        tsv = shard / "results.tsv"
        if not tsv.exists():
            continue
        with open(tsv, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        if not lines:
            continue
        if wrote_header:
            fout.writelines(lines[1:])
        else:
            fout.writelines(lines)
            wrote_header = True

print(f"Wrote {out_tsv}")