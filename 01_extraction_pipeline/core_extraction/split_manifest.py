# =============================================================================
# split_manifest.py — Split the ACL Anthology manifest into parallel shards
#
# Paper section: §3.3 "Automated Dataset Expansion"
# Role in pipeline: Step 2 — prepares shards for SLURM job array (run_shard.sh)
#
# Usage:
#   python pipeline/split_manifest.py data/manifest.jsonl <shards_dir> 600
#
# Arguments:
#   argv[1]: path to full manifest JSONL (data/manifest.jsonl — 119,964 papers)
#   argv[2]: output directory for shard files (manifest_000.jsonl, manifest_001.jsonl, ...)
#   argv[3]: number of papers per shard (600 papers/shard → 20 SLURM array jobs)
# =============================================================================
import json
import os
import sys

manifest = sys.argv[1]
out_dir = sys.argv[2]
chunk_size = int(sys.argv[3])

os.makedirs(out_dir, exist_ok=True)

with open(manifest) as f:
    lines = f.readlines()

for i in range(0, len(lines), chunk_size):
    shard_id = i // chunk_size
    out_path = os.path.join(out_dir, f"manifest_{shard_id:03d}.jsonl")

    with open(out_path, "w") as out:
        out.writelines(lines[i:i+chunk_size])

print(f"Created {shard_id+1} shards.")