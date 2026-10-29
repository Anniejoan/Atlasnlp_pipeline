#!/bin/bash
#SBATCH --job-name=atlasnlp_canary
#SBATCH --account=mihalcea0
#SBATCH --partition=standard
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/mihalcea_root/mihalcea0/jnwatu/atlasnlp_content_country_v2/logs/canary_%A_%a.out
#SBATCH --error=/scratch/mihalcea_root/mihalcea0/jnwatu/atlasnlp_content_country_v2/logs/canary_%A_%a.err

set -euo pipefail

export PATH="/sw/pkgs/arc/python3.11-anaconda/2024.02-1/bin:$PATH"

PIPELINE_DIR=/home/jnwatu/Worldbench/post_audit_pipeline/atlasnlp_content_country_v2
INPUT_CSV=/home/jnwatu/Worldbench/compiled_ACL/task_matrix_construction/unified_worldbench_views/language_cleanup_v2/worldbench_base_ground_truth_second_stage_cleaned.csv
SCRATCH_BASE=/scratch/mihalcea_root/mihalcea0/jnwatu/atlasnlp_content_country_v2
OUT_DIR="$SCRATCH_BASE/canary/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"

mkdir -p "$OUT_DIR"

cd "$PIPELINE_DIR"

python3 content_country_v2.py \
  --input "$INPUT_CSV" \
  --out "$OUT_DIR" \
  --allow_download \
  --num_shards 32 \
  --shard_index "$SLURM_ARRAY_TASK_ID" \
  --max_items 25 \
  --retries 6

echo "Shard $SLURM_ARRAY_TASK_ID canary complete."
