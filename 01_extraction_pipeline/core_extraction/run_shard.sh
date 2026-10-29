#!/bin/bash
# =============================================================================
# run_shard.sh — SLURM job array script for parallel LLM metadata extraction
#
# Paper section: §3.3 "Automated Dataset Expansion"
# Role in pipeline: Step 3 — submits one extraction job per manifest shard.
#
# Usage:
#   1. First split the manifest:
#        python pipeline/split_manifest.py data/manifest.jsonl <shards_dir> 600
#   2. Then submit this array job:
#        sbatch pipeline/run_shard.sh
#
# Each SLURM array task processes one manifest_NNN.jsonl shard and writes
# results to a shard_NNN/ output directory on scratch.
# After all jobs complete, merge results with:
#   python pipeline/merge_jsonl_shards.py
#   python pipeline/merge_tsv_shards.py
# =============================================================================
#SBATCH --job-name=dataset_extract
#SBATCH --partition=standard
#SBATCH --account=<PI_ACCOUNT>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=1-20
#SBATCH --output=/scratch/<PI_ACCOUNT_ROOT>/<PI_ACCOUNT>/jnwatu/worldbench_dataset_pipeline/logs/%A_%a.out
#SBATCH --error=/scratch/<PI_ACCOUNT_ROOT>/<PI_ACCOUNT>/jnwatu/worldbench_dataset_pipeline/logs/%A_%a.err

module load python

source ~/Worldbench/extractor_env/bin/activate

MANIFEST=/scratch/<PI_ACCOUNT_ROOT>/<PI_ACCOUNT>/jnwatu/worldbench_dataset_pipeline/manifests/manifest_$(printf "%03d" ${SLURM_ARRAY_TASK_ID}).jsonl
OUTDIR=/scratch/<PI_ACCOUNT_ROOT>/<PI_ACCOUNT>/jnwatu/worldbench_dataset_pipeline/runs/shard_$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

mkdir -p $OUTDIR

time python -u ~/Worldbench/pipeline/metadata_extractor.py \
  --manifest $MANIFEST \
  --schema ~/Worldbench/pipeline/schema.json \
  --out $OUTDIR