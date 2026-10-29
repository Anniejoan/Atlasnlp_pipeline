#!/bin/bash
#SBATCH --job-name=country_second_pass
#SBATCH --partition=standard
#SBATCH --account=mihalcea98
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --array=2-7
#SBATCH --output=/scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_country_second_pass/logs/%A_%a.out
#SBATCH --error=/scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_country_second_pass/logs/%A_%a.err

module load python

source ~/Worldbench/extractor_env/bin/activate

TASK_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

MANIFEST=/scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_country_second_pass/manifests/manifest_${TASK_ID}.csv
OUTDIR=/scratch/mihalcea_root/mihalcea0/jnwatu/worldbench_country_second_pass/runs/shard_${TASK_ID}

mkdir -p "$OUTDIR"

time python -u /home/jnwatu/Worldbench/compiled_ACL/country_second_pass/country_second_pass.py \
  --manifest "$MANIFEST" \
  --schema /home/jnwatu/Worldbench/compiled_ACL/country_second_pass/country_second_pass_schema.json \
  --out "$OUTDIR"