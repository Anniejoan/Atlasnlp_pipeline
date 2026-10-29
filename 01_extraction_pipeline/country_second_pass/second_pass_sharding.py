import math
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("country_second_pass_manifest.csv")
OUTDIR = Path("/scratch/<PI_ACCOUNT_ROOT>/<PI_ACCOUNT>/jnwatu/worldbench_country_second_pass/manifests")
N_SHARDS = 7

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
n = len(df)
rows_per_shard = math.ceil(n / N_SHARDS)

print(f"Total rows: {n}")
print(f"Rows per shard: {rows_per_shard}")

for i in range(N_SHARDS):
    start = i * rows_per_shard
    end = min((i + 1) * rows_per_shard, n)
    shard = df.iloc[start:end].copy()

    out_path = OUTDIR / f"manifest_{i+1:03d}.csv"
    shard.to_csv(out_path, index=False)
    print(f"{out_path} -> {len(shard)} rows")