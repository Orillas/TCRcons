#!/usr/bin/env python3
"""Quick test: DeepTCR via wrapper with GPU fix."""
import sys, os, logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ['numba','tensorflow','absl','matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import pandas as pd
from pathlib import Path
from tcrconsensus.io.parser import normalize

benchmark_path = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
df = pd.read_csv(benchmark_path, sep="\t", dtype=str)
rename_lower = {col: col.lower() for col in df.columns
                if col.lower() != col and col.lower() in ["cdr3_alpha","cdr3_beta","v_alpha","v_beta","j_alpha","j_beta","tcr_id","epitope"]}
if rename_lower:
    df = df.rename(columns=rename_lower)
df_norm = normalize(df.copy())
print(f"Loaded: {len(df_norm)} rows")

from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper
wrapper = DeepTCRWrapper()
config = {}
workdir = Path("/tmp/deeptcr_test_workdir")
workdir.mkdir(exist_ok=True)

result = wrapper.safe_execute(df_norm, workdir, config)
print(f"\nStatus: {result.status.value}")
print(f"Assignments: {len(result.assignments)}")
print(f"Runtime: {result.runtime_seconds:.1f}s")
if result.error_message:
    print(f"Error: {result.error_message}")
if result.assignments:
    n_clusters = len(set(a.cluster_id for a in result.assignments))
    print(f"Clusters: {n_clusters}")
    print("SUCCESS!")
