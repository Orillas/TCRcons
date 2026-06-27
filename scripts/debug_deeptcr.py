#!/usr/bin/env python3
"""Debug DeepTCR wrapper failure — standalone test."""

import sys
import os
import traceback
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import pandas as pd
from pathlib import Path
from tcrconsensus.io.parser import normalize

# Load benchmark data
benchmark_path = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
df = pd.read_csv(benchmark_path, sep="\t", dtype=str)

# Rename uppercase columns to lowercase (the fix from before)
rename_lower = {}
for col in df.columns:
    low = col.lower()
    if low != col and low in ["cdr3_alpha", "cdr3_beta", "v_alpha", "v_beta",
                               "j_alpha", "j_beta", "tcr_id", "epitope"]:
        rename_lower[col] = low
if rename_lower:
    df = df.rename(columns=rename_lower)
    print(f"Renamed columns: {rename_lower}")

df_norm = normalize(df.copy())
print(f"Loaded: {len(df_norm)} rows, columns: {list(df_norm.columns)}")

# Now try DeepTCR step by step
print("\n" + "="*60)
print("STEP 1: Prepare input")
print("="*60)

from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

wrapper = DeepTCRWrapper()
config = {}

try:
    prepared = wrapper.prepare_input(df_norm, config)
    print(f"  Records: {len(prepared['records'])}")
    print(f"  Clonotypes: {len(prepared['clonotype_to_tcr_ids'])}")
    print(f"  Total TCRs: {prepared['total_tcrs']}")
    if prepared['records']:
        print(f"  Sample record: {prepared['records'][0]}")
except Exception as e:
    print(f"  FAILED at prepare_input: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("STEP 2: Get_Data (DeepTCR data loading)")
print("="*60)

# Set up CUDA paths
venv_site = "/home/jilin/DeepTCR/.venv/lib/python3.10/site-packages"
nvidia_lib = ":".join(
    f"{venv_site}/nvidia/{pkg}/lib"
    for pkg in ["cublas", "cuda_cupti", "cuda_nvrtc", "cuda_runtime",
                "cudnn", "cufft", "curand", "cusolver", "cusparse", "nccl"]
)
os.environ["LD_LIBRARY_PATH"] = f"{nvidia_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tempfile
import shutil

data_dir = tempfile.mkdtemp()
try:
    records = prepared["records"]
    sample_dir = os.path.join(data_dir, "sample_0")
    os.makedirs(sample_dir, exist_ok=True)

    for rec in records:
        rec["cloneCount"] = 1

    input_df = pd.DataFrame(records)

    # Column ordering matching DeepTCR internal expectations
    tsv_col_order = []
    if "CDR3a" in input_df.columns:
        tsv_col_order.append("CDR3a")
    tsv_col_order.append("CDR3b")
    tsv_col_order.append("cloneCount")
    if "v_alpha" in input_df.columns:
        tsv_col_order.append("v_alpha")
    if "j_alpha" in input_df.columns:
        tsv_col_order.append("j_alpha")
    if "v_beta" in input_df.columns:
        tsv_col_order.append("v_beta")
    if "j_beta" in input_df.columns:
        tsv_col_order.append("j_beta")
    input_df = input_df[tsv_col_order]

    tsv_path = os.path.join(sample_dir, "tcr.tsv")
    input_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Wrote TSV: {tsv_path}")
    print(f"  Columns: {list(input_df.columns)}")
    print(f"  Rows: {len(input_df)}")
    print(f"  First 3 rows:")
    print(input_df.head(3).to_string())

    from DeepTCR.DeepTCR import DeepTCR_U

    model_dir = os.path.join(data_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    dtn = DeepTCR_U(os.path.join(model_dir, "deeptcr"))

    load_kw = {
        "directory": sample_dir,
        "Load_Prev_Data": False,
        "n_jobs": 4,
        "aa_column_beta": "CDR3b",
        "count_column": "cloneCount",
        "aggregate_by_aa": False,
        "aa_column_alpha": "CDR3a",
        "v_beta_column": "v_beta",
        "j_beta_column": "j_beta",
        "v_alpha_column": "v_alpha",
        "j_alpha_column": "j_alpha",
    }

    print(f"\n  Get_Data kwargs: {load_kw}")
    dtn.Get_Data(**load_kw)

    print(f"  Loaded beta_sequences: {len(dtn.beta_sequences)}")
    if hasattr(dtn, 'alpha_sequences') and dtn.alpha_sequences:
        print(f"  Loaded alpha_sequences: {len(dtn.alpha_sequences)}")
    if hasattr(dtn, 'v_beta') and dtn.v_beta is not None:
        print(f"  Loaded v_beta: {len(dtn.v_beta)}")
    if hasattr(dtn, 'j_beta') and dtn.j_beta is not None:
        print(f"  Loaded j_beta: {len(dtn.j_beta)}")
    if hasattr(dtn, 'v_alpha') and dtn.v_alpha is not None:
        print(f"  Loaded v_alpha: {len(dtn.v_alpha)}")
    if hasattr(dtn, 'j_alpha') and dtn.j_alpha is not None:
        print(f"  Loaded j_alpha: {len(dtn.j_alpha)}")

    # Check what data attributes exist
    print(f"\n  DeepTCR object attributes:")
    for attr in ['beta_sequences', 'alpha_sequences', 'v_beta', 'j_beta', 'v_alpha', 'j_alpha',
                 'features', 'seq_len', 'labels', 'class_types']:
        if hasattr(dtn, attr):
            val = getattr(dtn, attr)
            if val is not None:
                typ = type(val).__name__
                if hasattr(val, '__len__'):
                    print(f"    {attr}: {typ} len={len(val)}")
                else:
                    print(f"    {attr}: {val}")
            else:
                print(f"    {attr}: None")

    print("\n" + "="*60)
    print("STEP 3: Train VAE")
    print("="*60)

    n_seqs = len(dtn.beta_sequences)
    dtn.Train_VAE(
        latent_dim=64,
        epochs_min=10,
        suppress_output=False,  # show output for debugging
        batch_size=min(5000, n_seqs),
    )
    print("  VAE training complete")

    # Check features
    if hasattr(dtn, 'features'):
        print(f"  Features shape: {dtn.features.shape}")
        print(f"  Features dtype: {dtn.features.dtype}")
        print(f"  Features sample: {dtn.features[0][:5]}")
        print(f"  Features NaN count: {np.isnan(dtn.features).sum()}")
    else:
        print("  WARNING: No features attribute after Train_VAE!")

    print("\n" + "="*60)
    print("STEP 4: Hierarchical clustering")
    print("="*60)

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform, pdist
    from sklearn.metrics import silhouette_score
    import numpy as np

    distances = squareform(pdist(dtn.features))
    print(f"  Distance matrix shape: {distances.shape}")
    print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  Mean distance: {distances.mean():.4f}")

    Z = linkage(squareform(distances), method="ward")
    print(f"  Linkage matrix shape: {Z.shape}")

    t_list = np.arange(1, 100, 1)
    best_sil, best_t = -1.0, t_list[-1]
    valid_count = 0
    for t_val in t_list:
        idx = fcluster(Z, t_val, criterion="distance")
        n_uniq = len(np.unique(idx[idx >= 0]))
        if n_uniq <= 1 or n_uniq >= len(idx):
            continue
        sil = silhouette_score(dtn.features[idx >= 0], idx[idx >= 0])
        valid_count += 1
        if sil > best_sil:
            best_sil, best_t = sil, t_val

    print(f"  Valid silhouette computations: {valid_count}")
    print(f"  Best t={best_t}, silhouette={best_sil:.4f}")

    cluster_assignments = fcluster(Z, best_t, criterion="distance")
    n_clusters = len(set(cluster_assignments))
    print(f"  Final clusters: {n_clusters}")
    print(f"  Cluster sizes: min={min(np.bincount(cluster_assignments))}, max={max(np.bincount(cluster_assignments))}")

    # Check mapping
    print("\n" + "="*60)
    print("STEP 5: Map back to tcr_ids")
    print("="*60)

    ck_list = list(prepared["clonotype_to_tcr_ids"].keys())
    print(f"  Clonotype keys: {len(ck_list)}")
    print(f"  Cluster assignments: {len(cluster_assignments)}")

    if len(ck_list) != len(cluster_assignments):
        print(f"  WARNING: Mismatch! clonotypes={len(ck_list)} vs assignments={len(cluster_assignments)}")

    assignments = []
    for i, cluster_num in enumerate(cluster_assignments):
        if i >= len(ck_list):
            print(f"  WARNING: assignment index {i} beyond clonotype list")
            break
        ck = ck_list[i]
        tcr_ids = prepared["clonotype_to_tcr_ids"].get(ck, [])
        for tid in tcr_ids:
            assignments.append({"tcr_id": tid, "cluster": int(cluster_num)})

    print(f"  Total assignments: {len(assignments)}")
    print(f"\n  SUCCESS! DeepTCR produced {len(set(a['cluster'] for a in assignments))} clusters, {len(assignments)} assignments")

except Exception as e:
    print(f"\n  FAILED: {e}")
    traceback.print_exc()
finally:
    shutil.rmtree(data_dir, ignore_errors=True)
