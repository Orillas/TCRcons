#!/usr/bin/env python3
"""Debug DeepTCR — fix numpy bool issue, test clustering end-to-end."""

import sys
import os
import traceback
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Suppress noisy debug from numba etc
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import pandas as pd
import numpy as np
from pathlib import Path
from tcrconsensus.io.parser import normalize

# Load benchmark data
benchmark_path = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
df = pd.read_csv(benchmark_path, sep="\t", dtype=str)

rename_lower = {}
for col in df.columns:
    low = col.lower()
    if low != col and low in ["cdr3_alpha", "cdr3_beta", "v_alpha", "v_beta",
                               "j_alpha", "j_beta", "tcr_id", "epitope"]:
        rename_lower[col] = low
if rename_lower:
    df = df.rename(columns=rename_lower)

df_norm = normalize(df.copy())
print(f"Loaded: {len(df_norm)} rows")

# ---- Manually prepare DeepTCR input (skip wrapper for now) ----
print("\n" + "="*60)
print("STEP 1: Prepare input (manual)")
print("="*60)

import tempfile, shutil

records = []
for _, r in df_norm.iterrows():
    cdr3b = str(r.get("cdr3_beta", "") or "").strip()
    if not cdr3b:
        continue
    cdr3a = str(r.get("cdr3_alpha", "") or "").strip()
    vb = str(r.get("v_beta", "") or "").strip()
    jb = str(r.get("j_beta", "") or "").strip()
    va = str(r.get("v_alpha", "") or "").strip()
    ja = str(r.get("j_alpha", "") or "").strip()

    # Add *01 allele suffix
    for gene in ['vb', 'jb', 'va', 'ja']:
        val = locals()[gene]
        if val and '*' not in val:
            locals()[gene] = f"{val}*01"

    rec = {"CDR3b": cdr3b, "cloneCount": 1}
    if cdr3a:
        rec["CDR3a"] = cdr3a
    if vb:
        rec["v_beta"] = vb
    if jb:
        rec["j_beta"] = jb
    if va:
        rec["v_alpha"] = va
    if ja:
        rec["j_alpha"] = ja
    records.append(rec)

# Deduplicate
seen = set()
deduped = []
for rec in records:
    key = f"{rec.get('v_alpha','')}_{rec.get('CDR3a','')}_{rec.get('j_alpha','')}_{rec.get('v_beta','')}_{rec.get('CDR3b','')}_{rec.get('j_beta','')}"
    if key not in seen:
        seen.add(key)
        deduped.append(rec)

print(f"  Records: {len(records)}, after dedup: {len(deduped)}")

# Write TSV
data_dir = tempfile.mkdtemp()
try:
    sample_dir = os.path.join(data_dir, "sample_0")
    os.makedirs(sample_dir, exist_ok=True)

    input_df = pd.DataFrame(deduped)
    col_order = []
    if "CDR3a" in input_df.columns:
        col_order.append("CDR3a")
    col_order.append("CDR3b")
    col_order.append("cloneCount")
    for c in ["v_alpha", "j_alpha", "v_beta", "j_beta"]:
        if c in input_df.columns:
            col_order.append(c)
    input_df = input_df[col_order]
    input_df.to_csv(os.path.join(sample_dir, "tcr.tsv"), sep="\t", index=False)
    print(f"  TSV columns: {list(input_df.columns)}")

    # ---- CUDA setup ----
    venv_site = "/home/jilin/DeepTCR/.venv/lib/python3.10/site-packages"
    nvidia_lib = ":".join(
        f"{venv_site}/nvidia/{pkg}/lib"
        for pkg in ["cublas", "cuda_cupti", "cuda_nvrtc", "cuda_runtime",
                     "cudnn", "cufft", "curand", "cusolver", "cusparse", "nccl"]
    )
    os.environ["LD_LIBRARY_PATH"] = f"{nvidia_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    from DeepTCR.DeepTCR import DeepTCR_U

    print("\n" + "="*60)
    print("STEP 2: Get_Data")
    print("="*60)

    model_dir = os.path.join(data_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    dtn = DeepTCR_U(os.path.join(model_dir, "deeptcr"))

    dtn.Get_Data(
        directory=sample_dir,
        Load_Prev_Data=False,
        n_jobs=4,
        aa_column_beta="CDR3b",
        count_column="cloneCount",
        aggregate_by_aa=False,
        aa_column_alpha="CDR3a",
        v_beta_column="v_beta",
        j_beta_column="j_beta",
        v_alpha_column="v_alpha",
        j_alpha_column="j_alpha",
    )

    # FIX: use len() instead of truthiness for numpy arrays
    print(f"  beta_sequences: {len(dtn.beta_sequences)}")
    alpha_seqs = getattr(dtn, 'alpha_sequences', None)
    print(f"  alpha_sequences: {len(alpha_seqs) if alpha_seqs is not None and len(alpha_seqs) > 0 else 'None/empty'}")
    v_beta = getattr(dtn, 'v_beta', None)
    print(f"  v_beta: {type(v_beta).__name__}, len={len(v_beta) if v_beta is not None else 0}")

    print("\n" + "="*60)
    print("STEP 3: Train VAE")
    print("="*60)

    n_seqs = len(dtn.beta_sequences)
    dtn.Train_VAE(
        latent_dim=64,
        epochs_min=10,
        suppress_output=False,
        batch_size=min(5000, n_seqs),
    )
    print("  VAE training complete")

    if hasattr(dtn, 'features'):
        feat = dtn.features
        print(f"  Features shape: {feat.shape}, dtype: {feat.dtype}")
        print(f"  NaN: {np.isnan(feat).sum()}, range: [{feat.min():.4f}, {feat.max():.4f}]")
    else:
        print("  WARNING: no features!")
        sys.exit(1)

    print("\n" + "="*60)
    print("STEP 4: Hierarchical clustering")
    print("="*60)

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform, pdist
    from sklearn.metrics import silhouette_score

    distances = squareform(pdist(feat))
    print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}], mean: {distances.mean():.4f}")

    Z = linkage(squareform(distances), method="ward")
    print(f"  Linkage shape: {Z.shape}")

    # Find optimal threshold
    t_list = np.arange(1, 100, 1)
    best_sil, best_t = -1.0, t_list[-1]
    for t_val in t_list:
        idx = fcluster(Z, t_val, criterion="distance")
        n_uniq = len(np.unique(idx))
        if n_uniq <= 1 or n_uniq >= len(idx):
            continue
        sil = silhouette_score(feat, idx)
        if sil > best_sil:
            best_sil, best_t = sil, t_val

    cluster_assignments = fcluster(Z, best_t, criterion="distance")
    n_clusters = len(set(cluster_assignments))
    print(f"  Best t={best_t}, silhouette={best_sil:.4f}")
    print(f"  Clusters: {n_clusters}")
    sizes = np.bincount(cluster_assignments)
    print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, median={np.median(sizes):.0f}")
    print(f"  Size distribution: {dict(zip(*np.unique(sizes, return_counts=True)))}")

    print(f"\n  ✅ SUCCESS! DeepTCR produced {n_clusters} clusters from {len(cluster_assignments)} sequences")

except Exception as e:
    print(f"\n  ❌ FAILED: {e}")
    traceback.print_exc()
finally:
    shutil.rmtree(data_dir, ignore_errors=True)
