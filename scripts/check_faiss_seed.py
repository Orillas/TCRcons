import faiss
import numpy as np
import inspect

# 1. Check ClusteringParameters fields
cp = faiss.ClusteringParameters()
seed_fields = [attr for attr in dir(cp) if "seed" in attr.lower()]
print("ClusteringParameters seed-related fields:", seed_fields)

all_fields = [attr for attr in dir(cp) if not attr.startswith("_")]
print("All ClusteringParameters fields:", all_fields)
print("Default niter:", cp.niter)
print("Default redo:", cp.nredo)
print("Has seed field:", hasattr(cp, "seed"))

# 2. Test FAISS determinism WITHOUT setting np seed
print("\n=== FAISS K-means determinism test (no np seed set) ===")
data = np.random.randn(1000, 10).astype("float32")

labels_list = []
for run in range(3):
    km = faiss.Kmeans(10, 20, min_points_per_centroid=1)
    km.train(data)
    labels = km.index.search(data, 1)[1].flatten()
    labels_list.append(labels)

for i in range(1, 3):
    match = (labels_list[0] == labels_list[i]).sum()
    print(f"  Run 0 vs Run {i}: {match}/1000 match ({match/10:.1f}%)")

# 3. Test FAISS determinism WITH seed
print("\n=== FAISS K-means determinism test (WITH seed=42) ===")
labels_list2 = []
for run in range(3):
    km = faiss.Kmeans(10, 20, min_points_per_centroid=1, seed=42)
    km.train(data)
    labels = km.index.search(data, 1)[1].flatten()
    labels_list2.append(labels)

for i in range(1, 3):
    match = (labels_list2[0] == labels_list2[i]).sum()
    print(f"  Run 0 vs Run {i}: {match}/1000 match ({match/10:.1f}%)")

# 4. Now test actual clusTCR clustering variance
print("\n=== clusTCR actual clustering variance (3 runs) ===")
import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
from clustcr.clustering.clustering import Clustering
from clustcr.input.vdjdb import parse_vdjdb

vdjdb = parse_vdjdb("./clustcr/input/vdjdb/vdjdb_full.txt", q=1)
epitopes = vdjdb[["cdr3.beta","v.beta","antigen.epitope"]].dropna().drop_duplicates()
epitopes = epitopes.rename(columns={"cdr3.beta":"junction_aa","v.beta":"v_call","antigen.epitope":"epitope"})
cdr3 = epitopes["junction_aa"].drop_duplicates()
epi_for_metrics = epitopes[["junction_aa","epitope"]].drop_duplicates()
print(f"CDR3 sequences: {len(cdr3)}, Epitope pairs: {len(epi_for_metrics)}")

results = []
for run in range(3):
    result = Clustering(n_cpus=8).fit(cdr3)
    m = result.metrics(epi_for_metrics).summary()
    ret = m[m["metrics"]=="retention"]["actual"].values[0]
    pur = m[m["metrics"]=="purity"]["actual"].values[0]
    p90 = m[m["metrics"]=="purity_90"]["actual"].values[0]
    con = m[m["metrics"]=="consistency"]["actual"].values[0]
    results.append((ret, pur, p90, con))
    print(f"  Run {run+1}: Ret={ret:.4f} Pur={pur:.4f} P90={p90:.4f} Con={con:.4f}")

# Compute variance
import pandas as pd
df = pd.DataFrame(results, columns=["retention","purity","purity_90","consistency"])
print("\n  Variance across 3 runs:")
for col in df.columns:
    print(f"    {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, range=[{df[col].min():.4f}, {df[col].max():.4f}]")
