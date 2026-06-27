#!/usr/bin/env python3
"""DeepTCR hierarchical clustering, excluding singletons from all metrics."""

import sys, time, logging, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deeptcr-hier-no-sing")

DATA = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, sep="\t")
col_map = {c: c.lower() for c in df.columns}
df = df.rename(columns=col_map)
if "_pair" in df.columns:
    df = df.drop_duplicates(subset=["_pair"], keep="first")
cdr3_epi = df.groupby("cdr3_beta")["epitope"].nunique()
ambiguous = cdr3_epi[cdr3_epi > 1].index
if len(ambiguous) > 0:
    df = df[~df["cdr3_beta"].isin(ambiguous)]
df = df.reset_index(drop=True)
df["tcr_id"] = df.index.astype(str)
epitope_labels = {r["tcr_id"]: r["epitope"] for _, r in df.iterrows()}
logger.info("Data: %d TCRs, %d epitopes", len(df), df["epitope"].nunique())


def compute_ari(truth, pred):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(truth, pred)


def compute_bcubed(truth, pred, exclude_set=None):
    """Compute B-cubed on items that are in both truth and pred."""
    items = list(set(list(truth.keys()) + list(pred.keys())))
    if exclude_set:
        items = [x for x in items if x not in exclude_set]
    prec_sum = rec_sum = 0.0
    n = 0
    for item in items:
        tc = truth.get(item)
        pc = pred.get(item)
        if tc is None or pc is None:
            continue
        n += 1
        st = sum(1 for x in items if truth.get(x) == tc)
        sp = sum(1 for x in items if pred.get(x) == pc)
        co = sum(1 for x in items if truth.get(x) == tc and pred.get(x) == pc)
        prec_sum += co / sp if sp else 0
        rec_sum += co / st if st else 0
    p = prec_sum / n if n else 0
    r = rec_sum / n if n else 0
    f = 2 * p * r / (p + r) if (p + r) else 0
    return p, r, f


def purity_from_labels(truth, pred):
    clusters = {}
    for item, cid in pred.items():
        clusters.setdefault(cid, []).append(item)
    purities, sizes = [], []
    for members in clusters.values():
        if len(members) < 2:
            continue
        epi_counts = {}
        for m in members:
            epi = truth.get(m)
            if epi:
                epi_counts[epi] = epi_counts.get(epi, 0) + 1
        mc = max(epi_counts.values()) if epi_counts else 0
        purities.append(mc / len(members))
        sizes.append(len(members))
    if not sizes:
        return 0.0
    return sum(p * s for p, s in zip(purities, sizes)) / sum(sizes)


def pairwise_sensitivity(truth, pred, exclude_set=None):
    items = list(set(list(truth.keys()) + list(pred.keys())))
    if exclude_set:
        items = [x for x in items if x not in exclude_set]
    same_epi = co_clust = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ta, tb = truth.get(a), truth.get(b)
            if ta is not None and tb is not None and ta == tb:
                same_epi += 1
                if pred.get(a) == pred.get(b):
                    co_clust += 1
    return co_clust / same_epi if same_epi else 0


# Run
t0 = time.time()
from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

clusterer = DeepTCRWrapper(clustering_method="hierarchical")
prepared = clusterer.prepare_input(df, {})
raw = clusterer.run(prepared, OUTDIR / "_work_hier_no_sing")
assignments = clusterer.normalize(raw)
elapsed = time.time() - t0

cluster_labels = {a.tcr_id: a.cluster_id for a in assignments}
assigned_truth = {tid: epitope_labels[tid] for tid in cluster_labels if tid in epitope_labels}

# Identify singletons
cc = Counter(cluster_labels.values())
singleton_ids = set()
for tid, cid in cluster_labels.items():
    if cc[cid] == 1:
        singleton_ids.add(tid)

# All TCRs (including singletons)
ari_all = compute_ari(list(assigned_truth.values()), list(cluster_labels.values()))
bc_p_all, bc_r_all, bc_f1_all = compute_bcubed(assigned_truth, cluster_labels)
pur_all = purity_from_labels(assigned_truth, cluster_labels)
sens_all = pairwise_sensitivity(assigned_truth, cluster_labels)
retained_all = sum(1 for cid in cluster_labels.values() if cc[cid] >= 2)
retention_all = retained_all / len(assigned_truth)

# Excluding singletons
cluster_labels_ns = {k: v for k, v in cluster_labels.items() if k not in singleton_ids}
assigned_truth_ns = {tid: epitope_labels[tid] for tid in cluster_labels_ns if tid in epitope_labels}

ari_ns = compute_ari(list(assigned_truth_ns.values()), list(cluster_labels_ns.values()))
bc_p_ns, bc_r_ns, bc_f1_ns = compute_bcubed(assigned_truth_ns, cluster_labels_ns)
pur_ns = purity_from_labels(assigned_truth_ns, cluster_labels_ns)
sens_ns = pairwise_sensitivity(assigned_truth_ns, cluster_labels_ns)

# Report
n_clusters_all = len(set(cluster_labels.values()))
n_singletons = len(singleton_ids)
n_clustered_ns = len(assigned_truth_ns)
n_clusters_ns = len(set(cluster_labels_ns.values()))

print("")
print("=" * 70)
print("DeepTCR Hierarchical (silhouette threshold opt) -- v3_cd8")
print("=" * 70)
print(f"  ALL TCRs ({n_clusters_all} clusters, {n_singletons} singletons):")
print(f"    ARI:         {ari_all:.4f}")
print(f"    F1(bc):      {bc_f1_all:.4f}")
print(f"    Purity:      {pur_all:.4f}")
print(f"    Sensitivity: {sens_all:.4f}")
print(f"    Retention:   {retention_all:.4f}")
print(f"    N_clustered: {retained_all} / {len(assigned_truth)}")
print("")
print(f"  NO SINGLETONS ({n_clusters_ns} clusters, {n_clustered_ns} TCRs):")
print(f"    ARI:         {ari_ns:.4f}")
print(f"    F1(bc):      {bc_f1_ns:.4f}")
print(f"    Purity:      {pur_ns:.4f}")
print(f"    Sensitivity: {sens_ns:.4f}")
print(f"    Time:        {elapsed:.1f}s")
print("=" * 70)
print("")
print("Non-singleton cluster size distribution:")
for cid, cnt in sorted(cc.items(), key=lambda x: -x[1]):
    if cnt >= 2:
        print(f"  {cid}: {cnt}")

results = {
    "method": "deeptcr_hier_nosingleton",
    "all_tcrs": {
        "ari": round(ari_all, 4),
        "f1_bc": round(bc_f1_all, 4),
        "purity": round(pur_all, 4),
        "sensitivity": round(sens_all, 4),
        "retention": round(retention_all, 4),
        "n_clusters": n_clusters_all,
        "n_clustered": retained_all,
        "n_singletons": n_singletons,
    },
    "no_singletons": {
        "ari": round(ari_ns, 4),
        "f1_bc": round(bc_f1_ns, 4),
        "purity": round(pur_ns, 4),
        "sensitivity": round(sens_ns, 4),
        "n_clusters": n_clusters_ns,
        "n_clustered": n_clustered_ns,
    },
    "elapsed_s": round(elapsed, 1),
}
out_path = OUTDIR / "deeptcr_hier_nosingleton_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
