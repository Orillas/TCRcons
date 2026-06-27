#!/usr/bin/env python3
"""P0-3: Intersection/Union consensus baselines.

- Intersection: TCRs co-clustered by ALL methods (strictest)
- Union: TCRs co-clustered by ANY method (loosest)
- Random: Random clustering with same N_clusters

These are mandatory baselines per reviewer.md Section 9.
"""

import sys, time, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/scripts")
sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.refinement.refiner import refine
from exp_shared import (
    get_all_clusterers, run_all_methods,
    evaluate_clustering, assignments_to_labels, clusters_to_labels,
)
from sklearn.metrics import adjusted_rand_score

BENCHMARK = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_all.tsv"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Load data + run methods
# ═══════════════════════════════════════════════════════════════
print("=" * 78)
print("P0-3: INTERSECTION / UNION / RANDOM BASELINES")
print("=" * 78)

df = pd.read_csv(BENCHMARK, sep="\t", dtype=str)
rename_map = {c: c.lower() for c in df.columns if c != c.lower()}
if rename_map:
    df = df.rename(columns=rename_map)
df_norm = normalize(df.copy())
true_labels = df_norm["epitope"].values
tcr_ids = df_norm["tcr_id"].values
n_tcr = len(df_norm)

print(f"  {n_tcr} TCRs, {df_norm['epitope'].nunique()} epitopes")

config_obj = load_config()
config = dict(config_obj.__dict__)
clusterers = get_all_clusterers()

print("\nRunning all methods...")
t0 = time.time()
method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / "iob_methods")
print(f"All methods done in {time.time()-t0:.1f}s")

# Collect per-method cluster assignments
ALL_METHODS = sorted(method_results.keys())
tcr_id_to_idx = {tid: i for i, tid in enumerate(tcr_ids)}

# Build per-method cluster membership: tcr_id -> cluster_id
method_clusters = {}
for mname in ALL_METHODS:
    assigns, rt = method_results[mname]
    clust = defaultdict(set)
    for a in assigns:
        clust[a.cluster_id].add(a.tcr_id)
    method_clusters[mname] = dict(clust)
    print(f"  {mname}: {len(clust)} clusters")

# ═══════════════════════════════════════════════════════════════
# Helper: pairwise co-clustering from cluster assignments
# ═══════════════════════════════════════════════════════════════
def build_pairwise_cocluster(method_clusters, tcr_ids):
    """Build a dict: method -> set of (i,j) pairs that are co-clustered."""
    tcr_id_to_idx = {tid: i for i, tid in enumerate(tcr_ids)}
    method_pairs = {}
    for mname, clusters in method_clusters.items():
        pairs = set()
        for cid, members in clusters.items():
            indices = [tcr_id_to_idx[tid] for tid in members if tid in tcr_id_to_idx]
            for i, j in combinations(indices, 2):
                pairs.add((min(i, j), max(i, j)))
        method_pairs[mname] = pairs
        print(f"  {mname}: {len(pairs)} co-clustered pairs")
    return method_pairs

print("\nBuilding pairwise co-clustering...")
method_pairs = build_pairwise_cocluster(method_clusters, tcr_ids)

# ═══════════════════════════════════════════════════════════════
# Baseline 1: INTERSECTION (ALL methods must agree)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("BASELINE 1: INTERSECTION (all methods must co-cluster)")
print(f"{'='*78}")

all_pair_sets = list(method_pairs.values())
if all_pair_sets:
    intersection_pairs = all_pair_sets[0]
    for ps in all_pair_sets[1:]:
        intersection_pairs = intersection_pairs & ps
else:
    intersection_pairs = set()

print(f"  Intersection pairs: {len(intersection_pairs)}")

# Build clusters from intersection pairs using Union-Find
parent_int = list(range(n_tcr))
def find(x):
    while parent_int[x] != x:
        parent_int[x] = parent_int[parent_int[x]]
        x = parent_int[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent_int[ra] = rb

for i, j in intersection_pairs:
    union(i, j)

# Build clusters
int_groups = defaultdict(list)
for i in range(n_tcr):
    int_groups[find(i)].append(i)

# Assign labels: only keep clusters with >= 2 members
int_pred = np.full(n_tcr, -1)
cluster_id = 0
for root, members in int_groups.items():
    if len(members) >= 2:
        for m in members:
            int_pred[m] = cluster_id
        cluster_id += 1

int_metrics = evaluate_clustering(int_pred, true_labels, n_tcr, "intersection")
n_int_cls = len(set(int_pred) - {-1})
print(f"  Clusters: {n_int_cls}, Retained: {(int_pred != -1).sum()}/{n_tcr}")
print(f"  ARI={int_metrics['ari']:.4f}, AMI={int_metrics['ami']:.4f}, "
      f"Purity={int_metrics['purity']:.4f}, Sens={int_metrics['sensitivity']:.4f}, "
      f"F1={int_metrics['f1']:.4f}, Ret={int_metrics['retention']:.4f}")

# ═══════════════════════════════════════════════════════════════
# Baseline 2: UNION (ANY method co-clusters)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("BASELINE 2: UNION (any method co-clusters)")
print(f"{'='*78}")

union_pairs = set()
for ps in all_pair_sets:
    union_pairs = union_pairs | ps

print(f"  Union pairs: {len(union_pairs)}")

parent_uni = list(range(n_tcr))
def find_u(x):
    while parent_uni[x] != x:
        parent_uni[x] = parent_uni[parent_uni[x]]
        x = parent_uni[x]
    return x
def union_u(a, b):
    ra, rb = find_u(a), find_u(b)
    if ra != rb:
        parent_uni[ra] = rb

for i, j in union_pairs:
    union_u(i, j)

uni_groups = defaultdict(list)
for i in range(n_tcr):
    uni_groups[find_u(i)].append(i)

uni_pred = np.full(n_tcr, -1)
cluster_id = 0
for root, members in uni_groups.items():
    if len(members) >= 2:
        for m in members:
            uni_pred[m] = cluster_id
        cluster_id += 1

uni_metrics = evaluate_clustering(uni_pred, true_labels, n_tcr, "union")
n_uni_cls = len(set(uni_pred) - {-1})
print(f"  Clusters: {n_uni_cls}, Retained: {(uni_pred != -1).sum()}/{n_tcr}")
print(f"  ARI={uni_metrics['ari']:.4f}, AMI={uni_metrics['ami']:.4f}, "
      f"Purity={uni_metrics['purity']:.4f}, Sens={uni_metrics['sensitivity']:.4f}, "
      f"F1={uni_metrics['f1']:.4f}, Ret={uni_metrics['retention']:.4f}")

# ═══════════════════════════════════════════════════════════════
# Baseline 3: RANDOM clustering (sanity check)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("BASELINE 3: RANDOM CLUSTERING (sanity check)")
print(f"{'='*78}")

# Target: same number of clusters as consensus
# First get consensus cluster count for reference
all_assigns = []
for mname in ALL_METHODS:
    assigns, rt = method_results[mname]
    all_assigns.extend(assigns)
weights = empirical_weights(ALL_METHODS)
clusters_ref, edges_ref = balanced_consensus(all_assigns, weights)
clusters_ref = refine(clusters_ref, edges_ref, config)
n_target = len(clusters_ref)

# Random clustering: assign each TCR to a random cluster
rng = np.random.RandomState(42)
random_results = {}
for trial in range(5):
    random_pred = rng.randint(0, n_target, size=n_tcr)
    random_ari = adjusted_rand_score(true_labels, random_pred)
    random_results[f"trial_{trial}"] = random_ari
    print(f"  Trial {trial}: ARI = {random_ari:.4f} (n_clusters={n_target})")

random_mean_ari = np.mean(list(random_results.values()))
print(f"  Mean random ARI: {random_mean_ari:.4f}")

# ═══════════════════════════════════════════════════════════════
# Also get consensus for comparison
# ═══════════════════════════════════════════════════════════════
pred_cons = clusters_to_labels(clusters_ref, tcr_ids)
cons_metrics = evaluate_clustering(pred_cons, true_labels, n_tcr, "consensus")

# ═══════════════════════════════════════════════════════════════
# Comparison table
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("COMPARISON TABLE")
print(f"{'='*78}")
print(f"{'Method':<25s} {'ARI':>8s} {'AMI':>8s} {'Purity':>8s} {'Sens':>8s} {'F1':>8s} {'Ret':>8s} {'N_cls':>8s}")
print("-" * 85)

rows = [
    ("Consensus (empirical)", cons_metrics, len(clusters_ref)),
    ("Intersection (strict)", int_metrics, n_int_cls),
    ("Union (loose)", uni_metrics, n_uni_cls),
]

for name, m, nc in rows:
    print(f"{name:<25s} {m['ari']:8.4f} {m['ami']:8.4f} {m['purity']:8.4f} "
          f"{m['sensitivity']:8.4f} {m['f1']:8.4f} {m['retention']:8.4f} {nc:8d}")

print(f"{'Random (mean of 5)':<25s} {random_mean_ari:8.4f} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s} {n_target:8d}")

# Individual methods
print(f"\n{'Individual Methods:':<25s}")
for mname in ALL_METHODS:
    assigns, rt = method_results[mname]
    pred = assignments_to_labels(assigns, tcr_ids)
    m = evaluate_clustering(pred, true_labels, n_tcr, mname)
    nc = len(set(pred) - {-1})
    print(f"  {mname:<23s} {m['ari']:8.4f} {m['ami']:8.4f} {m['purity']:8.4f} "
          f"{m['sensitivity']:8.4f} {m['f1']:8.4f} {m['retention']:8.4f} {nc:8d}")

# Save
with open(OUTDIR / "p0_3_intersection_union.json", "w") as f:
    json.dump({
        "consensus": {**cons_metrics, "n_clusters": len(clusters_ref)},
        "intersection": {**int_metrics, "n_clusters": n_int_cls,
                         "intersection_pairs": len(intersection_pairs)},
        "union": {**uni_metrics, "n_clusters": n_uni_cls,
                  "union_pairs": len(union_pairs)},
        "random": {"mean_ari": random_mean_ari,
                   "trials": random_results,
                   "n_target_clusters": n_target},
    }, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'p0_3_intersection_union.json'}")
print("Done!")
