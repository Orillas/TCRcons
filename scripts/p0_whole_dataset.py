#!/usr/bin/env python3
"""P0-1: Whole-dataset evaluation + GIANA analysis.

Runs all 7 methods + consensus on v3_all as a WHOLE dataset
(no per-epitope splitting). This is the primary result for the paper.

Also compares:
  - Whole-dataset ARI for each method
  - Per-epitope ARI (one-vs-rest) for each method
  - Explains why GIANA wins per-epitope but loses whole-dataset
"""

import sys, time, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

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

BENCHMARK = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_all.tsv"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("=" * 78)
print("P0-1: WHOLE-DATASET PRIMARY RESULTS")
print("=" * 78)

df = pd.read_csv(BENCHMARK, sep="\t", dtype=str)
# Rename uppercase columns
rename_map = {c: c.lower() for c in df.columns if c != c.lower()}
if rename_map:
    df = df.rename(columns=rename_map)

df_norm = normalize(df.copy())
true_labels = df_norm["epitope"].values
tcr_ids = df_norm["tcr_id"].values
n_tcr = len(df_norm)
n_epi = df_norm["epitope"].nunique()
epi_counts = df_norm["epitope"].value_counts()

print(f"  {n_tcr} TCRs, {n_epi} epitopes")
print(f"  Epitope size: min={epi_counts.min()}, median={epi_counts.median():.0f}, max={epi_counts.max()}")
print(f"  TCRs per epitope (top 10):")
for epi, cnt in epi_counts.head(10).items():
    print(f"    {epi}: {cnt}")

# ═══════════════════════════════════════════════════════════════
# 2. Run all methods (whole-dataset)
# ═══════════════════════════════════════════════════════════════
config_obj = load_config()
config = dict(config_obj.__dict__)
clusterers = get_all_clusterers()
print(f"\nMethods: {sorted(clusterers.keys())}")

print("\n--- Running all methods (whole-dataset) ---")
t0 = time.time()
method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / "whole_ds_methods")
print(f"All methods done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# 3. Individual method metrics (whole-dataset)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("WHOLE-DATASET RESULTS (all epitopes mixed together)")
print("=" * 78)
print(f"{'Method':<15s} {'ARI':>8s} {'AMI':>8s} {'Purity':>8s} {'Sens':>8s} {'F1':>8s} {'Ret':>8s} {'N_cls':>8s}")
print("-" * 80)

indiv_metrics = {}
all_assigns = []
for mname in sorted(method_results.keys()):
    assigns, rt = method_results[mname]
    all_assigns.extend(assigns)
    pred = assignments_to_labels(assigns, tcr_ids)
    m = evaluate_clustering(pred, true_labels, n_tcr, mname)
    indiv_metrics[mname] = m
    n_clusters = len(set(pred) - {-1})
    print(f"{mname:<15s} {m.get('ari',0):8.4f} {m.get('ami',0):8.4f} {m.get('purity',0):8.4f} "
          f"{m.get('sensitivity',0):8.4f} {m.get('f1',0):8.4f} {m.get('retention',0):8.4f} {n_clusters:8d}")

# ═══════════════════════════════════════════════════════════════
# 4. Consensus (empirical weights, whole-dataset)
# ═══════════════════════════════════════════════════════════════
methods_list = sorted(set(a.method for a in all_assigns))
weights = empirical_weights(methods_list)

print(f"\nEmpirical weights:")
for k, v in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

clusters, edges = balanced_consensus(all_assigns, weights)
clusters = refine(clusters, edges, config)
pred_cons = clusters_to_labels(clusters, tcr_ids)
cons_metrics = evaluate_clustering(pred_cons, true_labels, n_tcr, "consensus")
n_cons_clusters = len(set(pred_cons) - {-1})

print(f"\n{'consensus':<15s} {cons_metrics.get('ari',0):8.4f} {cons_metrics.get('ami',0):8.4f} "
      f"{cons_metrics.get('purity',0):8.4f} {cons_metrics.get('sensitivity',0):8.4f} "
      f"{cons_metrics.get('f1',0):8.4f} {cons_metrics.get('retention',0):8.4f} {n_cons_clusters:8d}")

# Equal weights consensus
eq_weights = {m: 1.0/len(methods_list) for m in methods_list}
clusters_eq, edges_eq = balanced_consensus(all_assigns, eq_weights)
clusters_eq = refine(clusters_eq, edges_eq, config)
pred_eq = clusters_to_labels(clusters_eq, tcr_ids)
eq_metrics = evaluate_clustering(pred_eq, true_labels, n_tcr, "consensus_equal")
n_eq_clusters = len(set(pred_eq) - {-1})

print(f"{'consensus(EQ)':<15s} {eq_metrics.get('ari',0):8.4f} {eq_metrics.get('ami',0):8.4f} "
      f"{eq_metrics.get('purity',0):8.4f} {eq_metrics.get('sensitivity',0):8.4f} "
      f"{eq_metrics.get('f1',0):8.4f} {eq_metrics.get('retention',0):8.4f} {n_eq_clusters:8d}")

# ═══════════════════════════════════════════════════════════════
# 5. GIANA analysis: why per-epitope wins, whole-dataset loses
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("GIANA ANALYSIS: Per-epitope vs Whole-dataset")
print("=" * 78)

# GIANA cluster size distribution
giana_assigns = [a for a in all_assigns if a.method == "giana"]
giana_pred = assignments_to_labels(giana_assigns, tcr_ids)

# Cluster purity analysis
giana_clusters_map = {}
for a in giana_assigns:
    if a.cluster_id not in giana_clusters_map:
        giana_clusters_map[a.cluster_id] = []
    giana_clusters_map[a.cluster_id].append(a.tcr_id)

n_giana_clusters = len(giana_clusters_map)
giana_sizes = [len(v) for v in giana_clusters_map.values()]
print(f"\nGIANA: {n_giana_clusters} clusters")
print(f"  Cluster size: min={min(giana_sizes)}, median={np.median(giana_sizes):.0f}, max={max(giana_sizes)}")
print(f"  Large clusters (>50): {sum(1 for s in giana_sizes if s > 50)}")
print(f"  Singleton clusters: {sum(1 for s in giana_sizes if s == 1)}")

# How many epitopes per GIANA cluster? (cross-contamination)
giana_cross = []
for cid, members in giana_clusters_map.items():
    epi_set = set()
    for tid in members:
        idx = np.where(tcr_ids == tid)[0]
        if len(idx) > 0:
            epi_set.add(true_labels[idx[0]])
    giana_cross.append((cid, len(members), len(epi_set), epi_set))

cross_counts = [n_epi for _, _, n_epi, _ in giana_cross]
print(f"\n  Epitopes per cluster: min={min(cross_counts)}, median={np.median(cross_counts):.1f}, max={max(cross_counts)}")
print(f"  Pure clusters (1 epitope): {sum(1 for c in cross_counts if c == 1)}")
print(f"  Mixed clusters (>1 epitope): {sum(1 for c in cross_counts if c > 1)}")

# Same analysis for consensus (clusters is list[ConsensusCluster])
cons_cross = []
for cl in clusters:
    epi_set = set()
    for tid in cl.member_ids:
        idx = np.where(tcr_ids == tid)[0]
        if len(idx) > 0:
            epi_set.add(true_labels[idx[0]])
    cons_cross.append((cl.cluster_id, len(cl.member_ids), len(epi_set), epi_set))

cons_sizes = [len(cl.member_ids) for cl in clusters]
cons_cross_counts = [n_epi for _, _, n_epi, _ in cons_cross]
print(f"\nConsensus: {len(clusters)} clusters")
print(f"  Cluster size: min={min(cons_sizes)}, median={np.median(cons_sizes):.0f}, max={max(cons_sizes)}")
print(f"  Pure clusters (1 epitope): {sum(1 for c in cons_cross_counts if c == 1)}")
print(f"  Mixed clusters (>1 epitope): {sum(1 for c in cons_cross_counts if c > 1)}")
print(f"  Epitopes per cluster: min={min(cons_cross_counts)}, median={np.median(cons_cross_counts):.1f}, max={max(cons_cross_counts)}")

# Largest mixed clusters
print(f"\n  Top 5 largest MIXED GIANA clusters (over-merge evidence):")
mixed = [(cid, sz, n_epi, epis) for cid, sz, n_epi, epis in giana_cross if n_epi > 1]
mixed.sort(key=lambda x: -x[1])
for cid, sz, n_epi, epis in mixed[:5]:
    top_epis = Counter({e: 0 for e in epis})
    for tid in giana_clusters_map[cid]:
        idx = np.where(tcr_ids == tid)[0]
        if len(idx) > 0:
            top_epis[true_labels[idx[0]]] += 1
    top3 = top_epis.most_common(3)
    top_str = ", ".join(f"{e}({c})" for e, c in top3)
    print(f"    {cid}: {sz} TCRs, {n_epi} epitopes [{top_str}]")

# ═══════════════════════════════════════════════════════════════
# 6. Per-epitope analysis (one-vs-rest) for comparison
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("PER-EPITOPE ANALYSIS (for reference — NOT primary result)")
print("=" * 78)

# Load exp4 data for comparison
exp4_path = "/home/jilin/DeepTCR/tcrconsensus/results/exp4_improved/exp4_improved_results.tsv"
try:
    exp4 = pd.read_csv(exp4_path, sep="\t")
    exp4.loc[exp4["method"] == "improved_cc_empirical", "method"] = "consensus(emp)"
    exp4.loc[exp4["method"] == "control_cc_equal", "method"] = "consensus(eq)"

    key_methods = ["giana", "gliph2", "consensus(emp)", "consensus(eq)", "hd_baseline"]
    print(f"{'Method':<20s} {'MeanARI':>8s} {'MedARI':>8s} {'BestCount':>10s} {'Top3Count':>10s}")
    print("-" * 60)
    for method in key_methods:
        sub = exp4[exp4["method"] == method]
        if len(sub) > 0:
            ari = sub["ari"]
            # Count how many epitopes this method is best
            best_count = 0
            top3_count = 0
            for epi in exp4["target_epitope"].unique():
                epi_sub = exp4[exp4["target_epitope"] == epi]
                ranked = epi_sub.sort_values("ari", ascending=False)
                methods_ranked = ranked["method"].tolist()
                if methods_ranked[0] == method:
                    best_count += 1
                if method in methods_ranked[:3]:
                    top3_count += 1
            print(f"{method:<20s} {ari.mean():8.4f} {ari.median():8.4f} {best_count:10d} {top3_count:10d}")

    print("\n  NOTE: Per-epitope uses one-vs-rest isolation → favors methods")
    print("  with high intra-epitope purity (GIANA). Whole-dataset is the")
    print("  realistic use case and consensus dominates there.")
except Exception as e:
    print(f"  Could not load exp4: {e}")

# ═══════════════════════════════════════════════════════════════
# 7. Save summary
# ═══════════════════════════════════════════════════════════════
summary = {
    "dataset": "v3_all",
    "n_tcr": int(n_tcr),
    "n_epitope": int(n_epi),
    "whole_dataset": {
        **{k: v for k, v in indiv_metrics.items()},
        "consensus_empirical": cons_metrics,
        "consensus_equal": eq_metrics,
    },
    "giana_analysis": {
        "n_clusters": n_giana_clusters,
        "pure_clusters": sum(1 for c in cross_counts if c == 1),
        "mixed_clusters": sum(1 for c in cross_counts if c > 1),
        "max_epitopes_per_cluster": max(cross_counts),
    },
    "consensus_analysis": {
        "n_clusters": len(clusters),
        "pure_clusters": sum(1 for c in cons_cross_counts if c == 1),
        "mixed_clusters": sum(1 for c in cons_cross_counts if c > 1),
        "max_epitopes_per_cluster": max(cons_cross_counts),
    },
}

with open(OUTDIR / "p0_1_whole_dataset.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'p0_1_whole_dataset.json'}")
print("\nDone!")
