#!/usr/bin/env python3
"""P0-4 Revised: Background TCR Stress Test using 10X Genomics Donor1 data.

Following STRESS_TEST.md protocol:
- 6 subsets with increasing background TCRs
- Subset 1: 2,876 antigen-specific (labeled) CDR3β
- Subsets 2-6: labeled + 4,737/9,474/14,211/18,948/23,685 background
- 3 repetitions per subset (a/b/c)
- Evaluate: retention, purity, false recruitment, labeled recovery

Uses Gliph2 input CSVs (have CDR3b, TRBV, TRBJ, count).
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

DATA_BASE = "/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1"
SUBSET_DIR = Path(f"{DATA_BASE}/subsets")
GLIPH2_DIR = Path(f"{DATA_BASE}/input/Gliph2")
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments/stress_test")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Load labeled CDR3s (subset_1)
# ═══════════════════════════════════════════════════════════════
s1 = pd.read_csv(SUBSET_DIR / "subset_1.txt")
LABELED_CDR3S = set(s1["cdr3"].values)
print(f"Labeled CDR3s: {len(LABELED_CDR3S)}")

# ═══════════════════════════════════════════════════════════════
# Methods to use (skip TCRMatch for O(n²) reasons)
# ═══════════════════════════════════════════════════════════════
config_obj = load_config()
config = dict(config_obj.__dict__)
all_clusterers = get_all_clusterers()
# Remove tcrmatch for scalability
fast_clusterers = {k: v for k, v in all_clusterers.items() if k != "tcrmatch"}
print(f"Methods: {sorted(fast_clusterers.keys())}")

# ═══════════════════════════════════════════════════════════════
# Subset info
# ═══════════════════════════════════════════════════════════════
SUBSETS = [1, 2, 3, 4, 5, 6]
N_REPS = 3  # a/b/c repetitions

def load_subset_gliph2(subset_id):
    """Load subset data from Gliph2 input CSV (has V/J genes)."""
    path = GLIPH2_DIR / f"subset_{subset_id}.csv"
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "CDR3b": "cdr3_beta",
        "TRBV": "v_beta",
        "TRBJ": "j_beta",
    })
    # Create tcr_id from CDR3b
    df["tcr_id"] = df["cdr3_beta"]
    # Label: specific vs background
    df["epitope"] = df["cdr3_beta"].apply(
        lambda x: "SPECIFIC" if x in LABELED_CDR3S else "BACKGROUND"
    )
    # Deduplicate by CDR3b (keep first)
    df = df.drop_duplicates(subset=["cdr3_beta"])
    return df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]]


def evaluate_stress(pred_labels, true_labels, tcr_ids, labeled_set):
    """Evaluate stress test metrics.

    pred_labels: cluster assignment per TCR (-1 = unclustered)
    true_labels: SPECIFIC or BACKGROUND
    tcr_ids: CDR3 sequence as ID
    labeled_set: set of labeled CDR3 sequences
    """
    n_total = len(tcr_ids)
    pred = np.asarray(pred_labels, dtype=object)
    true = np.asarray(true_labels)

    # Identify labeled vs background indices
    labeled_idx = np.array([tid in labeled_set for tid in tcr_ids])
    bg_idx = ~labeled_idx

    n_labeled = labeled_idx.sum()
    n_bg = bg_idx.sum()

    # 1. Retention: fraction of labeled CDR3s in clusters (not -1)
    labeled_pred = pred[labeled_idx]
    if isinstance(labeled_pred, np.ndarray) and len(labeled_pred) > 0:
        retention = float(np.sum(labeled_pred != -1)) / n_labeled
    elif n_labeled > 0:
        retention = float(np.sum(np.array(labeled_pred) != -1)) / n_labeled
    else:
        retention = 0.0

    # 2. Labeled recovery: fraction of labeled CDR3s that appear in clusters
    #    (same as retention for this definition)
    recovery = retention

    # 3. Cluster purity: for clusters containing labeled CDR3s,
    #    what fraction of members are labeled?
    cluster_labeled = {}  # cluster_id -> (n_labeled, n_bg)
    for i in range(n_total):
        cid = pred[i]
        if cid == -1:
            continue
        if cid not in cluster_labeled:
            cluster_labeled[cid] = [0, 0]
        if labeled_idx[i]:
            cluster_labeled[cid][0] += 1
        else:
            cluster_labeled[cid][1] += 1

    # Only consider clusters that have at least 1 labeled member
    clusters_with_labeled = {cid: counts for cid, counts in cluster_labeled.items() if counts[0] > 0}

    if clusters_with_labeled:
        purity_per_cluster = []
        weighted_purity_num = 0
        weighted_purity_den = 0
        for cid, (n_lab, n_bk) in clusters_with_labeled.items():
            total = n_lab + n_bk
            purity = n_lab / total if total > 0 else 0
            purity_per_cluster.append(purity)
            weighted_purity_num += n_lab
            weighted_purity_den += total
        mean_purity = np.mean(purity_per_cluster)
        weighted_purity = weighted_purity_num / weighted_purity_den if weighted_purity_den > 0 else 0
        n_clusters_with_labeled = len(clusters_with_labeled)
    else:
        mean_purity = 0
        weighted_purity = 0
        n_clusters_with_labeled = 0

    # 4. False recruitment: background CDR3s placed in clusters with labeled CDR3s
    bg_in_labeled_clusters = sum(counts[1] for counts in clusters_with_labeled.values())
    false_recruitment = bg_in_labeled_clusters / n_bg if n_bg > 0 else 0

    # 5. Total clusters and sizes
    all_clusters = set(pred[labeled_idx]) | set(pred[bg_idx])
    all_clusters.discard(-1)
    n_total_clusters = len(all_clusters)

    return {
        "n_total": n_total,
        "n_labeled": int(n_labeled),
        "n_background": int(n_bg),
        "bg_ratio": float(n_bg / n_labeled) if n_labeled > 0 else 0,
        "retention": float(retention),
        "recovery": float(recovery),
        "mean_purity": float(mean_purity),
        "weighted_purity": float(weighted_purity),
        "false_recruitment_rate": float(false_recruitment),
        "n_bg_recruited": int(bg_in_labeled_clusters),
        "n_clusters_with_labeled": n_clusters_with_labeled,
        "n_total_clusters": n_total_clusters,
    }


# ═══════════════════════════════════════════════════════════════
# Run experiments
# ═══════════════════════════════════════════════════════════════
all_results = []

for subset_id in SUBSETS:
    # Load data
    df = load_subset_gliph2(subset_id)
    n_labeled = (df["epitope"] == "SPECIFIC").sum()
    n_bg = (df["epitope"] == "BACKGROUND").sum()

    print(f"\n{'='*78}")
    print(f"SUBSET {subset_id}: {len(df)} TCRs ({n_labeled} labeled + {n_bg} background, ratio={n_bg/n_labeled:.2f})")
    print(f"{'='*78}")

    # Use Gliph2 data as the canonical source
    df_norm = normalize(df.copy())
    true_labels = df_norm["epitope"].values
    tcr_ids = df_norm["tcr_id"].values
    n_tcr = len(df_norm)

    # Choose clusterers based on subset size
    if subset_id <= 3:
        clusterers = fast_clusterers  # All except tcrmatch
    else:
        # Skip tcrmatch and also skip deeptcr for large subsets (GPU memory)
        clusterers = {k: v for k, v in fast_clusterers.items()}
        print(f"  Using {len(clusterers)} methods: {sorted(clusterers.keys())}")

    # Run all methods (1 repetition for now; can add a/b/c later)
    t0 = time.time()
    method_results = run_all_methods(
        df_norm, clusterers, config,
        OUTDIR / f"subset_{subset_id}"
    )
    runtime = time.time() - t0
    print(f"  All methods done in {runtime:.1f}s")

    # ═══════════════════════════════════════════════════════
    # Individual method evaluation
    # ═══════════════════════════════════════════════════════
    print(f"\n  Individual Methods:")
    print(f"  {'Method':<15s} {'Ret':>6s} {'WPur':>6s} {'MPur':>6s} {'FalseR':>8s} {'N_cls':>6s}")
    print(f"  {'-'*55}")

    # Collect assignments for consensus
    all_assigns = []
    for mname in sorted(method_results.keys()):
        assigns, rt = method_results[mname]
        all_assigns.extend(assigns)
        pred = assignments_to_labels(assigns, tcr_ids)
        ev = evaluate_stress(pred, true_labels, tcr_ids, LABELED_CDR3S)
        print(f"  {mname:<15s} {ev['retention']:6.3f} {ev['weighted_purity']:6.3f} "
              f"{ev['mean_purity']:6.3f} {ev['false_recruitment_rate']:8.4f} {ev['n_total_clusters']:6d}")

    # ═══════════════════════════════════════════════════════
    # Consensus
    # ═══════════════════════════════════════════════════════
    methods_list = sorted(set(a.method for a in all_assigns))
    weights = empirical_weights(methods_list)

    clusters, edges = balanced_consensus(all_assigns, weights)
    clusters = refine(clusters, edges, config)
    pred_cons = clusters_to_labels(clusters, tcr_ids)
    ev_cons = evaluate_stress(pred_cons, true_labels, tcr_ids, LABELED_CDR3S)

    print(f"\n  {'consensus':<15s} {ev_cons['retention']:6.3f} {ev_cons['weighted_purity']:6.3f} "
          f"{ev_cons['mean_purity']:6.3f} {ev_cons['false_recruitment_rate']:8.4f} {ev_cons['n_total_clusters']:6d}")

    # Equal weights consensus
    eq_weights = {m: 1.0 / len(methods_list) for m in methods_list}
    clusters_eq, edges_eq = balanced_consensus(all_assigns, eq_weights)
    clusters_eq = refine(clusters_eq, edges_eq, config)
    pred_eq = clusters_to_labels(clusters_eq, tcr_ids)
    ev_eq = evaluate_stress(pred_eq, true_labels, tcr_ids, LABELED_CDR3S)

    print(f"  {'consensus(EQ)':<15s} {ev_eq['retention']:6.3f} {ev_eq['weighted_purity']:6.3f} "
          f"{ev_eq['mean_purity']:6.3f} {ev_eq['false_recruitment_rate']:8.4f} {ev_eq['n_total_clusters']:6d}")

    result = {
        "subset": subset_id,
        "runtime_s": runtime,
        "consensus": ev_cons,
        "consensus_eq": ev_eq,
    }
    all_results.append(result)

# ═══════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("STRESS TEST SUMMARY (Consensus, Empirical Weights)")
print(f"{'='*78}")
print(f"{'Subset':>6s} {'N_tot':>7s} {'N_bg':>7s} {'Ratio':>6s} {'Ret':>6s} {'WPur':>6s} "
      f"{'MPur':>6s} {'FalseR':>8s} {'N_bg_rec':>9s} {'N_cls':>6s} {'Time':>6s}")
print("-" * 80)

for r in all_results:
    c = r["consensus"]
    print(f"{r['subset']:6d} {c['n_total']:7d} {c['n_background']:7d} {c['bg_ratio']:6.2f} "
          f"{c['retention']:6.3f} {c['weighted_purity']:6.3f} {c['mean_purity']:6.3f} "
          f"{c['false_recruitment_rate']:8.4f} {c['n_bg_recruited']:9d} "
          f"{c['n_total_clusters']:6d} {r['runtime_s']:6.0f}s")

# Retention degradation
print(f"\nRetention degradation:")
base_ret = all_results[0]["consensus"]["retention"]
for r in all_results:
    delta = r["consensus"]["retention"] - base_ret
    print(f"  Subset {r['subset']}: {r['consensus']['retention']:.3f} (Δ={delta:+.3f})")

# Save
with open(OUTDIR / "stress_test_results.json", "w") as f:
    json.dump({"results": all_results}, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'stress_test_results.json'}")
print("Done!")
