#!/usr/bin/env python3
"""P0-4 Final: 10X Donor1 Epitope-Level Background Stress Test.

Data:
  - Labeled: 2,684 CDR3β from 10X Donor1 with 44 epitope labels (mean UMI > 10)
  - Background: Real 10X non-specific CDR3β (same donor repertoire)
  - 6 subsets: 0 / 4,737 / 9,474 / ... / 23,685 background CDR3s added

Evaluation:
  - ARI (44 epitope classes + BACKGROUND)
  - Labeled retention (fraction of specific CDR3s in clusters)
  - False recruitment (background CDR3s in specific clusters)
  - Weighted purity (labeled fraction in labeled-containing clusters)
"""

import sys, time, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

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

# ── Paths ──
DATA_BASE = Path("/home/jilin/DeepTCR/i3-unit-TCR_Unsupervised_Benchmark-469696e/Data/10X/Donor1")
GLIPH2_DIR = DATA_BASE / "input" / "Gliph2"
LABEL_JSON = "/home/jilin/DeepTCR/10X_Donor1_raw/cdr3_epitope_h5.json"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments/stress_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Load epitope labels ──
with open(LABEL_JSON) as f:
    label_data = json.load(f)
CDR3_EPI = label_data["cdr3_to_epitopes"]
print(f"Loaded {len(CDR3_EPI)} CDR3β → epitope mappings")

# ── Config ──
config_obj = load_config()
config = dict(config_obj.__dict__)
all_clusterers = get_all_clusterers()
# Skip TCRMatch for scalability
clusterers = {k: v for k, v in all_clusterers.items() if k != "tcrmatch"}
print(f"Methods: {sorted(clusterers.keys())}")

SUBSETS = [1, 2, 3, 4, 5, 6]

def load_subset_with_labels(subset_id):
    """Load subset from Gliph2 CSV, add epitope labels from h5 extraction."""
    df = pd.read_csv(GLIPH2_DIR / f"subset_{subset_id}.csv", sep="\t")
    df = df.rename(columns={"CDR3b": "cdr3_beta", "TRBV": "v_beta", "TRBJ": "j_beta"})
    df["tcr_id"] = df["cdr3_beta"]

    # Assign epitope: primary from CDR3_EPI, or "BACKGROUND"
    def get_label(cdr3):
        if cdr3 in CDR3_EPI:
            epis = CDR3_EPI[cdr3]
            return epis[0] if len(epis) == 1 else f"MULTI:{';'.join(sorted(epis))}"
        return "BACKGROUND"

    df["epitope"] = df["cdr3_beta"].apply(get_label)
    return df[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]].drop_duplicates(subset=["cdr3_beta"])


def evaluate_stress(pred, tcr_ids, true_labels):
    """Evaluate stress test with epitope-level ARI plus retention/purity."""
    n_total = len(tcr_ids)
    pred_arr = np.asarray(pred, dtype=object)
    true_arr = np.asarray(true_labels)

    # ARI on all data (epitope classes + BACKGROUND)
    ari = adjusted_rand_score(true_arr, pred_arr)

    # Split labeled vs background
    labeled_idx = np.array([tid in CDR3_EPI for tid in tcr_ids])
    bg_idx = ~labeled_idx
    n_labeled = labeled_idx.sum()
    n_bg = bg_idx.sum()

    # Retention
    if n_labeled > 0:
        lp = pred_arr[labeled_idx]
        retention = float(np.sum(lp != -1)) / n_labeled
    else:
        retention = 0.0

    # ARI on labeled subset only (ignoring BACKGROUND class)
    if n_labeled > 1:
        ari_labeled = adjusted_rand_score(true_arr[labeled_idx], pred_arr[labeled_idx])
    else:
        ari_labeled = 0.0

    # Weighted purity in labeled-containing clusters
    cluster_counts = defaultdict(lambda: [0, 0])  # cid → [n_labeled, n_bg]
    for i in range(n_total):
        cid = pred_arr[i]
        if cid == -1:
            continue
        if labeled_idx[i]:
            cluster_counts[cid][0] += 1
        else:
            cluster_counts[cid][1] += 1

    labeled_clusters = {cid: c for cid, c in cluster_counts.items() if c[0] > 0}
    if labeled_clusters:
        num = sum(c[0] for c in labeled_clusters.values())
        den = sum(sum(c) for c in labeled_clusters.values())
        weighted_purity = num / den if den > 0 else 0
        mean_purity = np.mean([c[0] / sum(c) for c in labeled_clusters.values()])
    else:
        weighted_purity = 0.0
        mean_purity = 0.0

    # False recruitment
    bg_recruited = sum(c[1] for c in labeled_clusters.values())
    false_rec = bg_recruited / n_bg if n_bg > 0 else 0.0

    # Cluster stats
    n_clusters = len(set(pred_arr) - {-1})

    return {
        "n_total": n_total, "n_labeled": int(n_labeled), "n_background": int(n_bg),
        "bg_ratio": float(n_bg / n_labeled) if n_labeled > 0 else 0,
        "ari": float(ari), "ari_labeled": float(ari_labeled),
        "retention": float(retention),
        "weighted_purity": float(weighted_purity), "mean_purity": float(mean_purity),
        "false_recruitment": float(false_rec), "n_bg_recruited": int(bg_recruited),
        "n_clusters": n_clusters,
    }


# ── Run ──
all_results = []

for subset_id in SUBSETS:
    df = load_subset_with_labels(subset_id)
    n_lab = (df["epitope"] != "BACKGROUND").sum()
    n_bg = (df["epitope"] == "BACKGROUND").sum()
    n_epi = df[df["epitope"] != "BACKGROUND"]["epitope"].nunique()

    print(f"\n{'='*78}")
    print(f"SUBSET {subset_id}: {len(df)} TCRs, {n_lab} labeled ({n_epi} epitopes), "
          f"{n_bg} background, ratio={n_bg/n_lab:.2f}")
    print(f"{'='*78}")

    df_norm = normalize(df.copy())
    true_labels = df_norm["epitope"].values
    tcr_ids = df_norm["tcr_id"].values

    t0 = time.time()
    method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / f"subset_{subset_id}")
    runtime = time.time() - t0
    print(f"  Methods done in {runtime:.1f}s")

    # Individual methods
    print(f"\n  {'Method':<15s} {'ARI':>8s} {'Ret':>6s} {'WPur':>6s} {'FalseR':>8s} {'N_cls':>6s}")
    print(f"  {'-'*55}")
    all_assigns = []
    for mname in sorted(method_results.keys()):
        assigns, rt = method_results[mname]
        all_assigns.extend(assigns)
        pred = assignments_to_labels(assigns, tcr_ids)
        ev = evaluate_stress(pred, tcr_ids, true_labels)
        print(f"  {mname:<15s} {ev['ari']:8.4f} {ev['retention']:6.3f} "
              f"{ev['weighted_purity']:6.3f} {ev['false_recruitment']:8.4f} {ev['n_clusters']:6d}")

    # Consensus
    methods_list = sorted(set(a.method for a in all_assigns))
    weights = empirical_weights(methods_list)
    clusters, edges = balanced_consensus(all_assigns, weights)
    clusters = refine(clusters, edges, config)
    pred_cons = clusters_to_labels(clusters, tcr_ids)
    ev_cons = evaluate_stress(pred_cons, tcr_ids, true_labels)
    print(f"  {'consensus':<15s} {ev_cons['ari']:8.4f} {ev_cons['retention']:6.3f} "
          f"{ev_cons['weighted_purity']:6.3f} {ev_cons['false_recruitment']:8.4f} {ev_cons['n_clusters']:6d}")

    # Consensus EQ
    eq_weights = {m: 1.0/len(methods_list) for m in methods_list}
    clusters_eq, edges_eq = balanced_consensus(all_assigns, eq_weights)
    clusters_eq = refine(clusters_eq, edges_eq, config)
    pred_eq = clusters_to_labels(clusters_eq, tcr_ids)
    ev_eq = evaluate_stress(pred_eq, tcr_ids, true_labels)
    print(f"  {'consensus(EQ)':<15s} {ev_eq['ari']:8.4f} {ev_eq['retention']:6.3f} "
          f"{ev_eq['weighted_purity']:6.3f} {ev_eq['false_recruitment']:8.4f} {ev_eq['n_clusters']:6d}")

    all_results.append({
        "subset": subset_id, "runtime_s": runtime,
        "consensus": ev_cons, "consensus_eq": ev_eq,
    })


# ── Summary ──
print(f"\n{'='*78}")
print("STRESS TEST SUMMARY — Consensus (Empirical Weights)")
print(f"{'='*78}")
print(f"{'Sub':>4s} {'N_tot':>6s} {'N_epi':>5s} {'N_bg':>6s} {'ARI':>8s} {'ARI(L)':>8s} "
      f"{'Ret':>6s} {'WPur':>6s} {'FalseR':>8s} {'N_cls':>6s} {'Time':>5s}")
print("-" * 80)

for r in all_results:
    c = r["consensus"]
    print(f"{r['subset']:4d} {c['n_total']:6d} {c['n_labeled']:5d} {c['n_background']:6d} "
          f"{c['ari']:8.4f} {c['ari_labeled']:8.4f} {c['retention']:6.3f} {c['weighted_purity']:6.3f} "
          f"{c['false_recruitment']:8.4f} {c['n_clusters']:6d} {r['runtime_s']:5.0f}s")

# ARI degradation
base_ari = all_results[0]["consensus"]["ari"]
print(f"\nARI degradation from baseline (ARI={base_ari:.4f}):")
for r in all_results:
    d = r["consensus"]["ari"] - base_ari
    print(f"  Subset {r['subset']}: ARI={r['consensus']['ari']:.4f} (Δ={d:+.4f})")

with open(OUTDIR / "stress_final_results.json", "w") as f:
    json.dump({"results": all_results, "baseline_ari": base_ari}, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'stress_final_results.json'}")
print("Done!")
