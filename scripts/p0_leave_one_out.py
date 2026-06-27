#!/usr/bin/env python3
"""P0-2: Leave-one-method-out ablation.

For each of 7 methods, remove it and run consensus with remaining 6.
Reports ARI delta vs full 7-method consensus.
"""

import sys, time, logging, json
import numpy as np
import pandas as pd
from pathlib import Path
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

BENCHMARK = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_all.tsv"
OUTDIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
print("=" * 78)
print("P0-2: LEAVE-ONE-METHOD-OUT ABLATION")
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

# ═══════════════════════════════════════════════════════════════
# Run all methods once (reuse assignments)
# ═══════════════════════════════════════════════════════════════
config_obj = load_config()
config = dict(config_obj.__dict__)
clusterers = get_all_clusterers()

print("\nRunning all methods...")
t0 = time.time()
method_results = run_all_methods(df_norm, clusterers, config, OUTDIR / "lomo_methods")
print(f"All methods done in {time.time()-t0:.1f}s")

# Collect assignments by method
assigns_by_method = {}
for mname, (assigns, rt) in method_results.items():
    assigns_by_method[mname] = assigns

ALL_METHODS = sorted(assigns_by_method.keys())
print(f"\nMethods: {ALL_METHODS}")

# ═══════════════════════════════════════════════════════════════
# Full 7-method consensus (baseline)
# ═══════════════════════════════════════════════════════════════
def run_consensus(methods_list, assigns_dict, tcr_ids, true_labels, n_tcr, config, label=""):
    """Run consensus with given method subset."""
    all_assigns = []
    for m in methods_list:
        all_assigns.extend(assigns_dict[m])

    weights = empirical_weights(methods_list)
    clusters, edges = balanced_consensus(all_assigns, weights)
    clusters = refine(clusters, edges, config)
    pred = clusters_to_labels(clusters, tcr_ids)
    metrics = evaluate_clustering(pred, true_labels, n_tcr, label)
    n_cls = len(set(pred) - {-1})
    return metrics, n_cls

full_metrics, n_full_cls = run_consensus(
    ALL_METHODS, assigns_by_method, tcr_ids, true_labels, n_tcr, config, "full_7"
)

print(f"\n{'='*78}")
print("FULL 7-METHOD CONSENSUS (baseline)")
print(f"{'='*78}")
print(f"  ARI={full_metrics['ari']:.4f}, AMI={full_metrics['ami']:.4f}, "
      f"Purity={full_metrics['purity']:.4f}, Sens={full_metrics['sensitivity']:.4f}, "
      f"F1={full_metrics['f1']:.4f}, Ret={full_metrics['retention']:.4f}, N_cls={n_full_cls}")

# ═══════════════════════════════════════════════════════════════
# Leave-one-out
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("LEAVE-ONE-METHOD-OUT RESULTS")
print(f"{'='*78}")
print(f"{'Removed':<15s} {'ARI':>8s} {'ΔARI':>8s} {'AMI':>8s} {'Purity':>8s} {'Sens':>8s} {'F1':>8s} {'N_cls':>8s} {'Impact':>10s}")
print("-" * 90)

lomo_results = {}

for removed_method in ALL_METHODS:
    remaining = [m for m in ALL_METHODS if m != removed_method]
    metrics, n_cls = run_consensus(
        remaining, assigns_by_method, tcr_ids, true_labels, n_tcr, config, f"no_{removed_method}"
    )

    delta_ari = metrics["ari"] - full_metrics["ari"]
    # Positive delta = removing this method IMPROVED consensus (bad method)
    # Negative delta = removing this method HURT consensus (important method)
    if delta_ari > 0.01:
        impact = "HARMFUL"
    elif delta_ari < -0.01:
        impact = "IMPORTANT"
    else:
        impact = "neutral"

    print(f"{removed_method:<15s} {metrics['ari']:8.4f} {delta_ari:>+8.4f} {metrics['ami']:8.4f} "
          f"{metrics['purity']:8.4f} {metrics['sensitivity']:8.4f} {metrics['f1']:8.4f} "
          f"{n_cls:8d} {impact:>10s}")

    lomo_results[removed_method] = {
        "metrics": metrics,
        "n_clusters": n_cls,
        "delta_ari": delta_ari,
        "impact": impact,
    }

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("SUMMARY")
print(f"{'='*78}")

# Rank by importance (most negative delta = most important)
ranked = sorted(lomo_results.items(), key=lambda x: x[1]["delta_ari"])
print("\nMethods ranked by importance (removal impact):")
for i, (method, data) in enumerate(ranked, 1):
    d = data["delta_ari"]
    marker = " *** CRITICAL" if d < -0.02 else (" * important" if d < -0.01 else "")
    print(f"  {i}. {method}: ΔARI = {d:+.4f}{marker}")

most_important = ranked[0][0]
print(f"\nMost important method: {most_important} (ΔARI = {ranked[0][1]['delta_ari']:+.4f})")

# Check: is any single method critical? (removal causes >0.05 ARI drop)
critical = [m for m, d in lomo_results.items() if d["delta_ari"] < -0.05]
if critical:
    print(f"\nWARNING: Critical methods found: {critical}")
    print("  Consensus is fragile — depends heavily on these methods.")
else:
    print("\nNo single method is critical (ΔARI > -0.05 for all).")
    print("  Consensus is ROBUST — no single-method dependency.")

# Save
with open(OUTDIR / "p0_2_leave_one_out.json", "w") as f:
    json.dump({
        "full_7": {**full_metrics, "n_clusters": n_full_cls},
        "leave_one_out": {k: {**v["metrics"], "delta_ari": v["delta_ari"],
                              "n_clusters": v["n_clusters"], "impact": v["impact"]}
                          for k, v in lomo_results.items()},
    }, f, indent=2, default=str)

print(f"\nSaved to {OUTDIR / 'p0_2_leave_one_out.json'}")
print("Done!")
