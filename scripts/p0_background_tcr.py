#!/usr/bin/env python3
"""P0-4: Background TCR stress test.

Inject increasing amounts of random/unrelated TCRs into the benchmark data
and measure:
  1. ARI on the labeled subset only
  2. False recruitment rate (% of background TCRs placed in clusters)
  3. Total ARI (including background as separate "noise" class)
"""

import sys, time, logging, json
import numpy as np
import pandas as pd
from pathlib import Path

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
# Generate background TCRs
# ═══════════════════════════════════════════════════════════════
def generate_background_tcrs(n, rng, ref_seqs=None):
    """Generate random CDR3β sequences that look realistic.

    Strategy: sample real CDR3 lengths, generate random amino acid sequences
    with realistic composition (biased toward small/medium residues).
    """
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    # Realistic CDR3 length distribution (mostly 10-18)
    lengths = rng.choice(range(8, 22), size=n, p=[
        0.02, 0.04, 0.08, 0.12, 0.14, 0.16, 0.14, 0.10,
        0.08, 0.05, 0.03, 0.02, 0.01, 0.01,
    ])

    seqs = []
    for l in lengths:
        seq = "".join(rng.choice(list(AMINO_ACIDS), size=l))
        seqs.append(seq)
    return seqs

# ═══════════════════════════════════════════════════════════════
# Load original data
# ═══════════════════════════════════════════════════════════════
print("=" * 78)
print("P0-4: BACKGROUND TCR STRESS TEST")
print("=" * 78)

df_orig = pd.read_csv(BENCHMARK, sep="\t", dtype=str)
rename_map = {c: c.lower() for c in df_orig.columns if c != c.lower()}
if rename_map:
    df_orig = df_orig.rename(columns=rename_map)
df_orig = normalize(df_orig.copy())
n_orig = len(df_orig)
n_epi = df_orig["epitope"].nunique()

print(f"  Original: {n_orig} TCRs, {n_epi} epitopes")

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
config_obj = load_config()
config = dict(config_obj.__dict__)
clusterers = get_all_clusterers()

BG_RATIOS = [0, 1.0, 2.0, 5.0]
# BG_RATIOS[0] = 0 means no background (baseline)
# Skip TCRMatch for non-zero ratios (O(n²) too slow with background TCRs)
SKIP_SLOW_METHODS = ["tcrmatch"]  # methods to skip when ratio > 0

rng = np.random.RandomState(42)

all_results = []

for ratio in BG_RATIOS:
    n_bg = int(n_orig * ratio)
    label = f"ratio={ratio:.1f}" if ratio > 0 else "baseline"

    print(f"\n{'='*78}")
    print(f"TEST: {label} ({n_bg} background TCRs, total {n_orig + n_bg})")
    print(f"{'='*78}")

    if n_bg == 0:
        # Use original data as-is
        df_test = df_orig.copy()
        bg_ids = []
    else:
        # Generate background TCRs
        bg_seqs = generate_background_tcrs(n_bg, rng)
        bg_ids = [f"bg_{i:06d}" for i in range(n_bg)]

        bg_df = pd.DataFrame({
            "tcr_id": bg_ids,
            "cdr3_beta": bg_seqs,
            "epitope": "BACKGROUND",
            "v_beta": "NA",
            "j_beta": "NA",
        })

        # Merge with original
        df_test = pd.concat([
            df_orig[["tcr_id", "cdr3_beta", "epitope", "v_beta", "j_beta"]],
            bg_df,
        ], ignore_index=True)

    df_norm_test = normalize(df_test.copy())
    true_all = df_norm_test["epitope"].values
    tcr_ids_all = df_norm_test["tcr_id"].values
    n_total = len(df_norm_test)

    # Run methods (skip slow O(n²) methods for non-zero ratios)
    t0 = time.time()
    if ratio > 0:
        fast_clusterers = {k: v for k, v in clusterers.items() if k not in SKIP_SLOW_METHODS}
        print(f"  Skipping slow methods for ratio>0: {SKIP_SLOW_METHODS}")
    else:
        fast_clusterers = clusterers
    method_results = run_all_methods(
        df_norm_test, fast_clusterers, config,
        OUTDIR / f"bg_ratio_{ratio:.1f}"
    )
    runtime = time.time() - t0

    # Collect assignments
    all_assigns = []
    for mname, (assigns, rt) in method_results.items():
        all_assigns.extend(assigns)

    methods_list = sorted(set(a.method for a in all_assigns))
    weights = empirical_weights(methods_list)

    # Consensus
    clusters, edges = balanced_consensus(all_assigns, weights)
    clusters = refine(clusters, edges, config)
    pred = clusters_to_labels(clusters, tcr_ids_all)

    # ── Metrics ──
    # 1. Full ARI (background as "BACKGROUND" class)
    full_metrics = evaluate_clustering(pred, true_all, n_total, f"bg_{ratio}")

    # 2. Labeled-subset ARI (only original TCRs, ignoring background)
    orig_idx = [i for i, tid in enumerate(tcr_ids_all) if tid in set(df_orig["tcr_id"].values)]
    orig_idx_set = set(orig_idx)
    pred_orig = pred[orig_idx]
    true_orig = true_all[orig_idx]
    subset_metrics = evaluate_clustering(pred_orig, true_orig, len(orig_idx), f"bg_{ratio}_subset")

    # 3. False recruitment: background TCRs placed in any cluster
    bg_idx = [i for i, tid in enumerate(tcr_ids_all) if tid.startswith("bg_")]
    if len(bg_idx) > 0:
        bg_pred = np.array([pred[i] for i in bg_idx])
        n_recruited = int(np.sum(bg_pred != -1))
        false_recruitment_rate = n_recruited / len(bg_idx)
    else:
        n_recruited = 0
        false_recruitment_rate = 0.0

    # 4. Cluster count
    n_clusters = len(set(pred) - {-1})

    print(f"  Runtime: {runtime:.1f}s")
    print(f"  Full ARI:       {full_metrics['ari']:.4f}")
    print(f"  Subset ARI:     {subset_metrics['ari']:.4f}  (labeled TCRs only)")
    print(f"  Subset Purity:  {subset_metrics['purity']:.4f}")
    print(f"  Subset Sens:    {subset_metrics['sensitivity']:.4f}")
    print(f"  False recruit:  {false_recruitment_rate:.4f} ({n_recruited}/{len(bg_idx)})")
    print(f"  N clusters:     {n_clusters}")

    all_results.append({
        "ratio": ratio,
        "n_bg": n_bg,
        "n_total": n_total,
        "runtime_s": runtime,
        "full_ari": full_metrics["ari"],
        "subset_ari": subset_metrics["ari"],
        "subset_purity": subset_metrics["purity"],
        "subset_sensitivity": subset_metrics["sensitivity"],
        "subset_f1": subset_metrics["f1"],
        "false_recruitment_rate": false_recruitment_rate,
        "n_recruited": int(n_recruited),
        "n_clusters": n_clusters,
    })

# ═══════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*78}")
print("BACKGROUND STRESS TEST SUMMARY")
print(f"{'='*78}")
print(f"{'Ratio':>6s} {'N_bg':>8s} {'N_total':>8s} {'SubARI':>8s} {'SubPur':>8s} {'SubSens':>8s} "
      f"{'FalseR':>8s} {'N_cls':>8s} {'Time':>8s}")
print("-" * 80)

for r in all_results:
    print(f"{r['ratio']:6.1f} {r['n_bg']:8d} {r['n_total']:8d} "
          f"{r['subset_ari']:8.4f} {r['subset_purity']:8.4f} {r['subset_sensitivity']:8.4f} "
          f"{r['false_recruitment_rate']:8.4f} {r['n_clusters']:8d} {r['runtime_s']:8.1f}")

# ARI degradation
baseline_ari = all_results[0]["subset_ari"]
print(f"\nARI degradation from baseline ({baseline_ari:.4f}):")
for r in all_results:
    delta = r["subset_ari"] - baseline_ari
    print(f"  ratio={r['ratio']:5.1f}: {r['subset_ari']:.4f} (Δ={delta:+.4f})")

# Save
with open(OUTDIR / "p0_4_background_stress.json", "w") as f:
    json.dump({"results": all_results, "baseline_ari": baseline_ari}, f, indent=2)

print(f"\nSaved to {OUTDIR / 'p0_4_background_stress.json'}")
print("Done!")
