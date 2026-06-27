#!/usr/bin/env python3
"""Reproducibility test: run majority_vote pipeline N times with different random seeds.

Tests whether the ARI improvement (0.207 → 0.534) is stable across runs.
Main sources of randomness: DeepTCR VAE training, clusTCR FAISS.
"""
import sys, os, logging, warnings, time, json
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import numpy as np
import pandas as pd
from pathlib import Path

from tcrconsensus.io.parser import normalize
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.refinement.refiner import refine

# Import all clusterers
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 2024]

def run_single_pipeline(df_norm, workdir, seed, use_empirical=True):
    """Run the full pipeline once with a specific seed."""
    np.random.seed(seed)
    
    results = {}
    
    # Run all clusterers
    clusterers = {
        "hd_baseline": HDBaselineClusterer(),
        "clustcr": ClusTCRWrapper(),
        "tcrdist3": TCRDist3Wrapper(),
        "gliph2": GLIPH2Wrapper(),
        "giana": GIANAWrapper(),
        "tcrmatch": TCRMatchWrapper(),
        "deeptcr": DeepTCRWrapper(),
    }
    
    all_assignments = []
    method_results = {}
    
    for name, wrapper in clusterers.items():
        method_workdir = workdir / f"seed_{seed}" / name
        method_workdir.mkdir(parents=True, exist_ok=True)
        result = wrapper.safe_execute(df_norm, method_workdir, {})
        if result.assignments:
            all_assignments.extend(result.assignments)
            method_results[name] = {
                "n_assignments": len(result.assignments),
                "n_clusters": len(set(a.cluster_id for a in result.assignments)),
                "runtime": result.runtime_seconds,
            }
    
    if not all_assignments:
        return None
    
    # Compute per-method ARI for comparison
    epitope_map = {}
    for _, row in df_norm.iterrows():
        tid = str(row.get("tcr_id", ""))
        epi = str(row.get("epitope", ""))
        if tid and epi:
            epitope_map[tid] = epi
    
    per_method_ari = {}
    methods_seen = sorted(set(a.method for a in all_assignments))
    for method in methods_seen:
        method_assigns = [a for a in all_assignments if a.method == method]
        if len(method_assigns) < 2:
            continue
        # Build cluster labels
        tcr_ids = list(set(a.tcr_id for a in method_assigns))
        tcr_idx = {t: i for i, t in enumerate(tcr_ids)}
        labels_pred = [-1] * len(tcr_ids)
        labels_true = [-1] * len(tcr_ids)
        assigned = set()
        for a in method_assigns:
            idx = tcr_idx[a.tcr_id]
            labels_pred[idx] = hash(a.cluster_id) % (10**8)
            if a.tcr_id in epitope_map:
                labels_true[idx] = hash(epitope_map[a.tcr_id]) % (10**8)
            assigned.add(idx)
        # Filter to assigned only
        idx_list = sorted(assigned)
        lp = [labels_pred[i] for i in idx_list]
        lt = [labels_true[i] for i in idx_list]
        try:
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(lt, lp)
            per_method_ari[method] = round(ari, 4)
        except:
            pass
    
    # Majority vote with empirical weights
    if use_empirical:
        methods = sorted(set(a.method for a in all_assignments))
        weights = empirical_weights(methods)
    else:
        weights = None
    
    clusters, edges = balanced_consensus(all_assignments, weights)
    
    if clusters:
        clusters = refine(clusters, edges, {})
    
    # Evaluate consensus
    if not clusters:
        return None
    
    # Build predictions
    tcr_ids = list(set(a.tcr_id for a in all_assignments))
    tcr_idx = {t: i for i, t in enumerate(tcr_ids)}
    
    pred = [-1] * len(tcr_ids)
    true_labels = [-1] * len(tcr_ids)
    
    for cc in clusters:
        cid = hash(cc.cluster_id) % (10**8)
        for tid in cc.tcr_ids:
            if tid in tcr_idx:
                pred[tcr_idx[tid]] = cid
            if tid in epitope_map:
                true_labels[tcr_idx[tid]] = hash(epitope_map[tid]) % (10**8)
    
    # Only evaluate TCRs that were assigned to clusters
    assigned_idx = [i for i in range(len(tcr_ids)) if pred[i] != -1]
    if len(assigned_idx) < 2:
        return None
    
    lp = [pred[i] for i in assigned_idx]
    lt = [true_labels[i] for i in assigned_idx]
    
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(lt, lp)
    
    # Purity
    from collections import Counter
    cluster_epitopes = {}
    for i in assigned_idx:
        cid = pred[i]
        epi = true_labels[i]
        if cid not in cluster_epitopes:
            cluster_epitopes[cid] = []
        cluster_epitopes[cid].append(epi)
    
    purity_sum = 0
    total = 0
    for cid, epis in cluster_epitopes.items():
        counts = Counter(epis)
        purity_sum += counts.most_common(1)[0][1]
        total += len(epis)
    purity = purity_sum / total if total > 0 else 0
    retention = len(assigned_idx) / len(tcr_ids)
    
    return {
        "seed": seed,
        "ari": round(ari, 4),
        "purity": round(purity, 4),
        "retention": round(retention, 4),
        "n_clusters": len(clusters),
        "n_tcrs": len(tcr_ids),
        "n_assigned": len(assigned_idx),
        "use_empirical": use_empirical,
        "weights": {k: round(v, 4) for k, v in (weights or {}).items()},
        "per_method_ari": per_method_ari,
        "method_results": method_results,
    }

# Load data
benchmark_path = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
df = pd.read_csv(benchmark_path, sep="\t", dtype=str)
rename_lower = {col: col.lower() for col in df.columns
                if col.lower() != col and col.lower() in ["cdr3_alpha","cdr3_beta","v_alpha","v_beta","j_alpha","j_beta","tcr_id","epitope"]}
if rename_lower:
    df = df.rename(columns=rename_lower)
df_norm = normalize(df.copy())
print(f"Dataset: {len(df_norm)} TCRs, {df_norm['epitope'].nunique()} epitopes")

workdir = Path("/tmp/reproducibility_test")

# =============================================
# Test 1: Empirical weights + CC + refinement (N_RUNS)
# =============================================
print(f"\n{'='*60}")
print(f"REPRODUCIBILITY TEST: {N_RUNS} runs with different seeds")
print(f"{'='*60}")

empirical_results = []
for i, seed in enumerate(SEEDS[:N_RUNS]):
    print(f"\n--- Run {i+1}/{N_RUNS} (seed={seed}) ---")
    start = time.time()
    result = run_single_pipeline(df_norm, workdir, seed, use_empirical=True)
    elapsed = time.time() - start
    if result:
        empirical_results.append(result)
        print(f"  ARI={result['ari']}, Purity={result['purity']}, "
              f"Retention={result['retention']}, Clusters={result['n_clusters']}, "
              f"Time={elapsed:.0f}s")
    else:
        print(f"  FAILED")

# =============================================
# Test 2: Equal weights + CC + refinement (N_RUNS) — for comparison
# =============================================
print(f"\n{'='*60}")
print(f"COMPARISON: Equal weights ({N_RUNS} runs)")
print(f"{'='*60}")

equal_results = []
for i, seed in enumerate(SEEDS[:N_RUNS]):
    print(f"\n--- Run {i+1}/{N_RUNS} (seed={seed}, equal weights) ---")
    start = time.time()
    result = run_single_pipeline(df_norm, workdir, seed + 10000, use_empirical=False)
    elapsed = time.time() - start
    if result:
        equal_results.append(result)
        print(f"  ARI={result['ari']}, Purity={result['purity']}, "
              f"Retention={result['retention']}, Clusters={result['n_clusters']}, "
              f"Time={elapsed:.0f}s")
    else:
        print(f"  FAILED")

# =============================================
# Summary
# =============================================
print(f"\n{'='*60}")
print(f"REPRODUCIBILITY SUMMARY")
print(f"{'='*60}")

if empirical_results:
    aris_e = [r['ari'] for r in empirical_results]
    purities_e = [r['purity'] for r in empirical_results]
    retentions_e = [r['retention'] for r in empirical_results]
    print(f"\nEmpirical weights (n={len(aris_e)}):")
    print(f"  ARI:       {np.mean(aris_e):.4f} ± {np.std(aris_e):.4f}  [min={min(aris_e):.4f}, max={max(aris_e):.4f}]")
    print(f"  Purity:    {np.mean(purities_e):.4f} ± {np.std(purities_e):.4f}")
    print(f"  Retention: {np.mean(retentions_e):.4f} ± {np.std(retentions_e):.4f}")
    print(f"  CV(ARI):   {np.std(aris_e)/np.mean(aris_e)*100:.1f}%")
    
    # Per-method ARI stability across runs
    print(f"\n  Per-method ARI (across runs):")
    all_methods = sorted(set(m for r in empirical_results for m in r['per_method_ari']))
    for method in all_methods:
        method_aris = [r['per_method_ari'].get(method, None) for r in empirical_results]
        method_aris = [a for a in method_aris if a is not None]
        if method_aris:
            print(f"    {method:15s}: {np.mean(method_aris):.4f} ± {np.std(method_aris):.4f}")

if equal_results:
    aris_eq = [r['ari'] for r in equal_results]
    purities_eq = [r['purity'] for r in equal_results]
    print(f"\nEqual weights (n={len(aris_eq)}):")
    print(f"  ARI:       {np.mean(aris_eq):.4f} ± {np.std(aris_eq):.4f}  [min={min(aris_eq):.4f}, max={max(aris_eq):.4f}]")
    print(f"  Purity:    {np.mean(purities_eq):.4f} ± {np.std(purities_eq):.4f}")
    print(f"  CV(ARI):   {np.std(aris_eq)/np.mean(aris_eq)*100:.1f}%")

if empirical_results and equal_results:
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(aris_e, aris_eq)
    print(f"\n  Empirical vs Equal: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Empirical > Equal in {sum(1 for a,b in zip(aris_e, aris_eq) if a > b)}/{min(len(aris_e), len(aris_eq))} runs")

# Save results
output = {
    "empirical_weights": empirical_results,
    "equal_weights": equal_results,
}
out_path = "/home/jilin/DeepTCR/tcrconsensus/results/reproducibility_test.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
