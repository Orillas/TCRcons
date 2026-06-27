#!/usr/bin/env python3
"""Reproducibility test: run majority_vote pipeline N times with different seeds."""
import sys, os, logging, warnings, time, json, traceback
# Force unbuffered output for nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
for noisy in ['numba', 'tensorflow', 'absl', 'matplotlib']:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import adjusted_rand_score

from tcrconsensus.io.parser import normalize
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 2024]

# Load data once
benchmark_path = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
df = pd.read_csv(benchmark_path, sep="\t", dtype=str)
rename_lower = {col: col.lower() for col in df.columns
                if col.lower() != col and col.lower() in ["cdr3_alpha","cdr3_beta","v_alpha","v_beta","j_alpha","j_beta","tcr_id","epitope"]}
if rename_lower:
    df = df.rename(columns=rename_lower)
df_norm = normalize(df.copy())
print(f"Dataset: {len(df_norm)} TCRs, {df_norm['epitope'].nunique()} epitopes")

# Build epitope lookup
epitope_map = {}
for _, row in df_norm.iterrows():
    tid = str(row.get("tcr_id", ""))
    epi = str(row.get("epitope", ""))
    if tid and epi:
        epitope_map[tid] = epi


def run_pipeline(seed, use_empirical=True):
    """Run full pipeline once. Returns dict with metrics."""
    np.random.seed(seed)
    workdir = Path(f"/tmp/repro_{seed}_{int(use_empirical)}")
    workdir.mkdir(parents=True, exist_ok=True)

    clusterers = [
        ("hd_baseline", HDBaselineClusterer()),
        ("clustcr", ClusTCRWrapper()),
        ("tcrdist3", TCRDist3Wrapper()),
        ("gliph2", GLIPH2Wrapper()),
        ("giana", GIANAWrapper()),
        ("tcrmatch", TCRMatchWrapper()),
        ("deeptcr", DeepTCRWrapper()),
    ]

    all_assignments = []
    method_info = {}

    for name, wrapper in clusterers:
        mdir = workdir / name
        mdir.mkdir(parents=True, exist_ok=True)
        try:
            result = wrapper.safe_execute(df_norm, mdir, {})
            if result.assignments:
                all_assignments.extend(result.assignments)
                method_info[name] = {
                    "n": len(result.assignments),
                    "k": len(set(a.cluster_id for a in result.assignments)),
                    "t": round(result.runtime_seconds, 1),
                }
                print(f"  {name}: {len(result.assignments)} assignments, {method_info[name]['k']} clusters")
            else:
                print(f"  {name}: NO assignments (status={result.status.value})")
                if result.error_message:
                    print(f"    error: {result.error_message[:200]}")
        except Exception as e:
            print(f"  {name}: EXCEPTION: {e}")
            traceback.print_exc()

    if not all_assignments:
        print("  No assignments at all!")
        return None

    # Per-method ARI
    methods = sorted(set(a.method for a in all_assignments))
    per_method_ari = {}
    for method in methods:
        ma = [a for a in all_assignments if a.method == method]
        if len(ma) < 2:
            continue
        tids = list(set(a.tcr_id for a in ma))
        idx_map = {t: i for i, t in enumerate(tids)}
        lp = [-1] * len(tids)
        lt = [-1] * len(tids)
        for a in ma:
            i = idx_map[a.tcr_id]
            lp[i] = hash(a.cluster_id) % (10**8)
            if a.tcr_id in epitope_map:
                lt[i] = hash(epitope_map[a.tcr_id]) % (10**8)
        valid = [j for j in range(len(tids)) if lp[j] != -1]
        if len(valid) >= 2:
            per_method_ari[method] = round(adjusted_rand_score(
                [lt[j] for j in valid], [lp[j] for j in valid]), 4)

    # Consensus
    if use_empirical:
        weights = empirical_weights(methods)
    else:
        weights = None

    print(f"  Computing consensus ({len(all_assignments)} assignments, {len(methods)} methods)...", flush=True)
    clusters, edges = balanced_consensus(all_assignments, weights)
    print(f"  Consensus: {len(clusters)} clusters, {len(edges)} edges", flush=True)
    if clusters:
        try:
            clusters = refine(clusters, edges, {})
            print(f"  After refinement: {len(clusters)} clusters", flush=True)
        except Exception as e:
            print(f"  Refinement error: {e}", flush=True)

    if not clusters:
        print("  No consensus clusters!", flush=True)
        return None

    # Evaluate
    all_tids = list(set(a.tcr_id for a in all_assignments))
    idx_map = {t: i for i, t in enumerate(all_tids)}
    pred = [-1] * len(all_tids)
    true_l = [-1] * len(all_tids)

    for cc in clusters:
        cid = hash(cc.cluster_id) % (10**8)
        for tid in cc.member_ids:
            if tid in idx_map:
                pred[idx_map[tid]] = cid
            if tid in epitope_map:
                true_l[idx_map[tid]] = hash(epitope_map[tid]) % (10**8)

    assigned = [i for i in range(len(all_tids)) if pred[i] != -1]
    print(f"  Evaluated: {len(assigned)}/{len(all_tids)} TCRs assigned to clusters", flush=True)
    if len(assigned) < 2:
        print("  Too few assigned TCRs!", flush=True)
        return None

    ari = adjusted_rand_score([true_l[i] for i in assigned], [pred[i] for i in assigned])

    # Purity
    cluster_epis = {}
    for i in assigned:
        cid = pred[i]
        cluster_epis.setdefault(cid, []).append(true_l[i])
    pur = sum(Counter(v).most_common(1)[0][1] for v in cluster_epis.values()) / len(assigned)
    ret = len(assigned) / len(all_tids)

    return {
        "seed": seed, "empirical": use_empirical,
        "ari": round(ari, 4), "purity": round(pur, 4),
        "retention": round(ret, 4), "n_clusters": len(clusters),
        "n_tcrs": len(all_tids), "n_assigned": len(assigned),
        "weights": {k: round(v, 4) for k, v in (weights or {}).items()},
        "per_method_ari": per_method_ari, "method_info": method_info,
    }


# ===== Main =====
all_results = []

for cond_name, use_emp in [("EMPIRICAL WEIGHTS", True), ("EQUAL WEIGHTS", False)]:
    print(f"\n{'='*60}")
    print(f"  {cond_name}")
    print(f"{'='*60}")

    for i, seed in enumerate(SEEDS[:N_RUNS]):
        print(f"\n--- Run {i+1}/{N_RUNS} (seed={seed}) ---", flush=True)
        t0 = time.time()
        try:
            res = run_pipeline(seed, use_empirical=use_emp)
            elapsed = time.time() - t0
            if res:
                all_results.append(res)
                print(f"  => ARI={res['ari']}, Purity={res['purity']}, "
                      f"Retention={res['retention']}, Clusters={res['n_clusters']}, "
                      f"Time={elapsed:.0f}s", flush=True)
            else:
                print(f"  => FAILED (no result)", flush=True)
        except Exception as e:
            print(f"  => EXCEPTION: {e}", flush=True)
            traceback.print_exc()

# ===== Summary =====
print(f"\n{'='*60}")
print(f"REPRODUCIBILITY SUMMARY")
print(f"{'='*60}")

emp = [r for r in all_results if r["empirical"]]
eq = [r for r in all_results if not r["empirical"]]

for label, data in [("Empirical", emp), ("Equal", eq)]:
    if not data:
        print(f"\n{label}: NO RESULTS")
        continue
    aris = [r["ari"] for r in data]
    purs = [r["purity"] for r in data]
    rets = [r["retention"] for r in data]
    cv = np.std(aris) / np.mean(aris) * 100 if np.mean(aris) > 0 else 0
    print(f"\n{label} weights (n={len(data)}):")
    print(f"  ARI:       {np.mean(aris):.4f} +/- {np.std(aris):.4f}  "
          f"[min={min(aris):.4f}, max={max(aris):.4f}]")
    print(f"  Purity:    {np.mean(purs):.4f} +/- {np.std(purs):.4f}")
    print(f"  Retention: {np.mean(rets):.4f} +/- {np.std(rets):.4f}")
    print(f"  CV(ARI):   {cv:.1f}%")

    # Per-method stability
    methods = sorted(set(m for r in data for m in r["per_method_ari"]))
    if methods:
        print(f"  Per-method ARI stability:")
        for m in methods:
            vals = [r["per_method_ari"].get(m) for r in data if m in r["per_method_ari"]]
            if vals:
                print(f"    {m:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (CV={np.std(vals)/np.mean(vals)*100:.1f}%)" if np.mean(vals) > 0 else f"    {m:15s}: N/A")

if emp and eq:
    from scipy import stats
    t, p = stats.ttest_ind([r["ari"] for r in emp], [r["ari"] for r in eq])
    wins = sum(1 for a, b in zip(emp, eq) if a["ari"] > b["ari"])
    print(f"\n  Empirical vs Equal: t={t:.3f}, p={p:.4f}")
    print(f"  Empirical wins: {wins}/{min(len(emp), len(eq))} runs")

# Save
out_path = "/home/jilin/DeepTCR/tcrconsensus/results/reproducibility_test.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
