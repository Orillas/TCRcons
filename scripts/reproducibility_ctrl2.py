#!/usr/bin/env python3
"""可复现性实验 - 对照2: CC + 等权 + merge 0.6 (去除Leiden，但不用经验权重)。

与改进后对比，验证经验权重的贡献。
"""
import sys
import os
import logging
import warnings
import time
import json
import traceback

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout, force=True,
)
for noisy in ["numba", "tensorflow", "absl", "matplotlib"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, "/home/jilin/DeepTCR/tcrconsensus/src")
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tcrconsensus.io.parser import normalize
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import build_consensus_graph, connected_components_clustering
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
BENCHMARK = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility")

print("=" * 70, flush=True)
print("对照2: CC + 等权 + merge 0.6", flush=True)
print("=" * 70, flush=True)

df = pd.read_csv(BENCHMARK, sep="\t", dtype=str)
rename = {c: c.lower() for c in df.columns
          if c.lower() != c and c.lower() in
          ["cdr3_alpha","cdr3_beta","v_alpha","v_beta","j_alpha","j_beta","tcr_id","epitope"]}
if rename:
    df = df.rename(columns=rename)
df_norm = normalize(df.copy())

epitope_map = {}
for _, row in df_norm.iterrows():
    tid = str(row.get("tcr_id", ""))
    epi = str(row.get("epitope", ""))
    if tid and epi:
        epitope_map[tid] = epi

print(f"数据集: {len(df_norm)} TCRs, {df_norm['epitope'].nunique()} epitopes", flush=True)


def evaluate(clusters, total_tcrs):
    members = []
    for cc in clusters:
        members.extend(cc.member_ids)
    if not members:
        return None
    tids = list(set(members))
    idx = {t: i for i, t in enumerate(tids)}
    pred = [-1] * len(tids)
    true_l = [-1] * len(tids)
    for cc in clusters:
        cid = hash(cc.cluster_id) % (10**8)
        for tid in cc.member_ids:
            if tid in idx:
                pred[idx[tid]] = cid
            if tid in epitope_map:
                true_l[idx[tid]] = hash(epitope_map[tid]) % (10**8)
    assigned = [i for i in range(len(tids)) if pred[i] != -1]
    if len(assigned) < 2:
        return None
    lp = [pred[i] for i in assigned]
    lt = [true_l[i] for i in assigned]
    ari = adjusted_rand_score(lt, lp)
    nmi = normalized_mutual_info_score(lt, lp)
    cluster_epis = {}
    for i in assigned:
        cluster_epis.setdefault(pred[i], []).append(true_l[i])
    purity = sum(Counter(v).most_common(1)[0][1] for v in cluster_epis.values()) / len(assigned)
    return {
        "ari": round(ari, 4), "nmi": round(nmi, 4),
        "purity": round(purity, 4), "n_clusters": len(clusters),
        "n_assigned": len(assigned),
        "retention": round(len(assigned) / total_tcrs, 4),
    }


def method_ari(assignments):
    tids = list(set(a.tcr_id for a in assignments))
    if len(tids) < 2:
        return None
    idx = {t: i for i, t in enumerate(tids)}
    lp = [-1] * len(tids)
    lt = [-1] * len(tids)
    for a in assignments:
        i = idx[a.tcr_id]
        lp[i] = hash(a.cluster_id) % (10**8)
        if a.tcr_id in epitope_map:
            lt[i] = hash(epitope_map[a.tcr_id]) % (10**8)
    valid = [j for j in range(len(tids)) if lp[j] != -1]
    if len(valid) < 2:
        return None
    return round(adjusted_rand_score([lt[j] for j in valid], [lp[j] for j in valid]), 4)


def run_cc_equal(seed):
    """对照2: CC + 等权 + merge 0.6"""
    np.random.seed(seed)
    workdir = OUT_DIR / f"cc_equal_seed{seed}"
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
    for name, wrapper in clusterers:
        mdir = workdir / name
        mdir.mkdir(parents=True, exist_ok=True)
        try:
            result = wrapper.safe_execute(df_norm, mdir, {})
            if result.assignments:
                all_assignments.extend(result.assignments)
                print(f"    {name:12s}: {len(result.assignments):5d} assignments", flush=True)
            else:
                print(f"    {name:12s}: FAILED", flush=True)
        except Exception as e:
            print(f"    {name:12s}: ERROR - {e}", flush=True)

    if not all_assignments:
        return None

    methods = sorted(set(a.method for a in all_assignments))
    per_method = {}
    for m in methods:
        ari = method_ari([a for a in all_assignments if a.method == m])
        if ari is not None:
            per_method[m] = ari

    # CC + 等权 (refine 使用当前默认 merge=0.6)
    weights = {m: 1.0 / len(methods) for m in methods}
    edges = extract_pairwise_support(all_assignments, weights)
    graph = build_consensus_graph(edges, threshold=0.3)
    clusters = connected_components_clustering(graph)

    if clusters:
        clusters = refine(clusters, edges, {})

    if not clusters:
        return None

    total_tcrs = len(set(a.tcr_id for a in all_assignments))
    metrics = evaluate(clusters, total_tcrs)
    if metrics is None:
        return None

    return {
        "seed": seed,
        "config": "cc_equal_weights",
        "per_method_ari": per_method,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        **metrics,
    }


# ===== 主循环 =====
ctrl2_results = []
for i, seed in enumerate(SEEDS[:N_RUNS]):
    print(f"\n--- Run {i+1}/{N_RUNS} (seed={seed}) ---", flush=True)
    t0 = time.time()
    try:
        res = run_cc_equal(seed)
        elapsed = time.time() - t0
        if res:
            ctrl2_results.append(res)
            print(
                f"  >>> ARI={res['ari']:.4f}, Purity={res['purity']:.4f}, "
                f"NMI={res['nmi']:.4f}, Ret={res['retention']:.4f}, "
                f"Clusters={res['n_clusters']}, Time={elapsed:.0f}s",
                flush=True,
            )
        else:
            print(f"  >>> FAILED ({elapsed:.0f}s)", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  >>> EXCEPTION ({elapsed:.0f}s): {e}", flush=True)
        traceback.print_exc()

# 汇总
print(f"\n{'=' * 70}", flush=True)
print(f"对照2汇总: CC + 等权 + merge 0.6", flush=True)
print(f"{'=' * 70}", flush=True)
if ctrl2_results:
    aris = [r["ari"] for r in ctrl2_results]
    print(f"  ARI: {np.mean(aris):.4f} ± {np.std(aris):.4f}", flush=True)

# 保存单独结果
ctrl2_path = OUT_DIR / "ctrl2_cc_equal_results.json"
with open(ctrl2_path, "w") as f:
    json.dump({"config": "cc_equal_weights", "results": ctrl2_results}, f, indent=2)
print(f"\n保存: {ctrl2_path}", flush=True)
print("对照2实验完成", flush=True)
