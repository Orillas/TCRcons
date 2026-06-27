#!/usr/bin/env python3
"""可复现性实验：tcrconsensus 改进前 vs 改进后 在高置信度数据上的对比。

改进后: connected_components + 经验权重 + merge 0.6
改进前: Leiden community + 等权 + merge 0.4  (原始管线)

5 个随机种子，主要随机源为 DeepTCR VAE 和 clusTCR FAISS。
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
from tcrconsensus.consensus.weights import empirical_weights
from tcrconsensus.consensus.coassociation import extract_pairwise_support
from tcrconsensus.consensus.graph import (
    build_consensus_graph,
    connected_components_clustering,
    community_clustering,
)
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.clusterers.clustcr_wrapper import ClusTCRWrapper
from tcrconsensus.clusterers.tcrdist3_wrapper import TCRDist3Wrapper
from tcrconsensus.clusterers.gliph2_wrapper import GLIPH2Wrapper
from tcrconsensus.clusterers.giana_wrapper import GIANAWrapper
from tcrconsensus.clusterers.tcrmatch_wrapper import TCRMatchWrapper
from tcrconsensus.clusterers.deeptcr_wrapper import DeepTCRWrapper

# ============ 参数 ============
N_RUNS = 5
SEEDS = [42, 123, 456, 789, 2024]
BENCHMARK = "/home/jilin/DeepTCR/tcrconsensus/results/paper_benchmark/paper_benchmark_v3_cd8.tsv"
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/reproducibility")

# ============ 加载数据 ============
print("=" * 70, flush=True)
print("TCRCONSENSUS 可复现性实验：改进前 vs 改进后", flush=True)
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
print(f"种子: {SEEDS}", flush=True)
print(f"改进后: CC + 经验权重 + merge 0.6", flush=True)
print(f"改进前: Leiden + 等权 + merge 0.4", flush=True)


# ============ 评估函数 ============
def evaluate(clusters, epitope_map, total_tcrs):
    """评估共识聚类结果，返回指标字典。"""
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
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "purity": round(purity, 4),
        "n_clusters": len(clusters),
        "n_assigned": len(assigned),
        "retention": round(len(assigned) / total_tcrs, 4),
    }


def method_ari(assignments, epitope_map):
    """计算单个方法的 ARI。"""
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


# ============ 共识策略 ============
def consensus_new(all_assignments, methods):
    """改进后: CC + 经验权重 + merge 0.6"""
    weights = empirical_weights(methods)
    edges = extract_pairwise_support(all_assignments, weights)
    graph = build_consensus_graph(edges, threshold=0.3)
    clusters = connected_components_clustering(graph)
    if clusters:
        # merge 阈值 0.6（当前默认）
        clusters = refine(clusters, edges, {})
    return clusters, weights


def consensus_old(all_assignments, methods):
    """改进前: Leiden + 等权 + merge 0.4"""
    weights = {m: 1.0 / len(methods) for m in methods}
    edges = extract_pairwise_support(all_assignments, weights)
    graph = build_consensus_graph(edges, threshold=0.3)
    clusters = community_clustering(graph, algorithm="leiden", resolution=1.0)
    if clusters:
        # merge 阈值 0.4（旧版默认）
        config = {"refine": {"merge": {"min_cross_association": 0.4}}}
        clusters = refine(clusters, edges, config)
    return clusters, weights


# ============ 单次运行 ============
def run_once(seed, consensus_fn, label):
    """运行一次完整管线。"""
    np.random.seed(seed)
    workdir = OUT_DIR / f"{label}_seed{seed}"
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
                print(
                    f"    {name:12s}: {len(result.assignments):5d} assignments",
                    flush=True,
                )
            else:
                print(f"    {name:12s}: FAILED", flush=True)
        except Exception as e:
            print(f"    {name:12s}: ERROR - {e}", flush=True)

    if not all_assignments:
        return None

    methods = sorted(set(a.method for a in all_assignments))

    # 逐方法 ARI
    per_method = {}
    for m in methods:
        ari = method_ari([a for a in all_assignments if a.method == m], epitope_map)
        if ari is not None:
            per_method[m] = ari

    # 共识
    total_tcrs = len(set(a.tcr_id for a in all_assignments))
    clusters, weights = consensus_fn(all_assignments, methods)
    if not clusters:
        return None

    metrics = evaluate(clusters, epitope_map, total_tcrs)
    if metrics is None:
        return None

    return {
        "seed": seed,
        "config": label,
        "per_method_ari": per_method,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        **metrics,
    }


# ============ 主实验 ============
all_results = []

# 两组实验
EXPERIMENTS = [
    ("改进后", consensus_new, "new_cc_empirical"),
    ("改进前", consensus_old, "old_leiden_equal"),
]

for cond_label, fn, tag in EXPERIMENTS:
    print(f"\n{'=' * 70}", flush=True)
    print(f"  {cond_label}", flush=True)
    print(f"{'=' * 70}", flush=True)

    for i, seed in enumerate(SEEDS[:N_RUNS]):
        print(f"\n  --- Run {i+1}/{N_RUNS} (seed={seed}) ---", flush=True)
        t0 = time.time()
        try:
            res = run_once(seed, fn, tag)
            elapsed = time.time() - t0
            if res:
                all_results.append(res)
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


# ============ 汇总 ============
print(f"\n{'=' * 70}", flush=True)
print(f"  可复现性汇总", flush=True)
print(f"{'=' * 70}", flush=True)

new_data = [r for r in all_results if r["config"] == "new_cc_empirical"]
old_data = [r for r in all_results if r["config"] == "old_leiden_equal"]

for label, data in [
    ("改进后 (CC + 经验权重 + merge 0.6)", new_data),
    ("改进前 (Leiden + 等权 + merge 0.4)", old_data),
]:
    if not data:
        print(f"\n  {label}: 无结果", flush=True)
        continue

    aris = [r["ari"] for r in data]
    purs = [r["purity"] for r in data]
    nmis = [r["nmi"] for r in data]
    rets = [r["retention"] for r in data]
    cv = np.std(aris) / np.mean(aris) * 100 if np.mean(aris) > 0 else 0

    print(f"\n  {label} (n={len(data)}):", flush=True)
    print(
        f"    ARI:       {np.mean(aris):.4f} ± {np.std(aris):.4f}  "
        f"[{min(aris):.4f} – {max(aris):.4f}]  CV={cv:.1f}%",
        flush=True,
    )
    print(f"    Purity:    {np.mean(purs):.4f} ± {np.std(purs):.4f}", flush=True)
    print(f"    NMI:       {np.mean(nmis):.4f} ± {np.std(nmis):.4f}", flush=True)
    print(f"    Retention: {np.mean(rets):.4f} ± {np.std(rets):.4f}", flush=True)

    # 逐方法 ARI 稳定性
    all_methods = sorted(set(m for r in data for m in r["per_method_ari"]))
    print(f"    逐方法 ARI:", flush=True)
    for m in all_methods:
        vals = [r["per_method_ari"][m] for r in data if m in r["per_method_ari"]]
        if vals and np.mean(vals) > 0:
            mcv = np.std(vals) / np.mean(vals) * 100
            print(
                f"      {m:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (CV={mcv:.1f}%)",
                flush=True,
            )

# 改进 vs 对照统计检验
if new_data and old_data and len(new_data) >= 2 and len(old_data) >= 2:
    from scipy import stats as sp_stats

    ari_new = [r["ari"] for r in new_data]
    ari_old = [r["ari"] for r in old_data]
    t_stat, p_val = sp_stats.ttest_ind(ari_new, ari_old)
    wins = sum(1 for a, b in zip(ari_new, ari_old) if a > b)
    delta = np.mean(ari_new) - np.mean(ari_old)
    pct = delta / np.mean(ari_old) * 100 if np.mean(ari_old) > 0 else 0

    print(f"\n  改进后 vs 改进前:", flush=True)
    print(f"    ARI: {np.mean(ari_new):.4f} vs {np.mean(ari_old):.4f}", flush=True)
    print(f"    提升: +{delta:.4f} ({pct:+.1f}%)", flush=True)
    print(f"    t-test: t={t_stat:.3f}, p={p_val:.4f}", flush=True)
    print(f"    每次种子改进后胜出: {wins}/{min(len(ari_new), len(ari_old))}", flush=True)

    # Wilcoxon (非参数，适合 n=5)
    if len(ari_new) >= 3:
        try:
            w_stat, w_p = sp_stats.wilcoxon(ari_new, ari_old, alternative="greater")
            print(f"    Wilcoxon (one-sided): W={w_stat:.1f}, p={w_p:.4f}", flush=True)
        except Exception:
            pass

# 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "reproducibility_results.json"
with open(out_path, "w") as f:
    json.dump(
        {
            "meta": {
                "n_runs": N_RUNS,
                "seeds": SEEDS[:N_RUNS],
                "dataset": BENCHMARK,
                "improved": "CC + empirical_weights + merge 0.6",
                "baseline": "Leiden + equal_weights + merge 0.4",
            },
            "results": all_results,
            "summary": {
                "improved": {
                    "n": len(new_data),
                    "ari_mean": float(np.mean([r["ari"] for r in new_data])) if new_data else None,
                    "ari_std": float(np.std([r["ari"] for r in new_data])) if new_data else None,
                    "purity_mean": float(np.mean([r["purity"] for r in new_data])) if new_data else None,
                },
                "baseline": {
                    "n": len(old_data),
                    "ari_mean": float(np.mean([r["ari"] for r in old_data])) if old_data else None,
                    "ari_std": float(np.std([r["ari"] for r in old_data])) if old_data else None,
                    "purity_mean": float(np.mean([r["purity"] for r in old_data])) if old_data else None,
                },
            },
        },
        f, indent=2,
    )
print(f"\n  结果保存: {out_path}", flush=True)
print("=" * 70, flush=True)
print("实验完成", flush=True)
