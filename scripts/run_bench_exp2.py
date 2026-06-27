#!/usr/bin/env python3
"""Exp2: Background Robustness using 10X noise subsets.

6 subsets with increasing noise ratio (98.9% - 99.6%).
Also tests per-epitope background injection on benchmark data.
"""

import sys, time, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/benchmark_data")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/bench_exp2")


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    for name, mod in [("clustcr", "clustcr_wrapper"), ("tcrdist3", "tcrdist3_wrapper"),
                      ("gliph2", "gliph2_wrapper")]:
        try:
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=["X"])
            cls = getattr(m, [c for c in dir(m) if "Wrapper" in c][0])
            clusterers[name] = cls()
        except: pass
    return clusterers


def clusters_to_labels(clusters, tcr_ids):
    label_map = {}
    for c in clusters:
        for mid in c.member_ids:
            label_map[mid] = c.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def evaluate(pred_labels, true_labels, n_total):
    valid = np.array([
        str(p) not in ("-1", "") and str(t) != "BACKGROUND"
        for p, t in zip(pred_labels, true_labels)
    ], dtype=bool)
    if valid.sum() < 2:
        return {}

    le_t, le_p = LabelEncoder(), LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    metrics = compute_all_metrics(le_p.transform(pred_str), le_t.transform(true_str), n_total)

    # False recruitment
    bg_mask = true_labels == "BACKGROUND"
    if bg_mask.sum() > 0:
        bg_pred = pred_labels[bg_mask]
        clustered_bg = sum(1 for p in bg_pred if str(p) not in ("-1", ""))
        metrics["false_recruitment_rate"] = clustered_bg / bg_mask.sum()
        metrics["n_bg_clustered"] = float(clustered_bg)
        metrics["n_bg_total"] = float(bg_mask.sum())

    return metrics


def run_consensus(df_norm, clusterers, config, workdir):
    all_a, methods = [], []
    for mname, cl in clusterers.items():
        r = cl.safe_execute(df_norm, workdir, config)
        if r.status.value == "success" and r.assignments:
            all_a.extend(r.assignments)
            methods.append(mname)
    if len(all_a) < 2:
        return [], {}
    weights = compute_method_weights(methods, "balanced", config)
    clusters, edges = balanced_consensus(all_a, weights)
    return refine(clusters, edges, config), {m: "ok" for m in methods}


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    # === Part A: 10X Noise Subsets ===
    log.info("=" * 60)
    log.info("Part A: 10X Noise Subsets")
    log.info("=" * 60)

    all_results = []
    for i in range(1, 7):
        sub = pd.read_csv(BENCHMARK_DIR / f"10x_subset_{i}.tsv", sep="\t", dtype=str)
        log.info(f"\nSubset {i}: {len(sub)} TCRs, {sub['is_signal'].sum()} signal")

        df_run = sub.rename(columns={"cdr3": "cdr3_beta"}).copy()
        if "v_beta" not in df_run.columns:
            df_run["v_beta"] = ""
        if "j_beta" not in df_run.columns:
            df_run["j_beta"] = ""

        tcr_ids = df_run["tcr_id"].values
        true_labels = sub["epitope"].values
        n_total = len(df_run)
        df_norm = normalize(df_run.copy())

        workdir = OUT_DIR / f"work/10x_subset_{i}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Consensus
        t0 = time.time()
        clusters, status = run_consensus(df_norm, clusterers, config, workdir / "consensus")
        elapsed = time.time() - t0
        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_total)
        m.update({"subset": i, "method": "consensus_balanced", "n_total": n_total,
                  "n_signal": int(sub["is_signal"].sum()), "n_bg": int((~sub["is_signal"].astype(bool)).sum()),
                  "noise_pct": float((~sub["is_signal"].astype(bool)).sum()) / n_total * 100,
                  "runtime_s": elapsed, "n_clusters": len(clusters)})
        all_results.append(m)
        log.info(f"  Consensus: ARI={m.get('ari',0):.4f}, FRR={m.get('false_recruitment_rate',0):.4f}, clusters={len(clusters)}")

        # HD baseline for comparison
        hd = HDBaselineClusterer()
        r = hd.safe_execute(df_norm, workdir / "hd", config)
        if r.status.value == "success" and r.assignments:
            label_map = {}
            for a in r.assignments:
                if a.tcr_id not in label_map:
                    label_map[a.tcr_id] = a.cluster_id
            pred_hd = np.array([label_map.get(tid, -1) for tid in tcr_ids])
            m_hd = evaluate(pred_hd, true_labels, n_total)
            m_hd.update({"subset": i, "method": "hd_baseline_only", "n_total": n_total,
                         "n_signal": int(sub["is_signal"].sum()), "n_bg": int((~sub["is_signal"].astype(bool)).sum()),
                         "noise_pct": float((~sub["is_signal"].astype(bool)).sum()) / n_total * 100})
            all_results.append(m_hd)

    # === Part B: Per-epitope background injection ===
    log.info("\n" + "=" * 60)
    log.info("Part B: Per-epitope background injection")
    log.info("=" * 60)

    bench = pd.read_csv(BENCHMARK_DIR / "benchmark_main.tsv", sep="\t", dtype=str)
    bench = bench.rename(columns={"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta", "Epitope": "epitope"})

    # Use non-signal 10X TCRs as background
    bg_pool = pd.read_csv(BENCHMARK_DIR / "10x_subset_6.tsv", sep="\t", dtype=str)
    bg_pool = bg_pool[~bg_pool["is_signal"].astype(bool)].copy()
    bg_pool = bg_pool.rename(columns={"cdr3": "cdr3_beta"})
    bg_pool["v_beta"] = ""
    bg_pool["j_beta"] = ""
    bg_pool["epitope"] = "BACKGROUND"
    bg_pool["tcr_id"] = ["bg_" + str(i).zfill(6) for i in range(len(bg_pool))]
    log.info(f"Background pool: {len(bg_pool)} TCRs from 10X subset_6")

    # Top 10 epitopes
    epi_counts = bench["epitope"].value_counts()
    top_epis = epi_counts.head(10).index.tolist()

    ratios = [10, 100, 1000]
    for epitope in top_epis:
        signal = bench[bench["epitope"] == epitope].copy()
        n_signal = len(signal)
        if n_signal < 20:
            continue

        for ratio in ratios:
            n_bg = min(int(n_signal * ratio), len(bg_pool))
            bg_sample = bg_pool.sample(n=n_bg, replace=False)

            mixed = pd.concat([
                signal[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
                bg_sample[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "epitope"]],
            ], ignore_index=True)

            true_labels = mixed["epitope"].values
            tcr_ids = mixed["tcr_id"].values
            n_total = len(mixed)
            df_norm = normalize(mixed.copy())

            workdir = OUT_DIR / f"work/inject/{epitope[:20]}_1{ratio}"
            workdir.mkdir(parents=True, exist_ok=True)

            clusters, _ = run_consensus(df_norm, clusterers, config, workdir)
            pred = clusters_to_labels(clusters, tcr_ids)
            m = evaluate(pred, true_labels, n_total)
            m.update({"epitope": epitope, "ratio": f"1:{ratio}", "n_signal": n_signal,
                      "n_background": n_bg, "n_total": n_total, "method": "consensus_balanced",
                      "n_clusters": len(clusters)})
            all_results.append(m)
            log.info(f"  {epitope} 1:{ratio}: ARI={m.get('ari',0):.4f}, FRR={m.get('false_recruitment_rate',0):.4f}")

    # Save
    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "robustness_results.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP2: BACKGROUND ROBUSTNESS RESULTS")
    print("=" * 80)
    print("\n--- 10X Noise Subsets ---")
    tenx = res[res["subset"].notna()] if "subset" in res.columns else pd.DataFrame()
    if len(tenx) > 0:
        print(tenx[["subset", "method", "ari", "purity", "false_recruitment_rate", "n_clusters"]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
