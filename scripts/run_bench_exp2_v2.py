#!/usr/bin/env python3
"""Exp2: Background Robustness Stress Test on core benchmark (4,779 pairs).

Part A: Per-epitope background injection at 1:10/100/1000 ratios.
Uses top epitopes from the core benchmark dataset.
"""

import sys, time, logging, random
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
    for name, mod in [("clustcr", "clustcr_wrapper"), ("gliph2", "gliph2_wrapper")]:
        try:
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=[name.title().replace("_","")+"Wrapper"])
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


def run_consensus(df_norm, clusterers, config, workdir):
    all_a = []
    for mname, clusterer in clusterers.items():
        r = clusterer.safe_execute(df_norm, workdir, config)
        if r.status.value == "success" and r.assignments:
            all_a.extend(r.assignments)
    if len(all_a) < 2:
        return []
    methods = list(set(a.method for a in all_a))
    weights = compute_method_weights(methods, "balanced", config)
    clusters, edges = balanced_consensus(all_a, weights)
    return refine(clusters, edges, config)


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

    bg_mask = true_labels == "BACKGROUND"
    if bg_mask.sum() > 0:
        bg_pred = pred_labels[bg_mask]
        clustered_bg = sum(1 for p in bg_pred if str(p) not in ("-1", ""))
        metrics["false_recruitment_rate"] = clustered_bg / bg_mask.sum()
    else:
        metrics["false_recruitment_rate"] = 0.0
    return metrics


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    # Load core benchmark
    df = pd.read_csv(BENCHMARK_DIR / "benchmark_core_4779.tsv", sep="\t", dtype=str)
    rename_map = {"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta",
                  "CDR3_alpha": "cdr3_alpha", "V_alpha": "v_alpha", "J_alpha": "j_alpha",
                  "Epitope": "epitope"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Top 10 epitopes by count
    epi_counts = df["epitope"].value_counts()
    top_epis = epi_counts.head(10).index.tolist()
    log.info(f"Top 10 epitopes: {top_epis}")

    # Background pool: sample from other epitopes' TCRs
    bg_pool = df[~df["epitope"].isin(top_epis)].copy()
    bg_pool["epitope"] = "BACKGROUND"
    bg_pool["tcr_id"] = ["bg_" + str(i).zfill(6) for i in range(len(bg_pool))]
    log.info(f"Background pool: {len(bg_pool)} TCRs from non-target epitopes")

    ratios = [10, 100, 1000]
    all_results = []

    for epitope in top_epis:
        signal = df[df["epitope"] == epitope].copy()
        n_signal = len(signal)
        if n_signal < 10:
            continue

        log.info(f"\nEpitope: {epitope} ({n_signal} signal TCRs)")

        for ratio in ratios:
            n_bg = min(int(n_signal * ratio), len(bg_pool))
            bg_sample = bg_pool.sample(n=n_bg, replace=False, random_state=42).copy()

            mixed = pd.concat([
                signal[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "cdr3_alpha", "v_alpha", "j_alpha", "epitope"]],
                bg_sample[["tcr_id", "cdr3_beta", "v_beta", "j_beta", "cdr3_alpha", "v_alpha", "j_alpha", "epitope"]],
            ], ignore_index=True)

            true_labels = mixed["epitope"].values
            n_total = len(mixed)
            df_norm = normalize(mixed.copy())

            workdir = OUT_DIR / f"work/{epitope}_{ratio}"
            workdir.mkdir(parents=True, exist_ok=True)

            log.info(f"  Ratio 1:{ratio} ({n_signal} signal + {n_bg} bg = {n_total})")

            t0 = time.time()
            clusters = run_consensus(df_norm, clusterers, config, workdir)
            elapsed = time.time() - t0

            pred = clusters_to_labels(clusters, df_norm["tcr_id"].values)
            metrics = evaluate(pred, true_labels, n_total)
            metrics.update({
                "epitope": epitope, "ratio": f"1:{ratio}", "n_signal": n_signal,
                "n_background": n_bg, "n_total": n_total, "runtime_s": elapsed,
                "n_clusters": len(clusters), "method": "consensus_balanced",
            })
            all_results.append(metrics)

            # Also run HD baseline alone
            hd = HDBaselineClusterer()
            r = hd.safe_execute(df_norm, workdir / "hd", config)
            if r.status.value == "success" and r.assignments:
                label_map = {}
                for a in r.assignments:
                    if a.tcr_id not in label_map:
                        label_map[a.tcr_id] = a.cluster_id
                pred_hd = np.array([label_map.get(tid, -1) for tid in df_norm["tcr_id"].values])
                hd_m = evaluate(pred_hd, true_labels, n_total)
                hd_m.update({"epitope": epitope, "ratio": f"1:{ratio}", "n_signal": n_signal,
                             "n_background": n_bg, "n_total": n_total, "method": "hd_baseline_only"})
                all_results.append(hd_m)

    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "robustness_results.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP2: BACKGROUND ROBUSTNESS RESULTS (core benchmark 4,779 pairs)")
    print("=" * 80)
    print(res.groupby(["epitope", "ratio", "method"]).agg({
        "ari": "mean", "purity": "mean", "retention": "mean",
        "false_recruitment_rate": "mean"
    }).to_string())
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
