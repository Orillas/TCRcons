#!/usr/bin/env python3
"""Exp1: Cross-Benchmark Comparison on benchmark dataset (28,021 TCRs, 560 epitopes).

Runs all single methods + consensus 3 modes on the high-confidence benchmark.
"""

import sys, time, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/benchmark_data")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/bench_exp1")


def get_clusterers():
    clusterers = {"hd_baseline": HDBaselineClusterer()}
    for name, mod in [("clustcr", "clustcr_wrapper"), ("tcrdist3", "tcrdist3_wrapper"),
                      ("gliph2", "gliph2_wrapper"), ("giana", "giana_wrapper"),
                      ("tcrmatch", "tcrmatch_wrapper")]:
        try:
            m = __import__(f"tcrconsensus.clusterers.{mod}", fromlist=[name.title().replace("_","")+"Wrapper"])
            cls = getattr(m, [c for c in dir(m) if "Wrapper" in c][0])
            clusterers[name] = cls()
        except Exception as e:
            log.warning(f"{name}: not available ({e})")
    return clusterers


def clusters_to_labels(clusters, tcr_ids):
    label_map = {}
    for c in clusters:
        for mid in c.member_ids:
            label_map[mid] = c.cluster_id
    return np.array([label_map.get(tid, -1) for tid in tcr_ids])


def evaluate(pred_labels, true_labels, n_total):
    valid = np.array([str(p) not in ("-1", "") for p in pred_labels], dtype=bool)
    if valid.sum() < 2:
        return {}
    le_t, le_p = LabelEncoder(), LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    return compute_all_metrics(le_p.transform(pred_str), le_t.transform(true_str), n_total)


def load_benchmark():
    """Load benchmark with column rename for normalize()."""
    df = pd.read_csv(BENCHMARK_DIR / "benchmark_main.tsv", sep="\t", dtype=str)
    df = df.rename(columns={
        "CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta",
        "CDR3_alpha": "cdr3_alpha", "V_alpha": "v_alpha", "J_alpha": "j_alpha",
        "Epitope": "epitope",
    })
    return df


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    df = load_benchmark()
    true_labels = df["epitope"].values
    tcr_ids = df["tcr_id"].values
    n_total = len(df)
    df_norm = normalize(df.copy())
    log.info(f"Dataset: {n_total} TCRs, {df['epitope'].nunique()} epitopes")

    workdir = OUT_DIR / "work"
    workdir.mkdir(exist_ok=True)

    # --- Run single methods ---
    all_results = []
    method_assignments = {}

    for mname, clusterer in clusterers.items():
        log.info(f"Running {mname}...")
        t0 = time.time()
        result = clusterer.safe_execute(df_norm, workdir / mname, config)
        elapsed = time.time() - t0

        if result.status.value == "success" and result.assignments:
            label_map = {}
            for a in result.assignments:
                if a.tcr_id not in label_map:
                    label_map[a.tcr_id] = a.cluster_id
            pred = np.array([label_map.get(tid, -1) for tid in tcr_ids])
            m = evaluate(pred, true_labels, n_total)
            m["method"] = mname
            m["runtime_s"] = elapsed
            m["n_clusters"] = len(set(label_map.values()))
            m["n_assigned"] = int(sum(1 for p in pred if str(p) not in ("-1", "")))
            all_results.append(m)
            method_assignments[mname] = result.assignments
            log.info(f"  {mname}: ARI={m.get('ari',0):.4f}, clusters={m['n_clusters']}, assigned={m['n_assigned']}")
        else:
            log.warning(f"  {mname}: FAILED")

    # --- Run consensus modes ---
    if len(method_assignments) >= 2:
        methods = list(method_assignments.keys())
        all_a = []
        for a in method_assignments.values():
            all_a.extend(a)

        for mode_name, mode_fn in [("consensus_conservative", conservative_consensus),
                                    ("consensus_balanced", balanced_consensus),
                                    ("consensus_coverage", coverage_consensus)]:
            log.info(f"Running {mode_name}...")
            weights = compute_method_weights(methods, mode_name.split("_")[-1], config)
            t0 = time.time()
            clusters, edges = mode_fn(all_a, weights)
            clusters = refine(clusters, edges, config)
            elapsed = time.time() - t0

            pred = clusters_to_labels(clusters, tcr_ids)
            m = evaluate(pred, true_labels, n_total)
            m["method"] = mode_name
            m["runtime_s"] = elapsed
            m["n_clusters"] = len(clusters)
            m["n_methods_used"] = len(methods)
            all_results.append(m)
            log.info(f"  {mode_name}: ARI={m.get('ari',0):.4f}, clusters={len(clusters)}")

    # --- Save ---
    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "cross_benchmark_results.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP1: CROSS-BENCHMARK RESULTS")
    print("=" * 80)
    cols = ["method", "ari", "ami", "nmi", "purity", "sensitivity", "retention", "v_measure", "n_clusters", "n_assigned"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
