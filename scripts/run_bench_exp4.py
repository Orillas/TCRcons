#!/usr/bin/env python3
"""Exp4: Adaptive Recommendation Generalization (leave-one-epitope-out).

Tests whether auto-selected mode generalizes to unseen epitopes.
Uses top epitopes with >= 50 TCRs.
"""

import sys, time, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tcrconsensus.io.parser import normalize
from tcrconsensus.config import load_config
from tcrconsensus.profiling.profiler import profile as compute_profile
from tcrconsensus.selection.selector import select_methods
from tcrconsensus.clusterers.hd_baseline import HDBaselineClusterer
from tcrconsensus.consensus.modes import balanced_consensus, conservative_consensus, coverage_consensus
from tcrconsensus.consensus.weights import compute_method_weights
from tcrconsensus.refinement.refiner import refine
from tcrconsensus.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/benchmark_data")
OUT_DIR = Path("/home/jilin/DeepTCR/tcrconsensus/results/bench_exp4")


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
    valid = np.array([str(p) not in ("-1", "") for p in pred_labels], dtype=bool)
    if valid.sum() < 2:
        return {}
    le_t, le_p = LabelEncoder(), LabelEncoder()
    true_str = true_labels[valid]
    pred_str = pred_labels[valid].astype(str)
    le_t.fit(np.unique(true_str))
    le_p.fit(np.unique(pred_str))
    return compute_all_metrics(le_p.transform(pred_str), le_t.transform(true_str), n_total)


def run_consensus(df_norm, clusterers, mode, config, workdir):
    all_a, methods = [], []
    for mname, cl in clusterers.items():
        r = cl.safe_execute(df_norm, workdir, config)
        if r.status.value == "success" and r.assignments:
            all_a.extend(r.assignments)
            methods.append(mname)
    if len(all_a) < 2:
        return []
    weights = compute_method_weights(methods, mode, config)
    fn = {"conservative": conservative_consensus, "coverage": coverage_consensus}.get(mode, balanced_consensus)
    clusters, edges = fn(all_a, weights)
    return refine(clusters, edges, config)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()._raw
    clusterers = get_clusterers()
    log.info(f"Clusterers: {list(clusterers.keys())}")

    df = pd.read_csv(BENCHMARK_DIR / "benchmark_main.tsv", sep="\t", dtype=str)
    df = df.rename(columns={"CDR3_beta": "cdr3_beta", "V_beta": "v_beta", "J_beta": "j_beta", "Epitope": "epitope"})

    # Only epitopes with >= 50 TCRs
    epi_counts = df["epitope"].value_counts()
    test_epis = epi_counts[epi_counts >= 50].index.tolist()
    log.info(f"Testing {len(test_epis)} epitopes (>= 50 TCRs)")

    all_results = []

    for held_epi in test_epis:
        test_df = df[df["epitope"] == held_epi].copy()
        train_df = df[df["epitope"] != held_epi].copy()
        n_test = len(test_df)
        log.info(f"\nHeld-out: {held_epi} ({n_test} TCRs)")

        test_norm = normalize(test_df.copy())
        tcr_ids = test_norm["tcr_id"].values
        true_labels = test_df["epitope"].values

        workdir = OUT_DIR / f"work/{held_epi[:30]}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Strategy 1: Auto-selected mode
        train_norm = normalize(train_df.copy())
        prof = compute_profile(train_norm, config)
        plan = select_methods(prof, "balanced", config)
        auto_mode = plan.consensus_mode.value

        t0 = time.time()
        clusters = run_consensus(test_norm, clusterers, auto_mode, config, workdir / "auto")
        elapsed = time.time() - t0
        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_test)
        m.update({"strategy": "auto_selected", "selected_mode": auto_mode,
                  "held_out_epitope": held_epi, "n_test": n_test, "runtime_s": elapsed})
        all_results.append(m)

        # Strategy 2: Fixed balanced
        clusters = run_consensus(test_norm, clusterers, "balanced", config, workdir / "balanced")
        pred = clusters_to_labels(clusters, tcr_ids)
        m = evaluate(pred, true_labels, n_test)
        m.update({"strategy": "fixed_balanced", "selected_mode": "balanced",
                  "held_out_epitope": held_epi, "n_test": n_test})
        all_results.append(m)

        # Strategy 3: Single HD baseline
        hd = HDBaselineClusterer()
        r = hd.safe_execute(test_norm, workdir / "hd", config)
        if r.status.value == "success" and r.assignments:
            label_map = {}
            for a in r.assignments:
                if a.tcr_id not in label_map:
                    label_map[a.tcr_id] = a.cluster_id
            pred = np.array([label_map.get(tid, -1) for tid in tcr_ids])
            m = evaluate(pred, true_labels, n_test)
            m.update({"strategy": "single_hd_baseline", "selected_mode": "single",
                      "held_out_epitope": held_epi, "n_test": n_test})
            all_results.append(m)

    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "generalization_results.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("EXP4: GENERALIZATION RESULTS")
    print("=" * 80)
    summary = res.groupby("strategy").agg({
        "ari": ["mean", "std"], "ami": ["mean", "std"],
        "purity": ["mean", "std"], "sensitivity": ["mean", "std"],
    }).reset_index()
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    run()
